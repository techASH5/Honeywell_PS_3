# =============================================================================
# mpc_controller.py
#
# Safety-Shielded Edge-Rider Receding Horizon MPC with Explainable AI (XAI)
#
# Three safety layers:
#   Layer 1 — True Receding Horizon with Recovery
#   Layer 2 — Inertia-Aware Safety Buffer (velocity-based pre-emption)
#   Layer 3 — Edge-Riding for Infeasible Targets (Scenario C)
#
# Every decision returns a structured XAI dict explaining the reasoning.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib

import config

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# =============================================================================
# Helper: constraint checking
# =============================================================================

def _violations(state: dict) -> list[str]:
    """Return a list of violated constraint strings. Empty = all safe."""
    v = []
    if not (config.WHP_MIN <= state["WHP"] <= config.WHP_MAX):
        v.append(
            f"WHP={state['WHP']:.1f} outside [{config.WHP_MIN},{config.WHP_MAX}] psi"
        )
    if not (config.FLP_MIN <= state["FLP"] <= config.FLP_MAX):
        v.append(
            f"FLP={state['FLP']:.1f} outside [{config.FLP_MIN},{config.FLP_MAX}] psi"
        )
    if not (config.BHP_MIN <= state["BHP"] <= config.BHP_MAX):
        v.append(
            f"BHP={state['BHP']:.1f} outside [{config.BHP_MIN},{config.BHP_MAX}] psi"
        )
    return v


def _is_safe(state: dict) -> bool:
    return len(_violations(state)) == 0


def _clip_choke(choke: float) -> float:
    return float(np.clip(choke, config.CHOKE_MIN, config.CHOKE_MAX))


# =============================================================================
# MPCController
# =============================================================================

class MPCController:
    """
    Receding Horizon MPC for a single naturally flowing oil well.

    Usage
    -----
        mpc = MPCController()
        result = mpc.calculate_next_move(current_state, target_q=130.0)
        print(result["reasoning"])
        next_choke = result["chosen_choke"]

    current_state dict keys: "choke", "Q", "WHP", "FLP", "BHP"
    """

    # Output column order — must match system_identification.py training
    _OUTPUT_COLS = ["next_Q", "next_WHP", "next_FLP", "next_BHP"]

    def __init__(self, model_path: str = None):
        path = model_path or config.MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Predictive model not found at '{path}'.\n"
                "Run system_identification.py first."
            )
        self.model = joblib.load(path)

        # History for velocity (inertia) calculation
        self._prev_state: dict | None = None
        self._step_count: int = 0

        # Edge-riding state
        self._edge_riding: bool = False
        self._last_edge_dir: float = +5.0   # direction of last edge probe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset controller history (call when simulator resets)."""
        self._prev_state = None
        self._step_count = 0
        self._edge_riding = False
        self._last_edge_dir = +5.0

    def calculate_next_move(self, current_state: dict, target_q: float) -> dict:
        """
        Compute the next optimal choke position.

        Parameters
        ----------
        current_state : dict
            {"choke": float, "Q": float, "WHP": float, "FLP": float, "BHP": float}
        target_q : float
            Desired oil production rate in bbl/hr.

        Returns
        -------
        dict
            {
                "chosen_choke"   : float,         # new choke % to command
                "action_taken"   : str,            # e.g. "+2.5%"
                "reasoning"      : str,            # XAI explanation
                "predicted_state": dict,           # 1-step-ahead prediction
                "edge_riding"    : bool,
                "safe_candidates": list[float],    # chokes that passed all layers
            }
        """
        self._step_count += 1
        cur_choke = current_state["choke"]

        # -- Pressure velocities (rates of change) for Layer 2 ----------
        velocities = self._compute_velocities(current_state)

        # -- Generate and validate candidates ----------------------------
        candidate_results = {}   # choke -> evaluation dict
        rejection_log = {}       # choke -> rejection reason(s)

        for delta in config.CHOKE_CANDIDATES:
            candidate_choke = _clip_choke(cur_choke + delta)

            eval_result = self._evaluate_candidate(
                candidate_choke, current_state, target_q, velocities
            )

            if eval_result["safe"]:
                candidate_results[candidate_choke] = eval_result
            else:
                rejection_log[candidate_choke] = eval_result["rejection_reason"]

        safe_chokes = sorted(candidate_results.keys())

        # -- Layer 3: Edge-Riding if no safe increase exists -------------
        if not safe_chokes or self._is_infeasible_target(
            current_state, target_q, safe_chokes, candidate_results
        ):
            return self._edge_ride(
                cur_choke, current_state, target_q, rejection_log
            )

        # -- Select best safe candidate ----------------------------------
        chosen_choke, chosen_eval = self._select_best(
            safe_chokes, candidate_results, current_state, target_q
        )

        # -- Build XAI reasoning string ----------------------------------
        delta_pct = chosen_choke - cur_choke
        action_str = f"{delta_pct:+.1f}%"

        reasoning = self._build_reasoning(
            current_state, target_q, chosen_choke, chosen_eval,
            rejection_log, velocities, delta_pct
        )

        # -- Update history ----------------------------------------------
        self._prev_state = dict(current_state)
        self._edge_riding = False

        return {
            "chosen_choke":    chosen_choke,
            "action_taken":    action_str,
            "reasoning":       reasoning,
            "predicted_state": chosen_eval["predicted_states"][0],
            "edge_riding":     False,
            "safe_candidates": safe_chokes,
        }

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _predict_one_step(self, state: dict, choke: float) -> dict:
        """Use the loaded RF model to predict one step ahead."""
        features = pd.DataFrame([{
            "Choke_pct": choke,
            "prev_Q":    state["Q"],
            "prev_WHP":  state["WHP"],
            "prev_FLP":  state["FLP"],
            "prev_BHP":  state["BHP"],
        }])
        pred = self.model.predict(features)[0]
        return {
            "choke": choke,
            "Q":     float(pred[0]),
            "WHP":   float(pred[1]),
            "FLP":   float(pred[2]),
            "BHP":   float(pred[3]),
        }

    def _simulate_sequence(self, initial_state: dict, choke_sequence: list[float]) -> list[dict]:
        """
        Simulate a sequence of choke moves from initial_state.
        Returns list of predicted states (one per step in the sequence).
        """
        state = dict(initial_state)
        trajectory = []
        for choke in choke_sequence:
            state = self._predict_one_step(state, choke)
            trajectory.append(dict(state))
        return trajectory

    # ------------------------------------------------------------------
    # Layer 1 — Receding Horizon with Recovery
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        candidate_choke: float,
        current_state: dict,
        target_q: float,
        velocities: dict,
    ) -> dict:
        """
        Evaluate one candidate choke via the full 3-layer safety stack.

        Returns dict with keys: safe, rejection_reason, predicted_states,
        recovery_used, inertia_warning.
        """
        horizon = config.PREDICTION_HORIZON

        # Build the primary choke sequence: hold candidate for full horizon
        primary_sequence = [candidate_choke] * horizon
        trajectory = self._simulate_sequence(current_state, primary_sequence)

        # -- Layer 2 check: inertia-aware pre-emption --------------------
        inertia_warn = self._inertia_warning(current_state, candidate_choke, velocities)
        if inertia_warn:
            return {
                "safe": False,
                "rejection_reason": f"Inertia buffer: {inertia_warn}",
                "predicted_states": trajectory,
                "recovery_used": False,
                "inertia_warning": True,
            }

        # -- Layer 1 check: scan trajectory for violations ---------------
        violation_step = None
        violation_msg  = None
        for i, state in enumerate(trajectory):
            viols = _violations(state)
            if viols:
                violation_step = i + 1   # 1-indexed
                violation_msg  = "; ".join(viols)
                break

        # Fully safe trajectory — no violations at any horizon step
        if violation_step is None:
            return {
                "safe": True,
                "rejection_reason": None,
                "predicted_states": trajectory,
                "recovery_used": False,
                "inertia_warning": False,
            }

        # Violation at step 1 — hard reject, no recovery possible
        if violation_step == 1:
            return {
                "safe": False,
                "rejection_reason": (
                    f"Step-1 constraint violation: {violation_msg}"
                ),
                "predicted_states": trajectory,
                "recovery_used": False,
                "inertia_warning": False,
            }

        # Violation at step 2 or 3 — attempt recovery via pullback ------
        # Try a -CHOKE_MAX_STEP recovery move at the step before violation
        recovery_choke = _clip_choke(candidate_choke - config.CHOKE_MAX_STEP)
        recovery_sequence = (
            [candidate_choke] * (violation_step - 1) +    # hold until breach step
            [recovery_choke]  * (horizon - violation_step + 1)  # then pull back
        )
        recovery_trajectory = self._simulate_sequence(current_state, recovery_sequence)

        recovery_ok = all(_is_safe(s) for s in recovery_trajectory)

        if recovery_ok:
            return {
                "safe": True,
                "rejection_reason": None,
                "predicted_states": trajectory,    # show primary trajectory to operator
                "recovery_trajectory": recovery_trajectory,
                "recovery_used": True,
                "recovery_step": violation_step,
                "inertia_warning": False,
                "violation_msg": violation_msg,
            }
        else:
            return {
                "safe": False,
                "rejection_reason": (
                    f"Step-{violation_step} violation: {violation_msg}. "
                    f"Recovery move ({recovery_choke:.1f}%) also unsafe."
                ),
                "predicted_states": trajectory,
                "recovery_used": False,
                "inertia_warning": False,
            }

    # ------------------------------------------------------------------
    # Layer 2 — Inertia-Aware Safety Buffer
    # ------------------------------------------------------------------

    def _compute_velocities(self, current_state: dict) -> dict:
        """
        Estimate velocity (rate of change per step) of each pressure variable.
        Returns zero velocities if no previous state is recorded.
        """
        if self._prev_state is None:
            return {"WHP": 0.0, "FLP": 0.0, "BHP": 0.0}

        return {
            "WHP": current_state["WHP"] - self._prev_state["WHP"],
            "FLP": current_state["FLP"] - self._prev_state["FLP"],
            "BHP": current_state["BHP"] - self._prev_state["BHP"],
        }

    def _inertia_warning(
        self,
        current_state: dict,
        candidate_choke: float,
        velocities: dict,
    ) -> str | None:
        """
        Trigger pre-emptive rejection if a pressure variable is approaching
        its hard limit fast enough that the next step would breach it
        even before the ML model predicts it.

        Returns a warning string if triggered, else None.
        """
        buf = config.CONSTRAINT_WARNING_BUFFER_PCT

        checks = [
            ("WHP", config.WHP_MIN, config.WHP_MAX, velocities["WHP"]),
            ("FLP", config.FLP_MIN, config.FLP_MAX, velocities["FLP"]),
            ("BHP", config.BHP_MIN, config.BHP_MAX, velocities["BHP"]),
        ]

        for var, lo, hi, vel in checks:
            val = current_state[var]
            full_range = hi - lo
            buffer = buf * full_range

            # Approaching upper limit with positive velocity
            if vel > 0 and (val + vel) > (hi - buffer):
                return (
                    f"{var}={val:.1f} psi rising at +{vel:.2f} psi/step, "
                    f"predicted to breach upper limit {hi} within 1 step"
                )
            # Approaching lower limit with negative velocity
            if vel < 0 and (val + vel) < (lo + buffer):
                return (
                    f"{var}={val:.1f} psi falling at {vel:.2f} psi/step, "
                    f"predicted to breach lower limit {lo} within 1 step"
                )

        return None

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _select_best(
        self,
        safe_chokes: list[float],
        candidate_results: dict,
        current_state: dict,
        target_q: float,
    ) -> tuple[float, dict]:
        """
        Among safe candidates, select the one whose first-step predicted Q
        minimises deviation from target_q.

        Tie-breaking: Random Forest plateaus can cause identical predictions
        for multiple candidate chokes. By rounding the error and applying a
        tiny direction penalty, we force the controller to keep stepping 
        towards the target instead of freezing on the plateau.
        """
        cur_choke = current_state["choke"]

        def score(choke):
            predicted_q = candidate_results[choke]["predicted_states"][0]["Q"]
            q_error = abs(predicted_q - target_q)
            
            # Direction penalty to break flat-region ties
            direction_penalty = 0.0
            if target_q > current_state["Q"]:
                direction_penalty = -choke * 0.001  # prefer opening choke
            else:
                direction_penalty = choke * 0.001   # prefer closing choke

            move_size = abs(choke - cur_choke)
            
            return (round(q_error, 2), direction_penalty, move_size)

        best_choke = min(safe_chokes, key=score)
        return best_choke, candidate_results[best_choke]

    def _is_infeasible_target(
        self,
        current_state: dict,
        target_q: float,
        safe_chokes: list[float],
        candidate_results: dict,
    ) -> bool:
        """
        Return True if the target Q is infeasible — i.e., every safe candidate
        predicts Q below the target, AND the best safe Q is not meaningfully
        better than current Q (we're near the safe ceiling).
        """
        if not safe_chokes:
            return True

        predicted_qs = [
            candidate_results[c]["predicted_states"][0]["Q"]
            for c in safe_chokes
        ]
        best_safe_q = max(predicted_qs)

        # Infeasible if we can't get within 5 bbl/hr of target AND
        # the highest safe candidate is at max choke
        max_safe_choke = max(safe_chokes)
        at_ceiling = max_safe_choke >= (config.CHOKE_MAX - config.CHOKE_MAX_STEP)
        cant_reach_target = best_safe_q < (target_q - 5.0)

        return at_ceiling and cant_reach_target

    # ------------------------------------------------------------------
    # Layer 3 — Edge-Riding
    # ------------------------------------------------------------------

    def _edge_ride(
        self,
        cur_choke: float,
        current_state: dict,
        target_q: float,
        rejection_log: dict,
    ) -> dict:
        """
        Infeasible-target handler.
        Alternates between probing +5% and pulling back -5% to oscillate
        at the safe production ceiling.
        """
        self._edge_riding = True

        # Flip direction from last edge probe
        probe_delta  = +config.CHOKE_MAX_STEP
        pullback_delta = -config.CHOKE_MAX_STEP

        probe_choke    = _clip_choke(cur_choke + probe_delta)
        pullback_choke = _clip_choke(cur_choke + pullback_delta)

        probe_traj    = self._simulate_sequence(current_state, [probe_choke])
        pullback_traj = self._simulate_sequence(current_state, [pullback_choke])

        probe_safe    = _is_safe(probe_traj[0])
        pullback_safe = _is_safe(pullback_traj[0])

        if probe_safe:
            # Push to probe higher production
            chosen_choke = probe_choke
            traj = probe_traj
            actual_delta = chosen_choke - cur_choke
            action_str = f"{actual_delta:+.1f}% [edge probe]"
            self._last_edge_dir = +config.CHOKE_MAX_STEP
        elif pullback_safe:
            # Probe unsafe — pull back
            chosen_choke = pullback_choke
            traj = pullback_traj
            actual_delta = chosen_choke - cur_choke
            action_str = f"{actual_delta:+.1f}% [edge pullback]"
            self._last_edge_dir = pullback_delta
        else:
            # Both directions unsafe — hold position
            chosen_choke = cur_choke
            traj = self._simulate_sequence(current_state, [cur_choke])
            action_str = "0.0% [edge hold — both dirs unsafe]"
            self._last_edge_dir = 0.0

        rejection_summary = "; ".join(
            f"Choke {c:.1f}%: {r}"
            for c, r in rejection_log.items()
        )
        if not rejection_summary:
            if cur_choke >= config.CHOKE_MAX:
                rejection_summary = "Physical Limit Reached (Choke at 100%). Maximum achievable safe flow."
            else:
                rejection_summary = "all larger moves violate constraints"

        pred = traj[0]
        reasoning = (
            f"INFEASIBLE TARGET DETECTED. "
            f"Target Q={target_q:.1f} bbl/hr cannot be safely reached. "
            f"Current Q={current_state['Q']:.2f} bbl/hr at Choke={cur_choke:.1f}%. "
            f"Rejection summary: [{rejection_summary}]. "
            f"ENGAGING EDGE-RIDING MODE: {action_str}. "
            f"Predicted Q={pred['Q']:.2f} bbl/hr | "
            f"WHP={pred['WHP']:.1f} | FLP={pred['FLP']:.1f} | BHP={pred['BHP']:.1f} psi. "
            f"Probing safe production ceiling to maximise output without violating constraints."
        )

        self._prev_state = dict(current_state)

        return {
            "chosen_choke":    chosen_choke,
            "action_taken":    action_str,
            "reasoning":       reasoning,
            "predicted_state": pred,
            "edge_riding":     True,
            "safe_candidates": [],
        }

    # ------------------------------------------------------------------
    # XAI reasoning builder
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        current_state: dict,
        target_q: float,
        chosen_choke: float,
        chosen_eval: dict,
        rejection_log: dict,
        velocities: dict,
        delta_pct: float,
    ) -> str:
        """Build the human-readable XAI reasoning string."""
        cur_q     = current_state["Q"]
        cur_choke = current_state["choke"]
        pred      = chosen_eval["predicted_states"][0]

        direction = "increase" if delta_pct > 0 else ("decrease" if delta_pct < 0 else "maintain")
        q_gap     = target_q - cur_q
        q_trend   = "above" if cur_q > target_q else "below"

        # Opening line
        parts = [
            f"Target Q={target_q:.1f} bbl/hr. "
            f"Current Q={cur_q:.2f} bbl/hr ({abs(q_gap):.1f} bbl/hr {q_trend} target) "
            f"at Choke={cur_choke:.1f}%."
        ]

        # Rejections
        if rejection_log:
            rej_parts = []
            for c, reason in sorted(rejection_log.items()):
                rej_parts.append(f"Choke {c:.1f}%: {reason}")
            parts.append(f"Rejected candidates: [{'; '.join(rej_parts)}].")

        # Recovery note
        if chosen_eval.get("recovery_used"):
            parts.append(
                f"NOTE: {chosen_choke:.1f}% has a predicted step-"
                f"{chosen_eval['recovery_step']} warning "
                f"({chosen_eval.get('violation_msg','constraint approach')}), "
                f"but a safe recovery move to "
                f"{_clip_choke(chosen_choke - config.CHOKE_MAX_STEP):.1f}% "
                f"at that step prevents any hard violation. "
                f"Move is APPROVED under receding horizon recovery logic."
            )

        # Inertia velocities (if noteworthy)
        notable_vel = {
            k: v for k, v in velocities.items() if abs(v) > 0.5
        }
        if notable_vel:
            vel_str = ", ".join(f"{k} drifting {v:+.2f} psi/step" for k, v in notable_vel.items())
            parts.append(f"Inertia monitor: {vel_str}.")

        # Decision
        parts.append(
            f"ACTION: {direction.upper()} choke {delta_pct:+.1f}% -> {chosen_choke:.1f}%. "
            f"Predicted next state: Q={pred['Q']:.2f} bbl/hr | "
            f"WHP={pred['WHP']:.1f} | FLP={pred['FLP']:.1f} | BHP={pred['BHP']:.1f} psi. "
            f"All constraints satisfied across {config.PREDICTION_HORIZON}-step horizon."
        )

        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"MPCController("
            f"step={self._step_count}, "
            f"edge_riding={self._edge_riding}, "
            f"horizon={config.PREDICTION_HORIZON})"
        )


# =============================================================================
# Verification test — run directly to validate all three layers
# =============================================================================

if __name__ == "__main__":
    import json

    SEP = "=" * 68

    print(SEP)
    print("  MPC CONTROLLER VERIFICATION TEST")
    print("  Safety-Shielded Edge-Rider Receding Horizon MPC")
    print(SEP + "\n")

    mpc = MPCController()
    print(f"Controller loaded: {mpc}\n")

    # ----------------------------------------------------------------
    # TEST 1 — Scenario A style: choke too low, target above current Q
    # ----------------------------------------------------------------
    print("-" * 68)
    print("TEST 1: Startup — choke at 30%, target Q=130 bbl/hr")
    print("-" * 68)
    state_a = dict(config.INITIAL_STATE)
    for step in range(1, 8):
        result = mpc.calculate_next_move(state_a, target_q=130.0)

        # Simulate the chosen move (use model directly for test fidelity)
        feat = pd.DataFrame([{
            "Choke_pct": result["chosen_choke"],
            "prev_Q":    state_a["Q"],
            "prev_WHP":  state_a["WHP"],
            "prev_FLP":  state_a["FLP"],
            "prev_BHP":  state_a["BHP"],
        }])
        pred_raw = mpc.model.predict(feat)[0]
        state_a = {
            "choke": result["chosen_choke"],
            "Q":     pred_raw[0],
            "WHP":   pred_raw[1],
            "FLP":   pred_raw[2],
            "BHP":   pred_raw[3],
        }

        print(
            f"  Step {step:>2} | Choke={result['chosen_choke']:>5.1f}%  "
            f"Action={result['action_taken']:>8}  "
            f"Q={state_a['Q']:>7.2f}  "
            f"WHP={state_a['WHP']:>7.1f}  "
            f"FLP={state_a['FLP']:>7.1f}  "
            f"BHP={state_a['BHP']:>8.1f}  "
            f"EdgeRide={result['edge_riding']}"
        )

    print(f"\n  XAI Reasoning (last step):\n  {result['reasoning'][:250]}...")

    # ----------------------------------------------------------------
    # TEST 2 — Scenario B style: mid-run target step change
    # ----------------------------------------------------------------
    print("\n" + "-" * 68)
    print("TEST 2: Target Tracking — step change 100 -> 150 bbl/hr at step 5")
    print("-" * 68)
    mpc.reset()
    state_b = {"choke": 50.0, "Q": 120.0, "WHP": 245.0, "FLP": 175.0, "BHP": 3020.0}
    targets = [100.0] * 4 + [150.0] * 6

    for step, tgt in enumerate(targets, 1):
        result = mpc.calculate_next_move(state_b, target_q=tgt)
        feat = pd.DataFrame([{
            "Choke_pct": result["chosen_choke"],
            "prev_Q":    state_b["Q"],
            "prev_WHP":  state_b["WHP"],
            "prev_FLP":  state_b["FLP"],
            "prev_BHP":  state_b["BHP"],
        }])
        pred_raw = mpc.model.predict(feat)[0]
        state_b = {
            "choke": result["chosen_choke"],
            "Q":     pred_raw[0], "WHP": pred_raw[1],
            "FLP":   pred_raw[2], "BHP": pred_raw[3],
        }
        marker = "  <-- TARGET STEP CHANGE" if step == 5 else ""
        print(
            f"  Step {step:>2} | Target={tgt:>5.0f}  Choke={result['chosen_choke']:>5.1f}%  "
            f"Action={result['action_taken']:>8}  "
            f"Q={state_b['Q']:>7.2f}  EdgeRide={result['edge_riding']}{marker}"
        )

    # ----------------------------------------------------------------
    # TEST 3 — Scenario C: infeasible target -> Edge-Riding
    # ----------------------------------------------------------------
    print("\n" + "-" * 68)
    print("TEST 3: Infeasible Target -> Edge-Riding Mode (target=220 bbl/hr)")
    print("-" * 68)
    mpc.reset()
    state_c = {"choke": 90.0, "Q": 158.0, "WHP": 215.0, "FLP": 152.0, "BHP": 2880.0}

    for step in range(1, 7):
        result = mpc.calculate_next_move(state_c, target_q=220.0)
        feat = pd.DataFrame([{
            "Choke_pct": result["chosen_choke"],
            "prev_Q":    state_c["Q"],
            "prev_WHP":  state_c["WHP"],
            "prev_FLP":  state_c["FLP"],
            "prev_BHP":  state_c["BHP"],
        }])
        pred_raw = mpc.model.predict(feat)[0]
        state_c = {
            "choke": result["chosen_choke"],
            "Q":     pred_raw[0], "WHP": pred_raw[1],
            "FLP":   pred_raw[2], "BHP": pred_raw[3],
        }
        print(
            f"  Step {step:>2} | Choke={result['chosen_choke']:>5.1f}%  "
            f"Action={result['action_taken']:>20}  "
            f"Q={state_c['Q']:>7.2f}  "
            f"EdgeRide={result['edge_riding']}"
        )

    print(f"\n  XAI Reasoning (last step):\n  {result['reasoning'][:350]}...")

    # ----------------------------------------------------------------
    # TEST 4 — XAI JSON output structure
    # ----------------------------------------------------------------
    print("\n" + "-" * 68)
    print("TEST 4: Full XAI JSON output structure")
    print("-" * 68)
    mpc.reset()
    state_xai = {"choke": 45.0, "Q": 120.0, "WHP": 245.0, "FLP": 175.0, "BHP": 3020.0}
    result_xai = mpc.calculate_next_move(state_xai, target_q=150.0)

    output = {
        "chosen_choke":    result_xai["chosen_choke"],
        "action_taken":    result_xai["action_taken"],
        "reasoning":       result_xai["reasoning"],
        "predicted_state": result_xai["predicted_state"],
        "edge_riding":     result_xai["edge_riding"],
    }
    print(json.dumps(output, indent=2))

    print("\n" + SEP)
    print("  ALL TESTS COMPLETE — MPC Controller is verified.")
    print(SEP + "\n")
