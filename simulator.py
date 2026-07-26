# =============================================================================
# simulator.py
# WellSimulator — Surrogate ML-based well simulator.
#
# Loads a pre-trained RandomForest model (predictive_model.pkl) and exposes
# the same interface the real Honeywell simulator will use.
#
# To swap in the real simulator: replace this file. config.py and the rest of
# the codebase require NO changes.
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

import config


class WellSimulator:
    """
    Surrogate well simulator trained on the Autonomous_Choke_Control_Simulated_Dataset.csv
    step-test data via system_identification.py.

    The model maps:
        (current_choke, prev_Q, prev_WHP, prev_FLP, prev_BHP)
        → (next_Q, next_WHP, next_FLP, next_BHP)

    Gaussian noise is added to each output to simulate real process variability.

    Usage
    -----
        sim = WellSimulator()
        outputs = sim.step(choke_position=45.0)
        # outputs → {"Q": ..., "WHP": ..., "FLP": ..., "BHP": ...}

        sim.reset()   # back to initial conditions
    """

    # Output variable order must match how system_identification.py trained the model
    OUTPUT_VARS = ["Q", "WHP", "FLP", "BHP"]

    def __init__(self, model_path: str = None):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to the .pkl file. Defaults to config.MODEL_PATH.
            Raises FileNotFoundError with a helpful message if not found.
        """
        self.model_path = model_path or config.MODEL_PATH

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Surrogate model not found at '{self.model_path}'.\n"
                "Run system_identification.py first to train and save the model:\n"
                "    python system_identification.py"
            )

        self.model = joblib.load(self.model_path)

        # Initialise state from config
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, choke_position: float) -> dict:
        """
        Advance the simulation by one control interval (1 hour).

        Parameters
        ----------
        choke_position : float
            New choke opening in percent (0–100).
            Clipped to [CHOKE_MIN, CHOKE_MAX] internally.

        Returns
        -------
        dict
            {"Q": float, "WHP": float, "FLP": float, "BHP": float}
            All values have small Gaussian noise applied.
        """
        choke_position = float(np.clip(choke_position, config.CHOKE_MIN, config.CHOKE_MAX))

        # Build feature vector as named DataFrame — must match training column order
        features = pd.DataFrame([{
            "Choke_pct": choke_position,
            "prev_Q":    self.state["Q"],
            "prev_WHP":  self.state["WHP"],
            "prev_FLP":  self.state["FLP"],
            "prev_BHP":  self.state["BHP"],
        }])

        # Predict next state
        prediction = self.model.predict(features)[0]  # shape (4,)

        # Apply Gaussian noise to simulate real variability
        noisy_output = {
            var: float(prediction[i]) + np.random.normal(0, config.NOISE_STD[var])
            for i, var in enumerate(self.OUTPUT_VARS)
        }

        # Update internal state
        self.state["choke"] = choke_position
        self.state.update(noisy_output)
        self.time_step += 1

        return dict(noisy_output)

    def reset(self):
        """Reset simulator to initial conditions defined in config.py."""
        self.state = dict(config.INITIAL_STATE)
        self.time_step = 0

    def get_state(self) -> dict:
        """
        Return full current state including choke position and time step.

        Returns
        -------
        dict
            {"time_step": int, "choke": float, "Q": float,
             "WHP": float, "FLP": float, "BHP": float}
        """
        return {
            "time_step": self.time_step,
            **self.state,
        }

    def __repr__(self) -> str:
        s = self.state
        return (
            f"WellSimulator(t={self.time_step}hr | "
            f"Choke={s['choke']:.1f}% | "
            f"Q={s['Q']:.1f} bbl/hr | "
            f"WHP={s['WHP']:.1f} psi | "
            f"FLP={s['FLP']:.1f} psi | "
            f"BHP={s['BHP']:.1f} psi)"
        )


# =============================================================================
# Quick smoke-test — run this file directly to verify the simulator works
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("WellSimulator — Smoke Test")
    print("=" * 60)

    try:
        sim = WellSimulator()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)

    print(f"\nInitial state:\n  {sim}\n")

    test_moves = [35.0, 40.0, 45.0, 50.0, 55.0]
    print(f"{'Step':>4}  {'Choke%':>7}  {'Q (bbl/hr)':>12}  {'WHP (psi)':>10}  {'FLP (psi)':>10}  {'BHP (psi)':>10}")
    print("-" * 62)

    for choke in test_moves:
        out = sim.step(choke)
        print(
            f"{sim.time_step:>4}  {choke:>7.1f}  "
            f"{out['Q']:>12.2f}  {out['WHP']:>10.2f}  "
            f"{out['FLP']:>10.2f}  {out['BHP']:>10.2f}"
        )

    print(f"\nFinal state:\n  {sim}")
    print("\n[PASS] Simulator is functional.")
