# =============================================================================
# system_identification.py
#
# Phase 2 - Run this once to:
#   1. Load the CSV step-test dataset
#   2. Generate a rich Plotly step-response analysis plot  → step_test_response.html
#   3. Calculate and print steady-state process gains
#   4. Train a RandomForestRegressor surrogate model on lagged features
#   5. Save the trained model as predictive_model.pkl
#
# Run:
#   python system_identification.py
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

import config

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH   = "Autonomous_Choke_Control_Simulated_Dataset.csv"
PLOT_PATH  = "step_test_response.html"
MODEL_PATH = config.MODEL_PATH   # "predictive_model.pkl"

OUTPUT_VARS  = ["OilRate_bbl_hr", "WHP_psi", "FLP_psi", "BHP_psi"]
FEATURE_COLS = ["Choke_pct", "prev_Q", "prev_WHP", "prev_FLP", "prev_BHP"]
TARGET_COLS  = ["next_Q",   "next_WHP", "next_FLP", "next_BHP"]

# Friendly display names for plots
DISPLAY = {
    "OilRate_bbl_hr": "Oil Flow Rate (bbl/hr)",
    "WHP_psi":        "Wellhead Pressure (psi)",
    "FLP_psi":        "Flowline Pressure (psi)",
    "BHP_psi":        "Bottom Hole Pressure (psi)",
    "Choke_pct":      "Choke Opening (%)",
}

# Colour palette — dark SCADA aesthetic
COLORS = {
    "choke":  "#00d4ff",   # cyan
    "Q":      "#00ff9d",   # green
    "WHP":    "#ff9500",   # amber
    "FLP":    "#c77dff",   # violet
    "BHP":    "#ff4d6d",   # red-pink
    "target": "#ffffff",   # white reference line
}

STEP_TIMES   = [0, 20, 40, 70, 90]   # hours where choke changes
STEP_CHOKES  = [30, 40, 55, 45, 65]  # corresponding choke values


# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[LOAD] {path}  ->  {len(df)} rows x {len(df.columns)} columns")
    print(f"       Columns : {list(df.columns)}")
    print(f"       Time    : {df['Time_hr'].min()} – {df['Time_hr'].max()} hr")
    print(f"       Choke % : {sorted(df['Choke_pct'].unique())}")
    return df


# =============================================================================
# 2.  STEP-RESPONSE PLOT
# =============================================================================

def build_step_response_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a 5-row Plotly subplot:
      Row 1 — Choke position with step-change annotations
      Row 2 — Oil Flow Rate
      Row 3 — Wellhead Pressure (WHP)
      Row 4 — Flowline Pressure (FLP)
      Row 5 — Bottom Hole Pressure (BHP)
    """

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Choke Opening (%)",
            "Oil Flow Rate (bbl/hr)",
            "Wellhead Pressure — WHP (psi)",
            "Flowline Pressure — FLP (psi)",
            "Bottom Hole Pressure — BHP (psi)",
        ],
        vertical_spacing=0.06,
        row_heights=[0.12, 0.22, 0.22, 0.22, 0.22],
    )

    t = df["Time_hr"]

    # ── Row 1: Choke ──
    fig.add_trace(go.Scatter(
        x=t, y=df["Choke_pct"],
        mode="lines", name="Choke %",
        line=dict(color=COLORS["choke"], width=2.5, shape="hv"),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
    ), row=1, col=1)

    # Step-change vertical annotations
    for st_t, st_c in zip(STEP_TIMES[1:], STEP_CHOKES[1:]):
        fig.add_vline(
            x=st_t, line_dash="dash",
            line_color="rgba(255,255,255,0.25)", line_width=1,
        )
        fig.add_annotation(
            x=st_t + 0.5, y=st_c + 4,
            text=f"-> {st_c}%",
            font=dict(color=COLORS["choke"], size=10),
            showarrow=False,
        )

    # ── Row 2: Oil Flow Rate ──
    fig.add_trace(go.Scatter(
        x=t, y=df["OilRate_bbl_hr"],
        mode="lines", name="Q (bbl/hr)",
        line=dict(color=COLORS["Q"], width=2),
    ), row=2, col=1)

    # ── Row 3: WHP ──
    fig.add_trace(go.Scatter(
        x=t, y=df["WHP_psi"],
        mode="lines", name="WHP (psi)",
        line=dict(color=COLORS["WHP"], width=2),
    ), row=3, col=1)

    # WHP constraint lines
    fig.add_hline(y=config.WHP_MAX, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=3, col=1)
    fig.add_hline(y=config.WHP_MIN, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=3, col=1)

    # ── Row 4: FLP ──
    fig.add_trace(go.Scatter(
        x=t, y=df["FLP_psi"],
        mode="lines", name="FLP (psi)",
        line=dict(color=COLORS["FLP"], width=2),
    ), row=4, col=1)

    fig.add_hline(y=config.FLP_MAX, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=4, col=1)
    fig.add_hline(y=config.FLP_MIN, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=4, col=1)

    # ── Row 5: BHP ──
    fig.add_trace(go.Scatter(
        x=t, y=df["BHP_psi"],
        mode="lines", name="BHP (psi)",
        line=dict(color=COLORS["BHP"], width=2),
    ), row=5, col=1)

    fig.add_hline(y=config.BHP_MAX, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=5, col=1)
    fig.add_hline(y=config.BHP_MIN, line_dash="dot",
                  line_color="rgba(255,77,109,0.6)", row=5, col=1)

    # ── Global layout ──
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                "<b>Autonomous Choke Controller — Open-Loop Step-Test Analysis</b><br>"
                "<sup>Honeywell Hackathon PS3 | System Identification Dataset</sup>"
            ),
            x=0.5,
            font=dict(size=18, color="#e8eaf0"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(family="Inter, sans-serif", color="#e8eaf0", size=12),
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.04,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(26,31,46,0.8)",
            bordercolor="#00d4ff",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=90, b=60),
        hovermode="x unified",
    )

    # Axis labels
    fig.update_yaxes(title_text="Choke (%)", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="bbl/hr",   row=2, col=1,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="psi",      row=3, col=1,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="psi",      row=4, col=1,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="psi",      row=5, col=1,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_xaxes(title_text="Time (hr)", row=5, col=1,
                     gridcolor="rgba(255,255,255,0.07)")

    return fig


# =============================================================================
# 3.  STEADY-STATE GAIN ANALYSIS
# =============================================================================

def compute_gains(df: pd.DataFrame) -> dict:
    """
    For each choke step, compare the mean of the last 5 rows before the step
    to the mean of the last 5 rows of that step's steady state.
    Returns a dict of per-variable gains: delta_var / delta_choke.
    """
    print("\n" + "=" * 60)
    print("  STEADY-STATE PROCESS GAIN ANALYSIS")
    print("=" * 60)

    segments = []
    for i, (t_start, choke_val) in enumerate(zip(STEP_TIMES, STEP_CHOKES)):
        t_end = STEP_TIMES[i + 1] if i + 1 < len(STEP_TIMES) else df["Time_hr"].max() + 1
        seg = df[(df["Time_hr"] >= t_start) & (df["Time_hr"] < t_end)]
        steady = seg.tail(5)  # last 5 rows = steady state for that choke level
        segments.append((choke_val, steady))

    all_gains = {v: [] for v in OUTPUT_VARS}

    for i in range(len(segments) - 1):
        choke_prev, ss_prev = segments[i]
        choke_next, ss_next = segments[i + 1]
        delta_choke = choke_next - choke_prev

        print(f"\n  Step: Choke {choke_prev}% -> {choke_next}%  (Delta = {delta_choke:+.0f}%)")

        for var in OUTPUT_VARS:
            ss_val_prev = ss_prev[var].mean()
            ss_val_next = ss_next[var].mean()
            delta_var   = ss_val_next - ss_val_prev
            gain        = delta_var / delta_choke
            all_gains[var].append(gain)
            unit = "bbl/hr/%" if "Rate" in var else "psi/%"
            print(
                f"    {DISPLAY[var]:<35}  "
                f"SS_prev={ss_val_prev:>8.2f}  "
                f"SS_next={ss_val_next:>8.2f}  "
                f"Delta={delta_var:>+8.2f}  "
                f"K={gain:>+7.3f} {unit}"
            )

    print("\n  Average Gains Across All Steps:")
    print("  " + "-" * 54)
    mean_gains = {}
    for var in OUTPUT_VARS:
        avg = np.mean(all_gains[var])
        mean_gains[var] = avg
        unit = "bbl/hr/%" if "Rate" in var else "psi/%"
        direction = "[DIRECT]" if avg > 0 else "[INVERSE]"
        print(f"    {DISPLAY[var]:<35}  K_avg = {avg:>+7.3f} {unit}  {direction}")

    print("\n  Process Insight:")
    print("  * Opening choke INCREASES oil flow (direct-acting)")
    print("  * Opening choke DECREASES pressures (inverse-acting)")
    print("  * Non-linear: gains vary across operating range")
    print("=" * 60)

    return mean_gains


# =============================================================================
# 4.  BUILD LAGGED FEATURES
# =============================================================================

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one-step-lag feature/target pairs:
        Features : [Choke_pct(t), Q(t-1), WHP(t-1), FLP(t-1), BHP(t-1)]
        Targets  : [Q(t),         WHP(t),  FLP(t),   BHP(t)  ]

    Row 0 is dropped (no previous row available).
    """
    feat = pd.DataFrame()
    feat["Choke_pct"] = df["Choke_pct"].values
    feat["prev_Q"]    = df["OilRate_bbl_hr"].shift(1)
    feat["prev_WHP"]  = df["WHP_psi"].shift(1)
    feat["prev_FLP"]  = df["FLP_psi"].shift(1)
    feat["prev_BHP"]  = df["BHP_psi"].shift(1)

    targ = pd.DataFrame()
    targ["next_Q"]   = df["OilRate_bbl_hr"].values
    targ["next_WHP"] = df["WHP_psi"].values
    targ["next_FLP"] = df["FLP_psi"].values
    targ["next_BHP"] = df["BHP_psi"].values

    # Drop first row (NaN from shift)
    feat = feat.iloc[1:].reset_index(drop=True)
    targ = targ.iloc[1:].reset_index(drop=True)

    print(f"[FEATURES] Training set: {len(feat)} samples x {len(feat.columns)} features")
    print(f"           Feature cols : {list(feat.columns)}")
    print(f"           Target  cols : {list(targ.columns)}")

    return feat, targ


# =============================================================================
# 5.  TRAIN & EVALUATE MODEL
# =============================================================================

def train_model(X: pd.DataFrame, y: pd.DataFrame):
    """
    Train a MultiOutputRegressor wrapping RandomForestRegressor.
    Evaluates with 5-fold CV and prints per-output R² scores.
    Saves model to MODEL_PATH.
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING — RandomForest Surrogate")
    print("=" * 60)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(rf, n_jobs=-1)

    print("\n  [1/3] Fitting model on full dataset...")
    model.fit(X, y)
    print("         Done.")

    # ── In-sample R² per output ──
    print("\n  [2/3] In-sample R² (train set):")
    y_pred_train = model.predict(X)
    for i, col in enumerate(y.columns):
        r2 = r2_score(y[col], y_pred_train[:, i])
        bar = "#" * int(r2 * 20)
        print(f"         {col:<12}  R2={r2:.4f}  [{bar:<20}]")

    # ── 5-fold cross-val per output (quick, dataset is small) ──
    print("\n  [3/3] 5-Fold Cross-Validation R² (generalisation estimate):")
    for i, col in enumerate(y.columns):
        single_rf = RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        scores = cross_val_score(single_rf, X, y[col], cv=5, scoring="r2")
        print(
            f"         {col:<12}  "
            f"mean={scores.mean():.4f}  "
            f"std={scores.std():.4f}  "
            f"folds={np.round(scores, 3)}"
        )

    # ── Save ──
    print(f"\n  Saving model → {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"  Model size : {size_kb:.1f} KB")
    print("=" * 60)

    return model


# =============================================================================
# 6.  POST-TRAIN SANITY PREDICTION
# =============================================================================

def sanity_check(model, X: pd.DataFrame, y: pd.DataFrame):
    """
    Print a few sample predictions vs actuals to visually verify model quality.
    """
    print("\n  Sample Predictions vs Actuals (first 5 rows):")
    print(f"  {'Feature':<55}  {'Predicted':>10}  {'Actual':>10}  {'Error':>8}")
    print("  " + "-" * 90)

    preds = model.predict(X.head(10))

    for row_idx in range(min(5, len(X))):
        for col_idx, col in enumerate(TARGET_COLS):
            pred  = preds[row_idx, col_idx]
            actual = y.iloc[row_idx][col]
            err   = pred - actual
            feat_summary = (
                f"Choke={X.iloc[row_idx]['Choke_pct']:.0f}%  "
                f"prevQ={X.iloc[row_idx]['prev_Q']:.1f}"
            )
            print(
                f"  [{row_idx}] {col:<8}  {feat_summary:<40}  "
                f"{pred:>10.3f}  {actual:>10.3f}  {err:>+8.3f}"
            )
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  SYSTEM IDENTIFICATION — Honeywell PS3")
    print("  Autonomous Choke Controller")
    print("=" * 60 + "\n")

    # ── 1. Load ──
    df = load_data(CSV_PATH)

    # ── 2. Step-response plot ──
    print(f"\n[PLOT] Building step-response visualisation...")
    fig = build_step_response_plot(df)
    fig.write_html(PLOT_PATH, include_plotlyjs="cdn")
    print(f"       Saved → {PLOT_PATH}")

    # ── 3. Gain analysis ──
    compute_gains(df)

    # ── 4. Feature engineering ──
    X, y = build_features(df)

    # ── 5. Train & evaluate ──
    model = train_model(X, y)

    # ── 6. Sanity check ──
    sanity_check(model, X, y)

    print("\n" + "=" * 60)
    print("  SYSTEM IDENTIFICATION COMPLETE")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Plot    : {PLOT_PATH}")
    print("  Next    : Run  python simulator.py  to verify the surrogate.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
