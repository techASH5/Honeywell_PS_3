import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Fix Windows terminal encoding crashes for emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import config
from mpc_controller import MPCController
from simulator import WellSimulator

st.set_page_config(
    page_title="Honeywell PS3 - Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
:root { --sidebar-width: 240px; }
.stDeployButton { display: none; }
.stApp { background-color: #0b0e14; }

/* Keep header invisible but present so Streamlit's internal state works */
header, [data-testid="stHeader"], .stAppHeader {
    background: transparent !important;
    height: 0px !important;
    min-height: 0px !important;
    overflow: visible !important;
}
/* Hide all native header junk */
.stDeployButton, [data-testid="stToolbar"], [data-testid="stActionSpace"],
[data-testid="stHeader"] > div { display: none !important; }
/* Custom injected expand button (via JS below) */
#eni-expand-btn {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 9999999;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
    font-size: 1.1rem;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background 0.2s;
}
#eni-expand-btn:hover { background: rgba(0, 212, 255, 0.3); }
[data-testid="stAppViewContainer"] {
    margin-top: 0px !important;
}
[data-testid="stMain"] {
    margin-top: 0px !important;
}

/* Remove Streamlit top padding — target all known selectors across versions */
[data-testid="block-container"],
[data-testid="stMainBlockContainer"],
.block-container,
section[data-testid="stMain"] > div > div {
    padding-top: 1rem !important;
    margin-top: 0 !important;
}

/* Sidebar absolute positioning and sizing (ONLY when expanded) */
section[data-testid="stSidebar"][aria-expanded="true"] { 
    background: linear-gradient(180deg, #090c10 0%, #111827 100%) !important; 
    border-right: 1px solid rgba(255,255,255,0.05) !important; 
    width: 240px !important; 
    min-width: 240px !important; 
    flex-shrink: 0 !important; 
}
section[data-testid="stSidebar"][aria-expanded="true"] > div { width: 240px !important; min-width: 240px !important; overflow-x: hidden !important; }
div[data-testid="stSidebarResizeHandle"] { left: 240px !important; }

/* Position logo normally at the top */
.sidebar-logo {
    display: flex;
    flex-direction: column;
    line-height: 1.1;
    margin-bottom: 20px;
    margin-top: 25px; /* Added padding to push it down from the absolute top edge */
}
.sidebar-logo .logo-main {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.05em;
}
.sidebar-logo .logo-sub {
    font-size: 0.6rem;
    color: #00f0ff;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    text-shadow: 0 0 8px rgba(0,240,255,0.3);
}

/* Position collapse icon absolutely to share the space with the logo */
[data-testid="stSidebarHeader"] {
    position: absolute !important;
    top: 29px !important; /* Perfect vertical alignment with the lowered logo */
    right: 5px !important;
    min-height: 0 !important;
    height: auto !important;
    padding: 0 !important;
    background: transparent !important;
    z-index: 999999;
}
[data-testid="stSidebarContent"] { 
    padding: 1rem 1rem !important; 
    padding-top: 0.2rem !important; 
}
section[data-testid="stSidebar"] .stRadio label { color: #a8b8d0 !important; font-size: 0.78rem !important; }
section[data-testid="stSidebar"] .stSlider label { color: #a8b8d0 !important; font-size: 0.78rem !important; }
section[data-testid="stSidebar"] .stSlider { padding: 0 !important; }
section[data-testid="stSidebar"] p { font-size: 0.78rem !important; }
section[data-testid="stSidebar"] button { font-size: 0.75rem !important; padding: 4px 8px !important; }

/* ── Digital Twin P&ID (Glassmorphism) ── */
.pid-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: radial-gradient(circle at 50% -20%, rgba(0, 240, 255, 0.05), rgba(11, 14, 20, 0) 70%), linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 30px 40px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 32px rgba(0,0,0,0.4);
}
.pid-pipe {
    flex-grow: 1;
    height: 6px;
    background: rgba(30, 58, 95, 0.3);
    position: relative;
    margin: 0 10px;
    border-radius: 3px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}
.pid-flow {
    position: absolute;
    top: 0; left: 0; height: 100%;
    width: 25%;
    background: linear-gradient(90deg, transparent, #00f0ff, transparent);
    box-shadow: 0 0 10px #00f0ff;
    animation: flowAnim 1.5s linear infinite;
}
@keyframes flowAnim {
    0% { left: -30%; }
    100% { left: 100%; }
}
.pid-node {
    background: rgba(26, 34, 54, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 50%;
    width: 135px;
    height: 135px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 10px rgba(255,255,255,0.03), 0 8px 20px rgba(0,0,0,0.3);
    z-index: 2;
    transition: all 0.3s ease;
}
.pid-node:hover {
    border-color: rgba(0, 240, 255, 0.3);
    box-shadow: inset 0 2px 10px rgba(255,255,255,0.05), 0 0 20px rgba(0, 240, 255, 0.15);
}
.pid-node.valve {
    border-radius: 12px;
    border: 1px solid rgba(0, 240, 255, 0.3);
    background: rgba(11, 25, 44, 0.5);
    width: 145px;
}
.pid-label {
    font-size: 0.70rem;
    color: #8be9fd;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.pid-value {
    font-size: 1.4rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    color: #ffffff;
    text-shadow: 0 0 10px rgba(255,255,255,0.1);
}
.pid-unit {
    font-size: 0.75rem;
    color: #6b8cae;
}
.pid-warning {
    border-color: #ffb86c !important;
    box-shadow: 0 0 20px rgba(255, 184, 108, 0.2) !important;
}
.pid-danger {
    border-color: #ff5555 !important;
    box-shadow: 0 0 20px rgba(255, 85, 85, 0.3) !important;
}

/* ── Terminal XAI log ── */
.xai-terminal {
    background-color: rgba(9, 13, 20, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.6;
    max-height: 420px;
    overflow-y: auto;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.xai-card {
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid #6272a4;
    margin-bottom: 10px;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    transition: background 0.2s;
}
.xai-card:hover { background: rgba(255, 255, 255, 0.04); }
.xai-card.safe { border-left-color: #50fa7b; }
.xai-card.warn { border-left-color: #ffb86c; }
.xai-card.edge { border-left-color: #ff5555; }
.xai-time { color: #6272a4; font-weight: 700; margin-right: 10px; }
.xai-action { color: #f8f8f2; font-weight: 700; }
.xai-reason { color: #a8b8d0; margin-top: 6px; display: block; }
.tag { padding: 3px 8px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; margin-right: 8px; letter-spacing: 0.05em; }
.tag.safe { background: #50fa7b; color: #000000; box-shadow: 0 0 10px rgba(80, 250, 123, 0.3); }
.tag.warn { background: #ffb86c; color: #000000; box-shadow: 0 0 10px rgba(255, 184, 108, 0.3); }
.tag.edge { background: #ff5555; color: #ffffff; box-shadow: 0 0 10px rgba(255, 85, 85, 0.3); }

/* ── Rest ── */
div[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 12px !important; }
.stButton > button { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s ease; }
.stButton > button:hover { background: rgba(0,240,255,0.1) !important; border-color: #00f0ff !important; color: #00f0ff !important; box-shadow: 0 0 15px rgba(0,240,255,0.2); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,240,255,0.5); }
.section-label { font-size: 0.65rem; font-weight: 700; color: #8be9fd; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 3px; }
</style>
""")

# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
def init_session_state():
    if "initialised" not in st.session_state:
        st.session_state.initialised   = True
        st.session_state.running       = False
        st.session_state.history       = []
        st.session_state.xai_log       = []
        st.session_state.time_step     = 0
        st.session_state.scenario      = "A"
        st.session_state.target_q      = float(config.SCENARIOS["A"]["target_q"])
        st.session_state.sim_state     = dict(config.INITIAL_STATE)
        st.session_state.prev_sim_state = None
        st.session_state.mpc_prev_state = None
        st.session_state.scenario_b_switched = False
        st.session_state.auto_mode     = True
        st.session_state.manual_choke  = 30.0

init_session_state()

@st.cache_resource
def load_mpc() -> MPCController:
    return MPCController()

@st.cache_resource
def load_sim() -> WellSimulator:
    return WellSimulator()

try:
    mpc = load_mpc()
    sim = load_sim()
except FileNotFoundError:
    st.error("Model not found. Run system_identification.py first.")
    st.stop()

# =============================================================================
# TIME-SERIES CHART
# =============================================================================
def make_trend_chart(history: list[dict], target_q: float) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.6)", height=650,
            title=dict(text="Trend Chart — awaiting data", font=dict(size=12, color="#4a6fa5")),
            margin=dict(l=50, r=50, t=40, b=40),
        )
        return fig

    df = pd.DataFrame(history)
    t = df["time_step"]
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[
            [{"secondary_y": True}],   # Row 1: Q and Choke
            [{"secondary_y": False}],  # Row 2: WHP and FLP
            [{"secondary_y": False}]   # Row 3: BHP
        ]
    )
    
    # ROW 1: Target Q, Actual Q, Choke
    fig.add_trace(go.Scatter(x=t, y=[target_q]*len(t), mode="lines", name="Target Q", line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dot")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=df["Q"], mode="lines+markers", name="Actual Q", line=dict(color="#00f0ff", width=2.5, shape='spline'), marker=dict(size=4, color="#00f0ff"), fill="tozeroy", fillcolor="rgba(0,240,255,0.05)"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=df["choke"], mode="lines", name="Choke %", line=dict(color="#b026ff", width=2, dash="solid", shape='spline'), opacity=0.85), row=1, col=1, secondary_y=True)
    
    # ROW 2: WHP and FLP
    fig.add_trace(go.Scatter(x=t, y=df["WHP"], mode="lines", name="WHP (psi)", line=dict(color="#ffb86c", width=2, shape='spline')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["FLP"], mode="lines", name="FLP (psi)", line=dict(color="#8be9fd", width=2, shape='spline')), row=2, col=1)
    
    # ROW 3: BHP
    fig.add_trace(go.Scatter(x=t, y=df["BHP"], mode="lines", name="BHP (psi)", line=dict(color="#ff5555", width=2, shape='spline'), fill="tozeroy", fillcolor="rgba(255,85,85,0.05)"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        height=500, margin=dict(l=40, r=40, t=30, b=30), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10, color="#a8b8d0", family='Outfit')),
        hovermode="x unified"
    )
    
    # Axis formatting
    axis_opts = dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9))
    fig.update_yaxes(title_text="Oil (bbl/hr)", title_font=dict(size=10, color="#00f0ff"), row=1, col=1, secondary_y=False, **axis_opts)
    fig.update_yaxes(title_text="Choke (%)", title_font=dict(size=10, color="#b026ff"), range=[0, 100], row=1, col=1, secondary_y=True, showgrid=False, zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Surface Press. (psi)", title_font=dict(size=10, color="#ffb86c"), row=2, col=1, **axis_opts)
    fig.update_yaxes(title_text="Subsurface (psi)", title_font=dict(size=10, color="#ff5555"), range=[2500, 3500], row=3, col=1, **axis_opts)
    fig.update_xaxes(title_text="Time (hr)", title_font=dict(size=10, color="#6b8cae"), row=3, col=1, **axis_opts)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9), row=2, col=1)
    
    return fig

# =============================================================================
# XAI LOG RENDERER
# =============================================================================
def render_xai_log(xai_log: list[dict]) -> str:
    if not xai_log:
        return '<div class="xai-terminal"><span class="xai-reason">Awaiting controller start...</span></div>'
    lines = []
    for entry in reversed(xai_log[-20:]):
        css_class = "safe"
        tag = "SAFE"
        if entry["edge_riding"]:
            css_class = "edge"
            tag = "EDGE RIDING"
        elif "WARNING" in entry["reasoning"] or "breach" in entry["reasoning"].lower():
            css_class = "warn"
            tag = "WARNING"
            
        lines.append(f'''
        <div class="xai-card {css_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <div>
                    <span class="xai-time">[T={entry["time"]:02d}h]</span>
                    <span class="tag {css_class}">{tag}</span>
                </div>
                <span class="xai-action" style="font-size: 0.75rem; padding: 2px 6px; background: rgba(255,255,255,0.03); border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);">{entry["action"]}</span>
            </div>
            <span class="xai-reason" style="font-size: 0.8rem; display: block; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
                <strong style="color: #ffffff;">Q={entry["q"]:.1f} bbl/hr</strong> &mdash; {entry["reasoning"]}
            </span>
        </div>
        ''')
    return '<div class="xai-terminal">' + "".join(lines) + '</div>'

# =============================================================================
# P&ID RENDERER
# =============================================================================
def get_node_class(val, min_val, max_val):
    rng = max_val - min_val
    if val < min_val + 0.05*rng or val > max_val - 0.05*rng: return "pid-danger"
    if val < min_val + 0.15*rng or val > max_val - 0.15*rng: return "pid-warning"
    return ""

def render_pid(state: dict):
    q_class = "pid-danger" if abs(state["Q"] - st.session_state.target_q) > 20 else ""
    whp_class = get_node_class(state["WHP"], config.WHP_MIN, config.WHP_MAX)
    flp_class = get_node_class(state["FLP"], config.FLP_MIN, config.FLP_MAX)
    bhp_class = get_node_class(state["BHP"], config.BHP_MIN, config.BHP_MAX)
    
    html = f'''
    <div class="pid-container">
        <div class="pid-node {bhp_class}">
            <div class="pid-label">Reservoir BHP</div>
            <div class="pid-value">{state["BHP"]:.1f}</div>
            <div class="pid-unit">psi</div>
        </div>
        <div class="pid-pipe"><div class="pid-flow"></div></div>
        
        <div class="pid-node {whp_class}">
            <div class="pid-label">Wellhead WHP</div>
            <div class="pid-value">{state["WHP"]:.1f}</div>
            <div class="pid-unit">psi</div>
        </div>
        <div class="pid-pipe"><div class="pid-flow"></div></div>
        
        <div class="pid-node valve">
            <div class="pid-label">CHOKE VALVE</div>
            <div class="pid-value">{state["choke"]:.1f}%</div>
            <div class="pid-unit" style="color:#00d4ff;">Restriction</div>
        </div>
        <div class="pid-pipe"><div class="pid-flow"></div></div>
        
        <div class="pid-node {flp_class}">
            <div class="pid-label">Flowline FLP</div>
            <div class="pid-value">{state["FLP"]:.1f}</div>
            <div class="pid-unit">psi</div>
        </div>
        <div class="pid-pipe"><div class="pid-flow"></div></div>
        
        <div class="pid-node {q_class}" style="border-color:#00ff9d; box-shadow:0 0 20px rgba(0,255,157,0.2);">
            <div class="pid-label">Production Q</div>
            <div class="pid-value" style="color:#00ff9d;">{state["Q"]:.1f}</div>
            <div class="pid-unit">bbl/hr</div>
        </div>
    </div>
    '''
    st.html(html)

# =============================================================================
# SIMULATION STEP
# =============================================================================
def run_one_step(disturbance_type=None):
    ss = st.session_state

    if ss.scenario == "B" and not ss.scenario_b_switched:
        if ss.time_step >= config.SCENARIOS["B"]["step_at_hour"]:
            ss.target_q = float(config.SCENARIOS["B"]["target_q_step"])
            ss.scenario_b_switched = True

    current_state = dict(ss.sim_state)
    
    # Inject Disturbance
    if disturbance_type == "BHP_SPIKE":
        current_state["BHP"] += 250.0
        ss.sim_state["BHP"] = current_state["BHP"]
        sim.state["BHP"] = current_state["BHP"]
    elif disturbance_type == "WHP_SPIKE":
        current_state["WHP"] += 150.0
        ss.sim_state["WHP"] = current_state["WHP"]
        sim.state["WHP"] = current_state["WHP"]

    if ss.mpc_prev_state is not None:
        mpc._prev_state = dict(ss.mpc_prev_state)
    else:
        mpc._prev_state = None

    if ss.auto_mode:
        try:
            result = mpc.calculate_next_move(current_state, target_q=ss.target_q)
            chosen_choke = result["chosen_choke"]
            action_taken = result["action_taken"]
            reasoning = result["reasoning"]
            edge_riding = result["edge_riding"]
        except Exception as e:
            ss.running = False
            st.error(f"MPC error: {e}")
            return
    else:
        # Manual Mode
        chosen_choke = ss.manual_choke
        delta = chosen_choke - current_state["choke"]
        action_taken = f"{delta:+.1f}% (MANUAL)"
        reasoning = f"Manual override to {chosen_choke}%."
        edge_riding = False

    try:
        sim.state = {
            "choke": current_state["choke"], "Q": current_state["Q"],
            "WHP": current_state["WHP"], "FLP": current_state["FLP"],
            "BHP": current_state["BHP"]
        }
        output = sim.step(chosen_choke)
    except Exception as e:
        ss.running = False
        st.error(f"Simulator error: {e}")
        return

    ss.time_step += 1
    new_state = {
        "choke": chosen_choke, "Q": output["Q"], "WHP": output["WHP"],
        "FLP": output["FLP"], "BHP": output["BHP"],
    }

    ss.history.append({
        "time_step": ss.time_step, "choke": chosen_choke, "Q": output["Q"],
        "WHP": output["WHP"], "FLP": output["FLP"], "BHP": output["BHP"],
        "target_q": ss.target_q, "action": action_taken, "edge_riding": edge_riding,
    })

    ss.xai_log.append({
        "time": ss.time_step,
        "action": action_taken,
        "q": output["Q"],
        "reasoning": reasoning[:200],
        "edge_riding": edge_riding
    })

    ss.mpc_prev_state = dict(current_state)
    ss.prev_sim_state = dict(current_state)
    ss.sim_state      = new_state


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    ss = st.session_state
    with st.sidebar:
        st.html("""
        <div class="sidebar-logo">
            <span class="logo-main">HONEYWELL</span>
            <span class="logo-sub">HACKATHON</span>
        </div>
        """)

        st.html('<div class="section-label">Mode</div>')
        mode = st.radio("Mode", ["AUTO (MPC)", "MANUAL OVERRIDE"], label_visibility="collapsed")
        ss.auto_mode = "AUTO" in mode

        if not ss.auto_mode:
            st.html('<div class="section-label" style="margin-top:10px; color:#ff9500;">Manual Choke %</div>')
            ss.manual_choke = st.slider("Choke", 0.0, 100.0, ss.sim_state["choke"], 2.5, label_visibility="collapsed")

        st.html('<div class="section-label" style="margin-top:10px;">Scenario</div>')
        scenarios = {"A": "A — Startup to Target", "B": "B — Target Tracking", "C": "C — Infeasible Target"}
        selected = st.radio("Select Scenario", list(scenarios.keys()), format_func=lambda k: scenarios[k], index=list(scenarios.keys()).index(ss.scenario), label_visibility="collapsed", disabled=ss.running)
        
        if selected != ss.scenario and not ss.running:
            ss.scenario = selected
            ss.target_q = float(config.SCENARIOS[selected].get("target_q") or config.SCENARIOS[selected].get("target_q_initial", 130.0))
            ss.scenario_b_switched = False

        st.html('<div class="section-label" style="margin-top:10px;">Target Flow Rate</div>')
        target_q = st.slider("Target Q (bbl/hr)", 80.0, 220.0, float(ss.target_q), 5.0, label_visibility="collapsed", disabled=ss.running)
        if not ss.running: ss.target_q = target_q
        
        st.html(f'<div style="text-align:center; font-size:1.1rem; font-weight:700; color:#00d4ff; margin: 2px 0 8px 0;">{ss.target_q:.0f} <span style="font-size:0.65rem; color:#4a6fa5;">bbl/hr</span></div>')

        st.html('<div class="section-label" style="margin-top:8px;">Control Panel</div>')
        col_s, col_r = st.columns(2)
        with col_s:
            if not ss.running:
                if st.button("▶ Start", use_container_width=True, type="primary"): ss.running = True
            else:
                if st.button("⏹ Stop", use_container_width=True): ss.running = False
        with col_r:
            if st.button("↺ Reset", use_container_width=True, disabled=ss.running):
                ss.running, ss.history, ss.xai_log, ss.time_step = False, [], [], 0
                ss.sim_state, ss.prev_sim_state, ss.mpc_prev_state = dict(config.INITIAL_STATE), None, None
                ss.scenario_b_switched = False
                mpc.reset(); sim.reset()
                ss.target_q = float(config.SCENARIOS[ss.scenario].get("target_q") or config.SCENARIOS[ss.scenario].get("target_q_initial", 130.0))

        st.html('<div class="section-label" style="margin-top:14px; color:#ff4d6d;">Chaos Injection</div>')
        if st.button("⚠️ Inject BHP Spike (+250)", use_container_width=True, disabled=not ss.running):
            run_one_step("BHP_SPIKE")
            st.rerun()
        if st.button("⚠️ Inject WHP Spike (+150)", use_container_width=True, disabled=not ss.running):
            run_one_step("WHP_SPIKE")
            st.rerun()

def make_single_chart(history: list[dict], target_q: float, mode: str) -> go.Figure:
    df = pd.DataFrame(history)
    t = df["time_step"]
    fig = go.Figure()
    
    if mode == "Q":
        fig.add_trace(go.Scatter(x=t, y=[target_q]*len(t), mode="lines", name="Target Q", line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=t, y=df["Q"], mode="lines+markers", name="Actual Q", line=dict(color="#00f0ff", width=3, shape='spline'), marker=dict(size=6, color="#00f0ff"), fill="tozeroy", fillcolor="rgba(0,240,255,0.05)"))
        fig.add_trace(go.Scatter(x=t, y=df["choke"], mode="lines", name="Choke %", line=dict(color="#b026ff", width=2, dash="solid", shape='spline'), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Choke (%)", title_font=dict(size=14, color="#b026ff"), overlaying="y", side="right", range=[0,100], tickfont=dict(size=13)))
        fig.update_yaxes(title_text="Oil (bbl/hr)", title_font=dict(size=14, color="#00f0ff"))
    elif mode == "SURFACE":
        fig.add_trace(go.Scatter(x=t, y=df["WHP"], mode="lines", name="WHP (psi)", line=dict(color="#ffb86c", width=3, shape='spline')))
        fig.add_trace(go.Scatter(x=t, y=df["FLP"], mode="lines", name="FLP (psi)", line=dict(color="#8be9fd", width=3, shape='spline')))
        fig.update_yaxes(title_text="Surface Press. (psi)", title_font=dict(size=14, color="#ffb86c"))
    else:
        fig.add_trace(go.Scatter(x=t, y=df["BHP"], mode="lines", name="BHP (psi)", line=dict(color="#ff5555", width=3, shape='spline'), fill="tozeroy", fillcolor="rgba(255,85,85,0.05)"))
        fig.update_yaxes(title_text="Subsurface (psi)", title_font=dict(size=14, color="#ff5555"), range=[2500, 3500])

    axis_opts = dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=13))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        height=600, margin=dict(l=50, r=50, t=40, b=40), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=14, color="#a8b8d0", family='Outfit')),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Time (hr)", title_font=dict(size=14, color="#6b8cae"), **axis_opts)
    fig.update_yaxes(**axis_opts)
    return fig

@st.dialog("🔍 Graph Explorer", width="large")
def graph_explorer():
    ss = st.session_state
    if not ss.history:
        st.warning("No data yet.")
        return
    tab1, tab2, tab3 = st.tabs(["Flow & Choke", "Surface Pressures", "Subsurface Pressure"])
    with tab1: st.plotly_chart(make_single_chart(ss.history, ss.target_q, "Q"), use_container_width=True)
    with tab2: st.plotly_chart(make_single_chart(ss.history, ss.target_q, "SURFACE"), use_container_width=True)
    with tab3: st.plotly_chart(make_single_chart(ss.history, ss.target_q, "BHP"), use_container_width=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    ss = st.session_state
    render_sidebar()

    # Header
    status_class = "status-running" if ss.running else "status-idle"
    status_text  = "● CONTROLLER ACTIVE" if ss.running else "○ STANDBY"
    if not ss.auto_mode:
        status_class = "status-stopped"
        status_text = "⚠️ MANUAL OVERRIDE"
    
    st.html(f'''
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; margin-top: 0;">
        <div>
            <div style="font-size:1.5rem; font-weight:700; color:#e8eaf0; letter-spacing:0.02em;">Autonomous Production Choke Controller</div>
            <div style="font-size:0.65rem; color:#4a6fa5; letter-spacing:0.05em; text-transform:uppercase;">Honeywell Hackathon PS3 | Scenario {ss.scenario}</div>
        </div>
        <div class="status-pill {status_class}">{status_text}</div>
    </div>
    ''')

    # Digital Twin P&ID
    st.html('<div class="section-label">LIVE DIGITAL TWIN</div>')
    render_pid(ss.sim_state)

    # Middle Row: Trends & Log
    st.html('<div class="section-label">TREND & CONTROLLER INTELLIGENCE</div>')
    col_chart, col_log = st.columns([1.2, 1])
    
    with col_chart:
        fig = make_trend_chart(ss.history, ss.target_q)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        if st.button("🔍 Expand Graphs", use_container_width=True):
            graph_explorer()
        
    with col_log:
        st.html(render_xai_log(ss.xai_log))
        if ss.history:
            st.html(f'<div style="font-size:0.9rem; color:#a8b8d0; margin-top:10px; background: rgba(0, 240, 255, 0.05); padding: 8px 12px; border-radius: 6px; border-left: 4px solid #00f0ff; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">⚡ Last Action: <span style="color:#00f0ff; font-weight:800;">{ss.history[-1]["action"]}</span></div>')

    # Bottom Row: History Table
    if ss.history:
        st.html('<div class="section-label" style="margin-top:10px;">LAST 5 CONTROL STATES</div>')
        df_display = pd.DataFrame(ss.history[-5:][::-1])[[
            "time_step", "choke", "Q", "WHP", "FLP", "BHP", "target_q", "action", "edge_riding"
        ]]
        for col in ["Q", "WHP", "FLP", "BHP"]: df_display[col] = df_display[col].round(2)
        df_display["choke"] = df_display["choke"].round(1)
        df_display["target_q"] = df_display["target_q"].round(1)
        
        df_display = df_display.rename(columns={
            "time_step": "Time (hr)", "choke": "Choke (%)", "Q": "Q (bbl/hr)",
            "WHP": "WHP (psi)", "FLP": "FLP (psi)", "BHP": "BHP (psi)",
            "target_q": "Target Q", "action": "Action", "edge_riding": "Edge Ride",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Simulation Loop
    if ss.running:
        import time
        time.sleep(1.0)
        run_one_step()
        st.rerun()

    # Inject: 1) padding-top killer, 2) resilient sidebar expand button
    components.html("""
<script>
(function() {
    var doc = window.parent.document;

    /* Kill Streamlit's inline padding-top (CSS !important can't override inline styles) */
    function killPadding() {
        ['[data-testid="stMainBlockContainer"]','[data-testid="block-container"]','.block-container'].forEach(function(sel) {
            var el = doc.querySelector(sel);
            if (el) {
                el.style.setProperty('padding-top', '1rem', 'important');
                el.style.setProperty('margin-top', '0', 'important');
            }
        });
    }
    killPadding();
    setInterval(killPadding, 800);

    /* Sidebar expand button — styled like Streamlit native collapse, polling survives reruns */
    function getOrCreateBtn() {
        var btn = doc.getElementById('eni-sidebar-btn');
        if (btn) return btn;
        btn = doc.createElement('button');
        btn.id = 'eni-sidebar-btn';
        btn.innerHTML = '&#xbb;';
        btn.title = 'Open Sidebar';
        btn.style.cssText = [
            'position:fixed','top:10px','left:10px','z-index:9999999',
            'width:2rem','height:2rem','border-radius:0.5rem',
            'background:rgb(19,23,32)','border:none',
            'color:rgb(250,250,250)','font-size:1.1rem',
            'display:none','align-items:center','justify-content:center',
            'cursor:pointer','line-height:1','padding:0',
            'opacity:0.7','transition:opacity 0.2s'
        ].join(';');
        btn.onmouseover = function(){ btn.style.opacity='1'; };
        btn.onmouseout  = function(){ btn.style.opacity='0.7'; };
        btn.onclick = function() {
            var trigger = doc.querySelector('[data-testid="collapsedControl"] button')
                       || doc.querySelector('[data-testid="stSidebarHeader"] button')
                       || doc.querySelector('[data-testid="stSidebar"] button');
            if (trigger) trigger.click();
        };
        doc.body.appendChild(btn);
        return btn;
    }

    /* Poll every 300ms — survives st.rerun() destroying/recreating iframes */
    setInterval(function() {
        var btn = getOrCreateBtn();
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            btn.style.display = (sidebar.getAttribute('aria-expanded') === 'false') ? 'flex' : 'none';
        }
    }, 300);

})();
</script>
""", height=0)

if __name__ == "__main__":
    main()
