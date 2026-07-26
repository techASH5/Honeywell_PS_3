# Project Overview: Autonomous Production Choke Controller (Honeywell Hackathon PS3)

## 1. Project Goal
Build a Python-based Model Predictive Controller (MPC) for a single naturally flowing oil well. The system must automatically adjust the production choke opening to hit target oil flow rates while strictly respecting pressure safety constraints and choke ramp-rate limits. 

Since the official Honeywell simulator is not yet provided, we will build a **Mock Simulator** trained on the provided CSV data. The architecture will be modular so the mock simulator can be easily swapped with the real one later.

## 2. Tech Stack
*   **Language:** Python 3.10+
*   **Dashboard:** Streamlit
*   **Data/Math:** Pandas, NumPy, Scikit-learn (for dynamic model identification)
*   **Charting:** Plotly

## 3. File Structure
The project must be strictly modular. Create the following files:
*   `config.py` -> Stores all constraints, limits, and initial states.
*   `simulator.py` -> Mock simulator class that mimics the Honeywell interface.
*   `system_identification.py` -> Script to run step-tests on the mock simulator, save plots, and train a predictive ML model (`model.pkl`).
*   `mpc_controller.py` -> The brute-force MPC logic.
*   `dashboard.py` -> Streamlit app for visualization and live control.
*   `requirements.txt` -> Dependencies.

## 4. Module Specifications

### `config.py`
Define the following configuration variables. *Note: Since exact constraints aren't provided, use these realistic placeholder values. They must be easy to change later.*
*   `TARGET_FLOW_INITIAL` = 100 (bbl/hr)
*   `CHoke_MIN` = 0 (%)
*   `CHOKE_MAX` = 100 (%)
*   `CHoke_MAX_STEP` = 5 (% per control interval)
*   `WHP_MIN` = 500, `WHP_MAX` = 1500 (psi)
*   `FLP_MIN` = 200, `FLP_MAX` = 800 (psi)
*   `BHP_MIN` = 2000, `BHP_MAX` = 4000 (psi)
*   `CONTROL_INTERVAL` = 1 (hour)

### `simulator.py` (The Mock Simulator)
*   **Context:** We only have a sample CSV (`Autonomous_Choke_Control_Simulated_Dataset.csv`). 
*   **Task:** Create a `WellSimulator` class. In the `__init__` method, load the CSV and train a quick `sklearn.linear_model.LinearRegression` (or Ridge) model to map `Choke Opening` -> `[Q, WHP, FLP, BHP]`. 
*   **Interface:** Implement a `step(choke_position)` method. It takes the new choke position (0-100), uses the internal ML model to predict the outputs, adds a tiny bit of Gaussian noise to simulate reality, and returns `[Q, WHP, FLP, BHP]`.
*   **State tracking:** The class should keep track of the current time step and current outputs.

### `system_identification.py`
*   **Task:** Run an automated step-test script.
*   **Logic:**
    1. Instantiate `WellSimulator`.
    2. Apply a sequence of step changes to the choke (e.g., 20% -> 40% -> 60% -> 80% -> 100%). Hold each step for 5 control intervals.
    3. Log all inputs (choke) and outputs (Q, WHP, FLP, BHP) into a Pandas DataFrame.
    4. Generate a Plotly figure with subplots showing the step responses. Save this as `step_test_response.html`.
    5. Train a dynamic model (e.g., use current choke + previous step's Q/WHP/FLP/BHP to predict current Q/WHP/FLP/BHP) using `LinearRegression`. Save this model as `predictive_model.pkl`. This model will be used by the MPC to "look ahead".

### `mpc_controller.py`
*   **Task:** Implement a simplified brute-force Model Predictive Controller.
*   **Class:** `MPCController`
*   **Inputs on init:** Load `predictive_model.pkl`, load constraints from `config.py`.
*   **Method:** `calculate_next_move(current_state, target_q)`
    *   **Current State:** Contains current `Q, WHP, FLP, BHP, current_choke`.
    *   **Candidate Generation:** Generate 5 possible next moves: `[-5, -2.5, 0, 2.5, 5]` added to `current_choke`.
    *   **Prediction:** For each candidate, use `predictive_model.pkl` to predict the next state.
    *   **Constraint Checking:** Reject any candidate that:
        1. Violates choke limits (0-100).
        2. Violates pressure limits (WHP, FLP, BHP min/max).
    *   **Cost Function:** From the safe candidates, calculate the absolute error between predicted `Q` and `target_q`. 
    *   **Selection:** Pick the candidate with the lowest error. If no candidates are safe, return 0 (no change).
    *   **Return:** The chosen `next_choke_position`.

### `dashboard.py` (Streamlit UI)
*   **Layout:** Wide layout. Title "Autonomous Choke Controller Dashboard".
*   **Sidebar:**
    *   Radio button for Scenarios: "Scenario A (Startup)", "Scenario B (Target Tracking)", "Scenario C (Infeasible Target)".
    *   Slider for "Target Oil Flow Rate (bbl/hr)" (dynamic based on scenario).
    *   Buttons: "Start Controller", "Stop Controller", "Reset Simulation".
*   **Main Area (3 Columns):**
    *   **Col 1: Live Gauges.** Use `streamlit-metrics` or Plotly gauges to show current Q, WHP, FLP, BHP, and Choke %. Turn gauges RED if constraints are violated.
    *   **Col 2: Trending Charts.** Two Plotly line charts:
        1. Target Q vs Actual Q over time.
        2. Choke Position over time.
    *   **Col 3: Controller Logs.** A scrolling text area (using `st.code` or a session_state list) showing what the MPC is deciding in real-time. Example log: `"Time: 5hr | Target: 150 | Current Q: 120 | Trying +5% choke | Predicted WHP safe | Move Accepted."`
*   **Execution Loop:** Use a `while` loop with `time.sleep(1)` inside Streamlit to simulate the 1-hour control intervals in real-time. Store historical data in `st.session_state`.

## 5. Execution Phasing for AI IDE
*   **Phase 1:** Create `config.py`, `requirements.txt`, and `simulator.py`. Ensure the mock simulator can be instantiated and `step()` works. (Assume `Autonomous_Choke_Control_Simulated_Dataset.csv` exists in the root folder).
*   **Phase 2:** Create `system_identification.py`. Run it to ensure it generates the step-test plot and saves `predictive_model.pkl`.
*   **Phase 3:** Create `mpc_controller.py`. Write a quick test script to ensure it returns valid moves and respects constraints.
*   **Phase 4:** Create `dashboard.py`. Ensure the Streamlit app runs, updates in real-time, and handles all 3 scenarios smoothly.
```