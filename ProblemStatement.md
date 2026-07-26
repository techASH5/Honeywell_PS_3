# Autonomous Production Choke Controller for a Single Naturally Flowing Oil Well

## Background
In oil and gas production, a **production choke** is used to regulate the flow of fluids from a well. Opening the choke generally increases production flow rate but reduces well pressure. Excessive choke opening may result in unsafe operating conditions, while conservative choke settings can lead to lost production opportunities.

Today, choke adjustments are often performed manually by operators based on observed process measurements and operational experience. As operators may be responsible for large numbers of wells, manual optimization becomes difficult to scale and may lead to inconsistent operation.

The objective of this hackathon is to develop an **autonomous choke control solution** for a **single naturally flowing well**. The controller should automatically determine the optimal choke position required to achieve a desired production target while maintaining the well within its safe operating envelope. Choke movements must also respect a maximum allowable ramp rate. The controller should achieve production targets only when these constraints can be maintained safely.

## Challenge
A simplified simulator representing a **single naturally flowing oil well** will be provided.

The simulator will have:
**Input**
- Production choke opening (%) u

**Outputs**
- Oil Flow Rate (Q)
- Wellhead Pressure (WHP)
- Flowline Pressure (FLP)
- Bottom Hole Pressure (BHP)

The well is assumed to be a **single naturally flowing oil well** with one production choke as the manipulated variable. Students are not required to develop the simulator. A simulator will be provided and should be treated as the source of process behavior.

The controller will execute at fixed control intervals.

**Control Interval (Ts): 1 hour**

At each control interval, the controller receives:
- Current Oil Flow Rate (Q)
- Wellhead Pressure (WHP)
- Flowline Pressure (FLP)
- Bottom Hole Pressure (BHP)
- Current Choke Position

and calculates the next choke position.

Students are expected to:
1. Study the process behavior by applying choke step changes.
2. Plot the response of flow and pressures.
3. Develop a simple dynamic model representing the process behavior.
4. Implement a predictive controller capable of selecting optimal choke movements.
5. Track production rate targets while ensuring that operating constraints are never violated.
6. Demonstrate controller performance under different operating scenarios.

A simplified MPC implementation based on brute-force candidate evaluation is acceptable. Use of optimization libraries is optional.

## Constraints
The controller must operate within the following limits:

**Choke Constraints**
- 0 % ≤ Choke Opening ≤ 100 %
- Maximum choke movement per control step ≤ 5 %

**Choke Ramp Rate Limit**
- Maximum Choke Movement = ±5 % per control interval

**Active Operating Constraints**
The following variables must always remain within their safe operating ranges:
1. Wellhead Pressure (WHP)
2. Flowline Pressure (FLP)
3. Bottom Hole Pressure (BHP)

Any candidate control action predicted to violate these limits must be rejected.

**Production Objective**
The controller should:
- Achieve the target oil production rate whenever feasible.
- If the target rate cannot be achieved safely, operate at the maximum achievable production rate without violating constraints.

**Additional Industrial Variables (Informational)**
Real-world autonomous choke control systems may additionally consider:
- Wellhead Temperature (WHT)
- Annulus Pressure (AP)
These variables are not active constraints in this challenge but should be recognized as part of a complete production operating envelope.

## Simulator Assumptions
To keep the challenge focused and manageable, the following assumptions apply:
- Single well only
- Naturally flowing well
- Single production choke
- No gas lift optimization
- No ESP optimization
- No facility network interactions
- No changing reservoir properties
- No changing GOR or water cut

The simulator should be treated as the source of process behavior.

## Demonstration Scenarios
Students should demonstrate controller performance for the following scenarios:

**Scenario A – Startup to Target**
Controller brings the well from startup conditions to a specified production target.

**Scenario B – Target Tracking**
Production target changes during operation.
Example: 100 bbl/hr → 150 bbl/hr
The controller should achieve the new target while respecting WHP, FLP, BHP and choke ramp-rate constraints.

**Scenario C – Infeasible Target**
Requested production target exceeds what can be achieved safely.
Controller should:
- Respect all pressure constraints
- Reject unsafe operating conditions
- Settle at the maximum achievable safe flow rate

## Expectations / Deliverables
Students should submit:
**Technical Deliverables**
- Python notebook or Python code
- Open-loop step-test analysis
- Dynamic model identification
- Autonomous choke controller implementation
- Results for all three scenarios

**Plots**
For each scenario provide trends for:
- Target Oil Rate
- Actual Oil Rate
- Wellhead Pressure
- Flowline Pressure
- Bottom Hole Pressure
- Choke Position

**Presentation**
Use the provided template but include additional details as below:
*Process Understanding & Model:* Step-test results, Model assumptions, Dynamic model developed
*Control Strategy:* Prediction methodology, Choke move selection logic, Constraint handling approach
*Results:* Scenario outcomes, Tracking performance, Safety performance, Lessons learned

## Domain Terminology & Process Overview
- **Production Choke:** A control valve installed at the wellhead that regulates the flow of fluids from the well.
- **Naturally Flowing Well:** Produces fluids using the reservoir's natural energy without artificial lift systems.
- **Oil Flow Rate (Q):** Amount of oil produced by the well (bbl/hr).
- **Wellhead Pressure (WHP):** Pressure measured at the wellhead. Active constraint.
- **Flowline Pressure (FLP):** Pressure measured in the flowline downstream of the wellhead. Active constraint.
- **Bottom Hole Pressure (BHP):** Pressure at the reservoir/wellbore interface. Active constraint.
- **Wellhead Temperature (WHT) / Annulus Pressure (AP):** Informational only.
- **Operating Envelope:** Defines the safe operating range of a well (WHP limits, FLP limits, BHP limits, Choke movement limits).
