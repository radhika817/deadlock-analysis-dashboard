# 🛡️ Deadlock Analysis and Resource Allocation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)
[![Pytest](https://img.shields.io/badge/pytest-37%2F37%20passing-brightgreen.svg)](test_logic.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive, visual dashboard to learn, simulate, and analyze how Operating Systems handle deadlocks through **Avoidance (Banker's Algorithm)**, **Detection & Recovery**, and **Prevention Policies**.

Developed for the Operating Systems curriculum (**Activity 2: AI-Assisted Dashboard Development & Deadlock Analysis**).

---

## ⚡ Quick Links
- 🌐 **Live Web App:** [deadlock-analysis-dashboard.streamlit.app](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)
- 🧪 **Test Suite:** 37 passing unit & integration tests (`pytest -v test_logic.py`)
- 📄 **Source Code:** [GitHub Repository](https://github.com/radhika817/deadlock-analysis-dashboard)

---

## 🚀 Getting Started

### Option 1: Live Demo (No Setup Needed)
Open the deployed dashboard directly in any browser:  
👉 **[https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)**

### Option 2: Run Locally (Python 3.10+)

```bash
# 1. Clone repository
git clone https://github.com/radhika817/deadlock-analysis-dashboard.git
cd deadlock-analysis-dashboard

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install dependencies & run app
pip install -r requirements.txt
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

### Option 3: Run with Docker
```bash
docker build -t deadlock-dashboard .
docker run -p 8501:8501 deadlock-dashboard
```

---

## 🎯 Dashboard Overview (4 Core Modules)

| Tab | Name | Key Functionality |
| :---: | :--- | :--- |
| **1** | **🛡️ Deadlock Avoidance** | Implements Dijkstra's Banker's Algorithm. Computes $\text{Need} = \text{Max} - \text{Alloc}$, runs step-by-step safety checks, grants safe requests, and rolls back unsafe requests. |
| **2** | **🔍 Deadlock Detection** | Analyzes both **Single-Instance** (Resource-Allocation & Wait-For Graphs with cycle detection) and **Multiple-Instance** (matrix reduction). Interactive victim termination and recovery. |
| **3** | **⚙️ Prevention & Policies** | Simulates **Resource Ordering** (breaks Circular Wait) and **Request-All-Upfront** (breaks Hold-and-Wait). Shows Coffman condition cards and trade-off metrics. |
| **4** | **📊 Scenarios & Results** | Auto-loads Presets A through F, displays an analytical evaluation table, and provides one-click CSV export. |

---

## 🚦 System Status Banners

The dashboard uses 7 distinct, color-coded badges to clearly distinguish system states:

| Badge | Meaning | Description |
| :--- | :---: | :--- |
| **`SAFE`** | 🟢 Green | A safe execution sequence exists where all processes can finish without deadlock. |
| **`UNSAFE`** | 🟠 Orange | No safe sequence can be guaranteed. *Note: Unsafe does not mean deadlocked—it signals risk, so requests are denied.* |
| **`WAITING / DEFERRED`** | 🔵 Blue | Request is within Need but exceeds Available. The process waits; system state is unchanged. |
| **`DEADLOCK DETECTED`** | 🔴 Red | Unresolvable circular wait or resource deficit confirmed. Deadlocked processes are identified. |
| **`DEADLOCK PREVENTED`** | 🟣 Purple | Prevention policy successfully eliminated a Coffman condition, ensuring zero deadlocks. |
| **`BOUNDARY / VALIDATION`** | 🔷 Teal | Valid edge case accepted (e.g., zero available, zero need). |
| **`INVALID INPUT`** | ⚪ Grey | Syntax or dimension error caught safely (e.g., negative numbers, Allocation > Max). |

---

## 🧪 Preset Scenarios (A through F)

You can instantly test textbook scenarios using the quick-preset buttons:

| Preset | Title | Scenario Summary | Expected Outcome |
| :---: | :--- | :--- | :---: |
| **A** | **Safe State** | Classic 5-process, 3-resource configuration with $\text{Available}=[3,3,2]$. | **`SAFE`**<br>`P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2` |
| **B** | **Deferred Request** | $P_4$ requests $[3,3,0]$. Request is within Need $[4,3,1]$, but exceeds Available $[2,3,0]$. | **`WAITING / DEFERRED`**<br>(State preserved) |
| **C** | **Unsafe Request** | $P_0$ requests $[0,2,0]$. Request $\le$ Need and Available, but temporary state leaves no safe sequence. | **`DENIED / UNSAFE`**<br>(State restored) |
| **D** | **Deadlock Detection** | **Single:** $P_1 \leftrightarrow P_2$ circular wait.<br>**Multi:** $P_1, P_2, P_3, P_4$ locked with $\text{Available}=[0,0,0]$. | **`DEADLOCK DETECTED`**<br>(Both models) |
| **E** | **Deadlock Prevention** | Applies Fixed Resource Ordering ($R_1 < R_2 < R_3 < R_4$). Rejects reverse request from $P_2 \to R_1$. | **`DEADLOCK PREVENTED`**<br>(Circular Wait broken) |
| **F** | **Boundary & Invalid** | Evaluates Zero Available/Need (valid boundaries) and catches negative vectors, out-of-range IDs, and Alloc > Max. | **`VALIDATION TEST`**<br>(6/6 cases pass) |

---

## 📈 Live Analytical Results

| Test Case | System Condition | Actual Evaluated Result | Match |
| :--- | :--- | :--- | :---: |
| **A: Safe State** | Classic 5-process example ($\text{Avail}=[3,3,2]$) | `SAFE: Safe sequence: P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2` | **PASS** |
| **B: Exceeds Available** | $P_4$ requests $[3,3,0]$ with $\text{Avail}=[2,3,0]$ | `WAITING: Process P4 must wait (insufficient available)` | **PASS** |
| **C: Unsafe Request** | $P_0$ requests $[0,2,0]$ with $\text{Avail}=[2,3,0]$ | `DENIED_UNSAFE: Request denied; original state restored` | **PASS** |
| **D: Detection** | Single cycle ($P_1 \leftrightarrow P_2$) & Multi deadlock ($P_1$–$P_4$) | `DEADLOCK DETECTED in single and multiple instance models` | **PASS** |
| **E: Prevention** | Linear Resource Ordering applied to Scenario D | `DEADLOCK PREVENTED: Circular Wait eliminated (0 deadlocked)` | **PASS** |
| **F: Boundary & Errors** | Zero Avail, Zero Need, Negative, Alloc > Max, $P_{99}$ | `2 VALID_BOUNDARY accepted; 4 INVALID safely rejected` | **PASS** |

---

## 🔬 Automated Test Suite

Run all automated unit, integration, and UI state tests with Pytest:

```bash
pytest -v test_logic.py
```

<details>
<summary><b>👉 Click to expand test suite details (37/37 passing)</b></summary>

```text
============================= test session starts ==============================
rootdir: /home/radhika/Desktop/Bankers
collected 37 items

test_logic.py::test_classic_need_matrix_is_correct PASSED                [  2%]
test_logic.py::test_classic_state_is_safe_and_valid_safe_sequence_found PASSED [  5%]
test_logic.py::test_classic_after_granting_p1_p4_requesting_3_3_0_returns_waiting PASSED [  8%]
test_logic.py::test_classic_after_granting_p1_p0_requesting_0_2_0_returns_denied_unsafe_and_state_unchanged PASSED [ 10%]
test_logic.py::test_classic_request_greater_than_need_returns_exceeds_need PASSED [ 13%]
test_logic.py::test_classic_negative_request_and_invalid_process_id_return_invalid PASSED [ 16%]
test_logic.py::test_single_instance_detection_scenario_d PASSED          [ 18%]
test_logic.py::test_multi_instance_detection_scenario_d PASSED           [ 21%]
test_logic.py::test_recovery_single_instance PASSED                      [ 24%]
test_logic.py::test_recovery_multi_instance PASSED                       [ 27%]
test_logic.py::test_resource_ordering_prevention PASSED                  [ 29%]
test_logic.py::test_request_all_upfront_prevention PASSED                [ 32%]
test_logic.py::test_validation_negative_numbers PASSED                   [ 35%]
test_logic.py::test_validation_non_integers PASSED                       [ 37%]
test_logic.py::test_validation_wrong_shapes PASSED                       [ 40%]
test_logic.py::test_validation_allocation_exceeds_maximum_returns_exact_process_and_resource PASSED [ 43%]
test_logic.py::test_validation_invalid_process_id PASSED                 [ 45%]
test_logic.py::test_validation_valid_boundary_available_all_zeros PASSED [ 48%]
test_logic.py::test_validation_valid_boundary_process_with_zero_need PASSED [ 51%]
test_logic.py::test_validation_valid_boundary_all_zero_request PASSED    [ 54%]
test_logic.py::test_validation_never_raises_exceptions PASSED            [ 56%]
test_logic.py::test_detect_single_instance_circular_wait_2_processes PASSED [ 59%]
test_logic.py::test_detect_single_instance_no_deadlock PASSED            [ 62%]
test_logic.py::test_detect_multi_instance_deadlock PASSED                [ 64%]
test_logic.py::test_recovery_by_terminating_process_makes_others_finish PASSED [ 67%]
test_logic.py::test_simulate_prevention_resource_ordering_breaks_circular_wait PASSED [ 70%]
test_logic.py::test_simulate_prevention_request_all_upfront_eliminates_hold_and_wait PASSED [ 72%]
test_logic.py::test_simulate_prevention_multi_instance_both_policies PASSED [ 75%]
test_logic.py::test_get_coffman_breakdown PASSED                         [ 78%]
test_logic.py::test_all_scenarios_runners_and_results_table PASSED       [ 81%]
test_logic.py::test_scenario_a_produces_valid_safe_sequence PASSED       [ 83%]
test_logic.py::test_scenario_b_request_within_need_exceeds_available_waiting_state_unchanged PASSED [ 86%]
test_logic.py::test_scenario_c_unsafe_request_denied_and_state_restored PASSED [ 89%]
test_logic.py::test_scenario_d_separate_deadlocked_scenario_detection_not_banker PASSED [ 91%]
test_logic.py::test_scenario_e_prevention_resource_ordering_prevents_circular_wait PASSED [ 94%]
test_logic.py::test_scenario_f_boundary_cases_and_invalid_inputs_validation PASSED [ 97%]
test_logic.py::test_streamlit_scenario_switching_and_distinct_results PASSED [100%]

======================== 37 passed, 1 warning in 3.00s =========================
```
</details>

---

## 📂 Project Structure

```text
.
├── app.py             # Streamlit application: UI layout, preset router, & themes
├── banker.py          # Avoidance: Need calculation, Safety check, & Request handler
├── detection.py       # Detection: Single-instance (RAG/WFG) & Multi-instance reduction
├── prevention.py      # Prevention: Resource ordering & Upfront allocation simulations
├── scenarios.py       # Presets A through F definitions and dynamic runners
├── validation.py      # Robust input validation & edge-case guards
├── test_logic.py      # 37-test suite (Unit tests + Streamlit AppTest)
├── requirements.txt   # Dependencies: streamlit, numpy, pandas, networkx, matplotlib
├── Dockerfile         # Docker deployment configuration
└── README.md          # Project documentation
```

---

## 🖼️ Screenshots

<details>
<summary><b>👉 Click to view dashboard screenshots</b></summary>

| Avoidance: Safe Sequence & Step Trace | Request Denial & Unsafe Rollback |
| :---: | :---: |
| ![Avoidance Safe State](screenshots/01_avoidance_safe_state.png) | ![Request Handling Rollback](screenshots/02_request_handling_rollback.png) |

| Detection: Single-Instance Cycle Highlight | Detection: Recovery via Victim Termination |
| :---: | :---: |
| ![Detection Graph Cycle](screenshots/03_detection_graph_cycle.png) | ![Detection Recovery](screenshots/04_detection_recovery.png) |

| Prevention: Coffman Breakdown & Trade-offs | Analytical Results Table & CSV Export |
| :---: | :---: |
| ![Prevention Policies](screenshots/05_prevention_policies.png) | ![Analytical Results Table](screenshots/06_analytical_results_table.png) |

</details>

---

## 🎓 Viva Cheat Sheet (Quick Function Guide)

- **`banker.calculate_need(max, alloc)`**: Computes remaining claims ($\text{Need} = \text{Max} - \text{Alloc}$).
- **`banker.safety_check(alloc, need, avail)`**: Finds an execution order where each process's need can be met by current work resources. Returns safe sequence or `UNSAFE`.
- **`banker.handle_request(...)`**: Tests $\text{Request} \le \text{Need}$ and $\text{Request} \le \text{Avail}$, simulates allocation, commits if safe (`GRANTED`), rolls back if unsafe (`DENIED_UNSAFE`).
- **`detection.detect_single_instance(...)`**: Finds directed cycles in the Wait-For Graph (Cycle $\iff$ Deadlock for single-instance).
- **`detection.detect_multi_instance(...)`**: Reduces matrix greedily based on $\text{Request} \le \text{Work}$. Remaining unfinished processes are deadlocked.
- **`prevention.simulate_prevention_single_instance(...)`**: Enforces $F(R_j) > F(R_i)$ to reject reverse requests, eliminating Circular Wait.

---

## 🤖 AI Usage Record

All AI assistance was validated and verified against OS academic standards:
- **Core Algorithms:** Banker's algorithm, graph cycle detection, and matrix reduction implemented and cross-referenced with Silberschatz OS concepts.
- **Visuals & UI:** Responsive Streamlit interface with 4 tabs, theme toggle, and custom status badges.
- **Debugging & State Integrity:** Centralized scenario dispatching in `apply_preset()` to ensure distinct, authentic outcomes across presets A–F with 37 automated tests.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
