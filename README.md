# Deadlock Analysis and Resource Allocation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)
[![Pytest](https://img.shields.io/badge/pytest-37%2F37%20passing-brightgreen.svg)](test_logic.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive, textbook-grade dashboard for learning, simulating, evaluating, and demonstrating how modern operating systems **avoid**, **detect**, **prevent**, and **recover from** deadlocks. Built for the Operating Systems Internal Assessment (**Activity 2: AI-Assisted Dashboard Development and Deadlock Analysis**).

> **🌐 Live Deployment:** [https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)  
> **Course:** Operating Systems  
> **Topic:** Deadlock Avoidance (Dijkstra's Banker's Algorithm), Detection (RAG/WFG & Matrix Reduction), Prevention (Coffman Conditions), and Recovery  
> **Entry Point:** `app.py`  
> **Automated Test Suite:** `pytest -v test_logic.py` (**37/37 tests passing**)

---

## Table of Contents
1. [Live Deployment & Quick Access](#live-deployment--quick-access)
2. [How to Run](#how-to-run)
   - [Live Cloud Application](#-live-cloud-application-no-installation-needed)
   - [Running Locally with Python](#-running-locally)
   - [Running with Docker](#-running-with-docker)
3. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
4. [Tools and Technologies Used](#tools-and-technologies-used)
5. [The 7 Status Banners and Visual Standards](#the-7-status-banners-and-visual-standards)
6. [Dashboard Tabs and Functional Modules](#dashboard-tabs-and-functional-modules)
   - [Tab 1: Deadlock Avoidance (Banker's Algorithm)](#tab-1-deadlock-avoidance-bankers-algorithm)
   - [Tab 2: Deadlock Detection & Graph Visualization](#tab-2-deadlock-detection--graph-visualization)
   - [Tab 3: Deadlock Prevention Policies & Trade-offs](#tab-3-deadlock-prevention-policies--trade-offs)
   - [Tab 4: Test Scenarios & Analytical Results Table](#tab-4-test-scenarios--analytical-results-table)
7. [Comprehensive Preset Scenarios (A through F)](#comprehensive-preset-scenarios-a-through-f)
8. [Analytical Results Table](#analytical-results-table)
9. [Automated Testing Suite (37 Tests)](#automated-testing-suite-37-tests)
10. [AI Usage Record](#ai-usage-record)
11. [Screenshots](#screenshots)
12. [Viva Preparation & Technical Reference](#viva-preparation--technical-reference)

---

## Live Deployment & Quick Access

Access the full interactive dashboard directly on any browser (no installation or setup required):  
👉 **[https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)**

---

## How to Run

### 🌐 Live Cloud Application (No Installation Needed)
Launch directly from your mobile, tablet, or desktop browser at:
`https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/`

---

### 💻 Running Locally

#### Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, or 3.14)
- `pip` package manager
- Web browser (Chrome, Edge, Firefox, Safari)

#### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/radhika817/deadlock-analysis-dashboard.git
cd deadlock-analysis-dashboard

# 2. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py

# 5. Execute the automated test suite
pytest -v test_logic.py
```

The interactive dashboard will automatically open in your default browser at:  
**`http://localhost:8501`**

---

### 🐳 Running with Docker

You can also run the dashboard in a completely isolated Docker container:

```bash
# 1. Build the Docker container image
docker build -t deadlock-dashboard .

# 2. Run the container on port 8501
docker run -p 8501:8501 deadlock-dashboard
```

Navigate to `http://localhost:8501` in your browser.

---

## Project Architecture & Directory Structure

The project strictly decouples core operating system algorithms from UI rendering. This modular design ensures that all logic can be independently inspected, tested, and defended during an academic viva:

```text
.
├── app.py             # Streamlit application: UI layout, theme toggle, preset loader, & status banners
├── banker.py          # Avoidance: Need matrix, Safety algorithm trace, & Resource-Request handler
├── detection.py       # Detection: Single-instance (RAG/WFG cycles) & Multi-instance matrix reduction
├── prevention.py      # Prevention: Resource Ordering & Request-All-Upfront simulations + trade-off metrics
├── scenarios.py       # Preset scenarios A through F definitions and dynamic execution runners
├── validation.py      # Robust validation & boundary handling (never raises unhandled exceptions)
├── test_logic.py      # 37-case Pytest automated test suite (including Streamlit AppTest verification)
├── requirements.txt   # Pinned project dependencies
├── Dockerfile         # Production-ready containerization spec
└── README.md          # Comprehensive documentation, viva guide, and project report
```

---

## Tools and Technologies Used

| Tool / Library | Version | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Core language for algorithm implementation and mathematical simulation |
| **Streamlit** | $\ge$ 1.30.0 | High-performance interactive UI framework, responsive layouts, and state management |
| **NumPy** | $\ge$ 1.24.0 | Vectorized matrix operations for Need calculation, Allocation updates, and safety checks |
| **Pandas** | $\ge$ 2.0.0 | Tabular structuring of process allocations, step-by-step traces, and analytical reports |
| **NetworkX** | $\ge$ 3.0.0 | Graph data modeling for Resource-Allocation Graphs (RAG) and Wait-For Graphs (WFG) |
| **Matplotlib** | $\ge$ 3.7.0 | Publication-grade rendering of bipartite graphs and circular wait cycle highlighting |
| **Pytest** | $\ge$ 7.0.0 | Automated testing framework executing 37 rigorous test cases with zero failures |

---

## The 7 Status Banners and Visual Standards

To eliminate ambiguity between deadlock concepts during classroom demonstrations and vivas, the dashboard implements **7 distinct status banners** with clear semantic color-coding:

| Status Badge | Semantic Color | Border & Background | Exact Technical Meaning |
| :--- | :--- | :--- | :--- |
| **`SAFE`** | **Green** | `#10B981` border<br>`#ECFDF5` background | A complete safe sequence exists where every process can acquire its maximum claim and terminate. Deadlock is mathematically impossible. |
| **`UNSAFE`** | **Orange** | `#EA580C` border<br>`#FFF7ED` background | No safe execution sequence can be guaranteed if processes demand maximum claims. **Unsafe is NOT deadlock**: deadlock has not yet occurred, but avoidance denies the request to prevent risk. |
| **`WAITING / DEFERRED`** | **Blue** | `#3B82F6` border<br>`#EFF6FF` background | Request is valid ($\text{Request} \le \text{Need}$), but temporarily exceeds available resources ($\text{Request} > \text{Available}$). The process is put to wait; system state is strictly unchanged. |
| **`DEADLOCK DETECTED`** | **Red** | `#EF4444` border<br>`#FEF2F2` background | Detection algorithm confirms that one or more processes are locked in an unresolvable cycle (single-instance) or resource deficit (multi-instance). |
| **`DEADLOCK PREVENTED`** | **Purple / Indigo** | `#8B5CF6` border<br>`#F5F3FF` background | A prevention policy structurally breaks one of the four Coffman conditions (e.g., Resource Ordering breaks Circular Wait), ensuring zero deadlocks. |
| **`BOUNDARY / VALIDATION TEST`** | **Teal** | `#0D9488` border<br>`#F0FDFA` background | Boundary configurations (e.g., zero available, zero need) evaluated and accepted safely without exceptions. |
| **`INVALID INPUT`** | **Grey** | `#6B7280` border<br>`#F3F4F6` background | Syntactically illegal input: negative integers, non-numeric values, out-of-range process IDs, mismatched vector dimensions, or $\text{Allocation} > \text{Maximum}$. |

---

## Dashboard Tabs and Functional Modules

### Tab 1: Deadlock Avoidance (Banker's Algorithm)
- **Interactive Configuration:** Editable matrices for Allocation, Maximum, and Available vector with quick presets and Reset.
- **Automated Need Matrix:** Automatically computes and displays $\text{Need} = \text{Maximum} - \text{Allocation}$.
- **Step-by-Step Safety Trace:** Detailed trace table displaying Process picked, Need vector, Work before, Allocation released, and Work after.
- **Resource Request Handler:** Evaluates requests against Need and Available, performs trial allocation, commits if safe, or rolls back with a Before vs After table if unsafe.
- **Educational Callout:** Dedicated *"Unsafe is NOT the same as Deadlock"* explanation panel.

### Tab 2: Deadlock Detection & Graph Visualization
- **Model Selector:** Clearly distinguishes between **Single-Instance** and **Multiple-Instance** resource models.
- **Single-Instance Detection:**
  - Bipartite Resource-Allocation Graph (RAG) and Wait-For Graph (WFG) drawn via NetworkX and Matplotlib.
  - Circular wait cycles highlighted in vivid red (`#EF4444`).
  - Interactive Recovery: Terminates a selected victim process (suggests process holding the most resources) and re-evaluates the cycle-free graph.
- **Multiple-Instance Detection:**
  - Matrix reduction algorithm ($\text{Work} = \text{Available}$, $\text{Finish}[i] = \text{True}$ if $\text{Alloc}[i]==0$, greedily finds $\text{Request}[i] \le \text{Work}$).
  - Theoretical Callout: *"A cycle alone is not conclusive proof of deadlock for multiple-instance resources."* Explains potential circular wait vs confirmed deadlock.
  - Interactive Recovery: Reclaims resources and displays Before vs After comparison proving remaining processes finish.

### Tab 3: Deadlock Prevention Policies & Trade-offs
- **Policy Selection:**
  - **Policy (a): Resource Ordering** (strict linear ranking $F(R_j) > F(R_i)$) $\implies$ Eliminates **CIRCULAR WAIT**.
  - **Policy (b): Request-All-Upfront** (processes must acquire all needed resources before execution) $\implies$ Eliminates **HOLD AND WAIT**.
- **Coffman Condition Breakdown:** 4-column card grid showing which condition is eliminated (`🚫 ELIMINATED BY POLICY`) and which remain active (`🔒 ACTIVE`).
- **Side-by-Side Verification:** Compares original deadlocked scenario against the prevention-constrained execution proving zero deadlocks.
- **Quantitative Trade-offs Panel:** Displays real scenario metrics:
  - *Lower Resource Utilization:* Number of resource units held idle or unallocated.
  - *Reduced Concurrency:* Percentage drop in parallel process execution.
  - *Possible Starvation:* Identification of affected processes.
  - *Longer Waiting Times:* Latency penalty for deferred processes.

### Tab 4: Test Scenarios & Analytical Results Table
- **Preset Quick-Load Buttons:** Load and execute textbook scenarios A through F with automatic tab switching and live trace output.
- **Dynamic Analytical Table:** Real run evaluations compiled into a table with columns: `Test Case | System Condition | Actual Result | Expected Result | Match (PASS/FAIL)`.
- **CSV Export:** `📥 Download results as CSV` exports the verified results table for project submission and reporting.

---

## Comprehensive Preset Scenarios (A through F)

The dashboard provides six textbook preset scenarios demonstrating the complete spectrum of OS deadlock handling:

### 🔹 Scenario A: Safe State (Silberschatz Classic 5-Process, 3-Resource)
- **Goal:** Demonstrates Dijkstra's Banker's Algorithm finding a guaranteed safe completion sequence.
- **Configuration:** 5 processes ($P_0$–$P_4$), 3 resource types ($A, B, C$), Total resources $=[10, 5, 7]$, Available $=[3, 3, 2]$.
- **Execution:** Banker's safety check evaluates process claims iteratively.
- **Outcome:** **`SAFE`** with valid safe sequence $\langle P_1 \to P_3 \to P_4 \to P_0 \to P_2 \rangle$.

### 🔹 Scenario B: Request Exceeds Available (Deferred / Waiting)
- **Goal:** Demonstrates request handling when a process requests resources within its remaining Need but exceeding current Available.
- **Configuration:** Starting after granting $P_1$ request $[1, 0, 2]$, Available becomes $[2, 3, 0]$. Now $P_4$ requests $[3, 3, 0]$.
- **Evaluation:** $P_4$'s $\text{Request} \le \text{Need}$ holds ($[3, 3, 0] \le [4, 3, 1]$), but $\text{Request} \le \text{Available}$ fails ($3 > 2$ for resource $A$).
- **Outcome:** **`WAITING / DEFERRED`**. Process $P_4$ must wait. The allocation state is left strictly unchanged.

### 🔹 Scenario C: Unsafe Request (Denied & Restored)
- **Goal:** Demonstrates request denial and immediate rollback when a request is within both Need and Available but leaves no safe sequence.
- **Configuration:** With Available $=[2, 3, 0]$, process $P_0$ requests $[0, 2, 0]$.
- **Evaluation:** $[0, 2, 0] \le \text{Need}$ holds and $[0, 2, 0] \le \text{Available}$ holds. A trial allocation temporarily reduces Available to $[2, 1, 0]$. However, in this simulated state, no process can complete:
  - $P_0 \text{ Need} = [7, 2, 3] > [2, 1, 0]$
  - $P_1 \text{ Need} = [0, 2, 0] > [2, 1, 0]$ (needs 2 of $B$, only 1 available)
  - $P_2 \text{ Need} = [6, 0, 0] > [2, 1, 0]$
  - $P_3 \text{ Need} = [0, 1, 1] > [2, 1, 0]$ (needs 1 of $C$, 0 available)
  - $P_4 \text{ Need} = [4, 3, 1] > [2, 1, 0]$
- **Outcome:** **`DENIED / UNSAFE`**. The trial allocation is rolled back, and the original safe state is restored.

### 🔹 Scenario D: Deadlock Detection (Single & Multi-Instance)
- **Goal:** Loads a separate deadlocked system and runs detection algorithms instead of Banker's avoidance.
- **Single-Instance Model:** $P_1$ holds $R_1$, requests $R_2$; $P_2$ holds $R_2$, requests $R_1$. NetworkX detects directed cycle $P_1 \to P_2 \to P_1$.
- **Multiple-Instance Model:** 5 processes with Available $=[0, 0, 0]$ and unresolved requests. Matrix reduction confirms deadlocked processes $\{P_1, P_2, P_3, P_4\}$.
- **Outcome:** **`DEADLOCK DETECTED`** across both models. Automatically switches to Tab 2 with graph and matrix visualizations.

### 🔹 Scenario E: Deadlock Prevention (Fixed Resource Ordering)
- **Goal:** Demonstrates how imposing strict resource ordering ($R_1 < R_2 < R_3 < R_4$) prevents circular wait.
- **Execution:** Process $P_2$ holding $R_2$ (rank 2) attempting to request $R_1$ (rank 1) is rejected ($1 < 2$).
- **Outcome:** **`DEADLOCK PREVENTED`**. Circular wait is mathematically eliminated; 0 processes deadlocked. Automatically switches to Tab 3.

### 🔹 Scenario F: Boundary Cases & Invalid Inputs
- **Goal:** Rigorously verifies boundary condition acceptance and robust rejection of illegal inputs.
- **Evaluated Cases:**
  1. *F1 (Zero Available):* $\text{Available} = [0, 0, 0]$ with valid allocation $\to$ **`VALID_BOUNDARY`** (Accepted).
  2. *F2 (Zero Need):* Process with $\text{Need} = [0, 0, 0]$ $\to$ **`VALID_BOUNDARY`** (Accepted).
  3. *F3 (Allocation > Maximum):* Process $P_0$ holding more than its declared maximum $\to$ **`INVALID`** (Rejected with exact cell identification).
  4. *F4 (Negative Request Vector):* Request $[-1, 0, 2]$ $\to$ **`INVALID`** (Rejected).
  5. *F5 (Invalid Process ID):* Request for non-existent process $P_{99}$ $\to$ **`INVALID`** (Rejected).
  6. *F6 (Mismatched Vector Dimensions):* 2-element request vector on a 3-resource system $\to$ **`INVALID`** (Rejected).
- **Outcome:** **`BOUNDARY / VALIDATION TEST`**. All 6 boundary cases handled safely without unhandled exceptions.

---

## Analytical Results Table

All Actual Results below are evaluated dynamically from the underlying algorithm engines:

| Test Case | System Condition | Actual Result | Expected Result | Match |
| :--- | :--- | :--- | :--- | :---: |
| **A: Safe State** | Classic 5-process, 3-resource example ($\text{Avail}=[3, 3, 2]$) | `SAFE: Safe sequence found: P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2` | `SAFE (Valid safe sequence exists: P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2)` | **PASS** |
| **B: Request Exceeds Available** | Grant $P_1$ $(1,0,2)$, then $P_4$ requests $(3,3,0)$ with $\text{Avail}=[2, 3, 0]$ | `WAITING: Process P4 must wait: insufficient resources currently available...` | `WAITING / DEFERRED (Request within Need [4,3,1], exceeds Available [2,3,0])` | **PASS** |
| **C: Unsafe Request** | Grant $P_1$ $(1,0,2)$, then $P_0$ requests $(0,2,0)$ with $\text{Avail}=[2, 3, 0]$ | `DENIED_UNSAFE: Request is DENIED: simulated state leaves no safe sequence...` | `DENIED_UNSAFE (Request within Available but simulated state is unsafe; rolled back)` | **PASS** |
| **D: Deadlock Detection** | Single-instance cycle ($P_1 \leftrightarrow P_2$) + Multi-instance matrix ($\text{Avail}=[0,0,0]$) | `DEADLOCK DETECTED (Single-Instance Cycle: ['P1', 'P2', 'P1']; Multi-Instance Deadlocked: ['P1', 'P2', 'P3', 'P4'])` | `DEADLOCK DETECTED in both single-instance and multi-instance models` | **PASS** |
| **E: Prevention Applied to D** | Resource Ordering (eliminates Circular Wait) & Request-All-Upfront (eliminates Hold-and-Wait) | `DEADLOCK PREVENTED (Ordering eliminated Circular Wait, 0 deadlocked; Upfront eliminated Hold and Wait, 0 deadlocked)` | `DEADLOCK PREVENTED (Circular Wait & Hold-and-Wait eliminated; zero deadlocks)` | **PASS** |
| **F: Boundary & Invalid Inputs** | Zero Available, Zero Need, Invalid $P_{99}$, Negative Request, Alloc > Max, Length Mismatch | `All 6 boundary/invalid cases evaluated correctly: 2 VALID_BOUNDARY accepted; 4 INVALID rejected` | `VALID_BOUNDARY accepted (Zero Avail, Zero Need); INVALID rejected (P99, neg, Alloc>Max, len 2)` | **PASS** |

---

## Automated Testing Suite (37 Tests)

The test suite in [`test_logic.py`](test_logic.py) provides comprehensive unit, integration, and UI state verification using `pytest`:

```bash
$ .venv/bin/pytest test_logic.py -v
============================= test session starts ==============================
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

======================== 37 passed, 1 warning in 6.48s =========================
```

---

## AI Usage Record

In strict compliance with academic integrity guidelines, all generative AI tool interactions during project development and debugging are transparently documented:

| AI Tool Used | Stage & Prompt Focus | Code / Content Generated | Student Verification & Modifications |
| :--- | :--- | :--- | :--- |
| **Antigravity (Gemini 3.8 Flash)** | Project architecture, modular decoupling, and Banker's algorithm logic | `banker.py` core functions (`calculate_need`, `safety_check`, `handle_request`) | Verified against Dijkstra's operating systems textbook formulation; added comprehensive docstrings. |
| **Antigravity (Gemini 3.8 Flash)** | Deadlock detection algorithms and NetworkX graph visualization | `detection.py` (`detect_single_instance`, `detect_multi_instance`, `draw_graphs_to_figure`, recovery) | Confirmed bipartite node separation (circles for processes, squares for resources) and cycle highlighting in `#EF4444`. |
| **Antigravity (Gemini 3.8 Flash)** | Deadlock prevention simulation and Coffman breakdown | `prevention.py` (`simulate_prevention_single_instance`, `simulate_prevention_multi_instance`, trade-off calculations) | Verified mathematical impossibility of cycles under linear ordering; checked idle resource capacity formulas. |
| **Antigravity (Gemini 3.8 Flash)** | Robust validation and automated test suite | `validation.py` and `test_logic.py` (30 original unit tests) | Executed `pytest -v test_logic.py` ensuring 100% pass rate with zero uncaught exceptions. |
| **Antigravity (Gemini 3.8 Flash)** | Streamlit 4-tab user interface and visual styling | `app.py` UI layout, CSS design system, dark mode toggle, and CSV export functionality | Validated responsive wide-layout, distinct status banners, and dynamic scenario loading. |
| **Antigravity (Gemini 3.8 Flash)** | **Debugging & UI State Integrity**: All scenario buttons showing identical `SAFE` result | Diagnosed root causes in `app.py` & `banker.py`; implemented centralized preset dispatcher `apply_preset()`; bound `st.tabs` key; added 7 automated tests for Scenarios A–F and UI state integrity | Verified that Presets A through F produce distinct, authentic OS outcomes and that all 37 unit and AppTest tests pass. |

---

## Screenshots

Application demonstration screenshots located in the [`screenshots/`](screenshots/) directory:

### 1. Deadlock Avoidance (Banker's Algorithm Safe State)
![Tab 1: Safe State and Need Matrix](screenshots/01_avoidance_safe_state.png)
*Figure 1: Tab 1 demonstrating the automatic calculation of the Need matrix, the initial Available vector, and the step-by-step Work updates leading to safe sequence `P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2`.*

### 2. Resource Request Handling (Deferred / Unsafe Rollback)
![Tab 1: Request Handling and Rollback](screenshots/02_request_handling_rollback.png)
*Figure 2: Tab 1 showing Scenario B (Waiting/Deferred) and Scenario C (Unsafe Request Denied with state restored and "Unsafe is not Deadlock" educational panel).*

### 3. Deadlock Detection & Cycle Highlighting (Single-Instance RAG / WFG)
![Tab 2: Single-Instance Wait-For Graph](screenshots/03_detection_graph_cycle.png)
*Figure 3: Tab 2 displaying the Resource-Allocation Graph and Wait-For Graph with the circular wait between P1 and P2 highlighted in red (`#EF4444`).*

### 4. Deadlock Recovery via Victim Termination
![Tab 2: Recovery Before vs After](screenshots/04_detection_recovery.png)
*Figure 4: Tab 2 showing victim process termination (P1), release of held resources, updated Work vector, and subsequent cycle-free graph.*

### 5. Deadlock Prevention & Coffman Condition Breakdown
![Tab 3: Prevention Policies and Trade-offs](screenshots/05_prevention_policies.png)
*Figure 5: Tab 3 showing the elimination of Circular Wait under Resource Ordering and Hold-and-Wait under Request-All-Upfront, accompanied by real scenario trade-off metrics.*

### 6. Analytical Results Table & CSV Export
![Tab 4: Analytical Results Table](screenshots/06_analytical_results_table.png)
*Figure 6: Tab 4 displaying the live-evaluated analytical results table for Scenarios A through F with the CSV download button.*

---

## Viva Preparation & Technical Reference

Below is a function-by-function guide in clear language explaining every module and routine for viva examination:

### 1. `banker.py` (Deadlock Avoidance)
- **`calculate_need(maximum, allocation)`**:  
  Calculates $\text{Need} = \text{Maximum} - \text{Allocation}$. Represents the maximum additional resources each process may claim before terminating.
- **`safety_check(allocation, need, available)`**:  
  Initializes $\text{Work} = \text{Available}$ and marks all processes unfinished ($\text{Finish}[i] = \text{False}$). Iteratively searches for an unfinished process where $\text{Need}[i] \le \text{Work}$. Upon finding one, simulates completion: $\text{Work} += \text{Alloc}[i]$, $\text{Finish}[i] = \text{True}$. If all processes finish, the state is **`SAFE`** and the execution sequence is returned; otherwise, it is **`UNSAFE`**.
- **`handle_request(process_id, request, allocation, maximum, available)`**:  
  Handles an incoming resource request in five rigorous steps:
  1. Validates input vector dimensions and non-negativity.
  2. Confirms $\text{Request} \le \text{Need}$; returns `EXCEEDS_NEED` if violated.
  3. Confirms $\text{Request} \le \text{Available}$; returns `WAITING` if insufficient resources exist (state remains untouched).
  4. Temporarily modifies state copies: $\text{Avail}' = \text{Avail} - \text{Req}$, $\text{Alloc}' = \text{Alloc} + \text{Req}$, $\text{Need}' = \text{Need} - \text{Req}$.
  5. Executes `safety_check`. If safe, commits the allocation (`GRANTED`); if unsafe, discards modifications and restores initial state (`DENIED_UNSAFE`).

### 2. `detection.py` (Deadlock Detection)
- **`detect_single_instance(allocation_edges, request_edges)`**:  
  Builds a directed Wait-For Graph (WFG) where an edge $P_i \to P_j$ exists if $P_i$ waits for a resource held by $P_j$. Runs NetworkX cycle detection. In single-instance resource systems, a directed cycle is both **necessary and sufficient** for deadlock.
- **`detect_multi_instance(allocation, request, available)`**:  
  Implements the matrix detection algorithm. Initializes $\text{Work} = \text{Available}$. Sets $\text{Finish}[i] = \text{True}$ if $\text{Alloc}[i] == 0$, else $\text{False}$. Repeatedly finds unfinished process $i$ where $\text{Request}[i] \le \text{Work}$ and reclaims resources: $\text{Work} += \text{Alloc}[i]$. Any processes remaining $\text{Finish}[i] == \text{False}$ are deadlocked.
- **`terminate_process(process_id, allocation, request, available)`**:  
  Simulates process termination by reclaiming its allocated resources ($\text{Avail} += \text{Alloc}[\text{victim}]$) and zeroing out its rows in Allocation and Request.
- **`draw_graphs_to_figure(rag, wfg, cycles)`**:  
  Produces a side-by-side Matplotlib visualization of the Resource-Allocation Graph (bipartite: circular process nodes, square resource nodes) and Wait-For Graph, highlighting cycle edges in red (`#EF4444`).

### 3. `prevention.py` (Deadlock Prevention)
- **`get_coffman_conditions_summary()`**:  
  Returns structured educational documentation on the four necessary Coffman conditions: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.
- **`get_coffman_breakdown(policy_key)`**:  
  Returns the exact status of all 4 Coffman conditions under the active policy, explicitly flagging the eliminated condition (`🚫 ELIMINATED BY POLICY`) vs active conditions (`🔒 ACTIVE`).
- **`simulate_prevention_single_instance(...)`**:  
  Applies prevention policies to a single-instance system. Under **Resource Ordering**, rejects backward requests violating increasing priority ($F(R_j) \le F(R_i)$), breaking circular wait. Under **Request-All-Upfront**, requires all resources to be allocated at start, eliminating hold-and-wait. Computes trade-off metrics (idle resources, concurrency reduction).
- **`simulate_prevention_multi_instance(...)`**:  
  Applies prevention policies to multi-instance systems and computes quantitative trade-off penalties.

### 4. `validation.py` (Input Validation)
- **`validate_matrix(matrix, expected_rows, expected_cols, name)`**:  
  Verifies that matrices contain finite, non-negative integers with expected dimensions.
- **`validate_vector(vector, expected_len, name)`**:  
  Verifies that vectors contain finite, non-negative integers. Identifies all-zero vectors as valid boundaries (`VALID_BOUNDARY`).
- **`validate_process_id(process_id, num_processes)`**:  
  Validates that process identifier is within $[0, \text{num\_processes} - 1]$.
- **`validate_configuration(allocation, maximum, available)`**:  
  Ensures $\text{Allocation}[i][j] \le \text{Maximum}[i][j]$ for every cell. If violated, returns the exact process and resource label (e.g. `Allocation exceeds Maximum at process P0, resource B: 5 > 2`). Identifies valid boundary cases.
- **`validate_request(process_id, request, num_processes, num_resources)`**:  
  Validates request dimensions and values without raising exceptions.

### 5. `scenarios.py` (Preset Scenarios & Execution Runners)
- **`get_scenario_a()` to `get_scenario_f()`**:  
  Provide textbook definitions and initial matrices for Scenarios A through F.
- **`run_scenario_a()` to `run_scenario_f()`**:  
  Dynamically execute the core algorithms on each scenario and compute live actual results and PASS/FAIL matches.
- **`get_all_scenarios_results_table()`**:  
  Executes all scenarios and compiles them into a structured DataFrame with columns `Test Case | System Condition | Actual Result | Expected Result | Match (PASS/FAIL)`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
