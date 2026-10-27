# Deadlock Analysis and Resource Allocation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)

An interactive dashboard for learning, simulating, and evaluating how operating systems **avoid**, **detect**, **prevent**, and **recover from** deadlocks. Built for the Operating Systems Internal Assessment (Activity 2: AI-Assisted Dashboard Development and Deadlock Analysis).

> **🌐 Live Deployment:** [https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)  
> **Course:** Operating Systems  
> **Topic:** Deadlock Avoidance (Banker's Algorithm), Detection, Prevention, and Recovery  
> **Entry Point:** `app.py`  
> **Test Suite:** `pytest -v test_logic.py` (30/30 tests passing)

---

## Table of Contents
1. [Live Deployment](#live-deployment)
2. [How to Run](#how-to-run)
3. [Project Structure](#project-structure)
4. [Tools and Technologies Used](#tools-and-technologies-used)
5. [The 5 Status Messages and Visual Standards](#the-5-status-messages-and-visual-standards)
6. [Dashboard Tabs and Features](#dashboard-tabs-and-features)
7. [Analytical Results Table (Scenarios A - F)](#analytical-results-table-scenarios-a---f)
8. [AI Usage Record](#ai-usage-record)
9. [Screenshots](#screenshots)
10. [Viva Preparation & Function Reference](#viva-preparation--function-reference)

---

## How to Run

### 🌐 Live Cloud App (No Installation Needed)
Access the live interactive application immediately on any device:  
👉 **[https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/](https://deadlock-analysis-dashboard-gmzoczappdqctv3szzrdim9.streamlit.app/)**

### 💻 Running Locally

#### Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, or 3.14)
- Web browser (Chrome, Firefox, Edge, Safari)

### Step-by-Step Execution

```bash
# 1. Clone or navigate to the repository directory
cd Bankers

# 2. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the interactive Streamlit dashboard
streamlit run app.py

# 5. Execute the automated unit test suite
pytest -v test_logic.py
```

The interactive dashboard will automatically open in your web browser at:
`http://localhost:8501`

---

## Project Structure

The project strictly decouples core algorithm logic from the user interface, ensuring code can be inspected, tested, and explained in an academic viva:

```text
.
├── app.py             # Streamlit web application & 4-tab user interface
├── banker.py          # Avoidance: Need calculation, Safety check, and Request handling
├── detection.py       # Detection: Single-instance (RAG/WFG cycles) & Multi-instance matrix algorithm
├── prevention.py      # Prevention: Resource Ordering & Request-All-Upfront simulations + trade-offs
├── validation.py      # Input validation & friendly boundary/error handling (never raises exceptions)
├── scenarios.py       # Preset scenarios A to F definitions and dynamic execution runners
├── test_logic.py      # Comprehensive 30-case Pytest test suite validating all algorithms
├── requirements.txt   # Pinned project dependencies
└── README.md          # Comprehensive documentation, viva guide, and project report
```

---

## Tools and Technologies Used

| Tool / Library | Version | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Core programming language for algorithm implementation |
| **Streamlit** | Latest | Interactive web application framework, responsive layout, and state management |
| **NumPy** | Latest | Matrix and vector vectorization for Need calculations and resource allocations |
| **Pandas** | Latest | Tabular representation of allocation matrices, step traces, and analytical reports |
| **NetworkX** | Latest | Directed graph modeling for Resource-Allocation Graphs (RAG) and Wait-For Graphs (WFG) |
| **Matplotlib** | Latest | Rendering publication-grade graph diagrams with highlighted deadlock cycles |
| **Pytest** | Latest | Automated testing framework verifying 30 test cases across avoidance, detection, prevention, and validation |

---

## The 5 Status Messages and Visual Standards

To prevent student confusion during analysis and vivas, the dashboard implements **5 distinct status banners** with strict semantic colors:

| Status Badge | Semantic Color | Border & Background | Exact Technical Meaning |
| :--- | :--- | :--- | :--- |
| **`SAFE`** | **Green** | `#10B981` border<br>`#ECFDF5` background | A complete safe sequence exists where every process can acquire its maximum claim and finish. Deadlock is mathematically impossible. |
| **`UNSAFE`** | **Orange** | `#EA580C` border<br>`#FFF7ED` background | No safe sequence can be guaranteed if processes demand maximum claims. **Unsafe is NOT deadlock**: deadlock has not occurred yet, but avoidance denies the request to prevent risk. |
| **`WAITING / DEFERRED`** | **Blue** | `#3B82F6` border<br>`#EFF6FF` background | Request is valid ($\text{Request} \le \text{Need}$), but temporarily exceeds available resources ($\text{Request} > \text{Available}$). The request is deferred; current state remains untouched. |
| **`DEADLOCK DETECTED`** | **Red** | `#EF4444` border<br>`#FEF2F2` background | Detection algorithm confirms that one or more processes are locked in an unresolvable circular wait or resource deficit and can never complete. |
| **`INVALID INPUT`** | **Grey** | `#6B7280` border<br>`#F3F4F6` background | Syntactically illegal input: negative integers, non-numeric values, invalid process identifiers, wrong-length vectors, or $\text{Allocation} > \text{Maximum}$. |

---

## Dashboard Tabs and Features

### Tab 1: Deadlock Avoidance (Dijkstra's Banker's Algorithm)
- **Interactive Configuration:** Editable matrices for Allocation, Maximum, and Available vector with quick presets and Reset.
- **Automated Need Matrix:** Automatically computes and displays $\text{Need} = \text{Maximum} - \text{Allocation}$.
- **Safety Algorithm Trace:** Step-by-step table showing Process picked, Need vector, Work before, Allocation released, and Work after.
- **Resource Request Handler:** Evaluates requests against Need and Available, runs temporary simulation, commits if safe, or rolls back with a Before vs After table if unsafe.
- **Educational Callout:** Dedicated *"Unsafe is NOT the same as Deadlock"* explanation panel.

### Tab 2: Deadlock Detection & Graph Visualization
- **Model Switcher:** Toggle between **Single-Instance** and **Multiple-Instance** resource models with clear theoretical banners.
- **Single-Instance Detection:**
  - Bipartite Resource-Allocation Graph (RAG) and reduced Wait-For Graph (WFG) drawn via NetworkX and Matplotlib.
  - Directed cycles highlighted in bright red (`#EF4444`).
  - Interactive Recovery: Terminates a victim process (suggests process holding the most resources) and re-draws the cycle-free graph.
- **Multiple-Instance Detection:**
  - Matrix algorithm ($\text{Work} = \text{Available}$, $\text{Finish}[i] = \text{True}$ if $\text{Alloc}[i]==0$, greedily finds $\text{Request}[i] \le \text{Work}$).
  - Theoretical Callout: *"A cycle alone is not conclusive proof of deadlock for multiple-instance resources."* Explains potential circular wait vs confirmed deadlock.
  - Interactive Recovery: Reclaims resources, updates Work, and displays Before vs After comparison proving remaining processes complete.

### Tab 3: Deadlock Prevention Policies & Trade-offs
- **Policy Selection:**
  - **Policy (a): Resource Ordering** (strict linear ranking $F(R_j) > F(R_i)$) $\implies$ Eliminates **CIRCULAR WAIT**.
  - **Policy (b): Request-All-Upfront** (processes must acquire all needed resources before start) $\implies$ Eliminates **HOLD AND WAIT**.
- **Coffman Condition Breakdown:** 4-column card grid showing which condition is eliminated (`🚫 ELIMINATED BY POLICY`) and which remain active (`🔒 ACTIVE`).
- **Side-by-Side Verification:** Compares original deadlocked scenario against the prevention-constrained execution proving zero deadlocks.
- **Quantitative Trade-offs Panel:** Displays real scenario metrics:
  - *Lower Resource Utilization:* Number of resource units held idle or unallocated.
  - *Reduced Concurrency:* Percentage drop in parallel process progress.
  - *Possible Starvation:* Identification of affected processes.
  - *Longer Waiting Times:* Latency penalty for deferred processes.

### Tab 4: Test Scenarios & Analytical Results Table
- **Preset Quick-Load Buttons:** Load and auto-run textbook scenarios A, B, C, D, E, and F with live trace output.
- **Dynamic Analytical Table:** Real run evaluations compiled into a table with columns: `Test Case | System Condition | Actual Result | Expected Result | Match (PASS/FAIL)`.
- **CSV Export:** `📥 Download results as CSV` exports the verified results table.

---

## Analytical Results Table (Scenarios A - F)

All Actual Results below are computed dynamically from real algorithm executions:

| Test Case | System Condition | Actual Result | Expected Result | Match (PASS/FAIL) |
| :--- | :--- | :--- | :--- | :---: |
| **A: Safe State** | Classic 5-process, 3-resource example (Available=[3,3,2]) | `SAFE: Safe sequence found: P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2` | `SAFE (Valid safe sequence exists: P1 ➔ P3 ➔ P4 ➔ P0 ➔ P2)` | **PASS** |
| **B: Request Exceeds Available** | Grant P1 (1,0,2), then P4 requests (3,3,0) with Available=[2,3,0] | `WAITING: Process P4 must wait: insufficient resources currently available...` | `WAITING / DEFERRED (Request within Need [4,3,1], exceeds Available [2,3,0])` | **PASS** |
| **C: Unsafe Request** | Grant P1 (1,0,2), then P0 requests (0,2,0) with Available=[2,3,0] | `DENIED_UNSAFE: Request is DENIED: simulated state leaves no safe sequence...` | `DENIED_UNSAFE (Request within Available but simulated state is unsafe; rolled back)` | **PASS** |
| **D: Deadlock Detection** | Single-instance circular wait (P1⇄P2) + Multi-instance matrix (P0-P4, Avail=[0,0,0]) | `DEADLOCK DETECTED (Single-Instance Cycle: ['P1', 'P2', 'P1']; Multi-Instance Deadlocked: ['P1', 'P2', 'P3', 'P4'])` | `DEADLOCK DETECTED in both single-instance and multi-instance models` | **PASS** |
| **E: Prevention Applied to Scenario D** | Resource Ordering (eliminates Circular Wait) & Request-All-Upfront (eliminates Hold-and-Wait) | `DEADLOCK PREVENTED (Ordering eliminated Circular Wait, 0 deadlocked; Upfront eliminated Hold and Wait, 0 deadlocked)` | `DEADLOCK PREVENTED (Circular Wait & Hold-and-Wait eliminated; zero deadlocks)` | **PASS** |
| **F: Boundary & Invalid Inputs** | Zero Available, Zero Need, Invalid Process ID (P99), Negative Request, Alloc > Max, Wrong Vector Length | `All 6 boundary/invalid cases evaluated correctly: 2 VALID_BOUNDARY accepted (F1: VALID_BOUNDARY, F2: VALID_BOUNDARY); 4 INVALID rejected (F3: INVALID, F4: INVALID, F5: INVALID, F6: INVALID)` | `VALID_BOUNDARY accepted (Zero Avail, Zero Need); INVALID rejected (P99, neg, Alloc>Max, len 2)` | **PASS** |

---

## AI Usage Record

In compliance with academic integrity guidelines, all generative AI tool interactions utilized during development are documented below:

| AI Tool Used | Date & Prompt Focus | Code / Content Generated | Student Verification & Modifications |
| :--- | :--- | :--- | :--- |
| **Antigravity (Gemini 3.8 Flash)** | Project architecture, modular decoupling, and Banker's algorithm logic | `banker.py` core functions (`calculate_need`, `safety_check`, `handle_request`) | Verified algorithm against Dijkstra's operating systems textbook formulation; added comprehensive docstrings. |
| **Antigravity (Gemini 3.8 Flash)** | Deadlock detection algorithms and NetworkX graph visualization | `detection.py` (`detect_single_instance`, `detect_multi_instance`, `draw_graphs_to_figure`, recovery functions) | Confirmed bipartite node separation (circles for processes, squares for resources) and cycle highlighting in `#EF4444`. |
| **Antigravity (Gemini 3.8 Flash)** | Deadlock prevention simulation and Coffman breakdown | `prevention.py` (`simulate_prevention_single_instance`, `simulate_prevention_multi_instance`, trade-off calculations) | Verified mathematical impossibility of cycles under linear ordering; checked idle resource capacity formulas. |
| **Antigravity (Gemini 3.8 Flash)** | Robust validation and automated test suite | `validation.py` and `test_logic.py` (30 pytest unit tests) | Executed `pytest -v test_logic.py` ensuring 100% pass rate with zero uncaught exceptions. |
| **Antigravity (Gemini 3.8 Flash)** | Streamlit 4-tab user interface and visual styling | `app.py` UI layout, CSS design system, and CSV export functionality | Validated responsive wide-layout, distinct 5-color status banners, and dynamic scenario loading. |

---

## Screenshots

Place application demonstration screenshots in the `screenshots/` directory using the filenames below:

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

## Viva Preparation & Function Reference

Below is a function-by-function guide in clear, simple language explaining what each module and function does:

### 1. `banker.py` (Deadlock Avoidance)
- **`calculate_need(maximum, allocation)`**:  
  Subtracts the Allocation matrix from the Maximum matrix ($\text{Need} = \text{Max} - \text{Alloc}$). Returns the remaining resources each process may still request before finishing.
- **`safety_check(allocation, need, available)`**:  
  Initializes $\text{Work} = \text{Available}$ and marks all processes unfinished ($\text{Finish}[i] = \text{False}$). Repeatedly finds an unfinished process whose remaining Need is less than or equal to Work ($\text{Need}[i] \le \text{Work}$). Simulates its completion by adding its allocated resources back to Work ($\text{Work} += \text{Alloc}[i]$) and marking it finished. If all processes finish, the state is **SAFE** and the execution order is returned as the **safe sequence**; otherwise, it is **UNSAFE**.
- **`handle_request(process_id, request, allocation, maximum, available)`**:  
  Handles a process's resource request in 5 steps: (1) validates input; (2) verifies $\text{Request} \le \text{Need}$ (checks if claim is exceeded); (3) verifies $\text{Request} \le \text{Available}$ (if not, returns `WAITING`); (4) temporarily allocates on copies ($\text{Avail} -= \text{Req}$, $\text{Alloc} += \text{Req}$, $\text{Need} -= \text{Req}$); (5) runs `safety_check`. If safe, commits the request (`GRANTED`); if unsafe, rolls back to original state (`DENIED_UNSAFE`).

### 2. `detection.py` (Deadlock Detection)
- **`detect_single_instance(allocation_edges, request_edges)`**:  
  Builds a directed Wait-For Graph (WFG) where an edge $P_i \to P_j$ exists if process $P_i$ is requesting a resource currently held by $P_j$. Uses NetworkX simple cycles algorithm to find directed cycles. If a cycle exists, deadlock is detected (in single-instance systems, cycle $\iff$ deadlock).
- **`detect_multi_instance(allocation, request, available)`**:  
  Runs the matrix detection algorithm. Sets $\text{Work} = \text{Available}$. Marks $\text{Finish}[i] = \text{True}$ if process $i$ currently holds zero resources ($\text{Alloc}[i] == 0$), else $\text{False}$. Greedily finds an unfinished process whose outstanding Request can be satisfied ($\text{Request}[i] \le \text{Work}$) and reclaims its resources ($\text{Work} += \text{Alloc}[i]$). Any process that remains $\text{Finish}[i] == \text{False}$ at the end is confirmed deadlocked.
- **`terminate_process(process_id, allocation, request, available)`**:  
  Terminates a chosen deadlocked process, reclaims all its allocated resources back into the Available pool ($\text{Avail} += \text{Alloc}[\text{victim}]$), and zeroes out its row in Allocation and Request.
- **`draw_graphs_to_figure(rag, wfg, cycles)`**:  
  Generates a side-by-side Matplotlib visualization of the Resource-Allocation Graph (bipartite: circular process nodes and square resource nodes) and the Wait-For Graph, highlighting cycle nodes and cycle edges in red (`#EF4444`).

### 3. `prevention.py` (Deadlock Prevention)
- **`get_coffman_conditions_summary()`**:  
  Returns structured educational documentation on the four necessary Coffman conditions: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.
- **`get_coffman_breakdown(policy_key)`**:  
  Returns the exact status of all 4 Coffman conditions under the active policy, explicitly flagging the eliminated condition (`🚫 ELIMINATED BY POLICY`) vs active conditions (`🔒 ACTIVE`).
- **`simulate_prevention_single_instance(...)`**:  
  Applies the chosen prevention policy to a single-instance deadlocked scenario. Under **Resource Ordering**, it rejects backward requests violating increasing priority ($F(R_j) \le F(R_i)$), breaking the circular wait. Under **Request-All-Upfront**, it allocates 0 resources to waiting processes, eliminating hold-and-wait. Computes concrete scenario trade-offs (idle resources, concurrency drop, latency).
- **`simulate_prevention_multi_instance(...)`**:  
  Applies prevention policies to a multi-instance deadlocked system. Identifies out-of-order claims or schedules upfront batch allocations, calculating real scenario metrics.

### 4. `validation.py` (Input Validation)
- **`validate_matrix(matrix, expected_rows, expected_cols, name)`**:  
  Ensures matrices contain finite, non-negative whole integers with the expected dimensions.
- **`validate_vector(vector, expected_len, name)`**:  
  Ensures vectors contain finite, non-negative whole integers. Distinguishes all-zero vectors as valid boundaries (`VALID_BOUNDARY`).
- **`validate_process_id(process_id, num_processes)`**:  
  Checks if a process identifier is a valid integer or label within range $[0, \text{num\_processes} - 1]$.
- **`validate_configuration(allocation, maximum, available)`**:  
  Checks dimensions and verifies that $\text{Allocation}[i][j] \le \text{Maximum}[i][j]$ for every cell. If violated, returns the exact process and resource label (e.g. `Allocation exceeds Maximum at process P0, resource B: 5 > 2`). Identifies valid boundary cases (e.g. Available all zeros, or process with zero Need).
- **`validate_request(process_id, request, num_processes, num_resources)`**:  
  Validates process index and request vector dimensions and values without raising exceptions.

### 5. `scenarios.py` (Preset Scenarios & Execution Runners)
- **`get_scenario_a()` to `get_scenario_f()`**:  
  Return textbook definitions and metadata for Scenarios A through F.
- **`run_scenario_a()` to `run_scenario_f()`**:  
  Live execution functions that run the underlying algorithms dynamically on each scenario and compute live actual results and PASS/FAIL matches.
- **`get_all_scenarios_results_table()`**:  
  Executes all scenarios A through F and formats them into a clean tabular data structure with columns `Test Case | System Condition | Actual Result | Expected Result | Match (PASS/FAIL)`.
