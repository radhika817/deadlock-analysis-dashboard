# Deadlock Analysis and Resource Allocation Dashboard

An interactive dashboard for learning and demonstrating how operating systems **avoid**, **detect**, **prevent** and **recover from** deadlocks. Built as part of the OS Internal Assessment (Activity 2: AI-Assisted Dashboard Development and Deadlock Analysis).

> **Course:** Operating Systems
> **Institute:** _<your institute name>_
> **Group:** _<group name / number>_
> **Members:** _<Name 1 (Roll No.)>, <Name 2 (Roll No.)>, <Name 3 (Roll No.)>, <Name 4 (Roll No.)>_

---

## Table of Contents
1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Installation and Running](#installation-and-running)
4. [Project Structure](#project-structure)
5. [Dashboard Modules](#dashboard-modules)
6. [Status Messages](#status-messages)
7. [Request Handling Logic](#request-handling-logic)
8. [Test Scenarios (A to F)](#test-scenarios-a-to-f)
9. [Analytical Results Table](#analytical-results-table)
10. [Sample Data](#sample-data)
11. [AI Usage Record](#ai-usage-record)
12. [Screenshots](#screenshots)
13. [Limitations](#limitations)
14. [Contributors](#contributors)

---

## Features

- Configurable number of processes and resource types, with editable **Allocation**, **Maximum** and **Available** inputs.
- Automatic **Need = Maximum − Allocation** calculation.
- **Banker's Algorithm** with step-by-step **Work vector** updates and the complete **safe sequence**.
- **Resource request validation** with simulate, grant or deny, and automatic **state restoration**.
- Separate modules for **Deadlock Avoidance**, **Deadlock Detection**, and **Prevention / Recovery**.
- Distinct status messages for Safe, Unsafe, Waiting/Deferred, Deadlock Detected and Invalid Input.
- Built-in explanation of why **an unsafe state is not necessarily a deadlock**.
- Preset buttons for test scenarios A to F and a Reset button.
- Input validation for boundary and invalid cases.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| Dashboard | Streamlit |
| Data handling | NumPy / Pandas |
| Visualization | Graphviz / NetworkX / Matplotlib (for graphs) |
| AI assistance | See [AI Usage Record](#ai-usage-record) |

## Installation and Running

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

**Example `requirements.txt`:**
```
streamlit
numpy
pandas
networkx
matplotlib
graphviz
```

## Project Structure

> Adjust this to match your actual files.

```
.
├── app.py                # Streamlit entry point (tabs / navigation)
├── banker.py             # Need matrix, safety algorithm, request handling
├── detection.py          # Deadlock detection (wait-for graph / matrix method)
├── prevention.py         # Prevention policies and recovery (process termination)
├── validation.py         # Input validation helpers
├── scenarios.py          # Preset data for scenarios A to F
├── requirements.txt
├── screenshots/          # Evidence of actual outputs
└── README.md
```

## Dashboard Modules

### 1. Deadlock Avoidance (Banker's Algorithm)
- Computes the Need matrix: `Need[i][j] = Max[i][j] − Allocation[i][j]`.
- Runs the **safety algorithm**:
  1. `Work = Available`, `Finish[i] = False` for all `i`.
  2. Find `i` with `Finish[i] = False` and `Need[i] ≤ Work`.
  3. `Work = Work + Allocation[i]`, `Finish[i] = True`; repeat.
  4. If all `Finish[i]` are True, the state is **safe** and the order found is the **safe sequence**.
- Displays the Work vector after each process completes.
- Handles resource requests (see [Request Handling Logic](#request-handling-logic)).

> The Banker's Algorithm checks whether *granting a request* keeps the system in a safe state. It does **not** detect an existing deadlock.

### 2. Deadlock Detection
- Runs on a **separate** scenario with processes waiting for resources held by one another.
- **Resource model used:** _<single-instance / multiple-instance, state clearly>_
- **Method:**
  - *Single-instance:* Wait-for graph and cycle detection. A cycle means deadlock.
  - *Multiple-instance:* Detection algorithm using Allocation, Request, Work and Finish. A cycle alone is **not** conclusive proof of deadlock here.
- Lists the deadlocked processes and visualizes the relationships.

### 3. Deadlock Prevention and Recovery
- **Prevention:** modifies the allocation policy to eliminate one necessary condition:
  - _Resource ordering_ breaks **circular wait**.
  - _Request all resources up front_ breaks **hold and wait**.
- Compares the original and modified scenarios and lists trade-offs (lower utilization, reduced concurrency, possible starvation).
- **Recovery:** terminates a selected process, releases its resources, and shows how the remaining processes can then progress.

## Status Messages

| Status | Meaning |
|---|---|
| **SAFE** | A complete safe sequence exists. |
| **UNSAFE** | No safe sequence exists. This does not mean deadlock has occurred; it means deadlock is possible. |
| **WAITING / DEFERRED** | Request is within the process's Need but exceeds Available. It cannot be granted now and is not invalid or permanently denied. |
| **DEADLOCK DETECTED** | Detection found processes that can never proceed. |
| **INVALID INPUT** | Bad process ID, negative values, non-numeric input, or allocation above maximum. |

## Request Handling Logic

For a request `Request_i` by process `P_i`:

1. **Invalid input?** (bad process ID, negative, non-numeric) → **Invalid Input**.
2. **`Request_i > Need_i`?** → Error: process exceeded its maximum claim.
3. **`Request_i > Available`?** → **Waiting / Deferred**. The state is unchanged.
4. Otherwise **temporarily allocate**:
   ```
   Available  = Available  − Request_i
   Allocation_i = Allocation_i + Request_i
   Need_i       = Need_i       − Request_i
   ```
5. Run the **safety algorithm**:
   - **Safe** → request granted, state updated.
   - **Unsafe** → request **denied**, original state **restored** and retained.

## Test Scenarios (A to F)

| Scenario | Description |
|---|---|
| **A: Safe State** | Need matrix, Work vector updates, full safe sequence and explanation. |
| **B: Request > Available** | Valid request within Need but above Available. Reported as waiting/deferred; allocation unchanged. |
| **C: Unsafe Request** | Request within Need and Available, but granting it leaves no safe sequence. Denied, state restored. |
| **D: Deadlock Detection** | Circular wait or detection-algorithm result, plus one recovery action. |
| **E: Prevention** | Resource ordering or request-all-upfront, with before/after comparison and trade-offs. |
| **F: Boundary / Invalid Inputs** | Zero Available, zero Need, invalid process ID, negative request, allocation above maximum. |

## Analytical Results Table

The following table records the actual outputs generated by running the dashboard and algorithm suite against test cases TC1 through TC6:

| Test Case | System Condition | Actual Result | Expected Result | Analysis / Conclusion |
|---|---|---|---|---|
| **TC1** | Safe state (Scenario A: 5 processes, 3 resource types, Available=[3, 3, 2]) | **SAFE**: Safe sequence `P1 → P3 → P4 → P0 → P2` found | Safe sequence found | `Need` calculated as `Max - Allocation`. Work vector starts at `[3, 3, 2]`. P1 completes and releases `[2, 0, 0]` making Work `[5, 3, 2]`, allowing P3, P4, P0, P2 to finish sequentially. System is guaranteed deadlock-free. |
| **TC2** | Request exceeds Available (Scenario B: P1 granted `[1, 0, 2]`, Available becomes `[2, 3, 0]`, P4 requests `[3, 3, 0]`) | **WAITING / DEFERRED**: Process P4 must wait; system state preserved unchanged | Request not granted immediately | P4's request `[3, 3, 0]` is within its Need `[4, 3, 1]`, but exceeds Available `[2, 3, 0]` on resource A ($3 > 2$). The request cannot be granted now, but is **not invalid or permanently denied**. Current allocation is strictly retained. |
| **TC3** | Request makes state unsafe (Scenario C: Available=`[2, 3, 0]`, P0 requests `[0, 2, 0]`) | **UNSAFE**: Request denied; simulated state rolled back; original state restored | Request denied; state unchanged | Request is within Need `[7, 4, 3]` and within Available `[2, 3, 0]`. However, temporary simulation results in Available `[2, 1, 0]`, in which no remaining process can guarantee completion. Banker's Algorithm denies the request and restores the safe state. |
| **TC4** | Deadlocked scenario (Scenario D: P1 holds R1 & requests R2; P2 holds R2 & requests R1) | **DEADLOCK DETECTED**: Deadlocked processes `['P1', 'P2']`; cycle `P1 → P2 → P1` found in Wait-For Graph | Deadlock identified by selected detection method | In single-instance resource systems, a directed cycle in the Wait-For Graph is necessary and sufficient proof of deadlock. Terminating victim P1 breaks the circular wait, releasing R1 and resolving the deadlock. |
| **TC5** | Prevention policy applied (Scenario E: Global Resource Ordering $R1=1 < R2=2 < R3=3$ applied to Scenario D) | **DEADLOCK PREVENTED**: Policy eliminates **Circular Wait**; P2's request for R1 rejected | Selected deadlock condition eliminated | Under resource ordering, processes may only request higher-ordered resources. P1 holding R1 can request R2 ($2 > 1$), but P2 holding R2 cannot request R1 ($1 < 2$). P2 must release R2 first, mathematically preventing cycle formation. |
| **TC6** | Boundary and invalid inputs (Scenario F: zero Available, zero Need, Alloc > Max, negative request, bad PID) | **HANDLED CORRECTLY**: Zero Available and zero Need identified as **VALID BOUNDARY**; negative values, Alloc > Max, and wrong shapes flagged as **INVALID INPUT** | Correct handling or validation message | Dashboard distinguishes valid edge cases (e.g. system temporarily exhausted) from illegal operations (e.g. Alloc > Max cell violation or negative numbers) with descriptive error messages instead of exceptions. |

## Sample Data

Classic textbook example (5 processes P0 to P4, resource types A, B, C with 10, 5, 7 total instances):

| Process | Allocation (A B C) | Max (A B C) | Need (A B C) |
|---|---|---|---|
| P0 | 0 1 0 | 7 5 3 | 7 4 3 |
| P1 | 2 0 0 | 3 2 2 | 1 2 2 |
| P2 | 3 0 2 | 9 0 2 | 6 0 0 |
| P3 | 2 1 1 | 2 2 2 | 0 1 1 |
| P4 | 0 0 2 | 4 3 3 | 4 3 1 |

**Available:** `3 3 2`  
**Total:** `10 5 7`

- **Scenario A:** Safe sequence exists: `P1 → P3 → P4 → P0 → P2`.
- **Scenario B:** After granting P1 `(1, 0, 2)`, Available becomes `(2, 3, 0)`. P4 requesting `(3, 3, 0)` is within Need but exceeds Available ➔ **WAITING / DEFERRED**.
- **Scenario C:** After granting P1 `(1, 0, 2)`, P0 requesting `(0, 2, 0)` is within Need and Available, but simulated state leaves no safe sequence ➔ **UNSAFE (DENIED & RESTORED)**.

## AI Usage Record

| AI tool used | Purpose | Output generated | Student verification |
|---|---|---|---|
| Antigravity (Gemini 3.8 Flash) | Architecture & code skeleton | Modular project structure (`app.py`, `banker.py`, `detection.py`, `prevention.py`, `validation.py`, `scenarios.py`, `test_logic.py`) | Verified separation of concerns; confirmed algorithm logic is decoupled from Streamlit UI. |
| Antigravity (Gemini 3.8 Flash) | Algorithm implementation & graph visualization | Dijkstra's safety algorithm, single-instance RAG/WFG cycle detection with NetworkX/Matplotlib, multi-instance matrix algorithm, and prevention simulations | Verified against textbook Banker's algorithm formulas and executed comprehensive 18-test pytest suite. |
| Antigravity (Gemini 3.8 Flash) | Test suite & analytical table generation | Automated pytest test cases for Scenarios A–F and analytical table population | Ran tests locally (`pytest test_logic.py`); confirmed 100% pass rate (18/18 tests passed). |

## Screenshots

Place demonstration screenshots in the `screenshots/` directory:
- `screenshots/01_safe_state_scenario_a.png` — Tab 1: Banker's safety check showing Need matrix, safe sequence, and Work updates.
- `screenshots/02_request_waiting_scenario_b.png` — Tab 1: Resource request exceeding Available showing WAITING / DEFERRED banner.
- `screenshots/03_unsafe_request_scenario_c.png` — Tab 1: Unsafe request showing DENIED, state restoration, and "Unsafe is not Deadlock" panel.
- `screenshots/04_deadlock_detection_scenario_d.png` — Tab 2: Single-instance Wait-For Graph with cycle highlighted in red and suggested victim.
- `screenshots/05_deadlock_recovery.png` — Tab 2: Recovery by victim termination and resolution verification.
- `screenshots/06_prevention_scenario_e.png` — Tab 3: Resource ordering policy comparison and trade-offs.
- `screenshots/07_analytical_table_scenario_f.png` — Tab 4: Completed analytical table with CSV/JSON export buttons.

## Viva Preparation Guide

1. **How is the Need matrix calculated?**
   - $\text{Need}[i][j] = \text{Maximum}[i][j] - \text{Allocation}[i][j]$. It represents the remaining resource instances process $P_i$ may request before completing.
2. **What is a safe sequence?**
   - A sequence of processes $\langle P_0, P_1, \dots, P_{n-1} \rangle$ such that for each $P_i$, its remaining $\text{Need}_i$ can be satisfied by current $\text{Available}$ plus resources released by all preceding processes $P_j$ ($j < i$).
3. **Why can a request be valid but still need to wait?**
   - If $\text{Request}_i \le \text{Need}_i$ (the process did not exceed its declared maximum), but $\text{Request}_i > \text{Available}$, the system simply does not currently have enough free resources. The process must wait until other processes release resources.
4. **Why does Banker's Algorithm deny a request that would make the state unsafe?**
   - In an unsafe state, there is no guarantee that at least one process can finish if all processes simultaneously demand their maximum claim. To avoid risk of entering an unrecoverable deadlock, the OS avoids transitioning into an unsafe state.
5. **Why is an unsafe state not necessarily a deadlock?**
   - An unsafe state is not a deadlock; it only means the OS *cannot guarantee* avoidance if worst-case claims occur. If processes voluntarily release resources before demanding their maximums, an unsafe state might never become a deadlock.
6. **How is deadlock detection different from deadlock avoidance?**
   - Avoidance (Banker's) is proactive: it checks *before* granting a request to ensure the state stays safe. Detection is reactive: it inspects current holdings and requests *after* they have occurred to see if any processes are currently deadlocked.
7. **Which deadlock condition does the prevention policy eliminate?**
   - **Resource Ordering** eliminates *Circular Wait* by enforcing a strict global numerical order on resource acquisition.
   - **Request-All-Upfront** eliminates *Hold and Wait* by requiring processes to obtain all resources at once or hold none while waiting.
8. **Which process is recovered or terminated, and which resources are released?**
   - In single-instance Scenario D, process $P_1$ (suggested victim holding $R_1$) is terminated, releasing $R_1$. Process $P_2$ can now acquire $R_1$ and complete, resolving the deadlock.
