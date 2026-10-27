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

> Fill the **Actual Result** and **Analysis** columns using real outputs from this dashboard only.

| Test Case | System Condition | Actual Result | Expected Result | Analysis / Conclusion |
|---|---|---|---|---|
| TC1 | Safe state | _<fill>_ | Safe sequence found | _<fill>_ |
| TC2 | Request exceeds Available | _<fill>_ | Request not granted immediately | _<fill>_ |
| TC3 | Request makes state unsafe | _<fill>_ | Request denied; state unchanged | _<fill>_ |
| TC4 | Deadlocked scenario | _<fill>_ | Deadlock identified by the selected detection method | _<fill>_ |
| TC5 | Prevention policy applied | _<fill>_ | Selected deadlock condition eliminated | _<fill>_ |
| TC6 | Boundary and invalid inputs | _<fill>_ | Correct handling or validation message | _<fill>_ |

## Sample Data

Classic textbook example (5 processes P0 to P4, resource types A, B, C with 10, 5, 7 total instances):

| Process | Allocation (A B C) | Max (A B C) |
|---|---|---|
| P0 | 0 1 0 | 7 5 3 |
| P1 | 2 0 0 | 3 2 2 |
| P2 | 3 0 2 | 9 0 2 |
| P3 | 2 1 1 | 2 2 2 |
| P4 | 0 0 2 | 4 3 3 |

**Available:** `3 3 2`

- **Scenario A:** a safe sequence exists (for example `P1 → P3 → P4 → P0 → P2`). Confirm with the dashboard output.
- **Scenario B:** after granting P1 `(1,0,2)`, Available becomes `(2,3,0)`. P4 requesting `(3,3,0)` is within its Need but exceeds Available.
- **Scenario C:** after granting P1 `(1,0,2)`, P0 requesting `(0,2,0)` is within Need and Available but leaves no safe sequence.
