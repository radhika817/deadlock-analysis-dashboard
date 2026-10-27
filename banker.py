"""
banker.py - Banker's Algorithm for Deadlock Avoidance.

This module implements:
1. Need matrix calculation: Need[i][j] = Maximum[i][j] - Allocation[i][j]
2. Safety Algorithm:
   - Evaluates whether the system is in a SAFE or UNSAFE state.
   - Computes step-by-step Work updates and produces a complete safe sequence.
3. Resource Request Handling:
   Strict 6-step evaluation:
     Step 1: Check for invalid inputs (bad process ID, negative values, wrong vector length).
     Step 2: Check if Request > Need (Process exceeds its maximum claim).
     Step 3: Check if Request > Available (Process must wait/defer; state unchanged).
     Step 4: Temporarily simulate allocation on a copy of the state.
     Step 5: Run safety algorithm on simulated state.
     Step 6: If Safe -> Grant and commit state. If Unsafe -> Deny and restore state.

All logic is independent of the UI and includes comprehensive docstrings for viva preparation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from validation import validate_request_input


def calculate_need_matrix(maximum: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """
    Computes the Need matrix for all processes and resources.
    
    Formula:
        Need[i][j] = Maximum[i][j] - Allocation[i][j]
        
    Args:
        maximum: 2D array representing the maximum resource demands.
        allocation: 2D array representing currently allocated resources.
        
    Returns:
        2D array of the same shape containing remaining resource needs.
    """
    max_arr = np.array(maximum, dtype=int)
    alloc_arr = np.array(allocation, dtype=int)
    return max_arr - alloc_arr


def run_safety_algorithm(
    allocation: np.ndarray,
    need: np.ndarray,
    available: np.ndarray,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs Dijkstra's Banker's Safety Algorithm to find a safe execution sequence.

    Algorithm steps:
    1. Let Work = Available (copy of available vector).
       Let Finish[i] = False for all processes i = 0, 1, ..., n-1.
    2. Find an index i such that:
         Finish[i] == False and Need[i] <= Work.
       If no such i exists, go to step 4.
    3. Work = Work + Allocation[i]
       Finish[i] = True
       Record step details.
       Go to step 2.
    4. If Finish[i] == True for all i, the system is in a SAFE state.
       Otherwise, the system is in an UNSAFE state.

    Args:
        allocation: 2D array of current allocations (n x m).
        need: 2D array of remaining needs (n x m).
        available: 1D array of available resources (m).
        process_labels: Optional labels for processes (e.g. ['P0', 'P1', ...]).
        resource_labels: Optional labels for resources (e.g. ['A', 'B', 'C']).

    Returns:
        A dictionary containing:
        - 'is_safe': bool
        - 'status': 'SAFE' or 'UNSAFE'
        - 'safe_sequence': list of process labels in safe order (empty if unsafe)
        - 'steps': list of dicts recording each execution step
        - 'unfinished_processes': list of process labels that could not finish
        - 'explanation': clear explanation for viva presentation
    """
    alloc_arr = np.array(allocation, dtype=int)
    need_arr = np.array(need, dtype=int)
    work = np.array(available, dtype=int).copy()

    n_proc, n_res = alloc_arr.shape
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    finish = [False] * n_proc
    safe_seq: List[str] = []
    steps_log: List[Dict[str, Any]] = []

    step_counter = 1
    while True:
        found_process = False
        for i in range(n_proc):
            if not finish[i]:
                # Check if Need[i] <= Work for all resource types
                if np.all(need_arr[i] <= work):
                    work_before = work.copy()
                    released = alloc_arr[i].copy()
                    work = work + released
                    finish[i] = True
                    safe_seq.append(p_labels[i])
                    found_process = True

                    steps_log.append({
                        "step": step_counter,
                        "process": p_labels[i],
                        "need": need_arr[i].tolist(),
                        "work_before": work_before.tolist(),
                        "allocation_released": released.tolist(),
                        "work_after": work.copy().tolist(),
                        "description": (
                            f"Step {step_counter}: Process {p_labels[i]} satisfies Need {need_arr[i].tolist()} "
                            f"<= Work {work_before.tolist()}. Runs to completion and releases "
                            f"Allocation {released.tolist()}. New Work = {work.tolist()}."
                        )
                    })
                    step_counter += 1
                    break  # Restart loop from first unfinished process
        if not found_process:
            break

    is_safe = all(finish)
    unfinished = [p_labels[i] for i in range(n_proc) if not finish[i]]

    if is_safe:
        status = "SAFE"
        seq_str = " -> ".join(safe_seq)
        explanation = (
            f"The system is in a SAFE state. A safe sequence exists: {seq_str}. "
            f"All {n_proc} processes can obtain their declared maximum resources, "
            f"finish execution, and return their allocated resources without deadlocking."
        )
    else:
        status = "UNSAFE"
        explanation = (
            f"The system is in an UNSAFE state. No safe sequence could be found. "
            f"Processes {', '.join(unfinished)} cannot satisfy their remaining Need "
            f"with the available Work vector {work.tolist()}. "
            f"IMPORTANT: An unsafe state does NOT mean a deadlock has already occurred; "
            f"it means the system cannot guarantee that deadlock will be avoided if processes "
            f"request their maximum resource claims."
        )

    return {
        "is_safe": is_safe,
        "status": status,
        "safe_sequence": safe_seq,
        "safe_sequence_str": " -> ".join(safe_seq) if is_safe else "None",
        "steps": steps_log,
        "unfinished_processes": unfinished,
        "final_work": work.tolist(),
        "explanation": explanation
    }


def handle_resource_request(
    allocation: np.ndarray,
    maximum: np.ndarray,
    available: np.ndarray,
    process_idx: Any,
    request_vector: Any,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Handles a resource request from a process following the strict 6-step protocol:

    Step 1: Validate input formatting (process ID, dimensions, non-negative integers).
    Step 2: Check if Request > Need (Process claims more than its declared maximum).
    Step 3: Check if Request > Available (Resource not currently free; process must WAIT/DEFER).
    Step 4: Temporarily allocate resources on a duplicate state copy:
              Available' = Available - Request
              Allocation'[i] = Allocation[i] + Request
              Need'[i] = Need[i] - Request
    Step 5: Run the safety algorithm on the simulated state.
    Step 6: Decision:
              - If SAFE: Grant request, permanently commit state changes.
              - If UNSAFE: Deny request, discard simulation, restore original state.

    Returns:
        A comprehensive result dictionary detailing the status, decision,
        before/after states, step log, and viva explanation.
    """
    alloc_orig = np.array(allocation, dtype=int).copy()
    max_orig = np.array(maximum, dtype=int).copy()
    avail_orig = np.array(available, dtype=int).copy()
    need_orig = calculate_need_matrix(max_orig, alloc_orig)

    n_proc, n_res = alloc_orig.shape
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    # --- STEP 1: Input Validation ---
    is_valid_input, p_idx, req_arr, err_msg = validate_request_input(
        process_idx, request_vector, n_proc, n_res, p_labels
    )
    if not is_valid_input:
        return {
            "status": "INVALID INPUT",
            "granted": False,
            "step_reached": 1,
            "reason": f"Input validation failed: {err_msg}",
            "process_label": str(process_idx),
            "request_vector": str(request_vector),
            "state_changed": False,
            "allocation": alloc_orig,
            "available": avail_orig,
            "need": need_orig,
            "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "after_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "safety_result": None,
            "viva_explanation": "Invalid requests (negative numbers, out-of-range IDs, non-integers) are rejected before any OS state evaluation."
        }

    proc_label = p_labels[p_idx]
    req_list = req_arr.tolist()

    # --- STEP 2: Request <= Need Check ---
    # Condition: If Request[i][j] > Need[i][j], error: process exceeded maximum claim.
    if np.any(req_arr > need_orig[p_idx]):
        violating_resources = [r_labels[j] for j in range(n_res) if req_arr[j] > need_orig[p_idx, j]]
        viol_details = [
            f"{r_labels[j]} (Requested: {req_arr[j]} > Remaining Need: {need_orig[p_idx, j]})"
            for j in range(n_res) if req_arr[j] > need_orig[p_idx, j]
        ]
        return {
            "status": "INVALID INPUT",
            "granted": False,
            "step_reached": 2,
            "reason": (
                f"Error: Process {proc_label} has exceeded its declared maximum claim! "
                f"Violations on resource(s): {', '.join(viol_details)}."
            ),
            "process_label": proc_label,
            "request_vector": req_list,
            "state_changed": False,
            "allocation": alloc_orig,
            "available": avail_orig,
            "need": need_orig,
            "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "after_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "safety_result": None,
            "viva_explanation": (
                f"In Banker's Algorithm, a process can NEVER request more than its declared maximum demand. "
                f"Since Request > Need for {proc_label}, this is treated as an illegal over-claim violation."
            )
        }

    # --- STEP 3: Request <= Available Check ---
    # Condition: If Request[i][j] > Available[j], process must wait (WAITING / DEFERRED).
    if np.any(req_arr > avail_orig):
        deficit_resources = [
            f"{r_labels[j]} (Requested: {req_arr[j]}, Available: {avail_orig[j]})"
            for j in range(n_res) if req_arr[j] > avail_orig[j]
        ]
        return {
            "status": "WAITING / DEFERRED",
            "granted": False,
            "step_reached": 3,
            "reason": (
                f"Process {proc_label} must wait because insufficient resources are currently available. "
                f"Shortage on: {', '.join(deficit_resources)}. The system state remains UNCHANGED."
            ),
            "process_label": proc_label,
            "request_vector": req_list,
            "state_changed": False,
            "allocation": alloc_orig,
            "available": avail_orig,
            "need": need_orig,
            "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "after_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "safety_result": None,
            "viva_explanation": (
                f"The request is valid and within maximum claims (Request <= Need), but the OS does not currently have "
                f"enough free instances. The process is placed in a waiting queue. The current allocation is preserved."
            )
        }

    # --- STEP 4: Simulated Allocation on State Copy ---
    sim_alloc = alloc_orig.copy()
    sim_avail = avail_orig.copy()
    sim_need = need_orig.copy()

    sim_avail -= req_arr
    sim_alloc[p_idx] += req_arr
    sim_need[p_idx] -= req_arr

    # --- STEP 5: Run Safety Algorithm on Simulated State ---
    safety_res = run_safety_algorithm(sim_alloc, sim_need, sim_avail, p_labels, r_labels)

    # --- STEP 6: Decision (Grant or Deny & Restore) ---
    if safety_res["is_safe"]:
        # Safe -> Grant and commit state
        return {
            "status": "SAFE",
            "granted": True,
            "step_reached": 6,
            "reason": (
                f"Request of {req_list} by {proc_label} GRANTED! "
                f"Simulated system state is SAFE with safe sequence: {safety_res['safe_sequence_str']}."
            ),
            "process_label": proc_label,
            "request_vector": req_list,
            "state_changed": True,
            "allocation": sim_alloc,
            "available": sim_avail,
            "need": sim_need,
            "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "after_state": {"allocation": sim_alloc, "available": sim_avail, "need": sim_need},
            "safety_result": safety_res,
            "viva_explanation": (
                f"Granting {proc_label}'s request leaves the system in a verified safe state. "
                f"There exists a sequence {safety_res['safe_sequence_str']} ensuring every process "
                f"can eventually finish without entering a deadlock."
            )
        }
    else:
        # Unsafe -> Deny, reject changes, and restore state
        return {
            "status": "UNSAFE",
            "granted": False,
            "step_reached": 6,
            "reason": (
                f"Request of {req_list} by {proc_label} DENIED! "
                f"Granting this request would leave the system in an UNSAFE state. "
                f"The original allocation and available vectors have been RESTORED."
            ),
            "process_label": proc_label,
            "request_vector": req_list,
            "state_changed": False,
            "allocation": alloc_orig,
            "available": avail_orig,
            "need": need_orig,
            "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "after_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
            "safety_result": safety_res,
            "viva_explanation": (
                f"The request is within Need and within Available, but granting it would leave insufficient free resources "
                f"for any remaining process to finish its maximum claim. To avoid potential deadlock, Banker's Algorithm "
                f"denies the request and restores the previous safe state."
            )
        }


def format_steps_as_dataframe(steps: List[Dict[str, Any]], resource_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Helper function to convert safety steps log into a clean pandas DataFrame for display.
    """
    if not steps:
        return pd.DataFrame(columns=["Step", "Process Picked", "Need[i]", "Work Before", "Allocation Released", "Work After"])

    rows = []
    for s in steps:
        rows.append({
            "Step": s["step"],
            "Process Picked": s["process"],
            "Need[i]": str(s["need"]),
            "Work Before": str(s["work_before"]),
            "Allocation Released": str(s["allocation_released"]),
            "Work After": str(s["work_after"])
        })
    return pd.DataFrame(rows)
