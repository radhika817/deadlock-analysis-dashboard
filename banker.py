"""
banker.py - Dijkstra's Banker's Algorithm for Deadlock Avoidance.

This module provides pure Python / NumPy functions for:
1. calculate_need(maximum, allocation)
   Computes Need = Maximum - Allocation.
2. safety_check(allocation, need, available)
   Executes the safety algorithm to determine whether a safe execution
   sequence exists, returning step-by-step Work updates.
3. handle_request(process_id, request, allocation, maximum, available)
   Evaluates a resource request against the strict 5-status protocol:
   INVALID, EXCEEDS_NEED, WAITING, GRANTED, or DENIED_UNSAFE.

Never modifies input arrays in place.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


def calculate_need(maximum: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """
    Computes the Need matrix for all processes and resources.

    Formula:
        Need = Maximum - Allocation

    Args:
        maximum: 2D array-like of shape (n_processes, n_resources) with maximum demands.
        allocation: 2D array-like of shape (n_processes, n_resources) with current allocations.

    Returns:
        np.ndarray: A new 2D integer array containing remaining resource needs.
    """
    # Create independent array copies to never modify inputs in place
    max_arr = np.array(maximum, dtype=int).copy()
    alloc_arr = np.array(allocation, dtype=int).copy()

    # Need = Maximum - Allocation
    need = max_arr - alloc_arr
    return need


def safety_check(
    allocation: np.ndarray,
    need: np.ndarray,
    available: np.ndarray
) -> Dict[str, Any]:
    """
    Runs Dijkstra's Banker's Safety Algorithm on the given system state.

    Algorithm:
    1. Work = Available (copy)
       Finish = all False (length n_processes)
    2. Repeatedly find an index i such that:
         Finish[i] == False and Need[i] <= Work
       If found:
         Work = Work + Allocation[i]
         Finish[i] = True
         Record step details
         Repeat step 2.
    3. If all Finish[i] == True, system is SAFE; else UNSAFE.

    Args:
        allocation: 2D array-like of current allocations (n x m).
        need: 2D array-like of remaining needs (n x m).
        available: 1D array-like of currently available resources (m).

    Returns:
        dict with:
        - 'is_safe' (bool): True if a safe sequence exists, False otherwise.
        - 'safe_sequence' (list of int): List of process indices in safe order.
        - 'steps' (list of dict): Step trace containing:
             'step', 'process', 'need', 'work_before', 'allocation_released', 'work_after'
        - 'unfinished' (list of int): List of process indices that could not finish.
    """
    # Ensure independent copies; never modify input arrays in place
    alloc_arr = np.array(allocation, dtype=int).copy()
    need_arr = np.array(need, dtype=int).copy()
    work = np.array(available, dtype=int).flatten().copy()

    n_proc, n_res = alloc_arr.shape

    # Step 1: Initialize Finish flags and tracing structures
    finish = [False] * n_proc
    safe_sequence: List[int] = []
    steps: List[Dict[str, Any]] = []

    step_number = 1

    # Step 2: Greedily find processes whose Need can be satisfied by current Work
    while True:
        found_process = False
        for i in range(n_proc):
            if not finish[i]:
                # Check if Need[i] <= Work for all resource types
                if np.all(need_arr[i] <= work):
                    work_before = work.copy()
                    released = alloc_arr[i].copy()

                    # Process completes and releases its allocated resources
                    work = work + released
                    finish[i] = True
                    safe_sequence.append(int(i))
                    found_process = True

                    steps.append({
                        "step": step_number,
                        "process": int(i),
                        "need": need_arr[i].tolist(),
                        "work_before": work_before.tolist(),
                        "allocation_released": released.tolist(),
                        "work_after": work.copy().tolist()
                    })
                    step_number += 1
                    break  # Restart loop to prioritize from first eligible process

        if not found_process:
            break

    # Step 3: Check whether all processes could finish
    is_safe = all(finish)
    unfinished = [int(i) for i in range(n_proc) if not finish[i]]

    return {
        "is_safe": is_safe,
        "safe_sequence": safe_sequence,
        "steps": steps,
        "unfinished": unfinished
    }


def handle_request(
    process_id: int,
    request: Union[List[int], np.ndarray],
    allocation: np.ndarray,
    maximum: np.ndarray,
    available: np.ndarray
) -> Dict[str, Any]:
    """
    Handles a resource request from a process using the strict order of checks:

    1. Invalid input check:
       - Bad process_id (not integer or out of bounds)
       - Negative values in request
       - Wrong request vector length (does not match number of resource types)
       -> Returns status "INVALID"
    2. Over-claim check:
       - request > Need for that process
       -> Returns status "EXCEEDS_NEED"
    3. Availability check:
       - request > Available
       -> Returns status "WAITING" (state unchanged, NOT invalid)
    4. Simulation:
       - Otherwise simulate on COPIES of the matrices and run safety_check
    5. Evaluation:
       - Safe -> Returns status "GRANTED" and commits new state
       - Unsafe -> Returns status "DENIED_UNSAFE" and preserves original state

    Never modifies input arrays in place.

    Returns:
        dict with:
        - 'status': "INVALID", "EXCEEDS_NEED", "WAITING", "GRANTED", or "DENIED_UNSAFE"
        - 'message': Detailed human-readable explanation
        - 'new_allocation': Updated allocation if GRANTED, else copy of original
        - 'new_available': Updated available if GRANTED, else copy of original
        - 'safety_result': Dict returned by safety_check if simulation was run, else None
    """
    # Defensive copies of all inputs to guarantee no in-place modification
    alloc_copy = np.array(allocation, dtype=int).copy()
    max_copy = np.array(maximum, dtype=int).copy()
    avail_copy = np.array(available, dtype=int).flatten().copy()

    n_proc, n_res = alloc_copy.shape

    # -------------------------------------------------------------
    # Check 1: Input Validation
    # -------------------------------------------------------------
    # Validate process_id
    if not isinstance(process_id, (int, np.integer)):
        return {
            "status": "INVALID",
            "message": f"Process ID must be an integer, got {type(process_id).__name__}.",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    p_id = int(process_id)
    if p_id < 0 or p_id >= n_proc:
        return {
            "status": "INVALID",
            "message": f"Invalid process ID {p_id}. Must be between 0 and {n_proc - 1}.",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    # Validate request vector
    try:
        req_arr = np.array(request, dtype=int).flatten()
    except Exception as e:
        return {
            "status": "INVALID",
            "message": f"Request vector could not be converted to integer array: {e}",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    if len(req_arr) != n_res:
        return {
            "status": "INVALID",
            "message": f"Request vector length mismatch: expected {n_res} resources, got {len(req_arr)}.",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    if (req_arr < 0).any():
        neg_idx = int(np.where(req_arr < 0)[0][0])
        return {
            "status": "INVALID",
            "message": f"Negative value {req_arr[neg_idx]} found in request at resource index {neg_idx}.",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    # Calculate Need on copies
    need = calculate_need(max_copy, alloc_copy)

    # -------------------------------------------------------------
    # Check 2: Request > Need (Process exceeded maximum claim)
    # -------------------------------------------------------------
    if np.any(req_arr > need[p_id]):
        violating = [j for j in range(n_res) if req_arr[j] > need[p_id, j]]
        viol_str = ", ".join([f"Resource {j}: requested {req_arr[j]} > Need {need[p_id, j]}" for j in violating])
        return {
            "status": "EXCEEDS_NEED",
            "message": f"Process P{p_id} exceeded its declared maximum claim (Need): {viol_str}.",
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    # -------------------------------------------------------------
    # Check 3: Request > Available (Process must wait, NOT invalid)
    # -------------------------------------------------------------
    if np.any(req_arr > avail_copy):
        deficit = [j for j in range(n_res) if req_arr[j] > avail_copy[j]]
        def_str = ", ".join([f"Resource {j}: requested {req_arr[j]} > Available {avail_copy[j]}" for j in deficit])
        return {
            "status": "WAITING",
            "message": (
                f"Process P{p_id} must wait: insufficient resources currently available ({def_str}). "
                "System state remains unchanged."
            ),
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": None
        }

    # -------------------------------------------------------------
    # Check 4: Simulate on COPIES
    # -------------------------------------------------------------
    sim_alloc = alloc_copy.copy()
    sim_avail = avail_copy.copy()
    sim_need = need.copy()

    sim_avail -= req_arr
    sim_alloc[p_id] += req_arr
    sim_need[p_id] -= req_arr

    # Run safety algorithm on the simulated state
    safety_result = safety_check(sim_alloc, sim_need, sim_avail)

    # -------------------------------------------------------------
    # Check 5: Evaluate Safety Decision
    # -------------------------------------------------------------
    if safety_result["is_safe"]:
        # Safe -> GRANTED and commit new state
        seq_str = " -> ".join([f"P{idx}" for idx in safety_result["safe_sequence"]])
        return {
            "status": "GRANTED",
            "message": (
                f"Request of {req_arr.tolist()} by Process P{p_id} is GRANTED! "
                f"Simulated state is SAFE with safe sequence: {seq_str}."
            ),
            "new_allocation": sim_alloc,
            "new_available": sim_avail,
            "safety_result": safety_result
        }
    else:
        # Unsafe -> DENIED_UNSAFE and restore original state untouched
        unfin_str = ", ".join([f"P{idx}" for idx in safety_result["unfinished"]])
        return {
            "status": "DENIED_UNSAFE",
            "message": (
                f"Request of {req_arr.tolist()} by Process P{p_id} is DENIED! "
                f"Granting it leaves the system in an UNSAFE state (processes {unfin_str} cannot finish). "
                "Original system state has been preserved untouched."
            ),
            "new_allocation": alloc_copy,
            "new_available": avail_copy,
            "safety_result": safety_result
        }


# ============================================================================
# COMPATIBILITY WRAPPERS & HELPERS FOR UI AND ADVANCED REPORTING
# ============================================================================

def calculate_need_matrix(maximum: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """Alias for calculate_need for backward compatibility."""
    return calculate_need(maximum, allocation)


def run_safety_algorithm(
    allocation: np.ndarray,
    need: np.ndarray,
    available: np.ndarray,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Wrapper around safety_check providing labeled output and viva explanations for UI.
    """
    res = safety_check(allocation, need, available)
    n_proc = len(allocation)
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]

    labeled_sequence = [p_labels[i] for i in res["safe_sequence"]]
    labeled_unfinished = [p_labels[i] for i in res["unfinished"]]

    # Enrich steps with process labels
    labeled_steps = []
    for s in res["steps"]:
        p_name = p_labels[s["process"]]
        labeled_steps.append({
            **s,
            "process": p_name,
            "description": (
                f"Step {s['step']}: Process {p_name} satisfies Need {s['need']} "
                f"<= Work {s['work_before']}. Executes to completion and returns "
                f"Allocation {s['allocation_released']}. Work becomes {s['work_after']}."
            )
        })

    is_safe = res["is_safe"]
    status_str = "SAFE" if is_safe else "UNSAFE"
    seq_str = " -> ".join(labeled_sequence) if is_safe else "None"

    if is_safe:
        explanation = (
            f"The system is in a SAFE state. A complete safe sequence exists: {seq_str}. "
            f"All {n_proc} processes can obtain their declared maximum resources, "
            f"finish execution, and return their allocated resources without deadlocking."
        )
    else:
        explanation = (
            f"The system is in an UNSAFE state. No safe sequence exists. "
            f"Processes {', '.join(labeled_unfinished)} cannot satisfy their remaining Need. "
            f"Note: An unsafe state does NOT mean a deadlock has occurred yet; "
            f"it means the system cannot guarantee avoidance if processes claim their maximum demands."
        )

    return {
        "is_safe": is_safe,
        "status": status_str,
        "safe_sequence": labeled_sequence,
        "safe_sequence_indices": res["safe_sequence"],
        "safe_sequence_str": seq_str,
        "steps": labeled_steps,
        "unfinished_processes": labeled_unfinished,
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
    Adapter bridging handle_request with UI data expectations.
    """
    n_proc, n_res = np.array(allocation).shape
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    # Parse process label to integer index if needed
    p_id = process_idx
    if isinstance(process_idx, str):
        cleaned = process_idx.strip().upper()
        if cleaned.startswith("P") and cleaned[1:].isdigit():
            p_id = int(cleaned[1:])
        elif cleaned.isdigit():
            p_id = int(cleaned)
        elif cleaned in p_labels:
            p_id = p_labels.index(cleaned)

    # Call pure handle_request
    core_res = handle_request(p_id, request_vector, allocation, maximum, available)

    # Map core status to UI status banners
    status_map = {
        "INVALID": "INVALID INPUT",
        "EXCEEDS_NEED": "INVALID INPUT",
        "WAITING": "WAITING / DEFERRED",
        "GRANTED": "SAFE",
        "DENIED_UNSAFE": "UNSAFE"
    }

    ui_status = status_map.get(core_res["status"], core_res["status"])
    is_granted = (core_res["status"] == "GRANTED")
    p_name = p_labels[p_id] if isinstance(p_id, int) and 0 <= p_id < n_proc else str(process_idx)

    alloc_orig = np.array(allocation, dtype=int).copy()
    max_orig = np.array(maximum, dtype=int).copy()
    avail_orig = np.array(available, dtype=int).copy()
    need_orig = calculate_need(max_orig, alloc_orig)

    new_alloc = core_res["new_allocation"]
    new_avail = core_res["new_available"]
    new_need = calculate_need(max_orig, new_alloc)

    # Format explanation for viva
    if core_res["status"] == "INVALID":
        viva_exp = "Invalid requests (negative numbers, out-of-range IDs, non-integers) are rejected before any OS state evaluation."
    elif core_res["status"] == "EXCEEDS_NEED":
        viva_exp = f"Process {p_name} requested more than its declared maximum claim (Request > Need), violating Banker's invariants."
    elif core_res["status"] == "WAITING":
        viva_exp = f"Process {p_name}'s request is within Need, but exceeds currently Available resources. Placed in waiting queue; allocation unchanged."
    elif core_res["status"] == "GRANTED":
        viva_exp = f"Granting {p_name}'s request leaves the system in a verified safe state. System state committed."
    else:
        viva_exp = f"Granting {p_name}'s request would leave the system unsafe with no guaranteed completion sequence. Request denied and original state restored."

    return {
        "status": ui_status,
        "core_status": core_res["status"],
        "granted": is_granted,
        "reason": core_res["message"],
        "process_label": p_name,
        "request_vector": list(request_vector) if hasattr(request_vector, '__iter__') else [request_vector],
        "state_changed": is_granted,
        "allocation": new_alloc,
        "available": new_avail,
        "need": new_need,
        "before_state": {"allocation": alloc_orig, "available": avail_orig, "need": need_orig},
        "after_state": {"allocation": new_alloc, "available": new_avail, "need": new_need},
        "safety_result": core_res["safety_result"],
        "viva_explanation": viva_exp
    }


def format_steps_as_dataframe(steps: List[Dict[str, Any]], resource_labels: Optional[List[str]] = None) -> Any:
    """Helper function to convert steps into a pandas DataFrame."""
    import pandas as pd
    if not steps:
        return pd.DataFrame(columns=["Step", "Process Picked", "Need[i]", "Work Before", "Allocation Released", "Work After"])

    rows = []
    for s in steps:
        rows.append({
            "Step": s["step"],
            "Process Picked": str(s["process"]),
            "Need[i]": str(s["need"]),
            "Work Before": str(s["work_before"]),
            "Allocation Released": str(s["allocation_released"]),
            "Work After": str(s["work_after"])
        })
    return pd.DataFrame(rows)
