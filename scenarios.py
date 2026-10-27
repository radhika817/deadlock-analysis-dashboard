"""
scenarios.py - Preset test scenarios A through F for the Deadlock Analysis Dashboard.

Contains pre-configured textbook scenarios and boundary cases:
- Scenario A: Safe State (Classic 5-process, 3-resource Banker's example)
- Scenario B: Request Exceeds Available (Within Need, but exceeds Available -> WAITING)
- Scenario C: Unsafe Request (Within Need and Available, but leaves state unsafe -> DENIED)
- Scenario D: Deadlock Detection (Single-instance cycle & Multiple-instance detection)
- Scenario E: Prevention Applied to Scenario D (Resource Ordering & Request-All-Upfront)
- Scenario F: Boundary and Invalid Inputs (Zero Available, Zero Need, Negative, Alloc > Max, etc.)
"""

from typing import Dict, Any
import numpy as np


def get_scenario_a() -> Dict[str, Any]:
    """
    Scenario A: Classic Safe State.
    5 processes (P0-P4), 3 resource types (A, B, C) with instances Total=(10, 5, 7).
    Available = (3, 3, 2).
    A safe sequence exists: e.g. P1 -> P3 -> P4 -> P0 -> P2.
    """
    return {
        "id": "A",
        "title": "Scenario A: Safe State",
        "description": "Classic textbook 5-process, 3-resource system in a proven safe state.",
        "num_processes": 5,
        "num_resources": 3,
        "process_labels": ["P0", "P1", "P2", "P3", "P4"],
        "resource_labels": ["A", "B", "C"],
        "allocation": np.array([
            [0, 1, 0],  # P0
            [2, 0, 0],  # P1
            [3, 0, 2],  # P2
            [2, 1, 1],  # P3
            [0, 0, 2],  # P4
        ], dtype=int),
        "maximum": np.array([
            [7, 5, 3],  # P0
            [3, 2, 2],  # P1
            [9, 0, 2],  # P2
            [2, 2, 2],  # P3
            [4, 3, 3],  # P4
        ], dtype=int),
        "available": np.array([3, 3, 2], dtype=int),
        "total": np.array([10, 5, 7], dtype=int),
        "expected_result": "SAFE",
        "expected_sequence": ["P1", "P3", "P4", "P0", "P2"],
        "viva_explanation": (
            "Available = [3, 3, 2]. P1 has Need [1, 2, 2] <= Available, so P1 can execute "
            "and release its Allocation [2, 0, 0], making Work = [5, 3, 2]. Next, P3 (Need [0, 1, 1]) "
            "can run, releasing [2, 1, 1] to make Work = [7, 4, 3], and so on. Every process completes."
        )
    }


def get_scenario_b() -> Dict[str, Any]:
    """
    Scenario B: Request Exceeds Available (WAITING / DEFERRED).
    Starting after P1 has been granted (1, 0, 2):
    Available is (2, 3, 0).
    Now P4 requests (3, 3, 0).
    Need for P4 is (4, 3, 1).
    Request <= Need holds, but Request <= Available fails (3 > 2 for A).
    Process P4 must wait. System state remains unchanged.
    """
    # Base state after P1 granted (1, 0, 2)
    alloc = np.array([
        [0, 1, 0],  # P0
        [3, 0, 2],  # P1 (was 2,0,0 + 1,0,2)
        [3, 0, 2],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2],  # P4
    ], dtype=int)
    max_m = np.array([
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3],
    ], dtype=int)
    avail = np.array([2, 3, 0], dtype=int)  # 3-1, 3-0, 2-2

    return {
        "id": "B",
        "title": "Scenario B: Request Exceeds Available (Waiting / Deferred)",
        "description": "P4 requests (3, 3, 0). Within Need (4, 3, 1), but exceeds Available (2, 3, 0).",
        "num_processes": 5,
        "num_resources": 3,
        "process_labels": ["P0", "P1", "P2", "P3", "P4"],
        "resource_labels": ["A", "B", "C"],
        "allocation": alloc,
        "maximum": max_m,
        "available": avail,
        "total": np.array([10, 5, 7], dtype=int),
        "request_process": 4,  # P4
        "request_process_label": "P4",
        "request_vector": np.array([3, 3, 0], dtype=int),
        "expected_result": "WAITING / DEFERRED",
        "viva_explanation": (
            "P4's request [3, 3, 0] <= Need [4, 3, 1] is true, so the process did not exceed its maximum claim. "
            "However, Request [3, 3, 0] <= Available [2, 3, 0] is false (3 > 2 for resource A). "
            "Therefore, P4 must wait until resources become available. This is NOT an error and NOT unsafe; "
            "the current allocation is strictly preserved."
        )
    }


def get_scenario_c() -> Dict[str, Any]:
    """
    Scenario C: Unsafe Request (State must be restored).
    Starting after P1 has been granted (1, 0, 2):
    Available is (2, 3, 0).
    Now P0 requests (0, 2, 0).
    P0 Need is (7, 4, 3).
    Request <= Need holds ([0, 2, 0] <= [7, 4, 3]).
    Request <= Available holds ([0, 2, 0] <= [2, 3, 0]).
    However, temporarily allocating gives Available = (2, 1, 0).
    At Available = (2, 1, 0), no process's remaining Need can be satisfied:
    P0 need: (7, 2, 3) > (2, 1, 0)
    P1 need: (0, 2, 0) > (2, 1, 0) (Needs 2 of B, only 1 available)
    P2 need: (6, 0, 0) > (2, 1, 0)
    P3 need: (0, 1, 1) > (2, 1, 0) (Needs 1 of C, 0 available)
    P4 need: (4, 3, 1) > (2, 1, 0)
    Safety algorithm fails. The request is DENIED and the original state is RESTORED.
    """
    alloc = np.array([
        [0, 1, 0],
        [3, 0, 2],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2],
    ], dtype=int)
    max_m = np.array([
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3],
    ], dtype=int)
    avail = np.array([2, 3, 0], dtype=int)

    return {
        "id": "C",
        "title": "Scenario C: Request Leaves System in Unsafe State (Denied & Restored)",
        "description": "P0 requests (0, 2, 0). Request <= Need and Request <= Available, but simulated state is UNSAFE.",
        "num_processes": 5,
        "num_resources": 3,
        "process_labels": ["P0", "P1", "P2", "P3", "P4"],
        "resource_labels": ["A", "B", "C"],
        "allocation": alloc,
        "maximum": max_m,
        "available": avail,
        "total": np.array([10, 5, 7], dtype=int),
        "request_process": 0,  # P0
        "request_process_label": "P0",
        "request_vector": np.array([0, 2, 0], dtype=int),
        "expected_result": "UNSAFE (DENIED & RESTORED)",
        "viva_explanation": (
            "Granting P0's request would reduce Available to [2, 1, 0]. In this simulated state, "
            "no process can guarantee completion because every process requires more resources than Available. "
            "Banker's Algorithm avoids deadlock by denying the request and rolling back to the previous safe state."
        )
    }


def get_scenario_d_single() -> Dict[str, Any]:
    """
    Scenario D (Single-instance model):
    Processes P1, P2, P3. Resources R1, R2, R3 (1 instance each).
    P1 holds R1, requests R2.
    P2 holds R2, requests R1.
    P3 holds R3, requests nothing (independent).
    Cycle in RAG / Wait-For Graph: P1 -> R2 -> P2 -> R1 -> P1.
    For single-instance, a cycle implies DEADLOCK.
    """
    return {
        "id": "D_single",
        "model": "single-instance",
        "title": "Scenario D: Deadlock Detection (Single-Instance Model)",
        "description": "Circular wait between P1 and P2 on resources R1 and R2. P3 is independent.",
        "processes": ["P1", "P2", "P3"],
        "resources": ["R1", "R2", "R3"],
        # Allocation edges: Resource -> Process (resource held by process)
        # Request edges: Process -> Resource (process waiting for resource)
        "allocation_edges": [
            ("R1", "P1"),
            ("R2", "P2"),
            ("R3", "P3"),
        ],
        "request_edges": [
            ("P1", "R2"),
            ("P2", "R1"),
        ],
        "expected_result": "DEADLOCK DETECTED",
        "deadlocked_processes": ["P1", "P2"],
        "cycles": [["P1", "P2", "P1"]],
        "suggested_victim": "P1",
        "viva_explanation": (
            "In a single-instance resource system, a cycle in the Resource Allocation Graph (or Wait-For Graph) "
            "is both a NECESSARY and SUFFICIENT condition for deadlock. P1 waits for R2 held by P2, while P2 waits "
            "for R1 held by P1. Neither can proceed."
        )
    }


def get_scenario_d_multi() -> Dict[str, Any]:
    """
    Scenario D (Multiple-instance model):
    5 processes (P0-P4), 3 resource types (A, B, C).
    Allocation and Request matrices with Available=(0, 0, 0).
    Processes P1 and P2 are deadlocked.
    """
    alloc = np.array([
        [0, 1, 0],  # P0
        [2, 0, 0],  # P1
        [3, 0, 3],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2],  # P4
    ], dtype=int)
    # P1 requests (2, 0, 2) which cannot be satisfied, P2 requests (0, 0, 1)...
    request = np.array([
        [0, 0, 0],  # P0 has finished its requests
        [2, 0, 2],  # P1 needs (2, 0, 2)
        [1, 0, 1],  # P2 needs (1, 0, 1)
        [1, 1, 0],  # P3
        [0, 0, 2],  # P4
    ], dtype=int)
    avail = np.array([0, 0, 0], dtype=int)

    return {
        "id": "D_multi",
        "model": "multiple-instance",
        "title": "Scenario D: Deadlock Detection (Multiple-Instance Matrix Method)",
        "description": "Multi-instance detection algorithm evaluates if any sequence of processes can finish.",
        "processes": ["P0", "P1", "P2", "P3", "P4"],
        "resources": ["A", "B", "C"],
        "allocation": alloc,
        "request": request,
        "available": avail,
        "expected_result": "DEADLOCK DETECTED",
        "viva_explanation": (
            "In multiple-instance systems, graph cycles are NOT conclusive. We must apply the matrix detection "
            "algorithm (finding processes whose Request <= Work). Here, after P0 releases [0, 1, 0], no other process's "
            "Request can be satisfied with Work=[0, 1, 0]. Processes {P1, P2, P3, P4} remain deadlocked."
        )
    }


def get_scenario_e() -> Dict[str, Any]:
    """
    Scenario E: Prevention Applied to Scenario D.
    Comparing Resource Ordering vs Request-All-Upfront.
    """
    return {
        "id": "E",
        "title": "Scenario E: Deadlock Prevention Policies",
        "description": "Applying Resource Ordering (eliminates Circular Wait) and Request-All-Upfront (eliminates Hold and Wait).",
        "resource_order": {"R1": 1, "R2": 2, "R3": 3},
        "original_scenario": "Scenario D Single-Instance (P1 holds R1, wants R2; P2 holds R2, wants R1)",
        "resource_ordering_result": {
            "policy": "Resource Ordering (Global Hierarchy: R1=1 < R2=2 < R3=3)",
            "condition_eliminated": "Circular Wait",
            "explanation": (
                "P1 holds R1 (rank 1) and requests R2 (rank 2) -> ALLOWED because 2 > 1.\n"
                "P2 holds R2 (rank 2) and requests R1 (rank 1) -> REJECTED because 1 < 2!\n"
                "To request R1, P2 must first release R2. Thus, the circular wait cycle is broken mathematically."
            ),
            "tradeoffs": [
                "Lower resource utilization (resources must be acquired in rigid order)",
                "Processes may hold resources earlier than actually needed",
                "Programmers must adhere to strict global ordering across all threads/processes"
            ]
        },
        "request_all_upfront_result": {
            "policy": "Request All Resources Upfront",
            "condition_eliminated": "Hold and Wait",
            "explanation": (
                "Processes cannot hold resources while waiting for others. "
                "P1 must acquire both {R1, R2} atomically before starting. If both are not free, it holds NONE.\n"
                "Hence, P1 holds nothing while waiting, eliminating the Hold-and-Wait condition entirely."
            ),
            "tradeoffs": [
                "Very low resource utilization (resources held throughout entire job lifetime)",
                "Starvation possible for processes requiring many resources",
                "Reduced concurrency across the operating system"
            ]
        }
    }


def get_scenario_f() -> Dict[str, Any]:
    """
    Scenario F: Boundary and Invalid Inputs.
    Provides several explicit test cases to prove dashboard robustness and validation logic.
    """
    return {
        "id": "F",
        "title": "Scenario F: Boundary Cases and Input Validation",
        "description": "Validation suite covering zero Available, zero Need, negative values, and Allocation > Max.",
        "cases": [
            {
                "id": "F1",
                "name": "Zero Available Resources",
                "type": "VALID BOUNDARY",
                "details": "Available = [0, 0, 0]. System is functioning, but all resources are currently allocated.",
                "valid": True,
                "available": np.array([0, 0, 0], dtype=int),
                "allocation": np.array([[1, 1, 1], [0, 0, 0]]),
                "maximum": np.array([[1, 1, 1], [1, 1, 1]]),
                "expected": "Valid configuration; safety algorithm will evaluate if processes with Need=0 can start."
            },
            {
                "id": "F2",
                "name": "Process with Zero Remaining Need",
                "type": "VALID BOUNDARY",
                "details": "P0 has Allocation = [2, 2, 2] and Maximum = [2, 2, 2], so Need = [0, 0, 0].",
                "valid": True,
                "available": np.array([1, 1, 1], dtype=int),
                "allocation": np.array([[2, 2, 2]]),
                "maximum": np.array([[2, 2, 2]]),
                "expected": "Valid configuration; P0 requires no additional resources and finishes immediately."
            },
            {
                "id": "F3",
                "name": "Allocation Exceeds Declared Maximum",
                "type": "INVALID INPUT",
                "details": "Allocation[P0][B] = 5, but Maximum[P0][B] = 2.",
                "valid": False,
                "available": np.array([3, 3, 2], dtype=int),
                "allocation": np.array([[0, 5, 0]]),
                "maximum": np.array([[7, 2, 3]]),
                "expected": "Invalid input error flagged with exact cell [P0, B]."
            },
            {
                "id": "F4",
                "name": "Negative Resource Value in Matrix/Request",
                "type": "INVALID INPUT",
                "details": "Request vector contains [-1, 2, 0].",
                "valid": False,
                "request": [-1, 2, 0],
                "expected": "Invalid input: negative resource quantities are strictly disallowed."
            },
            {
                "id": "F5",
                "name": "Invalid Process Identifier",
                "type": "INVALID INPUT",
                "details": "Request submitted for process 'P99' when only P0-P4 exist.",
                "valid": False,
                "process_idx": 99,
                "expected": "Invalid process identifier error."
            },
            {
                "id": "F6",
                "name": "Wrong Length Request Vector",
                "type": "INVALID INPUT",
                "details": "Request has 2 items when system has 3 resource types.",
                "valid": False,
                "request": [1, 2],
                "expected": "Dimension mismatch error: expected vector of length 3."
            }
        ]
    }


def get_default_state() -> Dict[str, Any]:
    """Returns the default pre-filled dashboard state (Scenario A)."""
    return get_scenario_a()
