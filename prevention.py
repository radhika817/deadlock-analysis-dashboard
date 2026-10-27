"""
prevention.py - Deadlock Prevention Policies and Trade-off Analysis.

This module implements:
1. Explanations of the four necessary Coffman conditions for deadlock:
   - Mutual Exclusion
   - Hold and Wait
   - No Preemption
   - Circular Wait
2. Deadlock Prevention Policies:
   - Policy A: Resource Ordering (Global Total Ordering of Resource Types)
     Eliminates: Circular Wait.
     Mechanism: A process holding resource Ri can only request Rj if Order(Rj) > Order(Ri).
   - Policy B: Request-All-Upfront (Conservative Allocation)
     Eliminates: Hold and Wait.
     Mechanism: A process must request and be allocated all its needed resources
     before it begins execution; otherwise it holds nothing while waiting.
3. Comparative Simulation & Trade-off Quantification:
   - Evaluates a scenario (such as Scenario D) before and after applying each policy.
   - Calculates resource idle time, concurrency reduction, and starvation risks.

All logic is independent of the UI and includes comprehensive docstrings for viva preparation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def get_coffman_conditions_summary() -> List[Dict[str, Any]]:
    """
    Returns educational documentation on the four Coffman conditions required for deadlock,
    including how each condition can theoretically be prevented.
    """
    return [
        {
            "name": "Mutual Exclusion",
            "definition": "At least one resource must be held in a non-shareable mode (only one process at a time).",
            "prevention_strategy": "Make resources shareable (e.g., read-only files). However, intrinsic hardware/locks cannot always be shared.",
            "broken_by": "Spooling or read-only access (impractical for exclusive hardware like printers/mutexes)."
        },
        {
            "name": "Hold and Wait",
            "definition": "A process must be holding at least one resource and waiting to acquire additional resources held by other processes.",
            "prevention_strategy": "Request-All-Upfront: A process must request and obtain all required resources before starting execution, or release all current holdings before requesting more.",
            "broken_by": "Request-All-Upfront Policy."
        },
        {
            "name": "No Preemption",
            "definition": "Resources cannot be forcibly taken from a process; they can only be released voluntarily after completion.",
            "prevention_strategy": "Resource Preemption: If a process requests a resource that cannot be immediately allocated, all resources currently held are preempted (released).",
            "broken_by": "Preemption protocols (suitable only for easily saveable states like CPU registers or memory pages)."
        },
        {
            "name": "Circular Wait",
            "definition": "A closed chain of processes exists such that each process holds at least one resource needed by the next process in the chain.",
            "prevention_strategy": "Resource Ordering: Impose a global linear ordering on all resource types. Processes may only request resources in strictly increasing order.",
            "broken_by": "Hierarchical Resource Ordering Policy."
        }
    ]


def evaluate_resource_ordering_prevention(
    resources: List[str],
    resource_order: Dict[str, int],
    process_requests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Simulates the Resource Ordering prevention policy.

    Rule: If process P holds resource R_held, it can only request R_req if:
          order(R_req) > order(R_held).
          If order(R_req) <= order(R_held), the request is REJECTED.
          The process must release R_held before requesting R_req.

    Args:
        resources: List of resource identifiers.
        resource_order: Mapping from resource label to integer priority (e.g. {'R1': 1, 'R2': 2, 'R3': 3}).
        process_requests: List of dicts: {'process': 'P1', 'holding': ['R1'], 'requesting': 'R2'}

    Returns:
        Evaluation report showing which requests are allowed/rejected, proving circular wait is impossible.
    """
    evaluation_log = []
    rejected_requests = []
    allowed_requests = []

    for req in process_requests:
        p = req["process"]
        held = req.get("holding", [])
        requesting = req["requesting"]

        req_order = resource_order.get(requesting, 0)
        max_held_order = max([resource_order.get(r, 0) for r in held]) if held else 0

        if held and req_order <= max_held_order:
            # Violation!
            violating_res = [r for r in held if resource_order.get(r, 0) >= req_order]
            msg = (
                f"REJECTED: Process {p} holds {violating_res} (order >= {req_order}) and attempts to request "
                f"{requesting} (order {req_order}). Strict increasing order violated. "
                f"Process must release {violating_res} first."
            )
            rejected_requests.append(p)
            evaluation_log.append({
                "process": p,
                "holding": held,
                "requesting": requesting,
                "allowed": False,
                "reason": msg
            })
        else:
            msg = (
                f"ALLOWED: Process {p} requests {requesting} (order {req_order}), which is strictly "
                f"greater than highest held resource order ({max_held_order})."
            )
            allowed_requests.append(p)
            evaluation_log.append({
                "process": p,
                "holding": held,
                "requesting": requesting,
                "allowed": True,
                "reason": msg
            })

    return {
        "policy": "Resource Ordering (Hierarchical Acquisition)",
        "condition_eliminated": "Circular Wait",
        "resource_order": resource_order,
        "evaluation_log": evaluation_log,
        "allowed_count": len(allowed_requests),
        "rejected_count": len(rejected_requests),
        "deadlock_prevented": True,
        "explanation": (
            "Because all processes must request resources in strictly increasing numerical order, "
            "it is mathematically impossible to form a directed cycle. Therefore, Circular Wait "
            "is completely eliminated, and deadlock cannot occur."
        ),
        "tradeoffs": [
            {
                "aspect": "Resource Utilization",
                "impact": "Sub-optimal / Lower",
                "detail": "Processes may acquire higher-ordered resources later than needed or must prematurely acquire lower-ordered ones."
            },
            {
                "aspect": "Programmer Burden",
                "impact": "High",
                "detail": "Application developers must strictly adhere to the system-wide global ordering in every function and thread."
            },
            {
                "aspect": "Concurrency",
                "impact": "Moderate",
                "detail": "Some operations must serialize according to the resource hierarchy."
            }
        ]
    }


def evaluate_request_all_upfront_prevention(
    processes: List[str],
    total_needed_per_proc: Dict[str, Dict[str, int]],
    available_total: Dict[str, int]
) -> Dict[str, Any]:
    """
    Simulates the Request-All-Upfront prevention policy.

    Rule: A process must request all its resources at once before starting.
          If all requested resources are not currently available, it gets NOTHING and waits.
          While waiting, it holds 0 resources.

    Eliminates: Hold and Wait.
    """
    current_avail = available_total.copy()
    execution_order = []
    waiting_processes = []
    log = []

    # Attempt to allocate to processes that can acquire everything upfront
    for p in processes:
        needed = total_needed_per_proc.get(p, {})
        can_satisfy_all = all(current_avail.get(r, 0) >= qty for r, qty in needed.items())

        if can_satisfy_all:
            # Grant all upfront
            for r, qty in needed.items():
                current_avail[r] -= qty
            execution_order.append(p)
            log.append({
                "process": p,
                "action": "GRANTED ALL UPFRONT",
                "details": f"Allocated all required resources {needed}. Runs to completion.",
                "remaining_avail": current_avail.copy()
            })
            # Simulated completion: releases all
            for r, qty in needed.items():
                current_avail[r] += qty
        else:
            waiting_processes.append(p)
            log.append({
                "process": p,
                "action": "DEFERRED / WAITING (HOLDS 0)",
                "details": f"Cannot acquire all resources {needed} at once. Allocated 0 resources while waiting.",
                "remaining_avail": current_avail.copy()
            })

    return {
        "policy": "Request All Resources Upfront",
        "condition_eliminated": "Hold and Wait",
        "execution_order": execution_order,
        "waiting_processes": waiting_processes,
        "log": log,
        "deadlock_prevented": True,
        "explanation": (
            "Processes never hold any resources while waiting for other resources. "
            "Since no process can be in a 'holding some and waiting for others' state, "
            "the Hold-and-Wait condition is mathematically eliminated."
        ),
        "tradeoffs": [
            {
                "aspect": "Resource Utilization",
                "impact": "Extremely Poor",
                "detail": "Resources allocated to a process sit idle until the specific phase where they are actually used."
            },
            {
                "aspect": "Starvation",
                "impact": "High Risk",
                "detail": "Processes needing multiple popular resources may wait indefinitely if resources are constantly occupied by short jobs."
            },
            {
                "aspect": "System Concurrency",
                "impact": "Severely Reduced",
                "detail": "Locks resources for the entire lifespan of a job, blocking other processes that only need them briefly."
            }
        ]
    }
