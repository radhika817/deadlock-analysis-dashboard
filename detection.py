"""
detection.py - Deadlock Detection for Single-Instance and Multiple-Instance Resource Models.

This module implements:
1. Single-Instance Model:
   - Resource-Allocation Graph (RAG) construction with processes and resources.
   - Wait-For Graph (WFG) reduction (processes waiting directly on processes).
   - Cycle detection with cycle nodes/edges highlighted in red.
   - Theorem: In single-instance systems, a cycle is necessary and sufficient for deadlock.
2. Multiple-Instance Model:
   - Matrix-based detection algorithm using Allocation, Request, Available, Work, and Finish.
   - Initial condition: Finish[i] = True if Allocation[i] == 0 (cannot hold resources needed by others).
   - Step-by-step progress tracking.
   - Theorem: In multiple-instance systems, a graph cycle is NOT conclusive proof of deadlock;
     the matrix detection algorithm must be applied.
3. Recovery by Process Termination:
   - Identifies suggested victim (e.g. deadlocked process holding maximum resources).
   - Simulates termination: releases its allocated resources into Available/Work.
   - Re-evaluates system to verify whether the deadlock has been resolved.

All logic is independent of the UI and includes comprehensive docstrings for viva preparation.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io


# ============================================================================
# 1. SINGLE-INSTANCE DETECTION (GRAPH-BASED)
# ============================================================================

def build_resource_allocation_graph(
    processes: List[str],
    resources: List[str],
    allocation_edges: List[Tuple[str, str]],
    request_edges: List[Tuple[str, str]]
) -> nx.DiGraph:
    """
    Constructs a directed bipartite Resource-Allocation Graph (RAG).

    Edge conventions:
    - Allocation edge: Resource -> Process (Resource is assigned to Process)
    - Request edge: Process -> Resource (Process is waiting for Resource)

    Args:
        processes: List of process labels (e.g. ['P1', 'P2', 'P3']).
        resources: List of resource labels (e.g. ['R1', 'R2', 'R3']).
        allocation_edges: List of (Resource, Process) tuples.
        request_edges: List of (Process, Resource) tuples.

    Returns:
        A NetworkX DiGraph with node attributes 'bipartite' ('process' or 'resource').
    """
    g = nx.DiGraph()

    for p in processes:
        g.add_node(p, bipartite="process", node_type="process")
    for r in resources:
        g.add_node(r, bipartite="resource", node_type="resource")

    for r, p in allocation_edges:
        g.add_edge(r, p, edge_type="allocation")

    for p, r in request_edges:
        g.add_edge(p, r, edge_type="request")

    return g


def build_wait_for_graph(
    processes: List[str],
    allocation_edges: List[Tuple[str, str]],
    request_edges: List[Tuple[str, str]]
) -> nx.DiGraph:
    """
    Constructs a Wait-For Graph (WFG) from RAG edges.

    In a single-instance system, if Process Pi requests Resource Rj,
    and Resource Rj is currently allocated to Process Pk,
    then Pi is waiting for Pk (directed edge Pi -> Pk).

    Args:
        processes: List of process labels.
        allocation_edges: List of (Resource, Process) tuples.
        request_edges: List of (Process, Resource) tuples.

    Returns:
        A NetworkX DiGraph containing only process nodes and wait-for edges.
    """
    wfg = nx.DiGraph()
    for p in processes:
        wfg.add_node(p)

    # Map each resource to the process currently holding it
    resource_holder: Dict[str, str] = {}
    for r, p in allocation_edges:
        resource_holder[r] = p

    # If Pi requests Rj, and Rj is held by Pk, add edge Pi -> Pk
    for pi, rj in request_edges:
        if rj in resource_holder:
            pk = resource_holder[rj]
            if pi != pk:
                wfg.add_edge(pi, pk)

    return wfg


def detect_single_instance_deadlock(
    processes: List[str],
    resources: List[str],
    allocation_edges: List[Tuple[str, str]],
    request_edges: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Performs deadlock detection on a single-instance resource system.

    Builds the Wait-For Graph and finds directed cycles.
    For single-instance systems, cycle <=> deadlock.

    Returns:
        Result dictionary with status, deadlocked processes, cycles, and suggested victim.
    """
    rag = build_resource_allocation_graph(processes, resources, allocation_edges, request_edges)
    wfg = build_wait_for_graph(processes, allocation_edges, request_edges)

    # Detect cycles in WFG
    cycles = list(nx.simple_cycles(wfg))
    deadlocked_set: Set[str] = set()
    for c in cycles:
        for p in c:
            deadlocked_set.add(p)

    deadlocked_processes = sorted(list(deadlocked_set))
    is_deadlocked = len(deadlocked_processes) > 0

    # Find suggested victim: process in deadlock holding the most resources
    suggested_victim = None
    if is_deadlocked:
        # Count allocations per deadlocked process
        holdings: Dict[str, int] = {p: 0 for p in deadlocked_processes}
        for r, p in allocation_edges:
            if p in holdings:
                holdings[p] += 1
        # Pick process holding the highest count
        suggested_victim = max(holdings.items(), key=lambda item: item[1])[0]

    status = "DEADLOCK DETECTED" if is_deadlocked else "NO DEADLOCK"

    explanation = (
        f"Single-instance resource model: Cycle detection was performed on the Wait-For Graph. "
        + (
            f"Detected {len(cycles)} cycle(s): {cycles}. "
            f"Processes {deadlocked_processes} are deadlocked. "
            f"In a single-instance system, a cycle is a necessary and sufficient proof of deadlock."
            if is_deadlocked else
            "No cycles found in the Wait-For Graph. The system is completely deadlock-free."
        )
    )

    return {
        "model": "single-instance",
        "status": status,
        "is_deadlocked": is_deadlocked,
        "deadlocked_processes": deadlocked_processes,
        "cycles": cycles,
        "suggested_victim": suggested_victim,
        "rag": rag,
        "wfg": wfg,
        "explanation": explanation
    }


def draw_graphs_to_figure(
    rag: nx.DiGraph,
    wfg: nx.DiGraph,
    cycles: List[List[str]],
    title_suffix: str = ""
) -> plt.Figure:
    """
    Draws both the Resource Allocation Graph and Wait-For Graph side-by-side using Matplotlib.
    Highlights cycle nodes and edges in red.

    Returns:
        A Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Draw Resource Allocation Graph (RAG) ---
    ax1.set_title(f"Resource-Allocation Graph (RAG) {title_suffix}", fontsize=13, fontweight="bold", pad=12)

    # Cycle edges in WFG mapped back to RAG
    cycle_nodes_set = set()
    for c in cycles:
        for n in c:
            cycle_nodes_set.add(n)

    # Layout for bipartite graph
    pos_rag = nx.spring_layout(rag, seed=42, k=1.2)

    process_nodes = [n for n, d in rag.nodes(data=True) if d.get("node_type") == "process"]
    resource_nodes = [n for n, d in rag.nodes(data=True) if d.get("node_type") == "resource"]

    # Colors
    p_colors = ["#EF4444" if n in cycle_nodes_set else "#3B82F6" for n in process_nodes]
    r_colors = ["#10B981" for _ in resource_nodes]

    # Draw process nodes (circles)
    nx.draw_networkx_nodes(rag, pos_rag, nodelist=process_nodes, node_color=p_colors,
                           node_shape="o", node_size=1600, ax=ax1, edgecolors="#1E293B", linewidths=2)
    # Draw resource nodes (squares)
    nx.draw_networkx_nodes(rag, pos_rag, nodelist=resource_nodes, node_color=r_colors,
                           node_shape="s", node_size=1600, ax=ax1, edgecolors="#1E293B", linewidths=2)

    # Edge colors: Allocation (green dashed/solid), Request (orange/red)
    alloc_edges = [(u, v) for u, v, d in rag.edges(data=True) if d.get("edge_type") == "allocation"]
    req_edges = [(u, v) for u, v, d in rag.edges(data=True) if d.get("edge_type") == "request"]

    nx.draw_networkx_edges(rag, pos_rag, edgelist=alloc_edges, edge_color="#10B981",
                           arrows=True, arrowstyle="-|>", arrowsize=20, width=2.5, ax=ax1,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(rag, pos_rag, edgelist=req_edges, edge_color="#F59E0B",
                           arrows=True, arrowstyle="-|>", arrowsize=20, width=2.5, ax=ax1,
                           connectionstyle="arc3,rad=0.08")

    nx.draw_networkx_labels(rag, pos_rag, font_size=11, font_weight="bold", font_color="white", ax=ax1)
    ax1.axis("off")

    # Legend for RAG
    ax1.plot([], [], 'o', color='#3B82F6', label='Process (Active)')
    ax1.plot([], [], 'o', color='#EF4444', label='Process (Deadlocked)')
    ax1.plot([], [], 's', color='#10B981', label='Resource (1-instance)')
    ax1.plot([], [], '-', color='#10B981', label='Alloc: Resource -> Process')
    ax1.plot([], [], '-', color='#F59E0B', label='Request: Process -> Resource')
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # --- 2. Draw Wait-For Graph (WFG) ---
    ax2.set_title(f"Wait-For Graph (WFG) {title_suffix}", fontsize=13, fontweight="bold", pad=12)

    pos_wfg = nx.circular_layout(wfg) if len(wfg.nodes) > 1 else {list(wfg.nodes)[0]: (0, 0)}

    wfg_node_colors = ["#EF4444" if n in cycle_nodes_set else "#3B82F6" for n in wfg.nodes()]

    # Highlight cycle edges in red
    cycle_edge_set = set()
    for c in cycles:
        for idx in range(len(c)):
            u = c[idx]
            v = c[(idx + 1) % len(c)]
            cycle_edge_set.add((u, v))

    wfg_edge_colors = ["#EF4444" if (u, v) in cycle_edge_set else "#64748B" for u, v in wfg.edges()]
    wfg_edge_widths = [3.5 if (u, v) in cycle_edge_set else 1.8 for u, v in wfg.edges()]

    nx.draw_networkx_nodes(wfg, pos_wfg, node_color=wfg_node_colors,
                           node_shape="o", node_size=1800, ax=ax2, edgecolors="#1E293B", linewidths=2)
    nx.draw_networkx_edges(wfg, pos_wfg, edge_color=wfg_edge_colors, width=wfg_edge_widths,
                           arrows=True, arrowstyle="-|>", arrowsize=22, ax=ax2,
                           connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_labels(wfg, pos_wfg, font_size=11, font_weight="bold", font_color="white", ax=ax2)
    ax2.axis("off")

    ax2.plot([], [], 'o', color='#3B82F6', label='Process (Normal)')
    ax2.plot([], [], 'o', color='#EF4444', label='Process in Cycle (Deadlocked)')
    ax2.plot([], [], '-', color='#EF4444', linewidth=3, label='Cycle Edge (Deadlock)')
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


# ============================================================================
# 2. MULTIPLE-INSTANCE DETECTION (MATRIX METHOD)
# ============================================================================

def detect_multi_instance_deadlock(
    allocation: np.ndarray,
    request: np.ndarray,
    available: np.ndarray,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes the classic Deadlock Detection Algorithm for Multiple-Instance Resources.

    Algorithm:
    1. Work = Available (copy).
    2. For i = 0, 1, ..., n-1:
         if Allocation[i] != 0:
             Finish[i] = False
         else:
             Finish[i] = True (processes holding nothing cannot cause deadlocks).
    3. Find an index i such that:
         Finish[i] == False and Request[i] <= Work.
       If no such i exists, go to Step 5.
    4. Work = Work + Allocation[i]
       Finish[i] = True
       Record step details.
       Go to Step 3.
    5. If Finish[i] == False for some i, then process Pi is DEADLOCKED.

    Args:
        allocation: 2D array of current allocations (n x m).
        request: 2D array of current requests (n x m).
        available: 1D array of available resources (m).
        process_labels: Optional list of process labels.
        resource_labels: Optional list of resource labels.

    Returns:
        Result dictionary with status, deadlocked processes, step log, and viva explanation.
    """
    alloc_arr = np.array(allocation, dtype=int)
    req_arr = np.array(request, dtype=int)
    work = np.array(available, dtype=int).copy()

    n_proc, n_res = alloc_arr.shape
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    # Step 2: Initialize Finish
    # Finish[i] = True if Allocation[i] is entirely 0; otherwise False
    finish = [np.all(alloc_arr[i] == 0) for i in range(n_proc)]

    steps_log: List[Dict[str, Any]] = []
    step_num = 1

    # Record initially finished processes (holding 0 resources)
    init_zero_procs = [p_labels[i] for i in range(n_proc) if finish[i]]

    while True:
        found = False
        for i in range(n_proc):
            if not finish[i]:
                # Check if Request[i] <= Work
                if np.all(req_arr[i] <= work):
                    work_before = work.copy()
                    released = alloc_arr[i].copy()
                    work = work + released
                    finish[i] = True
                    found = True

                    steps_log.append({
                        "step": step_num,
                        "process": p_labels[i],
                        "request": req_arr[i].tolist(),
                        "work_before": work_before.tolist(),
                        "allocation_released": released.tolist(),
                        "work_after": work.copy().tolist(),
                        "description": (
                            f"Process {p_labels[i]} satisfies Request {req_arr[i].tolist()} "
                            f"<= Work {work_before.tolist()}. Runs to completion, returning "
                            f"Allocation {released.tolist()}. Work becomes {work.tolist()}."
                        )
                    })
                    step_num += 1
                    break
        if not found:
            break

    deadlocked_processes = [p_labels[i] for i in range(n_proc) if not finish[i]]
    is_deadlocked = len(deadlocked_processes) > 0

    # Suggested victim
    suggested_victim = None
    if is_deadlocked:
        # Pick deadlocked process holding the largest total allocation sum
        victim_scores = {
            p_labels[i]: int(alloc_arr[i].sum())
            for i in range(n_proc) if not finish[i]
        }
        suggested_victim = max(victim_scores.items(), key=lambda item: item[1])[0]

    status = "DEADLOCK DETECTED" if is_deadlocked else "NO DEADLOCK"

    explanation = (
        f"Multiple-instance resource model: Evaluated using the matrix detection algorithm. "
        + (
            f"Processes {deadlocked_processes} could not complete because their current requests "
            f"exceed the remaining available Work vector {work.tolist()}. These processes are DEADLOCKED. "
            f"CRITICAL CONCEPT: A simple cycle in a multiple-instance graph is NOT conclusive proof of deadlock; "
            f"only the full matrix check determines true deadlock."
            if is_deadlocked else
            f"All processes were able to satisfy their requests and return allocated resources. "
            f"No deadlock exists in the system."
        )
    )

    return {
        "model": "multiple-instance",
        "status": status,
        "is_deadlocked": is_deadlocked,
        "deadlocked_processes": deadlocked_processes,
        "initially_zero_processes": init_zero_procs,
        "steps": steps_log,
        "final_work": work.tolist(),
        "suggested_victim": suggested_victim,
        "explanation": explanation
    }


# ============================================================================
# 3. DEADLOCK RECOVERY (PROCESS TERMINATION)
# ============================================================================

def recover_by_terminating_process_single(
    processes: List[str],
    resources: List[str],
    allocation_edges: List[Tuple[str, str]],
    request_edges: List[Tuple[str, str]],
    victim_process: str
) -> Dict[str, Any]:
    """
    Simulates recovery in a single-instance system by terminating a victim process.

    Removes all allocation and request edges associated with the terminated process.
    Re-runs single-instance deadlock detection to verify whether the cycle is broken.
    """
    new_processes = [p for p in processes if p != victim_process]

    # Find resources released by victim (edges R -> victim)
    released_resources = [r for r, p in allocation_edges if p == victim_process]

    new_alloc_edges = [(r, p) for r, p in allocation_edges if p != victim_process]
    new_req_edges = [(p, r) for p, r in request_edges if p != victim_process]

    new_rag = build_resource_allocation_graph(new_processes, resources, new_alloc_edges, new_req_edges)
    new_wfg = build_wait_for_graph(new_processes, new_alloc_edges, new_req_edges)

    new_cycles = list(nx.simple_cycles(new_wfg))
    deadlocked_set = set()
    for c in new_cycles:
        for p in c:
            deadlocked_set.add(p)
    new_deadlocked = sorted(list(deadlocked_set))
    is_resolved = len(new_deadlocked) == 0

    return {
        "victim_terminated": victim_process,
        "released_resources": released_resources,
        "remaining_processes": new_processes,
        "new_allocation_edges": new_alloc_edges,
        "new_request_edges": new_req_edges,
        "new_rag": new_rag,
        "new_wfg": new_wfg,
        "new_cycles": new_cycles,
        "new_deadlocked": new_deadlocked,
        "is_resolved": is_resolved,
        "status": "NO DEADLOCK (RESOLVED)" if is_resolved else "DEADLOCK PERSISTS",
        "explanation": (
            f"Process {victim_process} was terminated. Resources released: {released_resources}. "
            + (
                f"The circular wait is completely broken! No cycles remain in the Wait-For Graph. Deadlock resolved."
                if is_resolved else
                f"Deadlock still persists among remaining processes {new_deadlocked}. Additional termination required."
            )
        )
    }


def recover_by_terminating_process_multi(
    allocation: np.ndarray,
    request: np.ndarray,
    available: np.ndarray,
    process_labels: List[str],
    resource_labels: List[str],
    victim_process: str
) -> Dict[str, Any]:
    """
    Simulates recovery in a multiple-instance system by terminating a victim process.

    Releases victim's allocation into Available vector and re-runs detection.
    """
    alloc_arr = np.array(allocation, dtype=int).copy()
    req_arr = np.array(request, dtype=int).copy()
    avail_arr = np.array(available, dtype=int).copy()

    victim_idx = process_labels.index(victim_process)
    released_resources = alloc_arr[victim_idx].copy()

    # Reclaim resources
    new_avail = avail_arr + released_resources
    # Set victim alloc and request to 0
    alloc_arr[victim_idx] = 0
    req_arr[victim_idx] = 0

    # Re-run multi-instance detection
    post_detection = detect_multi_instance_deadlock(
        alloc_arr, req_arr, new_avail, process_labels, resource_labels
    )

    is_resolved = not post_detection["is_deadlocked"]

    return {
        "victim_terminated": victim_process,
        "released_vector": released_resources.tolist(),
        "work_before_release": avail_arr.tolist(),
        "work_after_release": new_avail.tolist(),
        "post_detection": post_detection,
        "is_resolved": is_resolved,
        "status": "NO DEADLOCK (RESOLVED)" if is_resolved else "DEADLOCK PERSISTS",
        "explanation": (
            f"Terminated process {victim_process}, releasing resources {released_resources.tolist()}. "
            f"Available Work increased from {avail_arr.tolist()} to {new_avail.tolist()}. "
            + (
                f"Detection algorithm confirmed all remaining processes can now complete! Deadlock resolved."
                if is_resolved else
                f"Deadlock still persists for processes: {post_detection['deadlocked_processes']}."
            )
        )
    }
