"""
test_logic.py - Unit tests for Deadlock Analysis and Resource Allocation algorithms.

Covers:
1. Need matrix calculation.
2. Banker's Safety Algorithm on Scenario A (Classic safe sequence).
3. Resource request handling:
   - Scenario B: Request exceeds Available (WAITING / DEFERRED, state unchanged).
   - Scenario C: Unsafe request (simulated -> UNSAFE -> DENIED & RESTORED).
   - Valid safe request (GRANTED -> state updated).
4. Deadlock Detection:
   - Single-instance Resource Allocation Graph and Wait-For Graph cycle detection.
   - Multi-instance matrix detection method.
5. Deadlock Recovery:
   - Process termination in single-instance graph (breaking circular wait).
   - Process termination in multi-instance system (resource reclamation & resolution).
6. Deadlock Prevention Policies:
   - Resource Ordering (breaking Circular Wait).
   - Request-All-Upfront (breaking Hold and Wait).
7. Validation & Boundary Conditions:
   - Zero Available (valid boundary).
   - Zero Need (valid boundary).
   - Allocation > Maximum (invalid input).
   - Negative values (invalid input).
   - Invalid process ID / vector length.
"""

import pytest
import numpy as np
import networkx as nx

from banker import (
    calculate_need,
    safety_check,
    handle_request,
    calculate_need_matrix,
    run_safety_algorithm,
    handle_resource_request
)
from detection import (
    detect_single_instance_deadlock,
    detect_multi_instance_deadlock,
    recover_by_terminating_process_single,
    recover_by_terminating_process_multi
)
from prevention import (
    evaluate_resource_ordering_prevention,
    evaluate_request_all_upfront_prevention,
    get_coffman_conditions_summary
)
from validation import (
    validate_matrix_shape_and_types,
    validate_vector,
    validate_system_state,
    validate_request_input
)
from scenarios import (
    get_scenario_a,
    get_scenario_b,
    get_scenario_c,
    get_scenario_d_single,
    get_scenario_d_multi,
    get_scenario_e,
    get_scenario_f
)


# ============================================================================
# 1. NEED MATRIX CALCULATION
# ============================================================================

def test_need_matrix_calculation():
    """Verify Need = Maximum - Allocation for textbook Scenario A."""
    sc_a = get_scenario_a()
    need = calculate_need_matrix(sc_a["maximum"], sc_a["allocation"])
    
    # Expected Need for P0: [7-0, 5-1, 3-0] = [7, 4, 3]
    # Expected Need for P1: [3-2, 2-0, 2-0] = [1, 2, 2]
    # Expected Need for P2: [9-3, 0-0, 2-2] = [6, 0, 0]
    # Expected Need for P3: [2-2, 2-1, 2-1] = [0, 1, 1]
    # Expected Need for P4: [4-0, 3-0, 3-2] = [4, 3, 1]
    expected_need = np.array([
        [7, 4, 3],
        [1, 2, 2],
        [6, 0, 0],
        [0, 1, 1],
        [4, 3, 1]
    ])
    np.testing.assert_array_equal(need, expected_need)


# ============================================================================
# 2. BANKER'S SAFETY ALGORITHM (SCENARIO A)
# ============================================================================

def test_safety_algorithm_scenario_a():
    """Scenario A should be SAFE with all 5 processes finishing in valid order."""
    sc_a = get_scenario_a()
    need = calculate_need_matrix(sc_a["maximum"], sc_a["allocation"])
    res = run_safety_algorithm(
        sc_a["allocation"], need, sc_a["available"],
        sc_a["process_labels"], sc_a["resource_labels"]
    )

    assert res["is_safe"] is True
    assert res["status"] == "SAFE"
    assert len(res["safe_sequence"]) == 5
    assert len(res["steps"]) == 5
    assert res["safe_sequence"][0] == "P1"  # P1 must be first with Available [3, 3, 2]
    assert "P3" in res["safe_sequence"]
    assert len(res["unfinished_processes"]) == 0


# ============================================================================
# 3. RESOURCE REQUEST HANDLING (SCENARIOS B, C, AND SAFE GRANT)
# ============================================================================

def test_request_exceeds_available_scenario_b():
    """
    Scenario B: P4 requests (3, 3, 0).
    Within Need (4, 3, 1), but exceeds Available (2, 3, 0).
    Must result in WAITING / DEFERRED, state unchanged.
    """
    sc_b = get_scenario_b()
    res = handle_resource_request(
        sc_b["allocation"], sc_b["maximum"], sc_b["available"],
        sc_b["request_process"], sc_b["request_vector"],
        sc_b["process_labels"], sc_b["resource_labels"]
    )

    assert res["status"] == "WAITING / DEFERRED"
    assert res["granted"] is False
    assert res["state_changed"] is False
    # Allocation and Available must match original
    np.testing.assert_array_equal(res["allocation"], sc_b["allocation"])
    np.testing.assert_array_equal(res["available"], sc_b["available"])


def test_unsafe_request_denied_and_restored_scenario_c():
    """
    Scenario C: P0 requests (0, 2, 0).
    Within Need and Available, but simulated state has no safe sequence.
    Must be DENIED and original state RESTORED.
    """
    sc_c = get_scenario_c()
    res = handle_resource_request(
        sc_c["allocation"], sc_c["maximum"], sc_c["available"],
        sc_c["request_process"], sc_c["request_vector"],
        sc_c["process_labels"], sc_c["resource_labels"]
    )

    assert res["status"] == "UNSAFE"
    assert res["granted"] is False
    assert res["state_changed"] is False
    # Verify state was restored
    np.testing.assert_array_equal(res["allocation"], sc_c["allocation"])
    np.testing.assert_array_equal(res["available"], sc_c["available"])
    assert res["safety_result"] is not None
    assert res["safety_result"]["is_safe"] is False


def test_valid_safe_request_granted():
    """P1 requests (1, 0, 2) from base Scenario A -> Must be GRANTED."""
    sc_a = get_scenario_a()
    res = handle_resource_request(
        sc_a["allocation"], sc_a["maximum"], sc_a["available"],
        1, [1, 0, 2],
        sc_a["process_labels"], sc_a["resource_labels"]
    )

    assert res["status"] == "SAFE"
    assert res["granted"] is True
    assert res["state_changed"] is True
    # Available was [3, 3, 2] - [1, 0, 2] = [2, 3, 0]
    np.testing.assert_array_equal(res["available"], np.array([2, 3, 0]))
    # P1 allocation was [2, 0, 0] + [1, 0, 2] = [3, 0, 2]
    np.testing.assert_array_equal(res["allocation"][1], np.array([3, 0, 2]))


def test_request_exceeds_need():
    """Request > Need must fail at step 2 with maximum claim exceeded error."""
    sc_a = get_scenario_a()
    # P1 Need is [1, 2, 2]. Request [2, 0, 0] exceeds A need (2 > 1).
    res = handle_resource_request(
        sc_a["allocation"], sc_a["maximum"], sc_a["available"],
        1, [2, 0, 0],
        sc_a["process_labels"], sc_a["resource_labels"]
    )

    assert res["status"] == "INVALID INPUT"
    assert res["step_reached"] == 2
    assert "exceeded its declared maximum claim" in res["reason"]
    assert res["granted"] is False


# ============================================================================
# 4. DEADLOCK DETECTION (SINGLE-INSTANCE & MULTI-INSTANCE)
# ============================================================================

def test_single_instance_detection_scenario_d():
    """Scenario D (Single-Instance): Cycle between P1 and P2 implies DEADLOCK."""
    sc_d = get_scenario_d_single()
    res = detect_single_instance_deadlock(
        sc_d["processes"], sc_d["resources"],
        sc_d["allocation_edges"], sc_d["request_edges"]
    )

    assert res["status"] == "DEADLOCK DETECTED"
    assert res["is_deadlocked"] is True
    assert set(res["deadlocked_processes"]) == {"P1", "P2"}
    assert len(res["cycles"]) >= 1


def test_multi_instance_detection_scenario_d():
    """Scenario D (Multiple-Instance): Matrix detection finds deadlocked processes."""
    sc_d = get_scenario_d_multi()
    res = detect_multi_instance_deadlock(
        sc_d["allocation"], sc_d["request"], sc_d["available"],
        sc_d["processes"], sc_d["resources"]
    )

    assert res["status"] == "DEADLOCK DETECTED"
    assert res["is_deadlocked"] is True
    # P0 can finish because Request is 0. Processes {P1, P2, P3, P4} remain deadlocked.
    assert "P0" not in res["deadlocked_processes"]
    assert "P1" in res["deadlocked_processes"]


# ============================================================================
# 5. RECOVERY BY PROCESS TERMINATION
# ============================================================================

def test_recovery_single_instance():
    """Terminating P1 in Scenario D breaks the cycle and resolves deadlock."""
    sc_d = get_scenario_d_single()
    rec = recover_by_terminating_process_single(
        sc_d["processes"], sc_d["resources"],
        sc_d["allocation_edges"], sc_d["request_edges"],
        victim_process="P1"
    )

    assert rec["is_resolved"] is True
    assert rec["status"] == "NO DEADLOCK (RESOLVED)"
    assert len(rec["new_cycles"]) == 0
    assert len(rec["new_deadlocked"]) == 0


def test_recovery_multi_instance():
    """Terminating a victim in multi-instance releases resources into Work."""
    alloc = np.array([
        [1, 0],
        [0, 1]
    ])
    req = np.array([
        [0, 1],
        [1, 0]
    ])
    avail = np.array([0, 0])
    
    rec = recover_by_terminating_process_multi(
        alloc, req, avail, ["P0", "P1"], ["R1", "R2"], victim_process="P0"
    )
    assert rec["victim_terminated"] == "P0"
    assert rec["released_vector"] == [1, 0]
    assert rec["work_after_release"] == [1, 0]
    # With [1, 0] available, P1 (needs [1, 0]) can now complete!
    assert rec["is_resolved"] is True


# ============================================================================
# 6. DEADLOCK PREVENTION
# ============================================================================

def test_resource_ordering_prevention():
    """Resource ordering prevents P2 from requesting R1 while holding R2."""
    order = {"R1": 1, "R2": 2, "R3": 3}
    requests = [
        {"process": "P1", "holding": ["R1"], "requesting": "R2"},  # 2 > 1 -> ALLOWED
        {"process": "P2", "holding": ["R2"], "requesting": "R1"},  # 1 < 2 -> REJECTED
    ]
    res = evaluate_resource_ordering_prevention(["R1", "R2", "R3"], order, requests)

    assert res["condition_eliminated"] == "Circular Wait"
    assert res["allowed_count"] == 1
    assert res["rejected_count"] == 1
    assert res["deadlock_prevented"] is True


def test_request_all_upfront_prevention():
    """Request-all-upfront eliminates Hold and Wait."""
    needed = {
        "P1": {"R1": 1, "R2": 1},
        "P2": {"R2": 1, "R3": 1}
    }
    avail = {"R1": 1, "R2": 1, "R3": 0}
    res = evaluate_request_all_upfront_prevention(["P1", "P2"], needed, avail)

    assert res["condition_eliminated"] == "Hold and Wait"
    assert "P1" in res["execution_order"]
    assert "P2" in res["waiting_processes"]


# ============================================================================
# 7. VALIDATION AND BOUNDARY CASES (SCENARIO F)
# ============================================================================

def test_zero_available_is_valid_boundary():
    """Available = [0, 0, 0] is a VALID boundary case, not an error."""
    alloc = np.array([[1, 0], [0, 1]])
    max_m = np.array([[1, 1], [1, 1]])
    avail = np.array([0, 0])

    is_valid, status, errors, warnings = validate_system_state(alloc, max_m, avail)
    assert is_valid is True
    assert status == "VALID"
    assert any("Boundary Condition Note: All Available resources are currently 0" in w for w in warnings)


def test_zero_need_is_valid_boundary():
    """Process with Need = [0, 0] is a VALID boundary case."""
    alloc = np.array([[2, 2]])
    max_m = np.array([[2, 2]])
    avail = np.array([1, 1])

    is_valid, status, errors, warnings = validate_system_state(alloc, max_m, avail)
    assert is_valid is True
    assert any("remaining Need = 0" in w for w in warnings)


def test_allocation_exceeds_maximum_is_invalid():
    """Allocation > Maximum must return INVALID INPUT with exact cell."""
    alloc = np.array([[0, 5, 0]])
    max_m = np.array([[7, 2, 3]])
    avail = np.array([3, 3, 2])

    is_valid, status, errors, warnings = validate_system_state(alloc, max_m, avail)
    assert is_valid is False
    assert status == "INVALID INPUT"
    assert len(errors) == 1
    assert "Allocation > Maximum violation at cell [P0, R1]" in errors[0]


def test_negative_values_invalid():
    """Negative values in matrix or request must fail validation."""
    is_valid, err = validate_matrix_shape_and_types([[-1, 2], [0, 1]], 2, 2)
    assert is_valid is False
    assert "Negative values are invalid" in err

    is_valid_req, _, _, req_err = validate_request_input(0, [-1, 0], 2, 2)
    assert is_valid_req is False
    assert "Negative value" in req_err


def test_invalid_process_id():
    """Process ID outside bounds must return clear validation error."""
    is_valid, _, _, err = validate_request_input(99, [1, 0], 2, 2)
    assert is_valid is False
    assert "Invalid process identifier '99'" in err


def test_wrong_request_vector_length():
    """Request vector length not matching number of resources must fail."""
    is_valid, _, _, err = validate_request_input(0, [1, 2, 3], 2, 2)
    assert is_valid is False
    assert "Expected length 2, got 3" in err
