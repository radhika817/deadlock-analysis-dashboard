"""
test_logic.py - Unit tests for Deadlock Analysis and Resource Allocation algorithms.

Test Suite Structure:
1. Classic Banker's Example Tests (Required by Assignment):
   - Need matrix calculation correctness.
   - Initial state safety and verification of valid safe sequence.
   - P1 granted (1, 0, 2), then P4 requests (3, 3, 0) -> WAITING.
   - P1 granted (1, 0, 2), then P0 requests (0, 2, 0) -> DENIED_UNSAFE and state unchanged.
   - Request exceeding Need -> EXCEEDS_NEED.
   - Negative request and invalid process ID -> INVALID.
2. Deadlock Detection Tests:
   - Single-instance RAG/WFG cycle detection.
   - Multi-instance matrix detection method.
3. Deadlock Recovery Tests:
   - Recovery by victim termination in single-instance and multi-instance systems.
4. Deadlock Prevention Tests:
   - Resource Ordering (eliminates Circular Wait).
   - Request-All-Upfront (eliminates Hold and Wait).
5. Validation & Boundary Tests:
   - Zero Available (valid boundary).
   - Zero Need (valid boundary).
   - Allocation > Maximum (invalid input).
   - Negative values / wrong vector length (invalid input).
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
    validate_matrix,
    validate_vector,
    validate_process_id,
    validate_configuration,
    validate_request,
    validate_matrix_shape_and_types,
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
# FIXTURES: CLASSIC TEXTBOOK BANKER'S EXAMPLE
# ============================================================================

@pytest.fixture
def classic_example():
    """
    Classic textbook example:
    5 processes (P0 to P4), 3 resource types (A, B, C).
    Allocation:
      P0: [0, 1, 0]
      P1: [2, 0, 0]
      P2: [3, 0, 2]
      P3: [2, 1, 1]
      P4: [0, 0, 2]
    Max:
      P0: [7, 5, 3]
      P1: [3, 2, 2]
      P2: [9, 0, 2]
      P3: [2, 2, 2]
      P4: [4, 3, 3]
    Available: [3, 3, 2]
    """
    alloc = np.array([
        [0, 1, 0],  # P0
        [2, 0, 0],  # P1
        [3, 0, 2],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2],  # P4
    ], dtype=int)

    maximum = np.array([
        [7, 5, 3],  # P0
        [3, 2, 2],  # P1
        [9, 0, 2],  # P2
        [2, 2, 2],  # P3
        [4, 3, 3],  # P4
    ], dtype=int)

    available = np.array([3, 3, 2], dtype=int)

    return {"allocation": alloc, "maximum": maximum, "available": available}


# ============================================================================
# 1. CLASSIC BANKER'S ALGORITHM TESTS
# ============================================================================

def test_classic_need_matrix_is_correct(classic_example):
    """
    Test 1: Verify Need matrix calculation.
    Formula: Need = Maximum - Allocation.
    Expected:
      P0: [7-0, 5-1, 3-0] = [7, 4, 3]
      P1: [3-2, 2-0, 2-0] = [1, 2, 2]
      P2: [9-3, 0-0, 2-2] = [6, 0, 0]
      P3: [2-2, 2-1, 2-1] = [0, 1, 1]
      P4: [4-0, 3-0, 3-2] = [4, 3, 1]
    """
    alloc = classic_example["allocation"]
    maximum = classic_example["maximum"]

    need = calculate_need(maximum, alloc)

    expected_need = np.array([
        [7, 4, 3],
        [1, 2, 2],
        [6, 0, 0],
        [0, 1, 1],
        [4, 3, 1]
    ], dtype=int)

    np.testing.assert_array_equal(need, expected_need)


def test_classic_state_is_safe_and_valid_safe_sequence_found(classic_example):
    """
    Test 2: Verify the initial system state is safe and a valid safe sequence is found.
    Validates that each process in the sequence can finish with Work >= Need.
    """
    alloc = classic_example["allocation"]
    maximum = classic_example["maximum"]
    available = classic_example["available"]

    need = calculate_need(maximum, alloc)
    result = safety_check(alloc, need, available)

    # 1. State must be reported as safe
    assert result["is_safe"] is True

    # 2. Must contain all 5 processes with no unfinished processes
    seq = result["safe_sequence"]
    assert len(seq) == 5
    assert set(seq) == {0, 1, 2, 3, 4}
    assert len(result["unfinished"]) == 0

    # 3. Independent mathematical proof: simulate the sequence step by step
    work = available.copy()
    for p in seq:
        # Every process in sequence must satisfy Need[p] <= Work
        assert np.all(need[p] <= work), f"Process P{p} cannot run with Work={work}"
        work += alloc[p]


def test_classic_after_granting_p1_p4_requesting_3_3_0_returns_waiting(classic_example):
    """
    Test 3: After granting P1 request (1, 0, 2):
    P4 requesting (3, 3, 0) must return WAITING.
    - P1 grant: alloc becomes [3, 0, 2], avail becomes [2, 3, 0].
    - P4 Need is [4, 3, 1]. Request [3, 3, 0] <= Need [4, 3, 1] is True.
    - But Request [3, 3, 0] <= Available [2, 3, 0] is False (3 > 2 for resource A).
    - Status must be WAITING, and state must remain unchanged.
    """
    alloc = classic_example["allocation"].copy()
    maximum = classic_example["maximum"].copy()
    avail = classic_example["available"].copy()

    # Step A: Grant P1 request of (1, 0, 2)
    p1_grant = handle_request(1, [1, 0, 2], alloc, maximum, avail)
    assert p1_grant["status"] == "GRANTED"

    alloc_post_p1 = p1_grant["new_allocation"]
    avail_post_p1 = p1_grant["new_available"]

    # Verify state after P1 grant: Available is [2, 3, 0]
    np.testing.assert_array_equal(avail_post_p1, np.array([2, 3, 0]))

    # Step B: P4 requests (3, 3, 0)
    p4_result = handle_request(4, [3, 3, 0], alloc_post_p1, maximum, avail_post_p1)

    assert p4_result["status"] == "WAITING"
    assert "insufficient resources currently available" in p4_result["message"]
    # State must be preserved completely unchanged
    np.testing.assert_array_equal(p4_result["new_allocation"], alloc_post_p1)
    np.testing.assert_array_equal(p4_result["new_available"], avail_post_p1)


def test_classic_after_granting_p1_p0_requesting_0_2_0_returns_denied_unsafe_and_state_unchanged(classic_example):
    """
    Test 4: After granting P1 request (1, 0, 2):
    P0 requesting (0, 2, 0) must return DENIED_UNSAFE and the original state is unchanged.
    - Available is [2, 3, 0].
    - P0 requests [0, 2, 0]: within Need [7, 4, 3] and within Available [2, 3, 0].
    - Simulated Available becomes [2, 1, 0].
    - In this simulated state, no remaining process can finish (all Need > Work).
    - Request must be DENIED_UNSAFE, and original state restored.
    """
    alloc = classic_example["allocation"].copy()
    maximum = classic_example["maximum"].copy()
    avail = classic_example["available"].copy()

    # Step A: Grant P1 request of (1, 0, 2)
    p1_grant = handle_request(1, [1, 0, 2], alloc, maximum, avail)
    assert p1_grant["status"] == "GRANTED"

    alloc_post_p1 = p1_grant["new_allocation"]
    avail_post_p1 = p1_grant["new_available"]

    # Step B: P0 requests (0, 2, 0)
    p0_result = handle_request(0, [0, 2, 0], alloc_post_p1, maximum, avail_post_p1)

    assert p0_result["status"] == "DENIED_UNSAFE"
    assert p0_result["safety_result"] is not None
    assert p0_result["safety_result"]["is_safe"] is False

    # State must be preserved completely unchanged
    np.testing.assert_array_equal(p0_result["new_allocation"], alloc_post_p1)
    np.testing.assert_array_equal(p0_result["new_available"], avail_post_p1)


def test_classic_request_greater_than_need_returns_exceeds_need(classic_example):
    """
    Test 5: Request greater than Need returns EXCEEDS_NEED.
    P1 Need is [1, 2, 2].
    P1 requesting [2, 0, 0] exceeds its declared claim on resource A (2 > 1).
    """
    alloc = classic_example["allocation"]
    maximum = classic_example["maximum"]
    available = classic_example["available"]

    result = handle_request(1, [2, 0, 0], alloc, maximum, available)

    assert result["status"] == "EXCEEDS_NEED"
    assert "exceeded its declared maximum claim" in result["message"]
    # State must be untouched
    np.testing.assert_array_equal(result["new_allocation"], alloc)
    np.testing.assert_array_equal(result["new_available"], available)


def test_classic_negative_request_and_invalid_process_id_return_invalid(classic_example):
    """
    Test 6: Negative request and invalid process ID return INVALID.
    """
    alloc = classic_example["allocation"]
    maximum = classic_example["maximum"]
    available = classic_example["available"]

    # 1. Negative value in request vector
    res_negative = handle_request(0, [-1, 0, 0], alloc, maximum, available)
    assert res_negative["status"] == "INVALID"
    assert "Negative value" in res_negative["message"]

    # 2. Process ID out of upper bound (only P0-P4 exist)
    res_bad_id_high = handle_request(99, [1, 0, 0], alloc, maximum, available)
    assert res_bad_id_high["status"] == "INVALID"
    assert "Invalid process ID" in res_bad_id_high["message"]

    # 3. Negative process ID
    res_bad_id_neg = handle_request(-1, [1, 0, 0], alloc, maximum, available)
    assert res_bad_id_neg["status"] == "INVALID"
    assert "Invalid process ID" in res_bad_id_neg["message"]

    # 4. Wrong length request vector (e.g. 2 resources instead of 3)
    res_bad_len = handle_request(0, [1, 0], alloc, maximum, available)
    assert res_bad_len["status"] == "INVALID"
    assert "mismatch" in res_bad_len["message"]


# ============================================================================
# 2. DEADLOCK DETECTION TESTS (SINGLE & MULTI-INSTANCE)
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
    assert "P0" not in res["deadlocked_processes"]
    assert "P1" in res["deadlocked_processes"]


# ============================================================================
# 3. RECOVERY BY PROCESS TERMINATION TESTS
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
    alloc = np.array([[1, 0], [0, 1]])
    req = np.array([[0, 1], [1, 0]])
    avail = np.array([0, 0])

    rec = recover_by_terminating_process_multi(
        alloc, req, avail, ["P0", "P1"], ["R1", "R2"], victim_process="P0"
    )
    assert rec["victim_terminated"] == "P0"
    assert rec["released_vector"] == [1, 0]
    assert rec["work_after_release"] == [1, 0]
    assert rec["is_resolved"] is True


# ============================================================================
# 4. DEADLOCK PREVENTION TESTS
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
# 5. VALIDATION AND BOUNDARY CASES
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
