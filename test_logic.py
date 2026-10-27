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
    detect_single_instance,
    detect_multi_instance,
    terminate_process,
    detect_single_instance_deadlock,
    detect_multi_instance_deadlock,
    recover_by_terminating_process_single,
    recover_by_terminating_process_multi
)
from prevention import (
    evaluate_resource_ordering_prevention,
    evaluate_request_all_upfront_prevention,
    get_coffman_conditions_summary,
    get_coffman_breakdown,
    simulate_prevention_single_instance,
    simulate_prevention_multi_instance
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
    get_scenario_f,
    run_scenario_a,
    run_scenario_b,
    run_scenario_c,
    run_scenario_d,
    run_scenario_e,
    run_scenario_f,
    get_all_scenarios_results_table
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

def test_validation_negative_numbers():
    """Negative numbers in matrices or vectors must return INVALID."""
    # Matrix with negative value
    is_v, msg, kind = validate_matrix([[1, -2], [0, 3]])
    assert is_v is False
    assert kind == "INVALID"
    assert "Negative values are invalid" in msg

    # Vector with negative value
    is_v, msg, kind = validate_vector([1, -1, 2])
    assert is_v is False
    assert kind == "INVALID"
    assert "Negative value -1 found" in msg


def test_validation_non_integers():
    """Non-integer decimal values in matrices or vectors must return INVALID."""
    is_v, msg, kind = validate_matrix([[1.5, 2], [0, 1]])
    assert is_v is False
    assert kind == "INVALID"
    assert "non-integer" in msg.lower()

    is_v, msg, kind = validate_vector([1, 2.7, 3])
    assert is_v is False
    assert kind == "INVALID"
    assert "non-integer" in msg.lower()


def test_validation_wrong_shapes():
    """Wrong dimensions or mismatched row/column counts must return INVALID."""
    # 1D array passed as matrix
    is_v, msg, kind = validate_matrix([1, 2, 3])
    assert is_v is False
    assert kind == "INVALID"
    assert "Expected a 2D matrix" in msg

    # Row count mismatch
    is_v, msg, kind = validate_matrix([[1, 2], [3, 4]], expected_rows=3, expected_cols=2)
    assert is_v is False
    assert kind == "INVALID"
    assert "Expected 3 rows" in msg

    # Column count mismatch
    is_v, msg, kind = validate_matrix([[1, 2], [3, 4]], expected_rows=2, expected_cols=3)
    assert is_v is False
    assert kind == "INVALID"
    assert "Expected 3 columns" in msg

    # Vector length mismatch
    is_v, msg, kind = validate_vector([1, 2], expected_len=3)
    assert is_v is False
    assert kind == "INVALID"
    assert "Expected length 3" in msg


def test_validation_allocation_exceeds_maximum_returns_exact_process_and_resource():
    """
    Allocation > Maximum for any cell must return INVALID and specify
    the exact process and resource labels.
    """
    alloc = np.array([[0, 5, 0]])  # Process P0, Resource B is 5
    max_m = np.array([[7, 2, 3]])  # Maximum for Resource B is 2
    avail = np.array([3, 3, 2])
    p_labels = ["P0"]
    r_labels = ["A", "B", "C"]

    is_v, msg, kind = validate_configuration(alloc, max_m, avail, p_labels, r_labels)

    assert is_v is False
    assert kind == "INVALID"
    # Exact process and resource must be identified
    assert "process P0" in msg
    assert "resource B" in msg
    assert "Allocation=5 > Maximum=2" in msg


def test_validation_invalid_process_id():
    """Invalid process IDs (out of bounds, negative, non-numeric) return INVALID."""
    # Out of upper bound
    is_v, msg, kind = validate_process_id(99, 5)
    assert is_v is False
    assert kind == "INVALID"
    assert "Out of range" in msg

    # Negative process ID
    is_v, msg, kind = validate_process_id(-1, 5)
    assert is_v is False
    assert kind == "INVALID"
    assert "Out of range" in msg

    # Non-numeric string
    is_v, msg, kind = validate_process_id("invalid_proc", 5)
    assert is_v is False
    assert kind == "INVALID"

    # Valid process ID
    is_v, msg, kind = validate_process_id(2, 5)
    assert is_v is True
    assert kind == "VALID"

    is_v, msg, kind = validate_process_id("P2", 5)
    assert is_v is True
    assert kind == "VALID"


def test_validation_valid_boundary_available_all_zeros():
    """
    Available = [0, 0, 0] is a VALID boundary case (system fully allocated).
    Must return is_valid=True and kind='VALID_BOUNDARY'.
    """
    alloc = np.array([[1, 2], [2, 1]])
    max_m = np.array([[2, 2], [2, 2]])
    avail = np.array([0, 0])

    is_v, msg, kind = validate_configuration(alloc, max_m, avail)
    assert is_v is True
    assert kind == "VALID_BOUNDARY"
    assert "Available vector is all zeros" in msg


def test_validation_valid_boundary_process_with_zero_need():
    """
    A process with Allocation == Maximum has Need == 0.
    Must return is_valid=True and kind='VALID_BOUNDARY'.
    """
    alloc = np.array([[2, 2], [0, 1]])
    max_m = np.array([[2, 2], [1, 2]])  # P0 has Need = [0, 0]
    avail = np.array([1, 1])

    is_v, msg, kind = validate_configuration(alloc, max_m, avail, process_labels=["P0", "P1"])
    assert is_v is True
    assert kind == "VALID_BOUNDARY"
    assert "remaining Need = 0" in msg
    assert "P0" in msg


def test_validation_valid_boundary_all_zero_request():
    """Request vector with all zeros is a valid boundary request."""
    is_v, msg, kind = validate_request(0, [0, 0, 0], num_processes=5, num_resources=3)
    assert is_v is True
    assert kind == "VALID_BOUNDARY"


def test_validation_never_raises_exceptions():
    """Validation functions must never crash or raise unhandled exceptions."""
    # Test with None, strings, empty lists, malformed inputs
    is_v, msg, kind = validate_matrix(None)
    assert is_v is False
    assert kind == "INVALID"

    is_v, msg, kind = validate_vector(None)
    assert is_v is False
    assert kind == "INVALID"

    is_v, msg, kind = validate_configuration("not a matrix", [], None)
    assert is_v is False
    assert kind == "INVALID"

    is_v, msg, kind = validate_request(None, None, 5, 3)
    assert is_v is False
    assert kind == "INVALID"


# ============================================================================
# DETECTION & RECOVERY TESTS
# ============================================================================

def test_detect_single_instance_circular_wait_2_processes():
    """
    Test 2-process circular wait:
    P1 holds R1, wants R2.
    P2 holds R2, wants R1.
    Wait-For Graph has cycle P1 -> P2 -> P1.
    Must detect deadlock, identify the cycle, and list deadlocked processes.
    """
    allocation_edges = [("R1", "P1"), ("R2", "P2")]
    request_edges = [("P1", "R2"), ("P2", "R1")]

    result = detect_single_instance(allocation_edges, request_edges)

    assert result["deadlock"] is True
    assert set(result["deadlocked_processes"]) == {"P1", "P2"}
    assert len(result["cycle"]) == 2
    assert set(result["cycle"]) == {"P1", "P2"}


def test_detect_single_instance_no_deadlock():
    """
    Test single-instance no-deadlock case:
    P1 holds R1, wants R2.
    P2 holds R2, wants R3.
    P3 holds R3, wants nothing (can finish and release R3).
    Wait-For Graph: P1 -> P2 -> P3 (DAG, acyclic).
    Must return deadlock=False, cycle=[], deadlocked_processes=[].
    """
    allocation_edges = [("R1", "P1"), ("R2", "P2"), ("R3", "P3")]
    request_edges = [("P1", "R2"), ("P2", "R3")]

    result = detect_single_instance(allocation_edges, request_edges)

    assert result["deadlock"] is False
    assert result["cycle"] == []
    assert result["deadlocked_processes"] == []


def test_detect_multi_instance_deadlock():
    """
    Test multi-instance deadlock detection:
    3 processes (P0, P1, P2) and 2 resource types (R0, R1).
    Available = [0, 0]
    Allocation:
      P0: [1, 0]
      P1: [0, 1]
      P2: [0, 0] (holding 0, Finish[2] is True initially)
    Request:
      P0: [0, 1] (wants R1, held by P1)
      P1: [1, 0] (wants R0, held by P0)
      P2: [0, 0]
    Both P0 and P1 are deadlocked because Available is [0, 0] and neither can be satisfied.
    """
    allocation = np.array([
        [1, 0],  # P0
        [0, 1],  # P1
        [0, 0]   # P2
    ])
    request = np.array([
        [0, 1],  # P0
        [1, 0],  # P1
        [0, 0]   # P2
    ])
    available = np.array([0, 0])

    result = detect_multi_instance(allocation, request, available)

    assert result["deadlock"] is True
    assert set(result["deadlocked_processes"]) == {0, 1}
    assert result["finish"][2] is True
    assert result["finish"][0] is False
    assert result["finish"][1] is False
    assert "work_updates" in result
    assert "steps" in result


def test_recovery_by_terminating_process_makes_others_finish():
    """
    Test recovery by process termination:
    In the multi-instance deadlock above, terminating P0 releases its resources [1, 0].
    Available increases from [0, 0] to [1, 0].
    Then P1 (requesting [1, 0]) can satisfy its request, execute, and finish.
    All processes finish and deadlock is resolved!
    """
    allocation = np.array([
        [1, 0],  # P0
        [0, 1],  # P1
        [0, 0]   # P2
    ])
    request = np.array([
        [0, 1],  # P0
        [1, 0],  # P1
        [0, 0]   # P2
    ])
    available = np.array([0, 0])

    # Initial detection confirms deadlock
    init_res = detect_multi_instance(allocation, request, available)
    assert init_res["deadlock"] is True
    assert set(init_res["deadlocked_processes"]) == {0, 1}

    # Terminate process 0 to recover
    term_res = terminate_process(0, allocation, request, available)

    assert term_res["released"] == [1, 0]
    assert np.array_equal(term_res["available"], [1, 0])
    assert np.array_equal(term_res["allocation"][0], [0, 0])
    assert np.array_equal(term_res["request"][0], [0, 0])

    # Re-run multi-instance detection on the post-termination state
    post_res = detect_multi_instance(
        term_res["allocation"],
        term_res["request"],
        term_res["available"]
    )

    # Deadlock must be resolved!
    assert post_res["deadlock"] is False
    assert post_res["deadlocked_processes"] == []
    # P1 completed in step 1 and released [0, 1], final work is [1, 1]
    assert post_res["work"] == [1, 1]
    assert len(post_res["steps"]) == 1
    assert post_res["steps"][0]["process"] == 1


def test_simulate_prevention_resource_ordering_breaks_circular_wait():
    """Applying Resource Ordering to a 2-process circular wait rejects backward request and eliminates Circular Wait."""
    processes = ["P1", "P2"]
    resources = ["R1", "R2"]
    alloc_edges = [("R1", "P1"), ("R2", "P2")]
    req_edges = [("P1", "R2"), ("P2", "R1")]
    order = {"R1": 1, "R2": 2}

    res = simulate_prevention_single_instance(
        processes, resources, alloc_edges, req_edges,
        policy="resource_ordering",
        resource_order=order
    )

    assert res["policy_name"] == "Policy A: Resource Ordering"
    assert res["condition_eliminated"] == "Circular Wait"
    assert res["is_original_deadlocked"] is True
    assert res["deadlock_prevented"] is True
    assert len(res["modified_deadlocked"]) == 0
    assert len(res["rejected_edges"]) == 1
    assert res["rejected_edges"][0] == ("P2", "R1")
    assert "tradeoffs" in res
    assert "resource_utilization" in res["tradeoffs"]
    assert "reduced_concurrency" in res["tradeoffs"]
    assert "starvation_risk" in res["tradeoffs"]
    assert "longer_waiting" in res["tradeoffs"]


def test_simulate_prevention_request_all_upfront_eliminates_hold_and_wait():
    """Applying Request-All-Upfront allocates 0 to waiting processes, eliminating Hold and Wait."""
    processes = ["P1", "P2"]
    resources = ["R1", "R2"]
    alloc_edges = [("R1", "P1"), ("R2", "P2")]
    req_edges = [("P1", "R2"), ("P2", "R1")]

    res = simulate_prevention_single_instance(
        processes, resources, alloc_edges, req_edges,
        policy="request_all_upfront"
    )

    assert res["policy_name"] == "Policy B: Request All Resources Upfront"
    assert res["condition_eliminated"] == "Hold and Wait"
    assert res["deadlock_prevented"] is True
    assert len(res["modified_deadlocked"]) == 0
    assert "P1" in res["execution_order"]
    assert "P2" in res["waiting_processes"]
    assert "tradeoffs" in res
    assert "resource_utilization" in res["tradeoffs"]


def test_simulate_prevention_multi_instance_both_policies():
    """Test prevention policies on multi-instance deadlocked matrices."""
    alloc = np.array([[1, 0], [0, 1], [0, 0]])
    req = np.array([[0, 1], [1, 0], [0, 0]])
    avail = np.array([0, 0])
    procs = ["P0", "P1", "P2"]
    res_names = ["R0", "R1"]

    # Resource ordering
    ro_res = simulate_prevention_multi_instance(
        procs, res_names, alloc, req, avail, policy="resource_ordering"
    )
    assert ro_res["condition_eliminated"] == "Circular Wait"
    assert ro_res["deadlock_prevented"] is True
    assert "tradeoffs" in ro_res

    # Request all upfront
    up_res = simulate_prevention_multi_instance(
        procs, res_names, alloc, req, avail, policy="request_all_upfront"
    )
    assert up_res["condition_eliminated"] == "Hold and Wait"
    assert up_res["deadlock_prevented"] is True
    assert "tradeoffs" in up_res


def test_get_coffman_breakdown():
    """Verify Coffman condition breakdown accurately flags eliminated conditions."""
    ro_breakdown = get_coffman_breakdown("resource_ordering")
    assert len(ro_breakdown) == 4
    # Condition 4 (Circular Wait) must be eliminated
    assert ro_breakdown[3]["is_eliminated"] is True
    assert ro_breakdown[1]["is_eliminated"] is False

    up_breakdown = get_coffman_breakdown("request_all_upfront")
    assert len(up_breakdown) == 4
    # Condition 2 (Hold and Wait) must be eliminated
    assert up_breakdown[1]["is_eliminated"] is True


def test_all_scenarios_runners_and_results_table():
    """Verify that all scenario runners A through F execute dynamically and pass."""
    # Scenario A
    res_a = run_scenario_a()
    assert res_a["match"] == "PASS"
    assert "SAFE" in res_a["actual_result"]

    # Scenario B
    res_b = run_scenario_b()
    assert res_b["match"] == "PASS"
    assert "WAITING" in res_b["actual_result"]

    # Scenario C
    res_c = run_scenario_c()
    assert res_c["match"] == "PASS"
    assert "DENIED_UNSAFE" in res_c["actual_result"] or "UNSAFE" in res_c["actual_result"]

    # Scenario D
    res_d = run_scenario_d()
    assert res_d["match"] == "PASS"
    assert "DEADLOCK DETECTED" in res_d["actual_result"]

    # Scenario E
    res_e = run_scenario_e()
    assert res_e["match"] == "PASS"
    assert "DEADLOCK PREVENTED" in res_e["actual_result"]

    # Scenario F
    res_f = run_scenario_f()
    assert res_f["match"] == "PASS"
    assert "VALID_BOUNDARY accepted" in res_f["actual_result"]

    # Check consolidated analytical table
    table_rows = get_all_scenarios_results_table()
    assert len(table_rows) == 6
    required_cols = {"Test Case", "System Condition", "Actual Result", "Expected Result", "Match (PASS/FAIL)"}
    for row in table_rows:
        assert required_cols.issubset(set(row.keys()))
        assert row["Match (PASS/FAIL)"] == "PASS"


# ============================================================================
# 6. SCENARIO A-F AUTOMATED VERIFICATION & STATE INTEGRITY TESTS
# ============================================================================

def test_scenario_a_produces_valid_safe_sequence():
    """
    Requirement 1: Scenario A must load a configuration that produces a valid safe sequence.
    5 processes, 3 resources, Available=[3, 3, 2].
    Need matrix is correctly computed and safety algorithm finds a valid safe sequence.
    """
    sc_a = get_scenario_a()
    assert sc_a["num_processes"] == 5
    assert sc_a["num_resources"] == 3
    assert np.array_equal(sc_a["available"], [3, 3, 2])

    need = calculate_need(sc_a["maximum"], sc_a["allocation"])
    res = safety_check(sc_a["allocation"], need, sc_a["available"])

    assert res["is_safe"] is True
    assert len(res["safe_sequence"]) == 5
    assert set(res["safe_sequence"]) == {0, 1, 2, 3, 4}
    assert res["unfinished"] == []
    # Verify Work increases monotonically as processes release allocations
    for s in res["steps"]:
        assert all(after >= before for before, after in zip(s["work_before"], s["work_after"]))


def test_scenario_b_request_within_need_exceeds_available_waiting_state_unchanged():
    """
    Requirement 2: Scenario B must submit a request within the process's Need but exceeding Available.
    It must show WAITING/DEFERRED and leave the allocation unchanged.
    """
    sc_b = get_scenario_b()
    p_id = sc_b["request_process"]
    req = sc_b["request_vector"]
    alloc_orig = sc_b["allocation"].copy()
    avail_orig = sc_b["available"].copy()
    max_orig = sc_b["maximum"].copy()

    need = calculate_need(max_orig, alloc_orig)

    # 1. Verify Request <= Need holds
    assert np.all(req <= need[p_id]), f"Request {req} must be <= Need {need[p_id]}"

    # 2. Verify Request <= Available fails (resource A deficit: 3 > 2)
    assert not np.all(req <= avail_orig), f"Request {req} must exceed Available {avail_orig}"
    assert req[0] > avail_orig[0]

    # 3. Execute handle_request and verify status is WAITING
    outcome = handle_request(p_id, req, alloc_orig, max_orig, avail_orig)
    assert outcome["status"] == "WAITING"
    assert "must wait" in outcome["message"]

    # 4. CRITICAL: Allocation and Available must remain completely unchanged
    assert np.array_equal(outcome["new_allocation"], alloc_orig)
    assert np.array_equal(outcome["new_available"], avail_orig)
    assert outcome["safety_result"] is None


def test_scenario_c_unsafe_request_denied_and_state_restored():
    """
    Requirement 3: Scenario C must submit a request within both Need and Available that would make the state unsafe.
    It must show DENIED/UNSAFE and restore the original state.
    """
    sc_c = get_scenario_c()
    p_id = sc_c["request_process"]
    req = sc_c["request_vector"]
    alloc_orig = sc_c["allocation"].copy()
    avail_orig = sc_c["available"].copy()
    max_orig = sc_c["maximum"].copy()

    need = calculate_need(max_orig, alloc_orig)

    # 1. Verify Request <= Need holds
    assert np.all(req <= need[p_id]), f"Request {req} must be <= Need {need[p_id]}"

    # 2. Verify Request <= Available holds
    assert np.all(req <= avail_orig), f"Request {req} must be <= Available {avail_orig}"

    # 3. Execute handle_request and verify status is DENIED_UNSAFE
    outcome = handle_request(p_id, req, alloc_orig, max_orig, avail_orig)
    assert outcome["status"] == "DENIED_UNSAFE"
    assert "DENIED" in outcome["message"]
    assert "UNSAFE" in outcome["message"]
    assert outcome["safety_result"]["is_safe"] is False

    # 4. CRITICAL: Original state must be restored untouched
    assert np.array_equal(outcome["new_allocation"], alloc_orig)
    assert np.array_equal(outcome["new_available"], avail_orig)


def test_scenario_d_separate_deadlocked_scenario_detection_not_banker():
    """
    Requirement 4: Scenario D must load a separate deadlocked scenario and run the appropriate deadlock detection algorithm.
    Clearly state whether resources have single or multiple instances.
    Verify detection does not merely display the Banker's Algorithm result.
    """
    # 1. Single-Instance Model: Cycle in WFG is necessary and sufficient
    sc_s = get_scenario_d_single()
    assert sc_s["model"] == "single-instance"
    res_s = detect_single_instance(sc_s["allocation_edges"], sc_s["request_edges"])
    assert res_s["deadlock"] is True
    assert len(res_s["cycle"]) > 0
    # In single-instance resources, each resource has exactly 1 unit; cycle <=> deadlock
    assert set(res_s["cycle"]).issubset(set(sc_s["processes"]))

    # 2. Multiple-Instance Model: Cycle alone is not conclusive; matrix detection algorithm required
    sc_m = get_scenario_d_multi()
    assert sc_m["model"] == "multiple-instance"
    res_m = detect_multi_instance(sc_m["allocation"], sc_m["request"], sc_m["available"])
    assert res_m["deadlock"] is True
    assert len(res_m["deadlocked_processes"]) > 0
    # Process 0 finishes and releases, but processes 1-4 remain deadlocked
    assert 0 not in res_m["deadlocked_processes"]
    assert {1, 2, 3, 4}.issubset(set(res_m["deadlocked_processes"]))

    # 3. Separation check: Banker's algorithm is avoidance (evaluates max claim transitions),
    # whereas detection inspects existing actual requests without requiring Max claims
    assert "request" in sc_m
    assert "maximum" not in sc_m


def test_scenario_e_prevention_resource_ordering_prevents_circular_wait():
    """
    Requirement 5: Scenario E must demonstrate a prevention policy, such as fixed resource ordering,
    and show how it prevents circular wait.
    """
    sc_d = get_scenario_d_single()
    order = {"R1": 1, "R2": 2, "R3": 3, "R4": 4}

    res_ro = simulate_prevention_single_instance(
        sc_d["processes"],
        sc_d["resources"],
        sc_d["allocation_edges"],
        sc_d["request_edges"],
        policy="resource_ordering",
        resource_order=order
    )

    # 1. Verify Circular Wait Coffman condition is eliminated
    assert res_ro["condition_eliminated"] == "Circular Wait"
    assert res_ro["deadlock_prevented"] is True
    assert len(res_ro["modified_deadlocked"]) == 0

    # 2. Verify backward request violating hierarchy was rejected
    assert len(res_ro["rejected_edges"]) > 0
    # P2 holding R2 (rank 2) requesting R1 (rank 1) violates 1 > 2
    assert ("P2", "R1") in res_ro["rejected_edges"]


def test_scenario_f_boundary_cases_and_invalid_inputs_validation():
    """
    Requirement 6: Scenario F must test valid boundary cases and invalid inputs,
    displaying the correct results or validation messages.
    """
    sc_f = get_scenario_f()
    cases = sc_f["cases"]
    assert len(cases) == 6

    # F1: Zero Available resources -> VALID_BOUNDARY accepted
    f1 = cases[0]
    is_v, msg, kind = validate_configuration(f1["allocation"], f1["maximum"], f1["available"])
    assert is_v is True
    assert kind == "VALID_BOUNDARY"

    # F2: Zero remaining Need -> VALID_BOUNDARY accepted
    f2 = cases[1]
    is_v, msg, kind = validate_configuration(f2["allocation"], f2["maximum"], f2["available"])
    assert is_v is True
    assert kind == "VALID_BOUNDARY"

    # F3: Allocation > Maximum -> INVALID rejected
    f3 = cases[2]
    is_v, msg, kind = validate_configuration(f3["allocation"], f3["maximum"], f3["available"])
    assert is_v is False
    assert kind == "INVALID"
    assert "P0" in msg and "exceeds" in msg

    # F4: Negative Request Vector -> INVALID rejected
    f4 = cases[3]
    is_v, msg, kind = validate_request(0, f4["request"], 5, 3)
    assert is_v is False
    assert kind == "INVALID"
    assert "Negative" in msg

    # F5: Invalid Process ID P99 -> INVALID rejected
    f5 = cases[4]
    is_v, msg, kind = validate_process_id(f5["process_idx"], 5)
    assert is_v is False
    assert kind == "INVALID"
    assert "out of range" in msg or "Invalid process ID" in msg

    # F6: Wrong Vector Length -> INVALID rejected
    f6 = cases[5]
    is_v, msg, kind = validate_request(0, f6["request"], 5, 3)
    assert is_v is False
    assert kind == "INVALID"
    assert "Expected length" in msg or "length" in msg


def test_streamlit_scenario_switching_and_distinct_results():
    """
    Streamlit UI Test: Verifies that scenario switching loads the correct data,
    requests do not incorrectly modify state, and deadlock detection does not
    merely display the Banker's Algorithm result.
    Verifies that preset buttons A through F produce distinct, correct OS outcomes.
    """
    from streamlit.testing.v1 import AppTest

    # --- Test Scenario A ---
    at_a = AppTest.from_file("app.py").run()
    btn_a = [b for b in at_a.sidebar.button if b.label == "A: Safe"][0]
    btn_a.click().run()
    assert at_a.session_state.active_preset == "A"
    assert at_a.session_state.main_tab == "🛡️ Deadlock Avoidance (Banker's)"
    assert at_a.session_state.last_safety_check["is_safe"] is True
    assert len(at_a.session_state.last_safety_check["safe_sequence"]) == 5

    # --- Test Scenario B ---
    at_b = AppTest.from_file("app.py").run()
    btn_b = [b for b in at_b.sidebar.button if b.label == "B: Wait"][0]
    btn_b.click().run()
    assert at_b.session_state.active_preset == "B"
    assert at_b.session_state.main_tab == "🛡️ Deadlock Avoidance (Banker's)"
    assert at_b.session_state.last_safety_check is None
    assert at_b.session_state.last_request_outcome is not None
    assert at_b.session_state.last_request_outcome["outcome"]["status"] == "WAITING"
    # Allocation unchanged
    assert np.array_equal(
        at_b.session_state.last_request_outcome["before_allocation"],
        at_b.session_state.last_request_outcome["outcome"]["new_allocation"]
    )

    # --- Test Scenario C ---
    at_c = AppTest.from_file("app.py").run()
    btn_c = [b for b in at_c.sidebar.button if b.label == "C: Unsafe"][0]
    btn_c.click().run()
    assert at_c.session_state.active_preset == "C"
    assert at_c.session_state.main_tab == "🛡️ Deadlock Avoidance (Banker's)"
    assert at_c.session_state.last_safety_check is None
    assert at_c.session_state.last_request_outcome is not None
    assert at_c.session_state.last_request_outcome["outcome"]["status"] == "DENIED_UNSAFE"
    # State restored
    assert np.array_equal(
        at_c.session_state.last_request_outcome["before_allocation"],
        at_c.session_state.last_request_outcome["outcome"]["new_allocation"]
    )

    # --- Test Scenario D ---
    at_d = AppTest.from_file("app.py").run()
    btn_d = [b for b in at_d.sidebar.button if b.label == "D: Detect"][0]
    btn_d.click().run()
    assert at_d.session_state.active_preset == "D"
    assert at_d.session_state.main_tab == "🔍 Deadlock Detection"
    assert at_d.session_state.det_single_last_result["is_deadlocked"] is True
    assert at_d.session_state.det_multi_last_result["deadlock"] is True

    # --- Test Scenario E ---
    at_e = AppTest.from_file("app.py").run()
    btn_e = [b for b in at_e.sidebar.button if b.label == "E: Prevent"][0]
    btn_e.click().run()
    assert at_e.session_state.active_preset == "E"
    assert at_e.session_state.main_tab == "⚙️ Prevention & Recovery"
    assert at_e.session_state.active_preset_run[1]["match"] == "PASS"

    # --- Test Scenario F ---
    at_f = AppTest.from_file("app.py").run()
    btn_f = [b for b in at_f.sidebar.button if b.label == "F: Bound"][0]
    btn_f.click().run()
    assert at_f.session_state.active_preset == "F"
    assert at_f.session_state.main_tab == "📊 Test Scenarios & Results Table"
    assert at_f.session_state.active_preset_run[1]["match"] == "PASS"





