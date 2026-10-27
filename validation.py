"""
validation.py - Input validation for the Deadlock Analysis and Resource Allocation Dashboard.

This module validates configurations, matrices, vectors, process IDs, and requests.
All functions:
1. Never raise exceptions (gracefully handle any errors or invalid types).
2. Return a 3-tuple: (is_valid, message, kind) where kind is one of:
   - "VALID"
   - "VALID_BOUNDARY"
   - "INVALID"

Identifies:
- Negative numbers, non-integers, wrong matrix/vector shapes
- Allocation > Maximum violations with exact process and resource labels
- Invalid process IDs (out of bounds, non-integer)
- Valid boundary cases (Available all zeros, processes with zero Need)
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


def validate_matrix(
    matrix: Any,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    name: str = "Matrix"
) -> Tuple[bool, str, str]:
    """
    Validates a 2D matrix for shape, integer values, and non-negativity.
    Never raises exceptions.

    Returns:
        (is_valid, message, kind) where kind is "VALID" or "INVALID".
    """
    try:
        if matrix is None:
            return False, f"{name}: Input cannot be None.", "INVALID"

        arr = np.array(matrix)
        if arr.ndim != 2:
            return False, f"{name}: Expected a 2D matrix, but got {arr.ndim}D array.", "INVALID"

        if not np.issubdtype(arr.dtype, np.number):
            return False, f"{name}: Contains non-numeric elements.", "INVALID"

        if expected_rows is not None and arr.shape[0] != expected_rows:
            return False, f"{name}: Expected {expected_rows} rows, but got {arr.shape[0]}.", "INVALID"

        if expected_cols is not None and arr.shape[1] != expected_cols:
            return False, f"{name}: Expected {expected_cols} columns, but got {arr.shape[1]}.", "INVALID"

        # Check for NaN / Inf
        if not np.isfinite(arr).all():
            return False, f"{name}: Contains NaN, null, or infinite values.", "INVALID"

        # Check for non-integers
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            return False, f"{name}: Contains non-integer values. All resource counts must be integers.", "INVALID"

        # Check for negative numbers
        if (arr < 0).any():
            neg_indices = np.argwhere(arr < 0)
            r, c = neg_indices[0]
            return False, f"{name}: Negative values are invalid. Found {arr[r, c]} at row {r}, column {c}.", "INVALID"

        return True, f"{name} is valid.", "VALID"
    except Exception as e:
        return False, f"{name}: Validation failed with error ({e}).", "INVALID"


def validate_vector(
    vector: Any,
    expected_len: Optional[int] = None,
    name: str = "Vector"
) -> Tuple[bool, str, str]:
    """
    Validates a 1D vector for expected length, integers, and non-negativity.
    Identifies all-zero vector as a valid boundary case.
    Never raises exceptions.

    Returns:
        (is_valid, message, kind) where kind is "VALID", "VALID_BOUNDARY", or "INVALID".
    """
    try:
        if vector is None:
            return False, f"{name}: Input cannot be None.", "INVALID"

        arr = np.array(vector).flatten()
        if expected_len is not None and len(arr) != expected_len:
            return False, f"{name}: Expected length {expected_len}, got {len(arr)}.", "INVALID"

        if not np.issubdtype(arr.dtype, np.number):
            return False, f"{name}: Contains non-numeric elements.", "INVALID"

        if not np.isfinite(arr).all():
            return False, f"{name}: Contains NaN, null, or infinite values.", "INVALID"

        if not np.all(np.equal(np.mod(arr, 1), 0)):
            return False, f"{name}: Contains non-integer values. All values must be whole integers.", "INVALID"

        if (arr < 0).any():
            neg_idx = int(np.where(arr < 0)[0][0])
            return False, f"{name}: Negative value {arr[neg_idx]} found at index {neg_idx}.", "INVALID"

        # Valid boundary check: all zeros
        if len(arr) > 0 and np.all(arr == 0):
            return True, f"{name} is all zeros (valid boundary case).", "VALID_BOUNDARY"

        return True, f"{name} is valid.", "VALID"
    except Exception as e:
        return False, f"{name}: Validation failed with error ({e}).", "INVALID"


def validate_process_id(process_id: Any, num_processes: int) -> Tuple[bool, str, str]:
    """
    Validates whether a process identifier is a valid integer within [0, num_processes - 1].
    Accepts integer, string integer (e.g. "2"), or label (e.g. "P2").
    Never raises exceptions.

    Returns:
        (is_valid, message, kind) where kind is "VALID" or "INVALID".
    """
    try:
        if process_id is None:
            return False, "Process ID cannot be None.", "INVALID"

        parsed_id = None
        if isinstance(process_id, str):
            cleaned = process_id.strip().upper()
            if cleaned.startswith("P") and cleaned[1:].lstrip("-").isdigit():
                parsed_id = int(cleaned[1:])
            elif cleaned.lstrip("-").isdigit():
                parsed_id = int(cleaned)
        elif isinstance(process_id, (int, np.integer)):
            parsed_id = int(process_id)

        if parsed_id is None:
            return False, f"Invalid process ID '{process_id}'. Must be an integer or label (e.g. P0).", "INVALID"

        if parsed_id < 0 or parsed_id >= num_processes:
            return False, (
                f"Invalid process ID {parsed_id}: Out of range. "
                f"Must be between 0 and {num_processes - 1}."
            ), "INVALID"

        return True, f"Process ID P{parsed_id} is valid.", "VALID"
    except Exception as e:
        return False, f"Process ID validation error ({e}).", "INVALID"


def validate_configuration(
    allocation: Any,
    maximum: Any,
    available: Any,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Tuple[bool, str, str]:
    """
    Validates system matrices (Allocation, Maximum, Available).
    Never raises exceptions.

    Checks:
    - Matrix and vector shapes and types.
    - Allocation <= Maximum for every cell; returns exact process and resource if violated.
    - Boundary conditions:
      - Available all zeros -> "VALID_BOUNDARY"
      - Any process with Need == 0 -> "VALID_BOUNDARY"
      - Normal valid -> "VALID"

    Returns:
        (is_valid, message, kind) where kind is "VALID", "VALID_BOUNDARY", or "INVALID".
    """
    try:
        if allocation is None or maximum is None or available is None:
            return False, "Configuration matrices/vectors cannot be None.", "INVALID"

    # Validate shapes and dimensions
    if alloc_arr.ndim != 2:
        return False, f"Allocation must be a 2D matrix, got {alloc_arr.ndim}D.", "INVALID"
    if max_arr.ndim != 2:
        return False, f"Maximum must be a 2D matrix, got {max_arr.ndim}D.", "INVALID"

    if alloc_arr.shape != max_arr.shape:
        return False, (
            f"Dimension mismatch between Allocation {alloc_arr.shape} "
            f"and Maximum {max_arr.shape}."
        ), "INVALID"

    n_proc, n_res = alloc_arr.shape

    # Validate allocation matrix contents
    is_alloc_valid, alloc_msg, _ = validate_matrix(alloc_arr, n_proc, n_res, name="Allocation matrix")
    if not is_alloc_valid:
        return False, alloc_msg, "INVALID"

    # Validate maximum matrix contents
    is_max_valid, max_msg, _ = validate_matrix(max_arr, n_proc, n_res, name="Maximum matrix")
    if not is_max_valid:
        return False, max_msg, "INVALID"

    # Validate available vector
    is_avail_valid, avail_msg, avail_kind = validate_vector(avail_arr, n_res, name="Available vector")
    if not is_avail_valid:
        return False, avail_msg, "INVALID"

    # Labels for clear cell reporting
    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    # Check: Allocation <= Maximum for every cell
    violations = alloc_arr > max_arr
    if np.any(violations):
        coords = np.argwhere(violations)[0]
        r, c = int(coords[0]), int(coords[1])
        return False, (
            f"Allocation exceeds Maximum at process {p_labels[r]}, resource {r_labels[c]}: "
            f"Allocation={alloc_arr[r, c]} > Maximum={max_arr[r, c]}."
        ), "INVALID"

    # Check Boundary Cases
    is_avail_zero = np.all(avail_arr == 0)
    need_arr = max_arr - alloc_arr
    zero_need_procs = [p_labels[i] for i in range(n_proc) if np.all(need_arr[i] == 0)]

    if is_avail_zero and zero_need_procs:
        return True, (
            f"Valid boundary state: Available is all zeros, and process(es) "
            f"{', '.join(zero_need_procs)} have remaining Need = 0."
        ), "VALID_BOUNDARY"

    if is_avail_zero:
        return True, (
            "Valid boundary state: Available vector is all zeros "
            "(all resources are temporarily allocated to active processes)."
        ), "VALID_BOUNDARY"

    if zero_need_procs:
        return True, (
            f"Valid boundary state: Process(es) {', '.join(zero_need_procs)} have remaining Need = 0 "
            "(process already holds its declared maximum demand and can finish)."
        ), "VALID_BOUNDARY"

    return True, "Configuration is valid.", "VALID"


def validate_request(
    process_id: Any,
    request: Any,
    num_processes: int,
    num_resources: int
) -> Tuple[bool, str, str]:
    """
    Validates a resource request.

    Checks:
    - Valid process ID
    - Valid request vector (length, non-negative integers)
    - Distinguishes boundary all-zero request as "VALID_BOUNDARY"

    Returns:
        (is_valid, message, kind) where kind is "VALID", "VALID_BOUNDARY", or "INVALID".
    """
    # 1. Validate process ID
    is_pid_valid, pid_msg, _ = validate_process_id(process_id, num_processes)
    if not is_pid_valid:
        return False, pid_msg, "INVALID"

    # 2. Validate request vector
    is_vec_valid, vec_msg, vec_kind = validate_vector(request, num_resources, name="Request vector")
    if not is_vec_valid:
        return False, vec_msg, "INVALID"

    if vec_kind == "VALID_BOUNDARY":
        return True, "Request vector is all zeros (valid boundary request).", "VALID_BOUNDARY"

    return True, "Request is valid.", "VALID"


# ============================================================================
# COMPATIBILITY ALIASES FOR EXISTING CALLS IN APP.PY AND SUITES
# ============================================================================

def validate_matrix_shape_and_types(
    matrix: Any,
    expected_rows: int,
    expected_cols: int,
    name: str = "Matrix"
) -> Tuple[bool, Optional[str]]:
    """Backward-compatible wrapper returning (is_valid, error_msg)."""
    is_valid, msg, _ = validate_matrix(matrix, expected_rows, expected_cols, name)
    return is_valid, None if is_valid else msg


def validate_system_state(
    allocation: np.ndarray,
    maximum: np.ndarray,
    available: np.ndarray,
    total: Optional[np.ndarray] = None,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Tuple[bool, str, List[str], List[str]]:
    """Backward-compatible wrapper for app.py."""
    is_valid, msg, kind = validate_configuration(allocation, maximum, available, process_labels, resource_labels)
    errors = [] if is_valid else [msg]
    warnings = [msg] if kind == "VALID_BOUNDARY" else []
    status_str = "VALID" if is_valid else "INVALID INPUT"
    return is_valid, status_str, errors, warnings


def validate_request_input(
    process_idx: Any,
    request_vector: Any,
    num_processes: int,
    num_resources: int,
    process_labels: Optional[List[str]] = None
) -> Tuple[bool, Optional[int], Optional[np.ndarray], Optional[str]]:
    """Backward-compatible wrapper for request validation."""
    is_valid, msg, _ = validate_request(process_idx, request_vector, num_processes, num_resources)
    if not is_valid:
        return False, None, None, msg

    # Parse int index
    parsed_id = process_idx
    if isinstance(process_idx, str):
        cleaned = process_idx.strip().upper()
        if cleaned.startswith("P") and cleaned[1:].isdigit():
            parsed_id = int(cleaned[1:])
        elif cleaned.isdigit():
            parsed_id = int(cleaned)
    p_id = int(parsed_id)
    req_arr = np.array(request_vector, dtype=int).flatten()
    return True, p_id, req_arr, None
