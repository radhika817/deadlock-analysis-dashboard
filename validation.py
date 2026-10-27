"""
validation.py - Input validation for the Deadlock Analysis and Resource Allocation Dashboard.

This module validates:
- Matrix shapes, types, and non-negative integers.
- Allocation <= Maximum consistency (with exact cell reporting).
- Available vs Total resource consistency (with friendly warnings).
- Process ID existence and bounds.
- Request vector length and non-negative integer values.
- Explicit distinction between valid boundary cases (e.g., zero Available, zero Need)
  and genuinely invalid inputs (e.g., negative numbers, Allocation > Max).

Designed to be robust and never crash the Streamlit dashboard or tests.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def validate_matrix_shape_and_types(
    matrix: Any,
    expected_rows: int,
    expected_cols: int,
    name: str = "Matrix"
) -> Tuple[bool, Optional[str]]:
    """
    Validates that a matrix is 2D, matches expected dimensions, and contains only
    non-negative integers.

    Args:
        matrix: 2D array-like structure.
        expected_rows: Expected number of processes (rows).
        expected_cols: Expected number of resource types (columns).
        name: Name of the matrix for error messages.

    Returns:
        (is_valid, error_message): True if valid, or False with detailed explanation.
    """
    try:
        arr = np.array(matrix)
    except Exception as e:
        return False, f"{name}: Unable to convert to numeric array. Details: {e}"

    if arr.ndim != 2:
        return False, f"{name}: Expected a 2D matrix of shape ({expected_rows}, {expected_cols}), got {arr.ndim}D."

    if arr.shape != (expected_rows, expected_cols):
        return False, f"{name}: Shape mismatch. Expected ({expected_rows}, {expected_cols}), got {arr.shape}."

    # Check for NaN / infinite
    if not np.isfinite(arr).all():
        return False, f"{name}: Contains NaN, null, or infinite values."

    # Check for integers (or floats that are exact whole numbers)
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        return False, f"{name}: Contains non-integer decimal values. All resource counts must be integers."

    # Check non-negative
    if (arr < 0).any():
        neg_indices = np.argwhere(arr < 0)
        first_neg = neg_indices[0]
        return False, (
            f"{name}: Negative values are invalid. Found value {arr[first_neg[0], first_neg[1]]} "
            f"at row {first_neg[0]} (P{first_neg[0]}), column {first_neg[1]}."
        )

    return True, None


def validate_vector(
    vector: Any,
    expected_len: int,
    name: str = "Vector"
) -> Tuple[bool, Optional[str]]:
    """
    Validates a 1D vector of expected length and non-negative integers.
    """
    try:
        arr = np.array(vector).flatten()
    except Exception as e:
        return False, f"{name}: Unable to convert to array. Details: {e}"

    if len(arr) != expected_len:
        return False, f"{name}: Expected length {expected_len}, got {len(arr)}."

    if not np.isfinite(arr).all():
        return False, f"{name}: Contains NaN, null, or infinite values."

    if not np.all(np.equal(np.mod(arr, 1), 0)):
        return False, f"{name}: Contains non-integer values. Resource counts must be integers."

    if (arr < 0).any():
        neg_idx = np.where(arr < 0)[0][0]
        return False, f"{name}: Negative value {arr[neg_idx]} found at index {neg_idx}."

    return True, None


def validate_system_state(
    allocation: np.ndarray,
    maximum: np.ndarray,
    available: np.ndarray,
    total: Optional[np.ndarray] = None,
    process_labels: Optional[List[str]] = None,
    resource_labels: Optional[List[str]] = None
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Validates the overall state configuration.
    
    Checks:
    1. Allocation <= Maximum for every process and resource cell.
    2. Optional Total resources consistency: Available == Total - sum(Allocation).
    3. Identifies valid boundary conditions (e.g., zero Available, zero Need).

    Returns:
        (is_valid, status_code, errors, warnings)
        status_code: 'VALID' or 'INVALID INPUT'
    """
    errors: List[str] = []
    warnings: List[str] = []

    alloc_arr = np.array(allocation, dtype=int)
    max_arr = np.array(maximum, dtype=int)
    avail_arr = np.array(available, dtype=int)

    n_proc, n_res = alloc_arr.shape

    p_labels = process_labels if process_labels and len(process_labels) == n_proc else [f"P{i}" for i in range(n_proc)]
    r_labels = resource_labels if resource_labels and len(resource_labels) == n_res else [f"R{j}" for j in range(n_res)]

    # 1. Allocation > Maximum check
    violations = alloc_arr > max_arr
    if np.any(violations):
        coords = np.argwhere(violations)
        for r, c in coords:
            errors.append(
                f"Allocation > Maximum violation at cell [{p_labels[r]}, {r_labels[c]}]: "
                f"Allocation is {alloc_arr[r, c]}, but Maximum is {max_arr[r, c]}."
            )

    # 2. Total resources check (if provided)
    if total is not None:
        total_arr = np.array(total, dtype=int).flatten()
        if len(total_arr) == n_res:
            allocated_sum = alloc_arr.sum(axis=0)
            expected_avail = total_arr - allocated_sum
            for j in range(n_res):
                if total_arr[j] < allocated_sum[j]:
                    errors.append(
                        f"Resource {r_labels[j]}: Total instances ({total_arr[j]}) is less than "
                        f"sum of currently allocated instances ({allocated_sum[j]})."
                    )
                elif avail_arr[j] != expected_avail[j]:
                    warnings.append(
                        f"Resource {r_labels[j]} Available inconsistency: Currently Available is "
                        f"{avail_arr[j]}, but Total ({total_arr[j]}) - Allocated ({allocated_sum[j]}) = {expected_avail[j]}."
                    )

    # 3. Informational boundary checks (not errors, but noted for viva clarity)
    if np.all(avail_arr == 0):
        warnings.append(
            "Boundary Condition Note: All Available resources are currently 0. "
            "This is a VALID boundary state (e.g. all resources temporarily allocated), not an error."
        )

    need = max_arr - alloc_arr
    zero_need_procs = [p_labels[i] for i in range(n_proc) if np.all(need[i] == 0)]
    if zero_need_procs:
        warnings.append(
            f"Boundary Condition Note: Process(es) {', '.join(zero_need_procs)} have remaining Need = 0. "
            "This is a VALID boundary state (process holds all its required resources and can finish immediately)."
        )

    is_valid = len(errors) == 0
    status_code = "VALID" if is_valid else "INVALID INPUT"
    return is_valid, status_code, errors, warnings


def validate_request_input(
    process_idx: Any,
    request_vector: Any,
    num_processes: int,
    num_resources: int,
    process_labels: Optional[List[str]] = None
) -> Tuple[bool, Optional[int], Optional[np.ndarray], Optional[str]]:
    """
    Validates a resource request input before running Banker's algorithm checks.

    Checks:
    - Process index is an integer within [0, num_processes - 1].
    - Request vector is 1D with length == num_resources.
    - All elements of request vector are non-negative integers.

    Returns:
        (is_valid, parsed_proc_idx, parsed_request_array, error_message)
    """
    p_labels = process_labels or [f"P{i}" for i in range(num_processes)]

    # Parse process_idx
    parsed_idx: Optional[int] = None
    if isinstance(process_idx, str):
        cleaned = process_idx.strip().upper()
        if cleaned.startswith("P") and cleaned[1:].isdigit():
            parsed_idx = int(cleaned[1:])
        elif cleaned.isdigit():
            parsed_idx = int(cleaned)
        elif cleaned in p_labels:
            parsed_idx = p_labels.index(cleaned)
    elif isinstance(process_idx, (int, np.integer)):
        parsed_idx = int(process_idx)

    if parsed_idx is None or parsed_idx < 0 or parsed_idx >= num_processes:
        return False, None, None, (
            f"Invalid process identifier '{process_idx}'. Must be an integer or label "
            f"between P0 and P{num_processes - 1}."
        )

    # Validate vector
    is_valid_vec, vec_err = validate_vector(request_vector, num_resources, name="Request vector")
    if not is_valid_vec:
        return False, parsed_idx, None, vec_err

    req_arr = np.array(request_vector, dtype=int).flatten()
    return True, parsed_idx, req_arr, None
