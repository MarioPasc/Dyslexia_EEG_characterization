import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import scipy.stats as stats

from pyddeeg.signal_processing.rqa_toolbox.rqa import compute_rqa_metrics_for_window

def compute_rqa_batch(
    signal: np.ndarray,
    raw_signal_window_size: int,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int,
    metrics_to_use: List[str],
    stride: int = 1,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute RQA metrics for a batch of sliding windows.
    Optimized version for Dask parallel processing.

    Parameters:
    -----------
    signal : np.ndarray
        1D array containing the signal to analyze
    raw_signal_window_size, embedding_dim, etc. : Various parameters
        RQA computation parameters
    stride : int
        Step size between consecutive windows
    batch_size : Optional[int]
        Number of windows to process in one batch (None = all)

    Returns:
    --------
    rqa_matrix : np.ndarray
        2D array with shape (num_windows, len(metrics_to_use))
    """
    n = len(signal)
    num_windows = (n - raw_signal_window_size) // stride + 1

    if batch_size is None or batch_size >= num_windows:
        batch_size = num_windows

    # Pre-allocate results array
    rqa_results = np.full((batch_size, len(metrics_to_use)), np.nan)

    for i in range(batch_size):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + raw_signal_window_size]

        metrics, _ = compute_rqa_metrics_for_window(
            window_signal=window_data,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line,
            metrics_to_use=metrics_to_use,
        )

        # Extract metrics in the same order as metrics_to_use
        for j, key in enumerate(metrics_to_use):
            rqa_results[i, j] = metrics.get(key, np.nan)

    return rqa_results


def compute_rqe_batch(
    rqa_matrix: np.ndarray, rqa_space_window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RQE and correlation for a batch of RQA metrics.
    Optimized version for Dask parallel processing.

    Parameters:
    -----------
    rqa_matrix : np.ndarray
        2D array of shape (num_windows, num_metrics)
    rqa_space_window_size : int
        Size of window for computing RQE correlation in the RQA space

    Returns:
    --------
    rqe_values : np.ndarray
        Array of RQE values
    corr_values : np.ndarray
        Array of correlation values
    """
    T, L = rqa_matrix.shape
    Q = max(1, T - rqa_space_window_size + 1)

    rqe_list = np.zeros(Q)
    corr_list = np.zeros(Q)

    for start_idx in range(Q):
        sub_block = rqa_matrix[start_idx : start_idx + rqa_space_window_size, :]
        # Transpose to get each metric as a row
        sub_block_T = sub_block.T

        # Collect pairwise correlations
        pairwise_rho = []
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                # Skip if either series has NaN values
                if (
                    np.isnan(sub_block_T[l1, :]).any()
                    or np.isnan(sub_block_T[l2, :]).any()
                ):
                    continue

                try:
                    rho, _ = stats.spearmanr(sub_block_T[l1, :], sub_block_T[l2, :])
                    if not np.isnan(rho):
                        pairwise_rho.append(abs(rho))
                except Exception:
                    # Handle potential errors in correlation computation
                    pass

        if len(pairwise_rho) == 0:
            rqe_list[start_idx] = 1.0
            corr_list[start_idx] = 0.0
        else:
            # RQE = product(1 + |rho|)
            product_val = 1.0
            for c in pairwise_rho:
                product_val *= 1.0 + c
            rqe_list[start_idx] = product_val

            # Corr = mean(|rho|)
            corr_list[start_idx] = np.mean(pairwise_rho)

    return rqe_list, corr_list


def process_single_channel_band(
    signal: np.ndarray,
    rqa_params: Dict,
    normalize_metrics: bool = False,
    return_rqe: bool = False,
    rqa_space_window_size: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single channel/band combination.
    This function is designed to be the unit of parallelization.

    Parameters:
    -----------
    signal : np.ndarray
        1D array representing a single channel/band
    rqa_params : Dict
        Dictionary of RQA parameters
    normalize_metrics : bool
        Whether to normalize metrics
    return_rqe : bool
        Whether to compute RQE and correlation
    rqa_space_window_size : int
        Size of window for computing RQE correlation in the RQA space.
        Only used if return_rqe is True.

    Returns:
    --------
    rqa_matrix : np.ndarray
        Matrix of RQA metrics for each time window
    rqe_values : np.ndarray
        RQE values if return_rqe is True, empty array otherwise
    corr_values : np.ndarray
        Correlation values if return_rqe is True, empty array otherwise
    """
    # Extract parameters
    embedding_dim = rqa_params.get("embedding_dim", 10)
    radius = rqa_params.get("radius", 0.8)
    time_delay = rqa_params.get("time_delay", 1)
    raw_signal_window_size = rqa_params.get("raw_signal_window_size", 100)
    min_diagonal_line = rqa_params.get("min_diagonal_line", 5)
    min_vertical_line = rqa_params.get("min_vertical_line", 1)
    min_white_vertical_line = rqa_params.get("min_white_vertical_line", 1)
    metrics_to_use = rqa_params.get("metrics_to_use", ["RR", "DET", "ENT", "TT"])
    stride = rqa_params.get("stride", 1)

    # Compute RQA metrics
    rqa_matrix = compute_rqa_batch(
        signal=signal,
        raw_signal_window_size=raw_signal_window_size,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        radius=radius,
        distance_metric="euclidean",
        min_diagonal_line=min_diagonal_line,
        min_vertical_line=min_vertical_line,
        min_white_vertical_line=min_white_vertical_line,
        metrics_to_use=metrics_to_use,
        stride=stride,
    )

    # Normalize metrics if requested
    if normalize_metrics:
        for j in range(rqa_matrix.shape[1]):
            col = rqa_matrix[:, j]
            if np.all(np.isnan(col)):
                continue

            valid_values = col[~np.isnan(col)]
            if len(valid_values) > 0:
                col_min = np.min(valid_values)
                col_max = np.max(valid_values)

                if col_max > col_min:
                    col[~np.isnan(col)] = (col[~np.isnan(col)] - col_min) / (
                        col_max - col_min
                    )
                    rqa_matrix[:, j] = col

    # Only compute RQE if requested
    if return_rqe:
        # Check if we have enough windows for RQE computation
        if rqa_matrix.shape[0] <= rqa_space_window_size:
            # Return empty arrays if not enough windows
            return rqa_matrix, np.array([]), np.array([])

        # Compute RQE and correlation
        rqe_values, corr_values = compute_rqe_batch(
            rqa_matrix, rqa_space_window_size=rqa_space_window_size
        )
        return rqa_matrix, rqe_values, corr_values
    else:
        # Skip RQE computation, return empty arrays
        return rqa_matrix, np.array([]), np.array([])
