import numpy as np
from typing import List, Optional


def extract_signal_windows(
    signal: np.ndarray,
    window_size: int,
    stride: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extracts overlapping sliding windows from a 1D input signal for Recurrence Quantification Analysis (RQA).

    Parameters
    ----------
    signal : np.ndarray
        1D array containing the signal to analyze.
    window_size : int
        Number of samples in each window.
    stride : Optional[int], default=None
        Step size (in samples) between consecutive windows. If None, defaults to 50% of window_size.
    max_windows : Optional[int], default=None
        Maximum number of windows to generate. If None, all possible windows are generated.

    Returns
    -------
    List[np.ndarray]
        List of 1D numpy arrays, each representing a windowed segment of the input signal.

    Notes
    -----
    - If the signal is shorter than window_size, an empty list is returned.
    - Windows are extracted from the start of the signal, with the specified stride.
    - The last window may be omitted if it would extend beyond the end of the signal.
    """
    if stride is None:
        stride = window_size // 2

    n = len(signal)
    if n < window_size or window_size <= 0 or stride <= 0:
        return []

    num_windows = (n - window_size) // stride + 1

    if max_windows is not None:
        num_windows = min(num_windows, max_windows)

    windows: List[np.ndarray] = []
    for i in range(num_windows):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + window_size]
        windows.append(window_data)

    return np.array(windows) if len(windows) > 0 else []
