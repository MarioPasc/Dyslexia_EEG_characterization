import numpy as np
from typing import List, Optional

def extract_signal_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int = 1,
    max_windows: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract sliding windows from an input signal for RQA analysis.
    
    Parameters:
    -----------
    signal : np.ndarray
        1D array containing the signal to analyze
    window_size : int
        Size of each window in samples
    stride : int
        Step size between consecutive windows in samples
    max_windows : Optional[int]
        Maximum number of windows to generate (None = generate all possible windows)
        
    Returns:
    --------
    windows : List[np.ndarray]
        List of signal windows to be used as window_signal in compute_rqa_metrics_for_window
    """
    n = len(signal)
    num_windows = (n - window_size) // stride + 1
    
    if max_windows is not None:
        num_windows = min(num_windows, max_windows)
    
    windows = []
    for i in range(num_windows):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + window_size]
        windows.append(window_data)
    
    return windows