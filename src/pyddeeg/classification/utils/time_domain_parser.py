import numpy as np
from typing import Tuple

from pyddeeg.classification.dataloaders import EEGDataset

def window_to_time_domain(
    window_data: np.ndarray,
    dataset: EEGDataset,
    window_dim: int = -1,
    time_resolution_ms: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert window-indexed data to time-domain representation.
    
    This function maps data indexed by window position to a continuous time domain
    representation. Each time point in the output is mapped to the nearest window
    in the input data.
    
    Parameters
    ----------
    window_data : np.ndarray
        Data tensor with window indices along a specific dimension
    window_dim : int
        Dimension in the tensor that corresponds to window indices
    window_centers_ms : np.ndarray
        Center time points of each window in milliseconds
    window_size_ms : int
        Size of each window in milliseconds
    time_resolution_ms : int, optional
        Resolution of the output time domain in milliseconds (default: 1ms)
        
    Returns
    -------
    time_data : np.ndarray
        Data interpolated to time domain
    time_axis : np.ndarray
        Time points in milliseconds corresponding to the time dimension
    """
    # Extract metadata from the dataset
    window_centers_ms = dataset.metadata["centres_ms"]
    window_size_ms = dataset.metadata["window_ms"]
    
    # Determine the temporal bounds of the signal
    half_window = window_size_ms // 2
    start_time = window_centers_ms[0] - half_window
    end_time = window_centers_ms[-1] + half_window
    
    # Create time axis
    time_axis = np.arange(start_time, end_time + time_resolution_ms, time_resolution_ms)
    
    # Find the nearest window center for each time point
    nearest_window_indices = np.argmin(
        np.abs(time_axis[:, np.newaxis] - window_centers_ms[np.newaxis, :]), 
        axis=1
    )
    
    # Move window dimension to the front for easier indexing
    window_data_reordered = np.moveaxis(window_data, window_dim, 0)
    
    # Use advanced indexing to get data for each time point
    time_data_reordered = window_data_reordered[nearest_window_indices]
    
    # Move time dimension back to original position
    time_data = np.moveaxis(time_data_reordered, 0, window_dim)
    
    return time_data, time_axis