#!/usr/bin/env python3
"""
Fast min-max normalization with electrode-level parallelization.

This module provides optimized functions to normalize neurophysiological data
by parallelizing across electrodes, making computation for individual files very fast.
"""
from typing import Tuple, Union, Optional, List
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import time


def minmax_normalize(
    data: np.ndarray,
    target_range: Tuple[float, float] = (-1.0, 1.0),
    electrode_axis: int = 1,
    points_axis: Optional[int] = 3,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    Normalize data to target range with parallelization across electrodes.
    
    Args:
        data: Input array of any shape - typically containing neurophysiological data
        target_range: Tuple with (min, max) values for normalization target range
        electrode_axis: The axis representing electrodes (to be parallelized)
        points_axis: The axis representing time points (used for chunking)
                     Set to None if no specific points axis exists
        chunk_size: Size of chunks for the points axis, auto-calculated if None
        
    Returns:
        Normalized numpy array with same shape as input
    """
    # Record start time for performance reporting
    start_time = time.time()
    
    # Extract target range
    target_min, target_max = target_range
    target_range_width = target_max - target_min
    
    # Get the shape of the data
    data_shape = data.shape
    n_dims = len(data_shape)
    
    # Validate axis parameters
    if electrode_axis >= n_dims:
        raise ValueError(f"electrode_axis {electrode_axis} out of bounds for array with {n_dims} dimensions")
    if points_axis is not None and points_axis >= n_dims:
        raise ValueError(f"points_axis {points_axis} out of bounds for array with {n_dims} dimensions")
    
    # Determine chunk size for Dask array
    chunks = list(data_shape)
    
    # Set chunk size for electrode axis to 1 for maximum parallelism
    chunks[electrode_axis] = 1
    
    # Set chunk size for points axis if specified
    if points_axis is not None:
        if chunk_size is None:
            # Auto-calculate a reasonable chunk size for points
            # Typically 1000 points or the full size if smaller
            chunks[points_axis] = min(1000, data_shape[points_axis])
        else:
            chunks[points_axis] = min(chunk_size, data_shape[points_axis])
    
    # Convert to Dask array with appropriate chunking for parallelism
    dask_data = da.from_array(data, chunks=tuple(chunks))
    
    print(f"Created Dask array with chunks: {chunks}")
    print(f"Starting normalization for data shape: {data_shape}")
    
    # Find global min and max
    min_val = dask_data.min()
    max_val = dask_data.max()
    
    # Compute them in parallel
    with ProgressBar():
        min_val, max_val = da.compute(min_val, max_val)
    
    data_range = max_val - min_val
    
    # Check for division by zero
    if data_range == 0:
        print("Warning: Data range is zero, returning array of constant values.")
        return np.full_like(data, target_min)
    
    # Create normalized data as a Dask array for parallel computation
    normalized_data = target_min + (target_range_width * (dask_data - min_val) / data_range)
    
    # Compute the normalized result with progress tracking
    print("Computing normalized data in parallel across electrodes...")
    with ProgressBar():
        result = normalized_data.compute()
    
    # Report performance
    elapsed_time = time.time() - start_time
    print(f"Normalization completed in {elapsed_time:.2f} seconds")
    
    return result


def normalize_multiple_files(
    file_list: List[np.ndarray],
    **kwargs
) -> List[np.ndarray]:
    """
    Apply min-max normalization to multiple files sequentially.
    
    Args:
        file_list: List of numpy arrays to normalize
        **kwargs: Keyword arguments passed to minmax_normalize function
        
    Returns:
        List of normalized arrays
    """
    return [minmax_normalize(file_data, **kwargs) for file_data in file_list]


def get_data_stats(array: np.ndarray) -> dict:
    """
    Get basic statistics from a numpy array.
    
    Args:
        array: Input numpy array
        
    Returns:
        Dictionary of statistics
    """
    return {
        "shape": array.shape,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "size_mb": array.nbytes / (1024 * 1024)
    }


if __name__ == "__main__":
    # Example usage with different array shapes
    print("Testing with different data shapes")
    
    # Example 1: CT data shape (34 patients, 32 electrodes, 5 bands, 9701 points, 15 metrics)
    ct_data = np.random.random(size=(34, 32, 5, 9701, 15)) * 100
    print("\nProcessing CT data (34 patients)...")
    norm_ct = minmax_normalize(
        ct_data, 
        electrode_axis=1,  # Electrodes are the second dimension (index 1)
        points_axis=3      # Points are the fourth dimension (index 3)
    )
    print("CT data stats after normalization:", get_data_stats(norm_ct))
    
    # Example 2: DD data shape (15 patients, 32 electrodes, 5 bands, 9701 points, 15 metrics)
    dd_data = np.random.random(size=(15, 32, 5, 9701, 15)) * 100
    print("\nProcessing DD data (15 patients)...")
    norm_dd = minmax_normalize(
        dd_data,
        electrode_axis=1,
        points_axis=3
    )
    print("DD data stats after normalization:", get_data_stats(norm_dd))
    
    # Example 3: Different shape entirely (10 subjects, 64 channels, 2000 timepoints)
    different_data = np.random.random(size=(10, 64, 2000)) * 50 - 25
    print("\nProcessing completely different data shape...")
    norm_different = minmax_normalize(
        different_data,
        electrode_axis=1,  # Channels are the second dimension (index 1)
        points_axis=2      # Timepoints are the third dimension (index 2)
    )
    print("Different data stats after normalization:", get_data_stats(norm_different))
    
    print("\nDone!")