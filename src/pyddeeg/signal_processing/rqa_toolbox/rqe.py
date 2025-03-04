#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def create_recurrence_matrix(time_series, embedding, lag, radius, distance="euclidean"):
    """
    Create a recurrence matrix from a time series.

    Parameters:
    -----------
    time_series : numpy.ndarray
        Input time series data
    embedding : int
        Embedding dimension
    lag : int
        Delay/lag parameter
    radius : float
        Threshold for recurrence
    distance : str, optional
        Distance metric for calculating recurrence (default: 'euclidean')

    Returns:
    --------
    R : numpy.ndarray
        Recurrence matrix
    """
    # Create the phase space reconstruction
    N = len(time_series)
    m = embedding

    # Form embedded vectors
    Y = np.zeros((N - (m - 1) * lag, m))
    for i in range(m):
        Y[:, i] = time_series[i * lag : N - (m - 1) * lag + i * lag]

    # Calculate distances between all pairs of points
    if distance == "meandist":
        # Normalize by the mean distance
        dist = squareform(pdist(Y, metric="euclidean"))
        threshold = radius * np.mean(dist) / 100.0
        R = dist <= threshold
    else:
        # Use raw distance
        dist = squareform(pdist(Y, metric=distance))
        R = dist <= radius

    return R


def calculate_det(recurrence_matrix, min_line_length):
    """
    Calculate determinism (DET) from recurrence matrix.

    Parameters:
    -----------
    recurrence_matrix : numpy.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum line length to consider

    Returns:
    --------
    det : float
        Determinism value
    """
    N = recurrence_matrix.shape[0]

    # Calculate diagonal lines
    diag_hist = {}
    for i in range(-(N - 1), N):
        diag = np.diag(recurrence_matrix, k=i)

        # Find all diagonal lines
        indices = np.where(diag > 0)[0]
        if len(indices) > 0:
            # Count consecutive ones
            diffs = np.diff(indices)
            # Starting point of each line
            breaks = np.where(diffs > 1)[0] + 1
            # Line lengths
            line_lengths = np.split(indices, breaks)

            for line in line_lengths:
                length = len(line)
                if length >= min_line_length:
                    if length in diag_hist:
                        diag_hist[length] += 1
                    else:
                        diag_hist[length] = 1

    # Calculate DET
    if len(diag_hist) == 0:
        return 0.0

    num = sum(length * count for length, count in diag_hist.items())
    denom = np.sum(recurrence_matrix)

    if denom == 0:
        return 0.0
    return num / denom


def calculate_lam(recurrence_matrix, min_line_length):
    """
    Calculate laminarity (LAM) from recurrence matrix.

    Parameters:
    -----------
    recurrence_matrix : numpy.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum vertical line length to consider

    Returns:
    --------
    lam : float
        Laminarity value
    """
    N = recurrence_matrix.shape[0]

    # Calculate vertical lines
    vert_hist = {}
    for j in range(N):
        col = recurrence_matrix[:, j]

        # Find all vertical lines
        indices = np.where(col > 0)[0]
        if len(indices) > 0:
            # Count consecutive ones
            diffs = np.diff(indices)
            # Starting point of each line
            breaks = np.where(diffs > 1)[0] + 1
            # Line lengths
            line_lengths = np.split(indices, breaks)

            for line in line_lengths:
                length = len(line)
                if length >= min_line_length:
                    if length in vert_hist:
                        vert_hist[length] += 1
                    else:
                        vert_hist[length] = 1

    # Calculate LAM
    if len(vert_hist) == 0:
        return 0.0

    num = sum(length * count for length, count in vert_hist.items())
    denom = np.sum(recurrence_matrix)

    if denom == 0:
        return 0.0
    return num / denom


def calculate_rqa_measures(
    time_series, embedding, lag, radius, min_line_length, distance="euclidean"
):
    """
    Calculate RQA measures for a time series.

    Parameters:
    -----------
    time_series : numpy.ndarray
        Input time series data
    embedding : int
        Embedding dimension
    lag : int
        Delay/lag parameter
    radius : float
        Threshold for recurrence
    min_line_length : int
        Minimum line length for DET and LAM
    distance : str, optional
        Distance metric (default: 'euclidean')

    Returns:
    --------
    rqa_measures : dict
        Dictionary with RQA measures
    """
    R = create_recurrence_matrix(time_series, embedding, lag, radius, distance)

    # Calculate RQA measures
    rqa = {}

    # Recurrence rate (RR)
    rqa["RR"] = np.sum(R) / R.size

    # Determinism (DET)
    rqa["DET"] = calculate_det(R, min_line_length)

    # Laminarity (LAM)
    rqa["LAM"] = calculate_lam(R, min_line_length)

    return rqa


def rqe_analysis(
    signal,
    timestamps,
    window_size=50,
    shift=1,
    lag=1,
    embedding=10,
    radius=80,
    min_line_length=5,
    distance="euclidean",
):
    """
    Perform RQE (Recurrence Quantification Epoch) analysis on a signal.

    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    timestamps : numpy.ndarray
        Timestamps corresponding to the signal
    window_size : int, optional
        Size of the rolling window (default: 50)
    shift : int, optional
        Shift between consecutive windows (default: 1)
    lag : int, optional
        Delay/lag parameter (default: 1)
    embedding : int, optional
        Embedding dimension (default: 10)
    radius : float, optional
        Threshold for recurrence (default: 80)
    min_line_length : int, optional
        Minimum line length for DET and LAM (default: 5)
    distance : str, optional
        Distance metric (default: 'euclidean')

    Returns:
    --------
    epochs_timestamps : numpy.ndarray
        Timestamps for each epoch
    rqa_measures : dict
        Dictionary with RQA measures for each epoch
    """
    N = len(signal)

    # Calculate number of epochs/windows
    q = N - window_size + 1

    # Dictionary to store RQA measures for each epoch
    rqa_measures = {"RR": np.zeros(q), "DET": np.zeros(q), "LAM": np.zeros(q)}

    epochs_timestamps = np.zeros(q)

    # Calculate RQA measures for each epoch/window
    for i in range(0, q, shift):
        window = signal[i : i + window_size]
        rqa = calculate_rqa_measures(
            window, embedding, lag, radius, min_line_length, distance
        )

        rqa_measures["RR"][i] = rqa["RR"]
        rqa_measures["DET"][i] = rqa["DET"]
        rqa_measures["LAM"][i] = rqa["LAM"]

        # Store the timestamp corresponding to the center of the window
        epochs_timestamps[i] = timestamps[i + window_size // 2]

    return epochs_timestamps, rqa_measures


def rqe_correlation_index(
    signal,
    timestamps,
    window_size=50,
    shift=1,
    lag=1,
    embedding=10,
    radius=80,
    min_line_length=5,
    distance="euclidean",
):
    """
    Calculate the RQE correlation index based on Spearman correlation of RQA measures.

    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    timestamps : numpy.ndarray
        Timestamps corresponding to the signal
    window_size : int, optional
        Size of the rolling window (default: 50)
    shift : int, optional
        Shift between consecutive windows (default: 1)
    lag : int, optional
        Delay/lag parameter (default: 1)
    embedding : int, optional
        Embedding dimension (default: 10)
    radius : float, optional
        Threshold for recurrence (default: 80)
    min_line_length : int, optional
        Minimum line length for DET and LAM (default: 5)
    distance : str, optional
        Distance metric (default: 'euclidean')

    Returns:
    --------
    corr_timestamps : numpy.ndarray
        Timestamps for the correlation index
    correlation_index : numpy.ndarray
        RQE correlation index for each window
    rqa_measures : dict
        Dictionary with RQA measures for each epoch
    """
    # Calculate RQA measures for the entire signal
    epochs_timestamps, rqa_measures = rqe_analysis(
        signal,
        timestamps,
        window_size,
        shift,
        lag,
        embedding,
        radius,
        min_line_length,
        distance,
    )

    # Get all RQA measure names
    measure_names = list(rqa_measures.keys())
    L = len(measure_names)

    # Number of windows
    q = len(rqa_measures[measure_names[0]])

    # Number of pairs of correlations
    p = L * (L - 1) // 2

    # Calculate the correct size for correlation_index array
    # We need to ensure windows are fully contained within the available data
    valid_windows = q - window_size + 1

    # Array to store the RQE correlation index for each window
    correlation_index = np.zeros(valid_windows)
    corr_timestamps = np.zeros(valid_windows)

    # Calculate RQE correlation index for each window
    for i in range(valid_windows):
        # Get the window for each RQA measure
        windows = {}
        for measure in measure_names:
            windows[measure] = rqa_measures[measure][i : i + window_size]

        # Calculate Spearman correlation for each pair
        prod = 1.0
        for j in range(L):
            for k in range(j + 1, L):
                measure1 = measure_names[j]
                measure2 = measure_names[k]

                # Check if either window has constant values
                if np.std(windows[measure1]) == 0 or np.std(windows[measure2]) == 0:
                    # Use a default value (e.g., 0) for the correlation when one window is constant
                    corr = 0
                else:
                    # Calculate Spearman correlation
                    try:
                        corr, _ = spearmanr(windows[measure1], windows[measure2])
                        if np.isnan(corr):
                            corr = 0
                    except:
                        corr = 0

                # Use absolute value and add to product
                prod *= 1 + abs(corr)

        correlation_index[i] = prod

        # Store the timestamp corresponding to the center of this window
        window_center_idx = i + window_size // 2
        corr_timestamps[i] = epochs_timestamps[window_center_idx]

    return corr_timestamps, correlation_index, rqa_measures


# Example usage function
def example_usage():
    """
    Example of how to use the RQE functions with a simple sine wave signal.
    """
    import matplotlib.pyplot as plt

    # Create a sample signal (sine wave with some noise)
    N = 1000
    timestamps = np.linspace(0, 10, N)
    signal = np.sin(2 * np.pi * 0.5 * timestamps) + 0.2 * np.random.randn(N)

    # Parameters from the paper
    window_size = 50
    shift = 1
    lag = 1
    embedding = 10
    radius = 80
    min_line_length = 5
    distance = "euclidean"

    # Perform RQE analysis
    epochs_timestamps, rqa_measures = rqe_analysis(
        signal,
        timestamps,
        window_size,
        shift,
        lag,
        embedding,
        radius,
        min_line_length,
        distance,
    )

    # Calculate RQE correlation index
    corr_timestamps, correlation_index, _ = rqe_correlation_index(
        signal,
        timestamps,
        window_size,
        shift,
        lag,
        embedding,
        radius,
        min_line_length,
        distance,
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, signal)
    plt.title("Original Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(epochs_timestamps, rqa_measures["DET"], label="DET")
    plt.plot(epochs_timestamps, rqa_measures["LAM"], label="LAM")
    plt.plot(epochs_timestamps, rqa_measures["RR"], label="RR")
    plt.title("RQA Measures")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(corr_timestamps, correlation_index)
    plt.title("RQE Correlation Index")
    plt.xlabel("Time")
    plt.ylabel("Index Value")

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     example_usage()
