#!/usr/bin/env python3

"""
Given a patient, channel, and time window, visualize the RQE computation process. This scripts returns
two plots:
    1. The first one is a visualization of each time window and the EEG signal within it, along with the
    RQA matrix and the RQA metrics computed for that window.
    2. The second plot shows the EEG signal in the domain time, marking the windows that were used for computing
    the RQA metric, whose evolution is shown in the second row of the plot, also marking the windows that were used
    to compute the Spearman correlations between them, leading to the RQE value, shown in the third row of the plot.
"""

import os
from typing import List, Tuple

import numpy as np

from pyddeeg.signal_processing.rqa_toolbox.rqe import compute_rqa_metrics_for_window

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

plt.rcParams["figure.dpi"] = 300


def visualize_rqe_process(
    data: np.ndarray,
    patient_idx: int,
    channel_name: str,
    band_idx: int = 4,  # Default to Gamma band
    start_time: int = 0,
    window_size: int = 500,
    raw_signal_window_size: int = 100,
    rqa_window_size: int = 55,  # Added parameter for RQA window size
    embedding_dim: int = 10,
    time_delay: int = 1,
    radius: float = 0.8,
    distance_metric: str = "euclidean",
    min_diagonal_line: int = 5,
    min_vertical_line: int = 1,
    min_white_vertical_line: int = 1,
    metrics_to_use: List[str] = ["RR", "DET", "L_max", "ENT", "LAM", "TT"],
    stride: int = 1,
    normalize_metrics: bool = True,  # New parameter for normalization
) -> Tuple[plt.Figure, plt.Figure, List[np.ndarray]]:
    """
    Visualize the RQE computation process for a single frequency band.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_electrodes, n_samples, n_bands)
    patient_idx : int
        Index of the patient to visualize
    channel_name : str
        Name of the electrode to visualize
    band_idx : int
        Index of the frequency band to visualize (0=Delta, 1=Theta, 2=Alpha, 3=Beta, 4=Gamma)
    start_time : int
        Starting time point for visualization
    window_size : int
        Number of samples to visualize
    raw_signal_window_size : int
        Size of sliding window for computing RQA metrics
    rqa_window_size : int
        Size of window for computing RQE correlation in the RQA space
    embedding_dim, time_delay, radius, etc. : Various RQA parameters
        Parameters for RQA computation
    normalize_metrics : bool
        Whether to normalize metrics to [0,1] range before computing correlation

    Returns:
    --------
    fig1, fig2, rec_matrices : Tuple[plt.Figure, plt.Figure, List[np.ndarray]]
        Figure objects for the two plots and list of recurrence matrices
    """
    from pyunicorn.timeseries import RecurrencePlot
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from pyddeeg.signal_processing.rqa_toolbox.rqe import compute_rqe_and_correlation

    # EEG bands names and channel names
    bands = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-12 Hz)",
        "Beta (12-30 Hz)",
        "Gamma (30-80 Hz)",
    ]

    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "C4",
        "T8",
        "TP9",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "TP10",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "PO9",
        "O1",
        "Oz",
        "O2",
        "PO10",
        "Cz",
    ]

    # Find electrode index
    try:
        electrode_idx = ch_names.index(channel_name)
    except ValueError:
        raise ValueError(
            f"Channel {channel_name} not found. Available channels: {', '.join(ch_names)}"
        )

    # Extract data for the specified patient, electrode, band and time window
    signal = data[
        patient_idx, electrode_idx, start_time : start_time + window_size, band_idx
    ]

    # Calculate how many sliding windows we'll have
    num_windows = (window_size - raw_signal_window_size) // stride + 1

    # Create time vector (in milliseconds)
    time = np.arange(start_time, start_time + window_size)

    # Compute the sliding windows and their recurrence plots
    windows = []
    rqa_metrics = []
    rec_matrices = []  # Store the actual matrices

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + raw_signal_window_size
        window_signal = signal[start_idx:end_idx]
        windows.append(window_signal)

        # Calculate RQA metrics for this window
        metrics = compute_rqa_metrics_for_window(
            window_signal=window_signal,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line,
            metrics_to_use=metrics_to_use,
        )
        rqa_metrics.append(metrics)

        # Compute and store the recurrence matrix
        if distance_metric.lower() == "meandist":
            from scipy.spatial.distance import pdist

            distances = pdist(window_signal.reshape(-1, 1), metric="euclidean")
            mean_dist = np.mean(distances) if len(distances) > 0 else 1.0
            normalized_radius = radius / mean_dist
            actual_radius = normalized_radius
            actual_metric = "euclidean"
        else:
            actual_radius = radius
            actual_metric = distance_metric

        try:
            rp = RecurrencePlot(
                time_series=window_signal,
                dim=embedding_dim,
                tau=time_delay,
                metric=actual_metric,
                threshold=actual_radius,
                silence_level=2,
            )
            # Get the matrix and store it
            rec_matrix = rp.recurrence_matrix()
            rec_matrices.append(rec_matrix)
        except Exception as e:
            print(f"Error computing recurrence plot for window {i}: {e}")
            # Create a dummy matrix of zeros
            rec_matrices.append(
                np.zeros((raw_signal_window_size, raw_signal_window_size))
            )

    # Convert RQA metrics to a numpy array for easier manipulation
    rqa_array = np.zeros((num_windows, len(metrics_to_use)))
    for i in range(num_windows):
        for j, metric in enumerate(metrics_to_use):
            val = rqa_metrics[i][metric]
            if val is not None:
                rqa_array[i, j] = val
            else:
                rqa_array[i, j] = np.nan

    # Store the original metrics for plotting
    rqa_array_original = rqa_array.copy()

    # Normalize each metric to [0,1] range if requested
    if normalize_metrics:
        # Normalize each column (metric) independently
        for j in range(rqa_array.shape[1]):
            col = rqa_array[:, j]
            # Skip if all values are NaN
            if np.all(np.isnan(col)):
                continue

            # Extract non-NaN values
            valid_values = col[~np.isnan(col)]
            if len(valid_values) > 0:
                col_min = np.min(valid_values)
                col_max = np.max(valid_values)

                # Avoid division by zero
                if col_max > col_min:
                    # Normalize non-NaN values
                    col[~np.isnan(col)] = (col[~np.isnan(col)] - col_min) / (
                        col_max - col_min
                    )
                    rqa_array[:, j] = col

    # This is the correct way to compute RQE and correlation values
    if num_windows > rqa_window_size:  # Ensure we have enough windows
        rqe_values, corr_values = compute_rqe_and_correlation(
            rqa_array, rqa_window_size
        )
    else:
        print(
            f"Warning: Not enough windows ({num_windows}) for RQA window size ({rqa_window_size})"
        )
        rqe_values = np.zeros(max(1, num_windows - rqa_window_size + 1))
        corr_values = np.zeros(max(1, num_windows - rqa_window_size + 1))

    # PLOT 1: Sliding windows and their recurrence plots
    # Create a figure with n_windows columns and 2 rows
    n_cols = min(10, num_windows)  # Limit to 10 columns max
    n_rows = 2

    fig1 = plt.figure(figsize=(n_cols * 2, 6))
    fig1.suptitle(
        f"RQA Computation Process for {bands[band_idx]} Band - Patient {patient_idx}, Electrode {channel_name}",
        fontsize=14,
    )

    gs = fig1.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # Plot up to the first 10 windows
    for i in range(min(n_cols, num_windows)):
        # Signal window plot
        ax_signal = fig1.add_subplot(gs[0, i])
        window_start = i * stride
        window_end = window_start + raw_signal_window_size
        window_time = time[window_start:window_end]

        ax_signal.plot(window_time, windows[i], "b-")
        ax_signal.set_title(f"Window {i+1}", fontsize=10)
        if i == 0:
            ax_signal.set_ylabel("Amplitude")
        ax_signal.set_xticklabels([])

        # Recurrence plot
        ax_rp = fig1.add_subplot(gs[1, i])

        # Directly use the stored matrix
        im = ax_rp.imshow(
            rec_matrices[i],
            cmap="binary",
            origin="lower",
            aspect="auto",
            interpolation="none",
        )
        ax_rp.set_xticks([])
        ax_rp.set_yticks([])

        # Add metrics as text - use original (non-normalized) metrics for display
        metrics = rqa_metrics[i]
        metrics_text = "\n".join(
            [
                f"{k}: {metrics[k]:.3f}" if metrics[k] is not None else f"{k}: None"
                for k in metrics_to_use[:6]
            ]
        )  # Limit to first 6 metrics for readability
        ax_rp.text(
            0.5,
            -0.3,
            metrics_text,
            transform=ax_rp.transAxes,
            fontsize=7,
            ha="center",
            va="top",
        )

    # Add common xlabel for bottom row
    fig1.text(0.5, 0.02, "Recurrence Plots with RQA Metrics", ha="center", fontsize=12)

    # PLOT 2: Evolution of RQA metrics, RQE and correlation
    fig2 = plt.figure(figsize=(12, 12))  # Increased height for three rows
    fig2.suptitle(
        f"RQA/RQE Evolution for {bands[band_idx]} Band - Patient {patient_idx}, Electrode {channel_name}",
        fontsize=14,
    )

    if normalize_metrics:
        fig2.text(
            0.5,
            0.01,
            "Note: RQA metrics were normalized to [0,1] range before RQE computation",
            ha="center",
            fontsize=10,
            style="italic",
        )

    gs = fig2.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.3)  # Now 3 rows

    # Original signal (row 0)
    ax_full_signal = fig2.add_subplot(gs[0])
    ax_full_signal.plot(time, signal, "k-")
    ax_full_signal.set_title("Raw EEG Signal")
    ax_full_signal.set_xlabel("Time (ms)")
    ax_full_signal.set_ylabel("Amplitude")

    # Highlight each window on the time domain
    for i in range(min(n_cols, num_windows)):
        window_start = i * stride
        window_end = window_start + raw_signal_window_size - 1
        ax_full_signal.axvspan(
            time[window_start], time[window_end], alpha=0.2, color=f"C{i%10}"
        )

    # RQA metrics evolution (row 1) - use original metrics for plotting
    ax_metrics = fig2.add_subplot(gs[1])

    # Calculate x positions for each window (use the middle of each window)
    window_centers = [
        time[i * stride + raw_signal_window_size // 2] for i in range(num_windows)
    ]

    # Plot each metric using the original (non-normalized) values
    for m_idx, metric in enumerate(metrics_to_use):
        metric_values = [
            rqa_metrics[i][metric]
            for i in range(num_windows)
            if rqa_metrics[i][metric] is not None
        ]
        window_positions = [
            window_centers[i]
            for i in range(num_windows)
            if rqa_metrics[i][metric] is not None
        ]

        if len(metric_values) > 0:
            ax_metrics.plot(window_positions, metric_values, "o-", label=metric)

    ax_metrics.set_title("RQA Metrics Evolution")
    ax_metrics.set_xlabel("Time (ms)")
    ax_metrics.set_ylabel("RQA Metric Value")
    ax_metrics.legend(
        loc="best", fontsize=8, ncol=2
    )  # Smaller font, 2 columns for legend
    ax_metrics.grid(True, alpha=0.3)

    # Now add an additional plot for RQE and correlation (row 2)
    ax_rqe = fig2.add_subplot(gs[2])

    # Calculate time points for RQE/correlation values (middle of each RQA window)
    if len(rqe_values) > 0:
        # Calculate the time point for each RQE/correlation value
        rqe_time_points = []
        for i in range(len(rqe_values)):
            center_idx = i + rqa_window_size // 2
            if center_idx < len(window_centers):
                rqe_time_points.append(window_centers[center_idx])
            else:
                rqe_time_points.append(window_centers[-1])

        # Plot RQE and correlation values
        ax_rqe.plot(rqe_time_points, rqe_values, "ro-", label="RQE", linewidth=2)
        ax_rqe.plot(rqe_time_points, corr_values, "bo-", label=r"|$\rho$|", linewidth=2)

        # Highlight RQA windows used for each RQE/correlation calculation
        for i in range(min(5, len(rqe_values))):  # Only show first 5 for clarity
            first_window_idx = i
            last_window_idx = i + rqa_window_size - 1

            if first_window_idx < len(window_centers) and last_window_idx < len(
                window_centers
            ):
                window_start = (
                    window_centers[first_window_idx] - raw_signal_window_size / 2
                )
                window_end = (
                    window_centers[last_window_idx] + raw_signal_window_size / 2
                )

                ax_metrics.axvspan(
                    window_start,
                    window_end,
                    alpha=0.1,
                    color=f"C{i%10}",
                    linestyle="--",
                )
                ax_rqe.axvspan(
                    window_start,
                    window_end,
                    alpha=0.1,
                    color=f"C{i%10}",
                    linestyle="--",
                )

    ax_rqe.set_title("RQE Index and Spearman Correlation")
    ax_rqe.set_xlabel("Time (ms)")
    ax_rqe.set_ylabel("Value")
    ax_rqe.legend(loc="upper right")
    ax_rqe.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig1, fig2, rec_matrices


def main() -> None:
    ROOT_PROCESSED = "/home/mariopasc/Python/Datasets/EEG/timeseries/processed"
    data_ct = np.load(os.path.join(ROOT_PROCESSED, "CT_UP_preprocess_2.npz"))["data"]
    data_dd = np.load(os.path.join(ROOT_PROCESSED, "DD_UP_preprocess_2.npz"))["data"]

    # Example usage with normalization:
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    band_idx = 4  # Gamma band

    fig1, fig2, rec = visualize_rqe_process(
        data=data_dd,
        patient_idx=0,
        channel_name="T7",
        band_idx=band_idx,
        start_time=0,
        window_size=200,
        raw_signal_window_size=50,
        rqa_window_size=10,
        embedding_dim=10,
        time_delay=1,
        radius=0.8,
        distance_metric="euclidean",
        min_diagonal_line=5,
        min_vertical_line=1,
        min_white_vertical_line=1,
        metrics_to_use=[
            "RR",
            "DET",
            "ENT",
            "TT",
            "V_max",
            "V_mean",
            "V_ENT",
            "W_max",
            "W_mean",
            "W_ENT",
            "PERM_ENT",
        ],
        stride=10,
        normalize_metrics=False,  # Enable normalization
    )

    fig1.savefig(
        f"./scripts/rqe/results/ct_t7_{band_names[band_idx]}_rqa_process.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
    )
    fig2.savefig(
        f"./scripts/rqe/results/ct_t7_{band_names[band_idx]}_rqa_evolution.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()
