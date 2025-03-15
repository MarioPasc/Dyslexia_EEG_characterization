#!/usr/bin/env python3

""" """

import os
from typing import List, Tuple, Dict

import numpy as np

from pyddeeg.signal_processing.rqa_toolbox.rqe import (
    compute_rqa_time_series,
    compute_rqe_and_correlation,
)

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

plt.rcParams["figure.dpi"] = 300


def plot_eeg_bands_with_rqe(
    data: np.ndarray,
    patient_idx: int,
    channel_name: str,
    start_time: int = 0,
    window_size: int = 5000,
    rqe_params: Dict = {
        "embedding_dim": 10,
        "radius": 0.8,
        "time_delay": 1,
        "raw_signal_window_size": 100,
        "rqa_space_window_size": 25,
        "min_diagonal_line": 5,
        "min_vertical_line": 1,
        "min_white_vertical_line": 1,
        "metrics_to_use": [
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
        "stride": 1,
    },
    normalize_metrics: bool = False,
    debug: bool = True,  # Add debug flag to print information
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize EEG bands with RQE analysis overlay for a specific patient and electrode.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_electrodes, n_samples, n_bands)
    patient_idx : int
        Index of the patient to visualize
    channel_name : str
        Name of the electrode to visualize
    start_time : int
        Starting time point for visualization
    window_size : int
        Number of samples to visualize
    rqe_params : Dict
        Parameters for RQE analysis
    normalize_metrics : bool
        Whether to normalize metrics before computing RQE
    debug : bool
        Whether to print debug information

    Returns:
    --------
    fig : plt.Figure
        The figure object
    axes : List[plt.Axes]
        List of axes objects
    """

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

    if debug:
        print(
            f"Processing data for patient {patient_idx}, electrode {channel_name} ({electrode_idx})"
        )
        print(f"Data shape: {data.shape}")
        print(f"Time window: {start_time} to {start_time + window_size}")
        print(f"RQE parameters: {rqe_params}")

    # Create time vector (in milliseconds)
    time = np.arange(start_time, start_time + window_size)

    # Extract data
    data_to_plot = data[
        patient_idx, electrode_idx, start_time : start_time + window_size, :
    ]

    if debug:
        print(f"Extracted data shape: {data_to_plot.shape}")

    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"EEG Bands with RQE Analysis - Patient {patient_idx} - Electrode {channel_name}",
        fontsize=14,
    )

    # Plot each band with RQE overlay
    for i in range(5):
        if debug:
            print(f"\nProcessing band {i}: {bands[i]}")

        # Original signal with alpha=0.5
        axes[i].plot(
            time, data_to_plot[:, i], "b-", linewidth=1, alpha=0.5, label="EEG Signal"
        )

        # Compute RQE for this band
        band_signal = data_to_plot[:, i]

        if debug:
            print(f"Band signal shape: {band_signal.shape}")
            print(
                f"Band signal min: {np.min(band_signal)}, max: {np.max(band_signal)}, mean: {np.mean(band_signal)}"
            )
            print(f"Computing RQA time series...")

        try:
            # Compute RQA time series
            rqa_matrix = compute_rqa_time_series(
                signal=band_signal,
                raw_signal_window_size=rqe_params["raw_signal_window_size"],
                embedding_dim=rqe_params["embedding_dim"],
                time_delay=rqe_params["time_delay"],
                radius=rqe_params["radius"],
                distance_metric="euclidean",
                min_diagonal_line=rqe_params["min_diagonal_line"],
                min_vertical_line=rqe_params["min_vertical_line"],
                min_white_vertical_line=rqe_params["min_white_vertical_line"],
                metrics_to_use=rqe_params["metrics_to_use"],
                stride=rqe_params["stride"],
            )

            if debug:
                print(f"RQA matrix shape: {rqa_matrix.shape}")
                print(f"RQA matrix contains NaN: {np.isnan(rqa_matrix).any()}")
                print(
                    f"RQA matrix non-NaN values: {np.count_nonzero(~np.isnan(rqa_matrix))}"
                )

            # If we should normalize the metrics
            if normalize_metrics:
                # Normalize each column (metric) independently
                for j in range(rqa_matrix.shape[1]):
                    col = rqa_matrix[:, j]
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
                            rqa_matrix[:, j] = col

            # Check if there are enough windows for RQE computation
            rqa_space_window_size = rqe_params["rqa_space_window_size"]
            if rqa_matrix.shape[0] <= rqa_space_window_size:
                if debug:
                    print(
                        f"WARNING: Not enough RQA windows ({rqa_matrix.shape[0]}) for RQA space window size ({rqa_space_window_size})"
                    )
                    print(f"Need at least {rqa_space_window_size + 1} windows")
                # Skip this band if not enough windows
                continue

            if debug:
                print(f"Computing RQE and correlation...")

            # Compute RQE and correlation
            rqe_vals, corr_vals = compute_rqe_and_correlation(
                rqa_matrix, rqa_space_window_size=rqa_space_window_size
            )

            if debug:
                print(f"RQE values shape: {rqe_vals.shape}")
                print(f"RQE values: {rqe_vals}")
                print(f"RQE min: {np.min(rqe_vals)}, max: {np.max(rqe_vals)}")
                print(f"Correlation values: {corr_vals}")

            # Create time vector for RQE values
            rqe_time = np.linspace(time[0], time[-1], len(rqe_vals))

            # Scale the values for better visibility
            if len(rqe_vals) > 0 and not np.all(np.isnan(rqe_vals)):
                max_rqe = np.nanmax(rqe_vals) if np.nanmax(rqe_vals) > 0 else 1
                max_corr = np.nanmax(corr_vals) if np.nanmax(corr_vals) > 0 else 1

                # Plot RQE and correlation
                axes[i].plot(
                    rqe_time,
                    rqe_vals / max_rqe,
                    "r-",
                    linewidth=1.5,
                    alpha=1.0,
                    label="RQE",
                )
                axes[i].plot(
                    rqe_time,
                    corr_vals / max_corr,
                    "g-",
                    linewidth=1.5,
                    alpha=1.0,
                    label=r"|$\rho$|",
                )

                # Add a small note about the scale
                axes[i].text(
                    0.02,
                    0.95,
                    rf"Max RQE: {max_rqe:.2f}, Max |$\rho$|: {max_corr:.2f}",
                    transform=axes[i].transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
            else:
                if debug:
                    print(f"WARNING: No valid RQE values to plot")

        except Exception as e:
            if debug:
                print(f"ERROR computing RQE for band {i}: {str(e)}")
                import traceback

                traceback.print_exc()

        axes[i].set_ylabel(f"{bands[i]}\nAmplitude")
        axes[i].grid(True, alpha=0.3)

        # Always add legend to all subplots for clarity
        axes[i].legend(loc="upper right")

        # Styling
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    # Add common x-label
    axes[-1].set_xlabel("Time (milliseconds)")

    return fig, axes


def main() -> None:
    ROOT_PROCESSED = "/home/mariopasc/Python/Datasets/EEG/timeseries/processed"
    data_ct = np.load(os.path.join(ROOT_PROCESSED, "CT_UP_preprocess_2.npz"))["data"]
    data_dd = np.load(os.path.join(ROOT_PROCESSED, "DD_UP_preprocess_2.npz"))["data"]

    # Example usage with more appropriate parameters:
    rqe_params = {
        "embedding_dim": 10,
        "radius": 0.8,  # Using same value as cell 6
        "time_delay": 1,
        "raw_signal_window_size": 100,  # Matches cell 6
        "rqa_space_window_size": 25,
        "min_diagonal_line": 5,
        "min_vertical_line": 1,
        "min_white_vertical_line": 1,
        "metrics_to_use": [
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
        "stride": 1,  # Using a larger stride like in cell 6
    }

    # For control group
    fig, axes = plot_eeg_bands_with_rqe(
        data_ct,
        patient_idx=0,
        channel_name="T7",
        start_time=0,
        window_size=5000,  # Reduced to make computation faster
        rqe_params=rqe_params,
        debug=False,  # Enable debug output
    )
    plt.show()
