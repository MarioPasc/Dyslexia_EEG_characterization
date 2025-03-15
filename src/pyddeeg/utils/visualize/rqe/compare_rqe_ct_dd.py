#!/usr/bin/env python3

"""
Compare EEG band with RQE analysis between control and dyslexic groups given
a specific patient, electrode, bandwidth, and time window within the EEG data.

The typical RQE input parameters are also required to compute the visualization.
The main function contanins a straightforward example of how to use the function.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from pyddeeg.signal_processing.rqa_toolbox.rqe import (
    compute_rqa_time_series,
    compute_rqe_and_correlation,
)

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

plt.rcParams["figure.dpi"] = 300


def plot_compare_eeg_band_with_rqe(
    data_ct: np.ndarray,
    data_dd: np.ndarray,
    patient_idx: int,
    channel_name: str,
    band_idx: int = 4,  # Default to Gamma band (4)
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
    debug: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Compare EEG band with RQE analysis between control and dyslexic groups.

    Parameters:
    -----------
    data_ct : np.ndarray
        Control group EEG data
    data_dd : np.ndarray
        Dyslexic group EEG data
    patient_idx, channel_name, etc. : Various parameters
        Same as in plot_single_eeg_band_with_rqe

    Returns:
    --------
    fig : plt.Figure
        The figure object
    axes : List[plt.Axes]
        List of axes objects [ax_ct, ax_dd]
    """
    # EEG bands names and channel names
    bands = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-12 Hz)",
        "Beta (12-30 Hz)",
        "Gamma (30-80 Hz)",
    ]

    if band_idx < 0 or band_idx > 4:
        raise ValueError(f"Band index must be between 0 and 4. Got {band_idx}")

    band_name = bands[band_idx]

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
        print(f"Analyzing frequency band: {band_name}")

    # Create time vector (in seconds)
    time = (
        np.arange(start_time, start_time + window_size) / 1000.0
    )  # Convert to seconds

    # Extract data for the specific band
    band_signal_ct = data_ct[
        patient_idx, electrode_idx, start_time : start_time + window_size, band_idx
    ]
    band_signal_dd = data_dd[
        patient_idx, electrode_idx, start_time : start_time + window_size, band_idx
    ]

    # Create figure with two subplots, one for each group
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f"{band_name} Band with RQE Analysis - Patient {patient_idx} - Electrode {channel_name}",
        fontsize=16,
    )

    # Process and plot each group
    for idx, (label, ax, band_signal, data_source) in enumerate(
        [
            ("Control", axes[0], band_signal_ct, "CT"),
            ("Dyslexic", axes[1], band_signal_dd, "DD"),
        ]
    ):
        # Plot original signal
        ax.plot(time, band_signal, "b-", linewidth=1, alpha=0.5, label="EEG Signal")
        ax.set_title(f"{label} Group", fontsize=14)

        # Create secondary y-axis for RQE values
        ax2 = ax.twinx()
        ax2.set_ylabel("RQE / Correlation Value", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 1.1)

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
                print(f"{label} Group - RQA matrix shape: {rqa_matrix.shape}")

            # Optional normalization
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

            # Check if there are enough windows for RQE computation
            rqa_space_window_size = rqe_params["rqa_space_window_size"]
            if rqa_matrix.shape[0] <= rqa_space_window_size:
                if debug:
                    print(
                        f"{label} Group - WARNING: Not enough RQA windows ({rqa_matrix.shape[0]}) for RQA space window size ({rqa_space_window_size})"
                    )

                ax.text(
                    0.5,
                    0.5,
                    f"Not enough windows for RQE computation\nNeed {rqa_space_window_size + 1}, have {rqa_matrix.shape[0]}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="red", alpha=0.2),
                )
            else:
                # Compute RQE and correlation
                rqe_vals, corr_vals = compute_rqe_and_correlation(
                    rqa_matrix, rqa_space_window_size=rqa_space_window_size
                )

                if debug:
                    print(f"{label} Group - RQE values shape: {rqe_vals.shape}")
                    print(
                        f"{label} Group - RQE min: {np.min(rqe_vals)}, max: {np.max(rqe_vals)}"
                    )

                # Calculate time points for RQE values
                raw_window_size = rqe_params["raw_signal_window_size"]
                stride = rqe_params["stride"]

                # Calculate window centers
                window_centers = [
                    (start_time + i * stride + raw_window_size / 2) / 1000.0
                    for i in range(rqa_matrix.shape[0])
                ]

                # Calculate time points for RQE values
                rqe_time_points = []
                for i in range(len(rqe_vals)):
                    center_idx = i + rqa_space_window_size // 2
                    if center_idx < len(window_centers):
                        rqe_time_points.append(window_centers[center_idx])
                    else:
                        rqe_time_points.append(window_centers[-1])

                # Scale the values for better visibility on the main axis
                if len(rqe_vals) > 0 and not np.all(np.isnan(rqe_vals)):
                    signal_range = np.max(band_signal) - np.min(band_signal)
                    signal_min = np.min(band_signal)

                    max_rqe = np.nanmax(rqe_vals) if np.nanmax(rqe_vals) > 0 else 1
                    max_corr = np.nanmax(corr_vals) if np.nanmax(corr_vals) > 0 else 1

                    # Create scaled versions for primary axis
                    rqe_scaled = (
                        signal_min
                        + signal_range * 0.6
                        + (rqe_vals / max_rqe) * (signal_range * 0.3)
                    )
                    corr_scaled = (
                        signal_min
                        + signal_range * 0.6
                        + (corr_vals / max_corr) * (signal_range * 0.3)
                    )

                    # Plot on main axis
                    ax.plot(
                        rqe_time_points,
                        rqe_scaled,
                        "r-",
                        linewidth=2,
                        alpha=0.9,
                        label="RQE",
                    )
                    ax.plot(
                        rqe_time_points,
                        corr_scaled,
                        "g-",
                        linewidth=2,
                        alpha=0.9,
                        label=r"|$\rho$|",
                    )

                    # Add scatter points
                    ax.scatter(rqe_time_points, rqe_scaled, color="r", s=20, alpha=0.7)
                    ax.scatter(rqe_time_points, corr_scaled, color="g", s=20, alpha=0.7)

                    # Also plot on secondary axis with original values
                    ax2.plot(
                        rqe_time_points, rqe_vals, "r--", alpha=0.4, linewidth=0.75
                    )
                    ax2.plot(
                        rqe_time_points, corr_vals, "g--", alpha=0.4, linewidth=0.75
                    )

                    # Add note about max values
                    ax.text(
                        0.02,
                        0.95,
                        rf"Max RQE: {max_rqe:.2f}, Max |$\rho$|: {max_corr:.2f}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )
                else:
                    if debug:
                        print(f"{label} Group - WARNING: No valid RQE values to plot")

        except Exception as e:
            if debug:
                print(f"{label} Group - ERROR computing RQE: {str(e)}")
                import traceback

                traceback.print_exc()

            ax.text(
                0.5,
                0.5,
                f"Error computing RQE:\n{str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=dict(facecolor="red", alpha=0.2),
            )

        # Styling and labels
        ax.set_ylabel(f"{label} {band_name} Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)

    # Only add x-label to bottom subplot
    axes[1].set_xlabel("Time (seconds)")

    # Add annotations about parameters in the upper right corner
    param_text = (
        f"Params: Embedding={rqe_params['embedding_dim']}, "
        f"Radius={rqe_params['radius']}, Time Delay={rqe_params['time_delay']}\n"
        f"Window={rqe_params['raw_signal_window_size']}, "
        f"Stride={rqe_params['stride']}, RQA Window={rqe_params['rqa_space_window_size']}"
    )
    fig.text(
        0.99,
        0.98,
        param_text,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
    )

    # Add note about normalization if used
    if normalize_metrics:
        fig.text(
            0.5,
            0.01,
            "Note: RQA metrics were normalized to [0,1] range before RQE computation",
            ha="center",
            fontsize=9,
            style="italic",
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.08)  # Make room for suptitle and note

    return fig, axes


def main() -> None:
    ROOT_PROCESSED = "/home/mariopasc/Python/Datasets/EEG/timeseries/processed"
    data_ct = np.load(os.path.join(ROOT_PROCESSED, "CT_UP_preprocess_2.npz"))["data"]
    data_dd = np.load(os.path.join(ROOT_PROCESSED, "DD_UP_preprocess_2.npz"))["data"]

    # Example usage:
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    band_idx = 4  # Gamma band (4=Gamma, 3=Beta, 2=Alpha, 1=Theta, 0=Delta)

    rqe_params = {
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
        "stride": 7,
    }

    patient_idx = 0
    channel_name = "T7"
    fig, axes = plot_compare_eeg_band_with_rqe(
        data_ct=data_ct,
        data_dd=data_dd,
        patient_idx=patient_idx,
        channel_name=channel_name,
        band_idx=band_idx,
        start_time=0,
        window_size=10000,
        rqe_params=rqe_params,
        normalize_metrics=True,
        debug=False,
    )

    plt.savefig(
        f"./scripts/rqe/results/compare_{band_names[band_idx]}_patient{patient_idx}_ {channel_name}.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
