#!/usr/bin/env python3

"""
Plot the RQA matrix for a specific, patient, channel, and time window,
for every band in the EEG signal.
"""

import os
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from pyunicorn.timeseries import RecurrencePlot

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

plt.rcParams["figure.dpi"] = 300


def plot_multiband_rqa(
    band_data: np.ndarray,
    time_range: Tuple[int, int],
    sample_rate: float = 1000.0,
    dim: int = 10,
    tau: int = 1,
    threshold: float = 0.5,
    metric: str = "euclidean",
    show: bool = True,
    save_path: Optional[Union[str, os.PathLike]] = None,
    save_format: str = "svg",
    band_labels: List[str] = [],
) -> List[Dict[str, float]]:
    """
    Generate and visualize Recurrence Plots for multiple EEG frequency bands.

    Parameters
    ----------
    band_data : np.ndarray
        Array of shape (n_bands, n_samples) containing the signal for each frequency band
    time_range : Tuple[int, int]
        Tuple indicating (start_idx, end_idx) for the data to analyze
    sample_rate : float, optional
        Sampling rate of the data in Hz, used for time axis labels. Default is 1000.0 Hz.
    dim : int, optional
        Embedding dimension for phase space reconstruction. Default is 10.
    tau : int, optional
        Time delay for phase space reconstruction. Default is 1.
    threshold : float, optional
        Distance threshold for determining recurrence. Default is 0.5.
    metric : str, optional
        Distance metric to use (e.g. 'euclidean'). Default is 'euclidean'.
    show : bool, optional
        If True, displays the plot. If False, the plot is not shown. Default is True.
    save_path : str or os.PathLike, optional
        If provided, the path where the figure is saved. Default is None (no saving).
    save_format : str, optional
        The format to use when saving (e.g., 'svg', 'png'). Default is 'svg'.
    band_labels : List[str], optional
        Labels for each frequency band. If None, uses default band names.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries containing RQA metrics for each band
    """
    start_idx, end_idx = time_range
    n_bands = band_data.shape[0]

    # Default band labels if none provided
    if band_labels is None or len(band_labels) == 0:
        band_labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    # Create time array in seconds
    time = np.arange(start_idx, end_idx) / sample_rate

    # Set up the figure with GridSpec
    fig = plt.figure(figsize=(4 * n_bands, 12))

    # Create a GridSpec with 2 rows and n_bands columns
    # Top row will be the time series signals
    # Bottom row will be the recurrence plots
    gs = plt.GridSpec(2, n_bands, height_ratios=[1, 4], hspace=0.1, wspace=0.2)

    # Lists to store RecurrencePlot objects and metrics
    rp_objects = []
    all_metrics = []

    # Scale for color normalization across all plots
    vmin, vmax = 0, 1

    # First pass - compute all recurrence matrices and determine metrics
    for i in range(n_bands):
        # Get data for this band
        data_segment = band_data[i, start_idx:end_idx]

        # Create RecurrencePlot object
        if metric.lower() == "meandist":
            from scipy.spatial.distance import pdist

            distances = pdist(data_segment.reshape(-1, 1), metric="euclidean")
            mean_dist = np.mean(distances)
            normalized_threshold = threshold / mean_dist
            actual_threshold = normalized_threshold
            actual_metric = "euclidean"
        else:
            actual_threshold = threshold
            actual_metric = metric

        rp = RecurrencePlot(
            data_segment,
            dim=dim,
            tau=tau,
            threshold=actual_threshold,
            metric=actual_metric,
            normalize=False,
        )
        rp_objects.append(rp)

        # Compute metrics
        metrics = {
            "Recurrence Rate": rp.recurrence_rate(),
            "Determinism": rp.determinism(l_min=2),
            "Laminarity": rp.laminarity(v_min=2),
            "Trapping Time": rp.trapping_time(v_min=2),
            "L_max": rp.max_diaglength(),
            "L_mean": rp.average_diaglength(l_min=2),
            "Entropy": rp.diag_entropy(l_min=2),
        }
        all_metrics.append(metrics)

    # Second pass - create the plots
    for i in range(n_bands):
        rp = rp_objects[i]
        data_segment = band_data[i, start_idx:end_idx]
        rec_matrix = rp.recurrence_matrix()

        # Time series in top row
        ax_top = fig.add_subplot(gs[0, i])
        ax_top.plot(time, data_segment, "k-", linewidth=0.8)
        ax_top.set_title(f"{band_labels[i]}", fontsize=12)
        ax_top.set_xlim(time[0], time[-1])

        # Turn off x ticks for top plots except the first one
        if i > 0:
            ax_top.set_yticklabels([])
        else:
            ax_top.set_ylabel("Amplitude")

        # Hide x ticks for top row
        ax_top.set_xticklabels([])

        # Recurrence plot in bottom row
        ax_rp = fig.add_subplot(gs[1, i])
        im = ax_rp.imshow(
            rec_matrix,
            cmap="binary",
            origin="lower",
            extent=[time[0], time[-1], time[0], time[-1]],  # type: ignore
            vmin=vmin,
            vmax=vmax,
        )

        # Label only the leftmost recurrence plot
        if i == 0:
            ax_rp.set_ylabel("Time (s)")
        else:
            ax_rp.set_yticklabels([])

        ax_rp.set_xlabel("Time (s)")

        # Add metrics as text in the bottom right corner
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in all_metrics[i].items()])
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
        ax_rp.text(
            0.95,
            0.05,
            metrics_text,
            transform=ax_rp.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=bbox_props,
        )

    plt.suptitle("EEG Frequency Band Recurrence Plots", fontsize=14)
    plt.tight_layout()
    # Save if a path is provided
    if save_path:
        full_path = f"{save_path}.{save_format}" if "." not in save_path else save_path  # type: ignore
        plt.savefig(full_path, format=save_format, bbox_inches="tight", dpi=300)

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_metrics


# Example usage:
def display_band_rqa(
    data: np.ndarray,
    patient_idx: int,
    channel_name: str,
    time_range: Tuple[int, int],
    save_path: str = "",
) -> List[Dict[str, float]]:
    """
    Display RQA plots for all frequency bands of a specific patient and channel.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_electrodes, n_samples, n_bands)
    patient_idx : int
        Index of the patient to visualize
    channel_name : str
        Name of the electrode to visualize
    time_range : Tuple[int, int]
        Range of samples to visualize (start_idx, end_idx)

    Returns:
    --------
    List[Dict[str, float]]
        RQA metrics for each frequency band
    """
    # Channel names list
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

    band_labels = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-12 Hz)",
        "Beta (12-30 Hz)",
        "Gamma (30-80 Hz)",
    ]

    # Find electrode index
    try:
        electrode_idx = ch_names.index(channel_name)
    except ValueError:
        raise ValueError(
            f"Channel {channel_name} not found. Available channels: {', '.join(ch_names)}"
        )

    start_idx, end_idx = time_range

    # Extract data and reshape to (n_bands, n_samples)
    band_data = data[patient_idx, electrode_idx, start_idx:end_idx, :].T

    # Call the multiband RQA plotting function
    metrics = plot_multiband_rqa(
        band_data=band_data,
        time_range=(0, end_idx - start_idx),  # Relative to extracted segment
        sample_rate=1000.0,
        dim=10,
        tau=1,
        threshold=0.5,
        metric="meandist",
        band_labels=band_labels,
        save_format="svg",
        save_path=save_path,
    )

    return metrics


def main() -> None:
    ROOT_PROCESSED = "/home/mariopasc/Python/Datasets/EEG/timeseries/processed"
    data_ct = np.load(os.path.join(ROOT_PROCESSED, "CT_UP_preprocess_2.npz"))["data"]
    data_dd = np.load(os.path.join(ROOT_PROCESSED, "DD_UP_preprocess_2.npz"))["data"]

    # Example call:
    metrics = display_band_rqa(
        data_dd,
        patient_idx=0,
        channel_name="T7",
        time_range=(000, 100),
        save_path="./rqe/results/ct_t7_rqa.svg",
    )
