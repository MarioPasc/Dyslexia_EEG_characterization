#!/usr/bin/env python3
"""
rqe_analysis.py

Attempt to replicate the paper's two-stage logic for computing an RQE
index from a single EEG signal, using the parameters from Table 3:

Embedding = 10
Radius = 80
Line = 5
Shift = 1
Epoch = 50
Distance = Meandist, Euclidean
No. of epochs = 642  (depends on actual signal length)

Stage 1:  Compute a time series for each RQA measure by sliding a small
          window of size across the raw EEG.
Stage 2:  Define bigger "rolling windows" across the RQA-measure domain,
          compute pairwise Spearman correlations among those sub-segments,
          then do the product (1 + |rho|).
"""

import numpy as np
from typing import List, Tuple, Any
import scipy.stats as stats
import matplotlib.pyplot as plt

from pyddeeg.utils.synthetic_signal_generator.synth_signal import synthetic_signal_gen

import scienceplots

plt.style.use(["science", "ieee", "std-colors"])

try:
    from pyunicorn.timeseries import RecurrencePlot
except ImportError:
    RecurrencePlot = None
    print("[DEBUG] pyunicorn not installed. Install via: pip install pyunicorn")


###############################################################################
# Helper Functions
###############################################################################


def compute_rqa_metrics_for_window(
    window_signal: np.ndarray,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int = 2,
    min_vertical_line: int = 2,
    min_white_vertical_line: int = 1,
    metrics_to_use: List[str] = [
        "RR",
        "DET",
        "L_max",
        "L_mean",
        "ENT",
        "LAM",
        "TT",
    ],
) -> dict:
    """
    Compute only the requested RQA metrics from a single window of the raw EEG signal
    using pyunicorn's RecurrencePlot.
    """
    if RecurrencePlot is None:
        raise ImportError(
            "pyunicorn is not available. Install with pip install pyunicorn."
        )

    if metrics_to_use is None:
        metrics_to_use = ["RR", "DET", "L_max", "L_mean", "ENT", "LAM", "TT"]

    if distance_metric.lower() == "meandist":
        from scipy.spatial.distance import pdist

        distances = pdist(window_signal.reshape(-1, 1), metric="euclidean")
        mean_dist = np.mean(distances)
        normalized_radius = radius / mean_dist
        actual_radius = normalized_radius
        distance_metric = "euclidean"
    else:
        actual_radius = radius

    rp = RecurrencePlot(
        time_series=window_signal,
        dim=embedding_dim,
        tau=time_delay,
        metric=distance_metric,
        threshold=actual_radius,
        silence_level=2,
    )

    # Define metric computation functions
    metric_functions = {
        "RR": lambda: rp.recurrence_rate(),
        "DET": lambda: rp.determinism(l_min=min_diagonal_line),
        "L_max": lambda: rp.max_diaglength(),
        "L_mean": lambda: rp.average_diaglength(l_min=min_diagonal_line),
        "ENT": lambda: rp.diag_entropy(l_min=min_diagonal_line),
        "LAM": lambda: rp.laminarity(v_min=min_vertical_line),
        "TT": lambda: rp.trapping_time(v_min=min_vertical_line),
        "V_max": lambda: rp.max_vertlength(),
        "V_mean": lambda: rp.average_vertlength(v_min=min_vertical_line),
        "V_ENT": lambda: rp.vert_entropy(v_min=min_vertical_line),
        "W_max": lambda: rp.max_white_vertlength(),
        "W_mean": lambda: rp.average_white_vertlength(w_min=min_white_vertical_line),
        "W_ENT": lambda: rp.white_vert_entropy(w_min=min_white_vertical_line),
        "CLEAR": lambda: (
            rp.complexity_entropy() if hasattr(rp, "complexity_entropy") else None
        ),
        "PERM_ENT": lambda: (
            rp.permutation_entropy() if hasattr(rp, "permutation_entropy") else None
        ),
    }

    # Only compute requested metrics
    metrics = {}
    for metric in metrics_to_use:
        if metric in metric_functions:
            metrics[metric] = metric_functions[metric]()
        else:
            metrics[metric] = None  # Handle unknown metrics gracefully

    return metrics


def compute_rqa_time_series(
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
    stride: int = 1,  # If stride == 1 then we are replicating the exact RQE definition from the paper.
) -> np.ndarray:
    """
    Slide a small window of size 'raw_signal_window_size' over the raw signal, compute RQA
    metrics for each position, and return them as a 2D array of shape (T, L).
    """
    n = len(signal)
    num_positions = (n - raw_signal_window_size) // stride + 1
    rqa_results = []

    for i in range(num_positions):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + raw_signal_window_size]
        rqa_vals = compute_rqa_metrics_for_window(
            window_signal=window_data,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line,
            metrics_to_use=metrics_to_use,  # Pass the metrics list
        )

        # Extract only the metrics we asked for
        values_of_interest = []
        for key in metrics_to_use:
            values_of_interest.append(rqa_vals.get(key))
        rqa_results.append(values_of_interest)

    return np.array(rqa_results)


def compute_rqe_and_correlation(
    rqa_matrix: np.ndarray, rqa_space_window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two measures for each 'big window' in the RQA time series:
      1) RQE = product of (1 + |rho|)
      2) Corr = average of |rho|
    """
    T, L = rqa_matrix.shape
    if rqa_space_window_size > T:
        raise ValueError(
            "rqa_space_window_size cannot exceed the length of the RQA time series."
        )

    Q = T - rqa_space_window_size + 1
    rqe_list = []
    corr_list = []

    for start_idx in range(Q):
        sub_block = rqa_matrix[start_idx : start_idx + rqa_space_window_size, :]
        # shape = (rqa_space_window_size, L)
        # We'll treat each column as a sub-series of length rqa_space_window_size
        sub_block_T = sub_block.T

        # Collect pairwise correlations
        pairwise_rho = []
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                rho, _ = stats.spearmanr(sub_block_T[l1, :], sub_block_T[l2, :])
                pairwise_rho.append(abs(rho))

        if len(pairwise_rho) == 0:
            # If L < 2
            rqe_val = 1.0
            corr_val = 0.0
        else:
            # RQE = product(1 + |rho|)
            product_val = 1.0
            for c in pairwise_rho:
                product_val *= 1.0 + c
            rqe_val = product_val

            # Corr = mean(|rho|)
            corr_val = np.mean(pairwise_rho)

        rqe_list.append(rqe_val)
        corr_list.append(corr_val)

    return np.array(rqe_list), np.array(corr_list)


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    """
    Scale the values of 'array' into [0, 1] by min-max normalization.
    If all values are the same, the result is an array of zeros.
    """
    array = np.nan_to_num(x=array)

    min_val = np.min(array)
    max_val = np.max(array)

    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(array)


###############################################################################
# Main demonstration function
###############################################################################


def main(
    embedding_dim: int = 10,
    radius: Any = 80.0,
    time_delay: int = 1,
    raw_signal_window_size: int = 50,
    distance_metric: str = "euclidean",
    rqa_space_window_size: int = 100,
    simulate_length: int = 700,
    min_diagonal_line: int = 5,
    min_vertical_line: int = 1,
    min_white_vertical_line: int = 1,
    metrics_to_use: List[str] = ["RR", "DET", "L_max", "ENT", "LAM", "TT"],
    stride: int = 0,
):
    """
    Demonstration function that:
      1) Generates a synthetic EEG signal of length 'simulate_length' with variability.
      2) Computes an RQA time series using 'raw_signal_window_size' as the rolling window on the raw signal.
      3) Computes RQE & Corr for a larger window of size 'rqa_space_window_size' sliding over the RQA domain.
      4) Plots:
         - Raw EEG signal
         - RQE vs. correlation
         - Time series of each RQA metric in metrics_to_use
    """
    # ------------------------------------------------------------------
    # 1) Generate a synthetic raw signal with some variability
    characteristics = {
        (0, 100): {"mean": 0.0, "std": 1.0},
        (101, 140): {"mean": 1.0, "std": 1.0},
        (141, 200): {"mean": 1.0, "std": 4.0},
        (201, 280): {"mean": 4.0, "std": 4.0},
        (281, 300): {"mean": 4.0, "std": 6.0},
        (301, 400): {"mean": -5.0, "std": 6.0},
        (401, 420): {"mean": 2.0, "std": 6.0},
        (421, 500): {"mean": 2.0, "std": 1.0},
        (501, 560): {"mean": 0.0, "std": 1.0},
        (561, 600): {"mean": 0.0, "std": 3.0},
        (601, 700): {"mean": 1.0, "std": 3.0},
    }

    # Generate the synthetic signal using scipy.stats.norm as the PDF.
    t, orig_sig, raw_signal, results = synthetic_signal_gen(
        characteristics=characteristics,
        base_mean=0,
        base_std=0.5,
        pdf_function=stats.norm,
    )

    print("[INFO] Synthetic signal generated.")
    print(f"       Length of raw signal = {simulate_length} samples.")

    # Replace the fixed radius with a percentage of the signal's range
    if isinstance(radius, float) and radius > 1.0:
        # If radius is large, interpret as percentage of range
        signal_range = np.max(raw_signal) - np.min(raw_signal)
        actual_radius = (radius / 100.0) * signal_range
    else:
        actual_radius = radius
    radius = actual_radius
    # ------------------------------------------------------------------
    # 2) Compute RQA time series
    rqa_matrix = compute_rqa_time_series(
        signal=raw_signal,
        raw_signal_window_size=raw_signal_window_size,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        radius=radius,
        distance_metric=distance_metric,
        min_diagonal_line=min_diagonal_line,
        min_vertical_line=min_vertical_line,
        min_white_vertical_line=min_white_vertical_line,
        metrics_to_use=metrics_to_use,
        stride=stride,
    )

    window_variances = []
    for start_idx in range(rqa_matrix.shape[0]):
        window_data = raw_signal[start_idx : start_idx + raw_signal_window_size]
        window_variances.append(np.var(window_data))

    print(
        f"[INFO] Window variance range: [{min(window_variances)}, {max(window_variances)}]"
    )

    T, L = rqa_matrix.shape
    print("[INFO] RQA time series computed.")
    print(f"       RQA matrix shape = (T={T}, L={L}).")
    print(f"       -> So we have {T} time points of RQA, each with {L} metrics.")
    if T <= 1:
        print("[WARNING] T <= 1. Not enough RQA points to form correlations.")

    if L < 2:
        print(
            "[WARNING] L < 2. You only have one metric, so correlation is not meaningful."
        )

    # ------------------------------------------------------------------
    # 3) Compute RQE & Corr in the RQA domain
    #    (the 'big window' for correlation must be <= T)
    if rqa_space_window_size > T:
        print(
            f"[WARNING] rqa_space_window_size={rqa_space_window_size} is larger than T={T}. Adjusting."
        )
        rqa_space_window_size = T

    rqe_vals, corr_vals = compute_rqe_and_correlation(
        rqa_matrix, rqa_space_window_size=rqa_space_window_size
    )
    Q = len(rqe_vals)

    # Provide guidance if Q is tiny
    print(
        f"[INFO] rqa_space_window_size={rqa_space_window_size} => we get Q={Q} windows in the RQA domain."
    )
    if Q < 2:
        print(
            "[WARNING] Q < 2. You only have one 'big window', so RQE/corr won't vary over time."
        )
        print(
            "          Try decreasing rqa_space_window_size or increasing the raw signal length or raw_signal_window_size."
        )

    # Build an approximate x-axis for the RQA-based series
    # For each RQA index i, approximate mapping to raw-signal domain
    # i -> i + raw_signal_window_size/2
    # For RQE/corr, each big window is rqa_space_window_size wide, so center is i + rqa_space_window_size/2
    big_half = rqa_space_window_size / 2.0
    rqe_time = []
    for i in range(Q):
        center_rqa = i + big_half
        # Approx mapping to raw-signal time index
        approx_raw = center_rqa + raw_signal_window_size / 2.0
        if approx_raw < simulate_length:
            rqe_time.append(approx_raw)
        else:
            rqe_time.append(simulate_length - 1)
    rqe_time = np.array(rqe_time)  # type: ignore

    # Normalize RQE & Corr
    rqe_norm = min_max_normalize(rqe_vals)
    corr_norm = min_max_normalize(corr_vals)

    # ------------------------------------------------------------------
    # 4) Plot
    #    We'll create 3 subplots:
    #       (a) raw EEG
    #       (b) RQE vs. correlation
    #       (c) each RQA metric vs. time
    #    For the RQA time, we similarly define an approximate X axis.
    #    For T points, we shift them by ~ raw_signal_window_size/2 to map to raw-signal domain.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # (a) Raw EEG
    ax_signal = axes[0]
    ax_signal.set_title("Raw EEG Signal (Simulated)")
    ax_signal.plot(t, raw_signal, label="EEG Signal")
    ax_signal.set_xlabel("Time index")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.legend(loc="lower left")

    # (b) RQE (blue) vs. correlation (red)
    ax_rqe = axes[1]
    ax_rqe.set_title(f"RQE (blue) vs. Correlation (red)")
    ax_rqe.plot(rqe_time, rqe_norm, label="RQE", color="blue")
    ax_rqe.plot(rqe_time, corr_norm, label="|rho| avg", color="red")
    ax_rqe.set_xlabel("Approx. Time in raw-signal index")
    ax_rqe.set_ylabel("RQE Index Value")
    ax_rqe.legend(loc="lower left")
    ax_rqe.set_ylim([-0.1, 1.1])

    # (c) Plot each RQA metric over time
    ax_rqa = axes[2]
    ax_rqa.set_title("RQA Metrics Over Time")
    # Approx. mapping for each of the T points in the RQA domain
    rqa_time = np.arange(T) + (raw_signal_window_size / 2.0)

    for m_idx, metric_name in enumerate(metrics_to_use):
        ax_rqa.plot(rqa_time, rqa_matrix[:, m_idx], label=metric_name)

    ax_rqa.set_xlabel("Approx. Time in raw-signal index")
    ax_rqa.set_ylabel("RQA Value")
    ax_rqa.legend(loc="best")

    plt.tight_layout()
    plt.savefig("scripts/rqe_analysis_with_metrics.svg")
    plt.show()


# ------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        embedding_dim=10,  # "Embedding"
        radius=70.0,  # "Radius"
        time_delay=1,  # "Shift"
        raw_signal_window_size=50,  # "Epoch"
        distance_metric="meandist",  # "Distance"
        min_diagonal_line=5,  # "Line"
        min_vertical_line=1,
        min_white_vertical_line=1,
        rqa_space_window_size=55,  # Big window for correlation in RQA domain
        simulate_length=700,  # Enough length to produce some variety
        stride=1,
    )
