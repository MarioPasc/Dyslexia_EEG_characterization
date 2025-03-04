#!/usr/bin/env python3
"""
Simulate a synthetic signal with Time-Varying Characteristics.

This module provides a function, `synthetic_signal_gen` which provides the functioanlity to generate a synthetic signal
which follows a given distribution. The final output is a time array and the corresponding signal array.

Example usage:
    characteristics_example = {
        (0, 100): {"mean": 0, "std": 1.0},
        (101, 140): {"mean": 1, "std": 1},
        (141, 200): {"mean": 1, "std": 4},
        (201, 280): {"mean": 4, "std": 4},
        (281, 300): {"mean": 4, "std": 6},
        (301, 400): {"mean": -5, "std": 6},
        (401, 420): {"mean": 2, "std": 6},
        (421, 500): {"mean": 2, "std": 1},
        (501, 560): {"mean": 0, "std": 1},
        (561, 600): {"mean": 0, "std": 3},
        (601, 700): {"mean": 1, "std": 3},
    }

    # Generate signals
    t, orig_sig, means, variances, final_sig = synthetic_signal_gen(
        characteristics_example
    )
"""

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from typing import Dict, Tuple
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])


def synthetic_signal_gen(
    characteristics: Dict[Tuple[int, int], Dict[str, float]],
    base_mean: float = 0.0,
    base_std: float = 1.0,
    pdf_function=stats.norm,
):
    """
    Generate a synthetic signal by first creating a single original random signal e ~ N(base_mean, base_std^2)
    over the entire time span, then applying piecewise mean and standard deviation from the 'characteristics'.

    Parameters:
        characteristics: A dictionary where each key is a tuple (start, end) representing
                        a time interval [start, end), and each value is a dict:
                            {'mean': float, 'std': float}
        base_mean: Mean of the baseline normal distribution for the original signal (default 0.0)
        base_std: Std. dev. of the baseline normal distribution for the original signal (default 1.0)

    Returns:
        t:             1D array of time stamps (integers).
        original_sig:  1D array of the original random signal e.
        piecewise_means: 1D array of piecewise means (for each time sample).
        piecewise_vars:  1D array of piecewise variances (std^2 for each time sample).
        final_sig:     1D array of the final signal after piecewise perturbation.
    """
    # 1) Determine the global start and end time from characteristics
    global_start = min(interval[0] for interval in characteristics.keys())
    global_end = max(interval[1] for interval in characteristics.keys())

    # 2) Create a time array (integer steps)
    t = np.arange(global_start, global_end)

    # 3) Generate the original random signal e ~ N(base_mean, base_std^2) over the entire time span
    original_sig = pdf_function.rvs(loc=base_mean, scale=base_std, size=len(t))

    # 4) Initialize arrays for the final signal, piecewise means, and piecewise variances
    final_sig = np.zeros_like(original_sig)
    piecewise_means = np.zeros_like(original_sig)
    piecewise_vars = np.zeros_like(original_sig)
    mean_only_sig = np.zeros_like(original_sig)
    std_only_sig = np.zeros_like(original_sig)

    results = {}

    # 5) For each segment, scale and shift the original signal
    for (start, end), params in characteristics.items():
        mask = (t >= start) & (t < end)
        seg_mean = params["mean"]
        seg_std = params["std"]

        # piecewise mean, piecewise variance
        piecewise_means[mask] = seg_mean
        piecewise_vars[mask] = seg_std

        # Only mean
        mean_only_sig[mask] = original_sig[mask] + seg_mean

        # Only std
        std_only_sig[mask] = original_sig[mask] * seg_std

        # final signal: e * seg_std + seg_mean
        final_sig[mask] = original_sig[mask] * seg_std + seg_mean

        results[f"{start},{end}"] = {
            "piecewise_means": seg_mean,
            "piecewise_vars": seg_std,
            "mean_only_sig": original_sig[mask] + seg_mean,
            "std_only_sig": original_sig[mask] * seg_std,
            "final_sig": original_sig[mask] * seg_std + seg_mean,
        }

    return t, original_sig, final_sig, results


def plot_piecewise_mean_std(
    t: np.ndarray, original_sig: np.ndarray, results: dict, save=None
):
    """
    Reconstruct the 'mean_only_sig' and 'std_only_sig' from the piecewise 'results' dict,
    then overlay their piecewise lines (means and stds).

    Creates two subplots:
      - Top:    Mean-only signal + piecewise means
      - Bottom: Std-only signal + piecewise std
    """
    # Prepare empty arrays to hold the reconstructed signals
    mean_only_sig = np.zeros_like(original_sig)
    std_only_sig = np.zeros_like(original_sig)
    piecewise_means = np.zeros_like(original_sig)
    piecewise_stds = np.zeros_like(original_sig)
    final_signal = np.zeros_like(original_sig)

    # Rebuild continuous arrays from 'results'
    for key, val in results.items():
        # Parse "start,end"
        start_str, end_str = key.split(",")
        start, end = int(start_str), int(end_str)
        mask = (t >= start) & (t < end)

        mean_only_sig[mask] = val["mean_only_sig"]
        std_only_sig[mask] = val["std_only_sig"]

        # 'piecewise_means' and 'piecewise_stds' are constants in each interval
        piecewise_means[mask] = val["piecewise_means"]
        piecewise_stds[mask] = val["piecewise_vars"]  # Actually the std

        final_signal[mask] = val["final_sig"]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    ax_original = axes[0, 0]
    ax_original.plot(t, original_sig, label="Original signal", color="C0")
    ax_original.set_title("Original Signal")
    ax_original.set_ylabel("Amplitude")
    ax_original.legend()

    ax_final = axes[1, 1]
    ax_final.plot(t, final_signal, label="Final signal", color="C0")
    ax_final.set_title("Final Signal")
    ax_final.set_xlabel("Time")
    ax_final.legend()

    ax_mean = axes[0, 1]
    # 1) Mean-only subplot
    ax_mean.plot(t, mean_only_sig, label="Mean-Only Signal", color="C1")
    # Overlay the piecewise means (as a step or line)
    ax_mean.plot(t, piecewise_means, label="Piecewise Mean", color="red", linewidth=2)
    ax_mean.set_title("Mean-Only Signal with Piecewise Means")
    ax_mean.legend()

    ax_std = axes[1, 0]
    # 2) Std-only subplot
    ax_std.plot(t, std_only_sig, label="Std-Only Signal", color="C1")
    # Overlay piecewise std
    ax_std.plot(t, piecewise_stds, label="Piecewise Std", color="red", linewidth=2)
    ax_std.set_title("Std-Only Signal with Piecewise Std")
    ax_std.set_xlabel("Time")
    ax_std.set_ylabel("Amplitude")
    ax_std.legend()

    plt.tight_layout()
    if save:
        plt.savefig(save, format="svg")
    plt.show()


if __name__ == "__main__":
    # Example usage with the table from your reference (time intervals in integer steps).
    characteristics_example = {
        (0, 100): {"mean": 0, "std": 1.0},
        (101, 140): {"mean": 1, "std": 1},
        (141, 200): {"mean": 1, "std": 4},
        (201, 280): {"mean": 4, "std": 4},
        (281, 300): {"mean": 4, "std": 6},
        (301, 400): {"mean": -5, "std": 6},
        (401, 420): {"mean": 2, "std": 6},
        (421, 500): {"mean": 2, "std": 1},
        (501, 560): {"mean": 0, "std": 1},
        (561, 600): {"mean": 0, "std": 3},
        (601, 700): {"mean": 1, "std": 3},
    }

    # Generate signals
    t, orig_sig, final_sig, results = synthetic_signal_gen(
        characteristics=characteristics_example,  # type: ignore
        base_mean=0,
        base_std=0.5,
        pdf_function=stats.norm,
    )
