import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def synthetic_signal_gen(
    characteristics: Dict[Tuple[int, int], Dict[str, float]],
    base_mean: float = 0.0,
    base_std: float = 1.0,
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
    original_sig = np.random.normal(loc=base_mean, scale=base_std, size=len(t))

    # 4) Initialize arrays for the final signal, piecewise means, and piecewise variances
    final_sig = np.zeros_like(original_sig)
    piecewise_means = np.zeros_like(original_sig)
    piecewise_vars = np.zeros_like(original_sig)

    # 5) For each segment, scale and shift the original signal
    for (start, end), params in characteristics.items():
        mask = (t >= start) & (t < end)
        seg_mean = params["mean"]
        seg_std = params["std"]

        # piecewise mean, piecewise variance
        piecewise_means[mask] = seg_mean
        piecewise_vars[mask] = seg_std**2

        # final signal: e * seg_std + seg_mean
        final_sig[mask] = original_sig[mask] * seg_std + seg_mean

    return t, original_sig, piecewise_means, piecewise_vars, final_sig


def plot_four_panel(
    t: np.ndarray,
    original_sig: np.ndarray,
    piecewise_means: np.ndarray,
    piecewise_vars: np.ndarray,
    final_sig: np.ndarray,
):
    """
    Create a 2×2 figure, clockwise:
      1. Original signal (top-left)
      2. Piecewise means (top-right)
      3. Piecewise variances (bottom-left)
      4. Final signal (bottom-right)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Clockwise: Original, Mean Jumps, Variances, and Final Signal")

    # Top-left: Original signal
    axes[0, 0].plot(t, original_sig, color="C0")
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Amplitude")

    # Top-right: Piecewise means
    axes[0, 1].plot(t, piecewise_means, color="C1")
    axes[0, 1].set_title("Means")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Mean Value")

    # Bottom-left: Piecewise variances
    axes[1, 0].plot(t, piecewise_vars, color="C2")
    axes[1, 0].set_title("Variances")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Variance")

    # Bottom-right: Final signal
    axes[1, 1].plot(t, final_sig, color="C3")
    axes[1, 1].set_title("Final Signal")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    t, orig_sig, means, variances, final_sig = synthetic_signal_gen(
        characteristics_example
    )

    # Plot the 2×2 figure
    plot_four_panel(t, orig_sig, means, variances, final_sig)
