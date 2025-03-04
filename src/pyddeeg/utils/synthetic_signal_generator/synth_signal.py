#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def synthetic_signal_gen(PDF, characteristics):
    """
    Generate an artificial signal over a specified time range using a given statistical distribution.

    Parameters:
        PDF (scipy.stats distribution): The probability distribution function to sample from.
        characteristics (dict): A dictionary where each key is a tuple (start, end) representing
            a time interval, and each value is a dict containing the parameters for the distribution,
            e.g., {'mean': value, 'std': value}.

    Returns:
        t (numpy.ndarray): Array of time stamps.
        signal (numpy.ndarray): Array of generated signal values corresponding to each time stamp.
    """
    # Determine the overall time span from the provided intervals.
    global_start = min(interval[0] for interval in characteristics.keys())
    global_end = max(interval[1] for interval in characteristics.keys())

    # Create a time array assuming integer steps; adjust if needed for your application.
    t = np.arange(global_start, global_end)
    signal = np.empty_like(t, dtype=float)

    # Loop over each interval and generate samples from the PDF with the given parameters.
    for (start, end), params in characteristics.items():
        # Create a mask for time values that fall within the current interval.
        mask = (t >= start) & (t < end)
        num_samples = np.sum(mask)
        # Generate samples using PDF.rvs with the provided 'mean' (loc) and 'std' (scale).
        samples = PDF.rvs(loc=params["mean"], scale=params["std"], size=num_samples)
        # Assign the generated samples to the corresponding positions in the signal array.
        signal[mask] = samples

    return t, signal


# Example usage:
if __name__ == "__main__":
    # Define the characteristics: time intervals and corresponding distribution parameters.
    characteristics = {
        (0, 100): {"mean": 1, "std": 2},
        (100, 200): {"mean": 0.5, "std": 0.5},
    }

    # Generate the synthetic signal using scipy.stats.norm as the PDF.
    t, signal = synthetic_signal_gen(stats.norm, characteristics)

    # Optionally, plot the synthetic signal.
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Synthetic Signal")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.title("Synthetic Signal Generation")
    plt.legend()
    plt.show()
