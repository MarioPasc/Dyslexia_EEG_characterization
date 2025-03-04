#!/usr/bin/env python3
"""
Simulate EEG Data with Time-Varying Characteristics.

This module provides a function, `synthetic_eeg`, which generates a synthetic single-channel EEG signal.
The signal is created over a global time interval defined by a dictionary where keys are tuples representing
time intervals (in seconds) and values are dictionaries with parameters for that segment:
    - 'amplitude': Amplitude of the sine wave.
    - 'frequency': Frequency of the sine wave in Hz.
    - 'noise_std': Standard deviation of the additive Gaussian noise.
    - 'phase': (Optional) Phase of the sine wave in radians (default is 0.0).

The generated signal is the sum of a sine wave and Gaussian noise for each interval. The final output is a
time array and the corresponding EEG signal array.

Example usage:
    characteristics = {
        (0, 10): {'amplitude': 50, 'frequency': 10, 'noise_std': 5},
        (10, 20): {'amplitude': 30, 'frequency': 8, 'noise_std': 3, 'phase': 0.5},
        (20, 30): {'amplitude': 40, 'frequency': 12, 'noise_std': 4},
    }
    t, eeg_signal = simulate_eeg(characteristics, sampling_rate=250.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def synthetic_eeg(
    characteristics: Dict[Tuple[float, float], Dict[str, float]],
    sampling_rate: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single-channel EEG signal with time-varying characteristics.

    For each interval specified in the `characteristics` dictionary, the function generates a sine wave with
    the specified amplitude, frequency, and phase. It then adds Gaussian noise with the provided standard deviation.

    Parameters:
        characteristics (Dict[Tuple[float, float], Dict[str, float]]):
            A dictionary where each key is a tuple (start_time, end_time) in seconds, and each value is a dictionary
            containing the parameters for that interval:
                - 'amplitude': Amplitude of the sine wave.
                - 'frequency': Frequency of the sine wave in Hz.
                - 'noise_std': Standard deviation of the Gaussian noise.
                - 'phase': (Optional) Phase of the sine wave in radians (default is 0.0 if not provided).
        sampling_rate (float): The sampling rate in Hz (samples per second). Default is 250 Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - t: A 1D NumPy array of time stamps (in seconds).
            - eeg_signal: A 1D NumPy array of simulated EEG signal values.
    """
    # Determine the overall time span from the dictionary keys.
    global_start: float = min(interval[0] for interval in characteristics.keys())
    global_end: float = max(interval[1] for interval in characteristics.keys())

    # Create a time vector with steps of 1/sampling_rate seconds.
    t: np.ndarray = np.arange(global_start, global_end, 1 / sampling_rate)
    eeg_signal: np.ndarray = np.empty_like(t)

    # Generate the signal for each specified time interval.
    for (start, end), params in characteristics.items():
        # Get phase parameter, defaulting to 0.0 if not provided.
        phase: float = params.get("phase", 0.0)
        # Find indices corresponding to the current interval.
        mask: np.ndarray = (t >= start) & (t < end)
        n_samples: int = np.sum(mask)
        # Extract parameters for the sine wave and noise.
        amplitude: float = params["amplitude"]
        frequency: float = params["frequency"]
        noise_std: float = params["noise_std"]
        # Time vector segment for the current interval.
        t_segment: np.ndarray = t[mask]
        # Generate sine wave and noise.
        sine_wave: np.ndarray = amplitude * np.sin(
            2 * np.pi * frequency * t_segment + phase
        )
        noise: np.ndarray = np.random.normal(loc=0.0, scale=noise_std, size=n_samples)
        # Combine sine wave and noise.
        eeg_signal[mask] = sine_wave + noise

    return t, eeg_signal


def main() -> None:
    """
    Main function to demonstrate the EEG signal simulation.
    """
    # Define time-varying characteristics for the simulated EEG signal.
    characteristics: Dict[Tuple[float, float], Dict[str, float]] = {
        (0, 10): {"amplitude": 4, "frequency": 4, "noise_std": 5},
        (10, 20): {"amplitude": 2, "frequency": 5, "noise_std": 3, "phase": 0.5},
        (20, 30): {"amplitude": 3, "frequency": 3, "noise_std": 4},
    }
    sampling_rate: float = 100.0  # in Hz

    # Generate the EEG signal.
    t, eeg_signal = synthetic_eeg(characteristics, sampling_rate)

    # Plot the simulated EEG signal.
    plt.figure(figsize=(12, 6))
    plt.plot(t, eeg_signal, label="Simulated EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Signal")
    plt.title("Simulated EEG Signal with Changing Characteristics")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
