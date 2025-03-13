# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def zerolag_bpfir2(
    sig: np.ndarray,
    fc: float,
    f0: float,
    f1: float,
    ncyc: int = 5,
    fs: int = 500,
    zerolag: bool = True,
    f_step: float = 0.5,
    show_response: bool = False,
) -> np.ndarray:
    """
    Zero-lag bandpass FIR filter.

    Two-way zero-phase lag finite impulse response (FIR) Least-Squares filter.
    Zero phase is achieved by passing the signal forward and backward through the filter.

    Args:
        sig: Signal to be filtered
        fc: Center frequency
        f0: Lower cutoff frequency
        f1: Upper cutoff frequency
        ncyc: Number of cycles (recommended 3 for low frequencies, 6 for high frequencies)
        fs: Sampling frequency
        zerolag: Whether to apply zero-lag filtering (forward and backward)
        f_step: Bandpass granularity
        show_response: Whether to plot filter frequency response

    Returns:
        Filtered signal
    """
    # Calculate filter order and number of taps
    order = ncyc * fs / fc
    ntaps = np.int32(np.ceil(order + 1) // 2 * 2 + 1)  # To get an odd number of taps

    # Define frequency bands
    interest_band = np.arange(0, fs / 2, f_step)
    start_idx = np.where(interest_band == f0)[0][0]
    stop_idx = np.where(interest_band == f1)[0][0]

    # Create desired response
    desired = np.hstack(
        (
            np.zeros(start_idx),
            np.ones(stop_idx - start_idx),
            np.zeros(len(interest_band) - stop_idx),
        )
    )

    # Design filter
    firfil = signal.firls(ntaps, interest_band, desired, fs=fs)

    # First - forward - pass
    filtered = signal.lfilter(firfil, 1, sig)

    if zerolag:
        # Second - backward - pass -> this nulls the lag introduced by the FIR filter!
        filtered = signal.lfilter(firfil, 1, filtered[::-1])[::-1]

    if show_response:
        # Show frequency response if required
        fig, ax = plt.subplots()
        w, response = signal.freqz(firfil)
        ax.plot((w / np.pi) * fs, 20 * np.log10(abs(response)), "b")
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.grid(True)
        ax.set(title=f"Band-pass {f0}-{f1} Hz\nfc={fc:.2f} Hz")
        fig.tight_layout()
        plt.savefig(f"../assets/bandpass_{f0}_{f1}.svg")

    return filtered
