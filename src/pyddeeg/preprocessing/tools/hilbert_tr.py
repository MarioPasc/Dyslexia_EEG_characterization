
"""pyddeeg.utils.hilbert_tr
==================================
Utility helpers to extract the *analytic signal* (instantaneous phase and
amplitude) from raw EEG traces.

The typical workflow is:
    1. (Optional) Zero‑phase band‑pass filtering in the band of interest.
    2. Apply the discrete Hilbert transform to obtain the analytic signal.
    3. Derive amplitude envelope and unwrapped instantaneous phase.

The helper is designed to be *stateless*, *JIT‑friendly* (NumPy only) and
usable inside multiprocessing frameworks such as Dask.

Example
-------
>>> from pyddeeg.utils.hilbert_tr import hilbert_transform
>>> phase, amp = hilbert_transform(signal, fs=500, band=(30, 80))
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from typing import Tuple, Sequence

__all__ = ["hilbert_transform"]

def _zero_phase_bandpass(
    x: np.ndarray,
    fs: float,
    band: Sequence[float],
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply a zero‑phase (forward–backward) Butterworth band‑pass filter.

    Parameters
    ----------
    x :
        Input signal.
    fs :
        Sampling frequency in Hz.
    band :
        2‑element sequence with *(low, high)* cut‑off frequencies in Hz.
    order :
        IIR filter order (per section).  A modest even order (4–8) is usually
        sufficient for EEG.
    axis :
        Time axis of *x*.

    Returns
    -------
    x_filt :
        Zero‑phase band‑pass filtered signal.
    """
    low, high = band
    nyq = 0.5 * fs
    if not 0 < low < high < nyq:
        raise ValueError("Invalid band limits. They must satisfy 0 < low < high < fs/2.")
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return filtfilt(sos[:, :3], sos[:, 3:], x, axis=axis, padlen=3 * max(len(s) for s in sos))

def hilbert_transform(
    x: np.ndarray,
    *,
    fs: float,
    band: Tuple[float, float] | None = None,
    filter_order: int = 4,
    axis: int = -1,
    unwrap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract instantaneous *phase* (rad) and *amplitude* from *x*.

    Parameters
    ----------
    x :
        The raw time‑series (any shape).  ``axis`` denotes the time axis.
    fs :
        Sampling frequency in Hz.
    band :
        ``(low, high)`` band‑pass limits (Hz).  If *None*, the signal is
        *not* filtered.  Use this to isolate an oscillatory band (e.g. 30–80 Hz).
    filter_order :
        Order of the Butterworth filter when *band* is provided.
    axis :
        Axis containing the time dimension.
    unwrap :
        If *True*, unwrap phase to avoid 2π discontinuities.

    Returns
    -------
    phase :
        Array with the same shape as *x* containing the (optionally unwrapped)
        instantaneous phase in radians.
    amplitude :
        Array with the same shape as *x* containing the amplitude envelope.
    """
    if band is not None:
        x_proc = _zero_phase_bandpass(x, fs, band, order=filter_order, axis=axis)
    else:
        x_proc = x

    analytic = hilbert(x_proc, axis=axis)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    if unwrap:
        phase = np.unwrap(phase, axis=axis)

    # Ensure float64 for downstream RQA
    return phase.astype(np.float64), amplitude.astype(np.float64)
