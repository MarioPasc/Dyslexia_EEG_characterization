# tau_tuning.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from nolitsa import delay as ndelay


def _first_local_minimum(curve: np.ndarray) -> int:
    """
    Return the index of the first local minimum in *curve*.
    If no local minimum exists, return ``np.argmin(curve)``.
    """
    for i in range(1, len(curve) - 1):
        if curve[i] < curve[i - 1] and curve[i] < curve[i + 1]:
            return i
    return int(np.argmin(curve))


def scan_tau_for_window(
    window: np.ndarray,
    *,
    max_lag: int = 100,
    bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the delayed mutual-information (MI) curve for **one** 1-D window.

    Parameters
    ----------
    window
        Signal segment (shape ``[samples]``).
    max_lag
        Upper bound for τ to test (inclusive).
    bins
        Histogram bins for MI estimation (passed to *nolitsa*).

    Returns
    -------
    taus
        Integer lags ``0 … max_lag``.
    mi_values
        MI(τ) for each lag.
    best_tau
        First local minimum of the MI curve (or global minimum fallback).
    """
    if window.ndim != 1:
        raise ValueError("window must be 1-D")

    mi_curve = ndelay.dmi(window, maxtau=max_lag, bins=bins)  # length = max_lag
    taus = np.arange(max_lag)
    best_tau = _first_local_minimum(mi_curve)
    return taus, mi_curve, best_tau


def tune_tau_per_window(
    windows: np.ndarray,
    *,
    max_lag: int = 100,
    bins: int = 64,
) -> List[Dict[str, np.ndarray | int]]:
    """
    Scan τ for **each** window in *windows* (shape ``[n_win, n_samples]``).

    Returns
    -------
    results
        ``[{'taus': np.ndarray, 'mi': np.ndarray, 'best_tau': int}, …]``
        One dictionary per window, in the same order.
    """
    if windows.ndim != 2:
        raise ValueError("windows must be 2-D (n_windows × samples)")

    summaries: List[Dict[str, np.ndarray | int]] = []
    for w in windows:
        taus, mi_vals, best = scan_tau_for_window(
            w,
            max_lag=max_lag,
            bins=bins,
        )
        summaries.append({"taus": taus, "mi": mi_vals, "best_tau": best})
    return summaries
