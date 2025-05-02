# m_tuning.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from nolitsa import dimension as ndim


def _pick_best_m(fnn_frac: np.ndarray, threshold: float = 0.01) -> int:
    """
    Pick the smallest *m* whose overall FNN fraction drops below *threshold*.
    Falls back to ``np.argmin(fnn_frac)`` if the threshold is never crossed.
    """
    idx = np.where(fnn_frac < threshold)[0]
    return int(idx[0] + 1) if idx.size else int(np.argmin(fnn_frac) + 1)


def scan_m_for_window(
    window: np.ndarray,
    *,
    tau: int = 1,
    max_dim: int = 10,
    R: float = 10.0,
    A: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute FNN curves for **one** 1-D window.

    Parameters
    ----------
    window
        Signal segment (shape ``[samples]``).
    tau
        Previously chosen delay.
    max_dim
        Largest embedding dimension to test.
    R, A
        Kennel et al. FNN tolerances (Test I and Test II).

    Returns
    -------
    dims
        Dimensions tested (1…max_dim).
    f1, f2, f3
        Fractions from Nolitsa (Test I, Test II, Test I ∨ II).
    best_m
        Dimension selected by the < 1 % rule (or global minimum fallback).
    """
    if window.ndim != 1:
        raise ValueError("window must be 1-D")

    dims = np.arange(1, max_dim + 1)
    f1, f2, f3 = ndim.fnn(window, dim=dims, tau=tau, R=R, A=A)
    best_m = _pick_best_m(f3)
    return dims, f1, f2, f3, best_m


def tune_m_per_window(
    windows: np.ndarray,
    *,
    taus: np.ndarray | int = 1,
    max_dim: int = 10,
    R: float = 10.0,
    A: float = 2.0,
) -> List[Dict[str, np.ndarray | int]]:
    """
    Scan *m* for **each** window (shape ``[n_win, n_samples]``).

    Parameters
    ----------
    windows
        Matrix of windows.
    taus
        Either a scalar τ applied to every window **or** a 1-D array of
        τ values (one per window) you obtained from the MI step.
    max_dim, R, A
        Passed to :func:`scan_m_for_window`.

    Returns
    -------
    summaries
        ``[{ 'dims': dims, 'f1': f1, 'f2': f2, 'f3': f3, 'best_m': m }, …]``
        Length equals ``n_win``.
    """
    if windows.ndim != 2:
        raise ValueError("windows must be 2-D (n_windows × samples)")

    if np.isscalar(taus):
        taus = np.full(windows.shape[0], taus, dtype=int)

    if len(taus) != windows.shape[0]:
        raise ValueError("taus must have the same length as n_windows")

    summaries: List[Dict[str, np.ndarray | int]] = []
    for w, t in zip(windows, taus):
        dims, f1, f2, f3, best = scan_m_for_window(
            w,
            tau=int(t),
            max_dim=max_dim,
            R=R,
            A=A,
        )
        summaries.append({"dims": dims, "f1": f1, "f2": f2, "f3": f3, "best_m": best})
    return summaries
