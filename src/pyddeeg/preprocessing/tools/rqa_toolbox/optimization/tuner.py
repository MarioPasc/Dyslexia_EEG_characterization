#!/usr/bin/env python3
"""tuner.py  ‒  Takens parameter optimiser
=========================================

This version **aligns exactly** with the implementation details of the
*Nolitsa* source code you provided (``delay.py`` & ``dimension.py``):

* **Delay (τ)** now uses ``delay.dmi`` — the *time‑delayed mutual
  information* array.  The former non‑existent ``delay.ami`` call has
  been removed.
* **Embedding dimension (m)** now harvests the *third* row returned by
  ``dimension.fnn`` (fraction false by *either* Kennel test) and works
  with the library’s required signature ``dim=[1, 2, …]``.

The public API is **unchanged**:
>>> tau, m, eps = tune_window(window)
>>> cache = tune_channel(full_signal, window_size, stride)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from nolitsa import delay, dimension
from scipy.spatial.distance import pdist
from pyddeeg.preprocessing.tools.rqa_toolbox.utils import extract_signal_windows

# ---------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------
Takens = Tuple[int, int, float]  # (tau, m, epsilon)

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _estimate_tau(signal: bytes, max_lag: int, bins: int) -> int:
    """Estimate τ via the first local minimum of delayed MI (Fraser–Swinney).

    The window is cached by its *bytes* value to avoid recomputation
    when identical windows overlap.
    """
    x = np.frombuffer(signal, dtype=np.float64)
    mi_curve = delay.dmi(x, maxtau=max_lag, bins=bins)  # length == max_lag
    for lag in range(1, len(mi_curve) - 1):
        if mi_curve[lag] < mi_curve[lag - 1] and mi_curve[lag] < mi_curve[lag + 1]:
            return lag
    return int(np.argmin(mi_curve))


def _estimate_m(signal: np.ndarray, tau: int, max_dim: int) -> int:
    """Smallest dimension whose overall false‑neighbour fraction < 1 %."""
    dims = list(range(1, max_dim + 1))
    f1, f2, f3 = dimension.fnn(signal, dim=dims, tau=tau, R=10.0, A=2.0, parallel=False)
    below = np.where(f3 < 0.01)[0]
    return int(dims[below[0]]) if below.size else int(dims[int(np.argmin(f3))])


def _estimate_eps(attractor: np.ndarray, rec_rate: float) -> float:
    dists = pdist(attractor, metric="euclidean")
    kth = int(len(dists) * rec_rate)
    return float(np.partition(dists, kth)[kth])


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def tune_window(
    window: np.ndarray,
    *,
    max_lag: int = 100,
    max_dim: int = 10,
    rec_rate: float = 0.15,
    bins: int = 64,
) -> Takens:
    """Return *(tau, m, eps)* tuned **for a single 1‑D window**."""
    if window.ndim != 1:
        raise ValueError("Window must be 1‑D")

    tau = _estimate_tau(window.tobytes(), max_lag, bins)
    m = _estimate_m(window, tau, max_dim)
    attractor = delay.utils.reconstruct(window, dim=m, tau=tau)
    eps = _estimate_eps(attractor, rec_rate)
    return tau, m, eps


def tune_channel(
    signal: np.ndarray,
    *,
    window_size: int,
    stride: int,
    max_lag: int = 100,
    max_dim: int = 10,
    rec_rate: float = 0.15,
    bins: int = 64,
) -> Dict[int, Takens]:
    """Return a dict mapping *window index* → tuned (τ, m, ε)."""
    windows = extract_signal_windows(signal, window_size, stride)
    return {
        idx: tune_window(
            w, max_lag=max_lag, max_dim=max_dim, rec_rate=rec_rate, bins=bins
        )
        for idx, w in enumerate(windows)
    }
