#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for Dyslexia-EEG visualisation scripts.

Change-log
----------
2025-05-14  •   add `hop_ms` parameter (defaults to window_ms / 2 for 50 % overlap)
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from typing import Tuple, Dict, Any


def load_npz(path: str | Path) -> Dict[str, Any]:
    """Load a ``np.savez_compressed`` file with pickle objects handled safely."""
    arr = np.load(Path(path), allow_pickle=True)
    return {k: (v.tolist() if v.dtype == "O" else v) for k, v in arr.items()}


def time_axis(
    n_windows: int,
    window_label: str,
    *,
    hop_ms: int | None = None,
) -> np.ndarray:
    """
    Map window index → *centre time* (seconds).

    Parameters
    ----------
    n_windows
        Total number of extracted windows.
    window_label
        String like ``"750"`` or ``"750ms"``.  The first integer found
        is interpreted as **window length in ms**.
    hop_ms
        Hop size in ms.  *If ``None`` (default) → 50 % overlap => ``window_ms / 2``.*

    Returns
    -------
    1-D array of size *(n_windows,)* giving the time-stamp at the **centre**
    of each window, expressed in seconds.
    """
    window_ms = int(re.search(r"\d+", window_label).group())  # type: ignore
    hop = hop_ms or window_ms // 2
    centres_ms = np.arange(n_windows) * hop + window_ms / 2
    return centres_ms / 1_000  # --> seconds
