# -*- coding: utf-8 -*-
"""
Cluster-based permutation test across *time windows*.

Reimplements the code that used to live in nested_evaluate so that it can be
run lazily from an external results file.
"""
from __future__ import annotations
from typing import Any, Dict, Sequence

import numpy as np
from mne.stats import permutation_cluster_test


def cluster_permutation_test(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    tail: int = 1,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    decision_scores : ndarray, shape (subjects, windows)
        The per-subject cross-validated probabilities returned by
        ``nested_evaluate``.
    labels : ndarray, shape (subjects,)
        1 = DD   0 = CT
    n_perm, alpha, tail, random_state, n_jobs
        Directly forwarded to :func:`mne.stats.permutation_cluster_test`.

    Returns
    -------
    dict
        Keys (shapes depend on data):

        - ``T_obs``            – observed T statistic
        - ``clusters``         – list of index arrays
        - ``p_values``         – cluster-level p values
        - ``H0``               – null distribution
        - ``significant_mask`` – boolean mask (n_windows,)
        - ``H0_max_sizes``     – length of each cluster (H0)
        - ``alpha``            – the cluster-level alpha you passed in
    """
    dd = decision_scores[labels == 1]
    ct = decision_scores[labels == 0]

    T_obs, clusters, p_vals, H0 = permutation_cluster_test(
        [dd, ct],
        n_permutations=n_perm,
        n_jobs=n_jobs,
        tail=tail,
        seed=random_state,
    )

    n_windows = decision_scores.shape[1]
    sig_mask = np.zeros(n_windows, bool)
    for c, p in zip(clusters, p_vals):
        if p < alpha:
            sig_mask[c] = True

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "p_values": p_vals,
        "H0": H0,
        "significant_mask": sig_mask,
        "H0_max_sizes": np.array([len(c) for c in clusters]),
        "alpha": alpha,
    }
