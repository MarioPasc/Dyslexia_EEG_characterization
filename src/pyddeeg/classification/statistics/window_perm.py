# -*- coding: utf-8 -*-
"""
Per-window permutation test on ROC-AUC with FDR control.
"""
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


def per_window_auc_test(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int = 1_000,
    alpha: float = 0.05,
    tail: str = "greater",
    fdr_method: str = "fdr_bh",
    random_state: int | None = None,
) -> Dict[str, Any]:
    """
    One-sample permutation test for ROC-AUC against chance (0.5) at **each**
    time window.

    Returns
    -------
    dict
        ``auc_obs``            – observed AUC (n_windows,)
        ``p_values``           – uncorrected p (n_windows,)
        ``p_values_fdr``       – FDR-corrected p
        ``significant_mask``   – bool mask after FDR
        ``null_distribution``  – shape (n_perm, n_windows)
    """
    rng = RandomState(random_state)
    n_subj, n_win = decision_scores.shape

    # observed AUC
    auc_obs = np.array(
        [roc_auc_score(labels, decision_scores[:, w]) for w in range(n_win)]
    )

    # null distribution
    null_dist = np.zeros((n_perm, n_win))
    for p in range(n_perm):
        perm_y = rng.permutation(labels)
        null_dist[p] = [
            roc_auc_score(perm_y, decision_scores[:, w]) for w in range(n_win)
        ]

    if tail == "greater":
        p_vals = ((null_dist >= auc_obs).sum(0) + 1) / (n_perm + 1)
    elif tail == "two-sided":
        eff_obs = np.abs(auc_obs - 0.5)
        eff_null = np.abs(null_dist - 0.5)
        p_vals = ((eff_null >= eff_obs).sum(0) + 1) / (n_perm + 1)
    else:  # pragma: no cover
        raise ValueError('tail must be "greater" or "two-sided"')

    reject, p_fdr, _, _ = multipletests(p_vals, alpha=alpha, method=fdr_method)

    return dict(
        auc_obs=auc_obs,
        p_values=p_vals,
        p_values_fdr=p_fdr,
        significant_mask=reject,
        null_distribution=null_dist,
        alpha=alpha,
        fdr_method=fdr_method,
    )
