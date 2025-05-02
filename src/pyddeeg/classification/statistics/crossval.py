# -*- coding: utf-8 -*-
"""
5×2-fold consistency: t-test of AUC > 0.5 across outer folds.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


def cv_consistency_test(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    outer_indices: List[Tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    decision_scores : (subjects, windows)
    labels          : (subjects,)
    outer_indices   : list of (train_idx, test_idx) from ``nested_evaluate``

    Returns
    -------
    dict with per-fold AUC, one-sample t-stats, FDR-corrected p, …
    """
    n_folds = len(outer_indices)
    n_win = decision_scores.shape[1]
    auc_fold = np.empty((n_folds, n_win))

    for i, (_, te) in enumerate(outer_indices):
        for w in range(n_win):
            auc_fold[i, w] = roc_auc_score(labels[te], decision_scores[te, w])

    t_stats, p_vals = ttest_1samp(auc_fold, popmean=0.5, axis=0, alternative="greater")
    reject, p_fdr, _, _ = multipletests(p_vals, alpha=alpha, method=fdr_method)

    return dict(
        auc_per_fold=auc_fold,
        mean_auc=auc_fold.mean(0),
        var_auc=auc_fold.var(0, ddof=1),
        t_stats=t_stats,
        p_values=p_vals,
        p_values_fdr=p_fdr,
        significant_mask=reject,
        alpha=alpha,
        fdr_method=fdr_method,
    )
