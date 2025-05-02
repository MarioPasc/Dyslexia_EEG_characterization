# -*- coding: utf-8 -*-
"""
Second-level statistics on selector outputs / feature importances.
"""
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from scipy.stats import binomtest, ttest_1samp
from statsmodels.stats.multitest import multipletests


def aggregate_selector_stats(
    selector_stats: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Fold-wise selector outputs → grand means.

    Returns
    -------
    mean_score, mean_p_value, selection_frequency
    """
    scr = selector_stats["scores"]  # (fold × window × feat)
    pvl = selector_stats["p_values"]
    msk = selector_stats["mask"]

    return dict(
        mean_score=np.nanmean(scr, 0),
        mean_p_value=np.nanmean(pvl, 0),
        selection_frequency=msk.mean(0),
    )


def feature_selection_binomial_test(
    selector_stats: Dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
) -> Dict[str, Any]:
    """
    Binomial test: is a feature selected more often than *chance*
    across folds × windows?
    """
    msk = selector_stats["mask"]
    n_trials = np.prod(msk.shape[:2])  # folds × windows
    k = msk.shape[2]

    succ = msk.sum((0, 1))
    p_vals = np.array(
        [binomtest(k=x, n=n_trials, p=1 / k, alternative="greater") for x in succ]
    )
    reject, p_fdr, _, _ = multipletests(p_vals, alpha=alpha, method=fdr_method)

    return dict(
        successes=succ,
        n_trials=n_trials,
        sel_frequency=succ / n_trials,
        p_values=p_vals,
        p_values_fdr=p_fdr,
        significant_mask=reject,
        alpha=alpha,
        fdr_method=fdr_method,
    )


def feature_score_ttest(
    selector_stats: Dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
) -> Dict[str, Any]:
    """
    One-sample t-test of *feature scores* > 0 across all folds+windows.
    """
    scores = selector_stats["scores"].reshape(-1, selector_stats["scores"].shape[-1])
    t_stats, p_vals = ttest_1samp(
        scores, popmean=0.0, axis=0, nan_policy="omit", alternative="greater"
    )
    reject, p_fdr, _, _ = multipletests(p_vals, alpha=alpha, method=fdr_method)
    return dict(
        t_stats=t_stats,
        p_values=p_vals,
        p_values_fdr=p_fdr,
        significant_mask=reject,
        alpha=alpha,
        fdr_method=fdr_method,
    )
