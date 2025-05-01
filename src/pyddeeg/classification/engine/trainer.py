# -*- coding: utf-8 -*-
"""
trainer.py – nested k × h CV with on-the-fly permutation test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from mne.stats import permutation_cluster_test

from pyddeeg import RQA_METRICS  # ▸ list[str] with metric names
from pyddeeg.classification.optimization.tuner import tune_one_electrode_parallel
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification import MultiWindowEstimator

__all__: Sequence[str] = ("nested_evaluate",)


# ------------------------------------------------------------------ #
#                          helper utilities                          #
# ------------------------------------------------------------------ #
def _slice_tensors(
    dd: np.ndarray,
    ct: np.ndarray,
    global_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dd/ct slices given *global* subject indices."""
    n_dd = len(dd)
    dd_idx = global_idx[global_idx < n_dd]
    ct_idx = global_idx[global_idx >= n_dd] - n_dd
    return dd[dd_idx], ct[ct_idx]


def _selector_stats(pipe) -> Dict[str, Any]:
    """
    Extract *scores*, *p-values* and *boolean mask* from a fitted pipeline.

    Returns
    -------
    dict with keys ``scores``, ``p_values``, ``mask``
    """
    sel = pipe.named_steps["selector"]
    # SelectKBest exposes .scores_ / .pvalues_.  Fallback to None if absent.
    scores = getattr(sel, "scores_", None)
    p_vals = getattr(sel, "pvalues_", None)
    mask = sel.get_support()
    return dict(scores=scores, p_values=p_vals, mask=mask)


# ------------------------------------------------------------------ #
#                           public driver                            #
# ------------------------------------------------------------------ #
def nested_evaluate(
    *,
    dataset: EEGDataset,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    base_estimator,
    selector_cfg_path: str | Path,
    optuna_configuration: str | Path | Dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs_tuner: int = 1,
    n_jobs_windows: int = 1,
    storage_dir: str | Path | None = None,
    n_perm: int = 10_000,
) -> Dict[str, Any]:
    """
    End-to-end k × h nested CV **plus** cluster-based permutation test.

    Parameters
    ----------
    dataset
        Loader object already containing outer / inner folds.
    hyperparam_cfg
        Search-space for Optuna.
    base_estimator
        Instantiated sklearn classifier (e.g. ``LogisticRegression()``).
    selector_cfg_path
        YAML describing the feature selector.
    n_perm
        Number of permutations for `mne.stats.permutation_cluster_test`.

    Returns
    -------
    dict
        * decision_scores        : (subjects × windows) float
        * labels                 : (subjects,) int
        * fold_auc               : (k × 2 × windows) float  – ROC / PR
        * params_per_outer       : list[k][windows]  – tuned params
        * selector_stats         : dict with keys ``scores``, ``p_values``,
                                   ``mask`` (each: windows × n_features)
        * perm_test              : dict from ``permutation_cluster_test``
        * outer_indices          : list[(train_idx, test_idx)]
    """
    dd, ct = dataset.dd, dataset.ct
    X_full = np.concatenate([dd, ct])
    y = np.concatenate([np.ones(len(dd)), np.zeros(len(ct))])
    n_subjects, _, n_windows = X_full.shape
    k = len(dataset.outer_splits)

    decision_scores = np.full((n_subjects, n_windows), np.nan)
    fold_auc = np.empty((k, 2, n_windows))
    params_per_outer: List[List[Dict[str, Any]]] = []

    groups = np.arange(len(y))  # unique group per subject
    outer_iter = enumerate(dataset.outer_splits)

    # helper: convert global → outer-local indices
    def _remap_to_outer(indices, mapping):
        """Return indices relative to ``tr_idx``."""
        return np.asarray([mapping[x] for x in indices], dtype=int)

    # ------------------------------------------------------------ #
    #    allocate selector diagnostics *per outer fold*            #
    # ------------------------------------------------------------ #
    n_feat = X_full.shape[1]  # RQA metrics
    selector_scores = np.full((k, n_windows, n_feat), np.nan)
    selector_pvals = np.full_like(selector_scores, np.nan, dtype=float)
    selector_masks = np.zeros((k, n_windows, n_feat), dtype=bool)

    # ------------------------------------------------------------ #
    #                    outer-loop cross-validation               #
    # ------------------------------------------------------------ #

    for i, (tr_idx, te_idx) in outer_iter:

        X_tr, y_tr = X_full[tr_idx], y[tr_idx]
        groups_tr = groups[tr_idx]

        # ---------- build inner-CV list with *relative* indices ------------- #
        lookup = {g: j for j, g in enumerate(tr_idx)}  # global → local map
        inner_rel = [
            (_remap_to_outer(tr, lookup), _remap_to_outer(val, lookup))
            for tr, val in dataset.inner_splits[i]
        ]

        # ---------- per-window Optuna tuning -------------------------------- #
        tuning = tune_one_electrode_parallel(
            X=X_tr,
            y=y_tr,
            groups=groups_tr,
            inner_splits=inner_rel,  # ✓ now aligned with X_tr
            metadata={**dataset.metadata, "elec": dataset.metadata.get("elec", "T7")},
            hyperparam_cfg=hyperparam_cfg,
            base_estimator=base_estimator,
            n_jobs=n_jobs_tuner,
            random_state=random_state,
            optuna_cfg=optuna_configuration,
            selector_cfg=selector_cfg_path,
            storage_dir=storage_dir,
        )
        best_params = [t["best_params"] for t in tuning]
        params_per_outer.append(best_params)

        # ------------------- re-train & test ---------------------- #
        est = MultiWindowEstimator(
            base_cls=type(base_estimator),
            params_per_window=best_params,
            selector_cfg_path=selector_cfg_path,
            random_state=random_state,
        ).fit(
            X_full[tr_idx],
            y[tr_idx],
            n_jobs=n_jobs_windows,
            show_progress=False,
        )
        proba = est.predict_proba(X_full[te_idx])[:, :, 1]
        decision_scores[te_idx] = proba

        for w in range(n_windows):
            fold_auc[i, 0, w] = roc_auc_score(y[te_idx], proba[:, w])
            fold_auc[i, 1, w] = average_precision_score(y[te_idx], proba[:, w])

        # ------------------------------------------------------------------- #
        # NEW —— harvest selector statistics for this fold & all windows      #
        # ------------------------------------------------------------------- #
        sel_stats_fold = [_selector_stats(pipe) for pipe in est.estimators_]
        selector_scores[i] = np.vstack([s["scores"] for s in sel_stats_fold])
        selector_pvals[i] = np.vstack([s["p_values"] for s in sel_stats_fold])
        selector_masks[i] = np.vstack([s["mask"] for s in sel_stats_fold])

    # ------------------------------------------------------------ #
    #                 permutation-cluster statistics               #
    # ------------------------------------------------------------ #
    dd_scores = decision_scores[y == 1]
    ct_scores = decision_scores[y == 0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        T_obs, clusters, p_vals, H0 = permutation_cluster_test(
            [dd_scores, ct_scores],
            tail=1,
            n_permutations=n_perm,
            n_jobs=1,
            seed=random_state,
        )
    perm_test = dict(T_obs=T_obs, clusters=clusters, p_values=p_vals, H0=H0)

    return dict(
        decision_scores=decision_scores,
        labels=y,
        fold_auc=fold_auc,
        params_per_outer=params_per_outer,
        selector_stats=dict(
            names=np.asarray(RQA_METRICS),
            scores=selector_scores,  # shape (fold, window, feature)
            p_values=selector_pvals,
            mask=selector_masks,
        ),
        perm_test=perm_test,
        outer_indices=dataset.outer_splits,
    )
