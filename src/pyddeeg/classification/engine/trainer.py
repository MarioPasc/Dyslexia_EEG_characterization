# -*- coding: utf-8 -*-
"""
trainer.py - nested k x h CV with rich research-ready outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from mne.stats import permutation_cluster_test

from pyddeeg import RQA_METRICS  # list[str] with metric names
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


def _get_model_coef(pipe) -> np.ndarray:
    """
    Return the *full-feature-space* coefficient vector for a fitted
    (selector ▸ scaler ▸ estimator) pipeline.

    NaNs are filled when the innermost estimator lacks ``coef_``/``feature_importances_``.
    """
    est = pipe.named_steps["classifier"]
    if hasattr(est, "coef_"):
        raw = est.coef_.ravel()  # LogisticRegression, LinearSVM, …
    elif hasattr(est, "feature_importances_"):
        raw = est.feature_importances_.ravel()  # Tree-based
    else:  # Non-linear kernels, etc.
        return np.full(pipe["selector"].get_support().sum(), np.nan)

    # Map back to original feature space
    full = np.full(pipe["selector"].get_support().shape, np.nan, dtype=float)
    full[pipe["selector"].get_support()] = raw
    return full


def _aggregate_selector_stats(
    scores: np.ndarray, p_vals: np.ndarray, mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """Fold-wise → grand aggregates."""
    mean_score = np.nanmean(scores, axis=0)  # (window x feat)
    mean_p = np.nanmean(p_vals, axis=0)
    sel_freq = mask.mean(axis=0)  # proportion selected
    return dict(
        mean_score=mean_score, mean_p_value=mean_p, selection_frequency=sel_freq
    )


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
    End-to-end k x h nested CV **plus** cluster-based permutation test.

    Notes
    -----
    Besides the *legacy* keys, the function now returns many extra artefacts
    (fold-level predictions, coefficients, permutation masks, …) to facilitate
    post-hoc analyses and visualisation.

    Returns
    -------
    dict
        ├─ decision_scores          (subjects x windows) float
        ├─ per_fold_scores          list[k] → (idx, proba) with
        │                           idx : ndarray[n_test]  - global indices
        │                           proba: ndarray[n_test x windows]
        ├─ labels                   (subjects,) int  (1 = DD, 0 = CT)
        ├─ fold_auc                 (k x 2 x windows) float  - ROC / PR
        ├─ params_per_outer         list[k][windows]  - tuned params
        ├─ selector_stats           raw fold-wise diagnostics
        │   ├─ names                (n_features,) str
        │   ├─ scores               (k x windows x n_features) float
        │   ├─ p_values             same
        │   └─ mask                 bool
        ├─ feature_importance       aggregates (mean_score, mean_p_value,
        │                                      selection_frequency)
        ├─ coefficients             (k x windows x n_features) float
        ├─ perm_test                original dict + “significant_mask”
        ├─ outer_indices            list[(train_idx, test_idx)]
        ├─ inner_indices            list[k] of list[(tr, val)]
        ├─ optuna_study             list[k][windows] - Optuna trials DF
        ├─ electrode                str  (dataset.metadata['elec'])
        └─ dataset_ref              the original EEGDataset (for raw features)
    """
    dd, ct = dataset.dd, dataset.ct
    X_full = np.concatenate([dd, ct])
    y = np.concatenate([np.ones(len(dd)), np.zeros(len(ct))])
    n_subjects, _, n_windows = X_full.shape
    k = len(dataset.outer_splits)

    # (1) containers -------------------------------------------------------- #
    decision_scores = np.full((n_subjects, n_windows), np.nan)
    per_fold_scores: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_auc = np.empty((k, 2, n_windows))
    params_per_outer: List[List[Dict[str, Any]]] = []
    optuna_records: List[List[Any]] = []  # raw optuna trial DF/Study

    groups = np.arange(len(y))  # unique group per subject
    outer_iter = enumerate(dataset.outer_splits)

    # selector diagnostics
    n_feat = X_full.shape[1]  # RQA metrics
    selector_scores = np.full((k, n_windows, n_feat), np.nan)
    selector_pvals = np.full_like(selector_scores, np.nan, dtype=float)
    selector_masks = np.zeros((k, n_windows, n_feat), dtype=bool)

    # coefficients
    coefficients = np.full_like(selector_scores, np.nan, dtype=float)

    # helpers
    def _remap_to_outer(indices, mapping):
        """Return indices relative to ``tr_idx``."""
        return np.asarray([mapping[x] for x in indices], dtype=int)

    # --------------------------------------------------------------------- #
    #                         outer-loop cross-validation                   #
    # --------------------------------------------------------------------- #
    for i, (tr_idx, te_idx) in outer_iter:
        X_tr, y_tr = X_full[tr_idx], y[tr_idx]
        groups_tr = groups[tr_idx]

        # ------ build inner-CV list with *relative* indices -------------- #
        lookup = {g: j for j, g in enumerate(tr_idx)}  # global → local map
        inner_rel = [
            (_remap_to_outer(tr, lookup), _remap_to_outer(val, lookup))
            for tr, val in dataset.inner_splits[i]
        ]

        # ------ per-window Optuna tuning --------------------------------- #
        tuning = tune_one_electrode_parallel(
            X=X_tr,
            y=y_tr,
            groups=groups_tr,
            inner_splits=inner_rel,
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
        # optuna_records.append([t["study"] for t in tuning])  # raw Study

        # ------------------- re-train & test ---------------------------- #
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
        proba = est.predict_proba(X_full[te_idx])[:, :, 1]  # (n_test x windows)
        decision_scores[te_idx] = proba
        per_fold_scores.append((te_idx, proba))

        # ---------------- metrics per window ---------------------------- #
        for w in range(n_windows):
            fold_auc[i, 0, w] = roc_auc_score(y[te_idx], proba[:, w])
            fold_auc[i, 1, w] = average_precision_score(y[te_idx], proba[:, w])

        # ---------------- selector diagnostics -------------------------- #
        sel_stats_fold = [_selector_stats(pipe) for pipe in est.estimators_]
        selector_scores[i] = np.vstack([s["scores"] for s in sel_stats_fold])
        selector_pvals[i] = np.vstack([s["p_values"] for s in sel_stats_fold])
        selector_masks[i] = np.vstack([s["mask"] for s in sel_stats_fold])

        # ---------------- coefficients / importances -------------------- #
        coefs_fold = [_get_model_coef(pipe) for pipe in est.estimators_]
        coefficients[i] = np.vstack(coefs_fold)

    # --------------------------------------------------------------------- #
    #                        aggregate feature stats                         #
    # --------------------------------------------------------------------- #
    feat_aggr = _aggregate_selector_stats(
        selector_scores, selector_pvals, selector_masks
    )

    # --------------------------------------------------------------------- #
    #                               RETURN                                  #
    # --------------------------------------------------------------------- #
    return dict(
        # legacy ---------------------------------------------------------- #
        decision_scores=decision_scores,
        labels=y,
        fold_auc=fold_auc,
        params_per_outer=params_per_outer,
        selector_stats=dict(
            names=np.asarray(RQA_METRICS),
            scores=selector_scores,  # (fold x window x feat)
            p_values=selector_pvals,
            mask=selector_masks,
        ),
        outer_indices=dataset.outer_splits,
        per_fold_scores=per_fold_scores,
        inner_indices=dataset.inner_splits,
        optuna_study=optuna_records,
        feature_importance=feat_aggr,
        coefficients=coefficients,
        electrode=dataset.metadata.get("elec"),
        dataset_ref=dataset,  # raw features accessible on demand
    )
