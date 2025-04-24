# -*- coding: utf-8 -*-
"""
trainer.py – re-train & re-evaluate tuned window-wise models
============================================================
* Accepts **one electrode**, its dataset and **one model-spec per window**.
* Wraps them in an MNE ``SlidingEstimator`` so that **exactly the same
  outer StratifiedGroupKFold** (stored in ``dataset.cv``) is used for
  evaluation – **no data leakage**.
* Returns everything you will likely want for downstream analysis
  (decision scores, per-fold AUC curves, feature masks, CV splits, …)
  in a single dict that can safely be written with
  ``np.savez_compressed(out_file, **results)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import warnings

import numpy as np
import mne
from mne.decoding import Scaler, Vectorizer, SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_test
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline

from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification import WindowParamEstimator

__all__: Tuple[str, ...] = (
    "build_sliding_estimator",
    "evaluate_frozen_models",
    "permutation_test_decision_scores",
)

# -----------------------------------------------------------------------------#
#                                helpers                                        #
# -----------------------------------------------------------------------------#


def _make_epochs(elec: str, X: np.ndarray, sfreq: float) -> mne.EpochsArray:
    """Convert (subjects × metrics × windows) tensor into a single‐epoch MNE object."""
    info = mne.create_info(
        [f"{elec}_{m}" for m in RQA_METRICS], sfreq=sfreq, ch_types="eeg"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Missing")
        epochs = mne.EpochsArray(X, info, tmin=0.0)
        epochs.set_montage("standard_1020", on_missing="ignore")
    return epochs


def _make_selector_mask(pipe: Pipeline) -> np.ndarray:
    """Return boolean mask of selected features for *one* window pipeline."""
    # 1) OptunaEstimator case (tuning stage)
    if hasattr(pipe, "named_steps") and "optunaestimator" in pipe.named_steps:
        sel = pipe.named_steps["optunaestimator"].pipeline_.named_steps["selector"]
        return sel.get_support()

    # 2) WindowParamEstimator → look inside .fitted_
    if hasattr(pipe, "fitted_"):
        return pipe.fitted_.named_steps["selector"].get_support()

    # 3) Any plain sklearn Pipeline passed directly
    if hasattr(pipe, "named_steps"):
        for step in pipe.named_steps.values():
            if hasattr(step, "get_support"):
                return step.get_support()

    raise RuntimeError("No selector with get_support() found in window pipeline.")


# -----------------------------------------------------------------------------#
#                          model construction                                   #
# -----------------------------------------------------------------------------#


def build_sliding_estimator(
    *,
    params_per_window: List[Dict[str, Any]],
    base_estimator_cls,
) -> SlidingEstimator:
    """
    Build a **fresh** SlidingEstimator that will (re-)train one model per window
    with *exactly* the supplied hyper-parameters.

    Parameters
    ----------
    params_per_window
        List with length == n_windows. Each dict **must** contain the key ``"k"``
        (number of RQA metrics to keep) plus any hyper-params of *base_estimator*.
    base_estimator_cls
        Estimator **class** (e.g. ``HistGradientBoostingClassifier``). Will be
        instantiated once per window.

    Returns
    -------
    se : SlidingEstimator (un-fitted)
    """
    return SlidingEstimator(
        WindowParamEstimator(
            base_cls=base_estimator_cls, params_per_window=params_per_window
        ),
        n_jobs=-1,
    )


# -----------------------------------------------------------------------------#
#                          main evaluation routine                              #
# -----------------------------------------------------------------------------#


def evaluate_frozen_models(
    *,
    elec: str,
    dataset: EEGDataset,
    params_per_window: List[Dict[str, Any]],
    base_estimator_cls,
    pos_label: int = 1,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Re-train the tuned models on the outer CV and gather all evaluation metrics.

    Returns
    -------
    results : dict
        {
          "decision_scores"   : ndarray (subjects × windows),
          "labels"            : ndarray (subjects,),
          "fold_auc"          : ndarray (n_folds × 2 × windows),
          "selected_features" : ndarray (windows × 15 RQA metrics),
          "params_per_window" : list[dict],
          "cv_indices"        : list[tuple[np.ndarray, np.ndarray]],
          "t_centres_ms"      : ndarray | None
        }
    """
    # ----------------------------- data ----------------------------------- #
    X = np.concatenate([dataset.dd, dataset.ct])  # shape  (N, 15, T)
    y = np.concatenate([np.ones(len(dataset.dd)), np.zeros(len(dataset.ct))])
    groups = np.arange(len(y))  # one id per subject
    cv: GroupKFold = dataset.cv  # already stratified
    cv_indices = [(tr, te) for tr, te in cv.split(X, y, groups=groups)]

    epochs = _make_epochs(
        dataset.metadata["elec"], X, sfreq=float(dataset.metadata["sfreq"])
    )

    # ----------------------------- model ---------------------------------- #
    scaler = Scaler(epochs.info, scalings="median")
    vectorizer = Vectorizer()
    se = build_sliding_estimator(
        params_per_window=params_per_window, base_estimator_cls=base_estimator_cls
    )

    base_pipeline = make_pipeline(scaler, vectorizer, se)

    # ----------------------------- prediction ----------------------------- #
    proba = cross_val_predict(
        base_pipeline,
        X=epochs.get_data(),
        y=y,
        cv=cv_indices,
        groups=groups,
        method="predict_proba",
        n_jobs=n_jobs,
    )  # (N, T, 2)
    decision_scores = proba[:, :, pos_label]

    # ----------------------------- AUC curves ----------------------------- #
    roc_se = SlidingEstimator(base_pipeline, scoring="roc_auc", n_jobs=n_jobs)
    pr_se = SlidingEstimator(base_pipeline, scoring="average_precision", n_jobs=n_jobs)

    auc_roc = cross_val_multiscore(
        roc_se, epochs.get_data(), y, cv=cv_indices, n_jobs=n_jobs
    )
    auc_pr = cross_val_multiscore(
        pr_se, epochs.get_data(), y, cv=cv_indices, n_jobs=n_jobs
    )
    fold_auc = np.stack([auc_roc, auc_pr], axis=1)  # (folds, 2, T)

    # ----------------------------- feature masks -------------------------- #
    # Fit once so we can dig into each window pipeline
    base_pipeline.fit(epochs.get_data(), y)
    selected_features = np.vstack(
        [
            _make_selector_mask(pipe)
            for pipe in base_pipeline.named_steps["slidingestimator"].estimators_
        ]
    )

    # ----------------------------- bundle --------------------------------- #
    return {
        "decision_scores": decision_scores,
        "labels": y,
        "fold_auc": fold_auc,
        "selected_features": selected_features,
        "params_per_window": params_per_window,
        "cv_indices": cv_indices,
        "t_centres_ms": dataset.metadata.get("t_centres_ms"),
    }


# -----------------------------------------------------------------------------#
#                       cluster-based permutation test                          #
# -----------------------------------------------------------------------------#


def permutation_test_decision_scores(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 5000,
    tail: int = 1,
    seed: int | None = None,
    **mne_kwargs,
) -> Dict[str, Any]:
    """
    Run Maris-Oostenveld cluster permutation on the ROC-AUC curves.

    Parameters
    ----------
    decision_scores : ndarray (subjects × windows)
    labels          : ndarray (subjects,)
    tail            : 1 (DD>CT), 0 (two-sided) or -1 (DD<CT)
    **mne_kwargs    : forwarded to ``mne.stats.permutation_cluster_test``

    Returns
    -------
    dict with keys ``T_obs, clusters, p_values, H0``.
    """
    dd = decision_scores[labels == 1]
    ct = decision_scores[labels == 0]

    if dd.size == 0 or ct.size == 0:
        raise ValueError("Each class must contain at least one subject.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        T_obs, clusters, pvals, H0 = permutation_cluster_test(
            [dd, ct],
            tail=tail,
            n_permutations=n_permutations,
            n_jobs=1,
            seed=seed,
            **mne_kwargs,
        )

    return {"T_obs": T_obs, "clusters": clusters, "p_values": pvals, "H0": H0}
