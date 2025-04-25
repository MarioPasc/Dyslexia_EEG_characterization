# -*- coding: utf-8 -*-
"""
trainer.py
==========

End-to-end **evaluation** of window-wise models:

*   re-trains the tuned hyper-parameters on *outer* Group-CV splits
    (no data leakage);
*   computes ROC-AUC & PR-AUC *per fold × window*;
*   gathers decision-scores, feature-selection masks and CV indices
    for downstream statistics or visualisation.

The file depends only on:

* ``MultiWindowEstimator`` – our custom SlidingEstimator clone;
* ``pyddeeg.classification.dataloaders.EEGDataset`` – wraps EEG tensors
  and a ready-to-use ``StratifiedGroupKFold`` splitter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import warnings

import mne
import numpy as np
from mne.stats import permutation_cluster_test
from sklearn.metrics import average_precision_score, roc_auc_score

from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification import MultiWindowEstimator

from tqdm.auto import tqdm

from joblib import Parallel, delayed

__all__: Sequence[str] = (
    "evaluate_frozen_models",
    "permutation_test_decision_scores",
)


# -----------------------------------------------------------------------------#
#                                small helpers                                 #
# -----------------------------------------------------------------------------#
def _epochs_from_tensor(
    *, elec: str, X: np.ndarray, sfreq: float | int
) -> mne.EpochsArray:
    """Wrap ``(subjects × metrics × windows)`` into a single-epoch MNE object."""
    info = mne.create_info(
        ch_names=[f"{elec}_{metric}" for metric in RQA_METRICS],
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Missing")
        epochs = mne.EpochsArray(X, info, tmin=0.0)
        epochs.set_montage("standard_1020", on_missing="ignore")
    return epochs


def _stack_classes_selected(estimator: MultiWindowEstimator) -> np.ndarray:
    """
    Extract the boolean *feature-selection* mask for **each window**.

    Returns
    -------
    mask : ndarray, shape (n_windows, 15)
    """
    return np.vstack(
        [pipe.named_steps["selector"].get_support() for pipe in estimator.estimators_]
    )


# -----------------------------------------------------------------------------#
#                           cross-validated evaluation                         #
# -----------------------------------------------------------------------------#
def evaluate_frozen_models(
    *,
    dataset: EEGDataset,
    params_per_window: list[dict],
    base_estimator_cls,
    n_jobs_windows: int = 1,
) -> Dict[str, Any]:
    """
    Re-train the tuned window-wise models on the outer Group-CV & collect metrics.

    Parameters
    ----------
    elec
        Electrode label (used only for informative channel names).
    dataset
        Wrapper with ``dd``, ``ct`` tensors and a ready Group-CV splitter.
    params_per_window
        List of hyper-param dicts (see :class:`MultiWindowEstimator` docs).
    base_estimator_cls
        Estimator class used during tuning (e.g. ``LogisticRegression``).

    Returns
    -------
    results : dict
        Keys
        ----
        decision_scores : ndarray (subjects × windows)
        labels          : ndarray (subjects,)
        fold_auc        : ndarray (n_folds × 2 × windows)  – [ROC, PR] AUC
        selected_features
        params_per_window
        cv_indices      : list[(train_idx, test_idx)]
        t_centres_ms    : optional 1-D array from dataset metadata
    """
    # --- prepare data & CV ------------------------------------------ #
    X = np.concatenate([dataset.dd, dataset.ct])  # (N, metrics, windows)
    y = np.concatenate([np.ones(len(dataset.dd)), np.zeros(len(dataset.ct))])
    groups = np.arange(len(y))
    cv = dataset.cv

    n_subjects, _, n_windows = X.shape
    decision_scores = np.full((n_subjects, n_windows), np.nan)
    fold_auc = np.empty((cv.get_n_splits(), 2, n_windows))

    splits = list(cv.split(X, y, groups))

    # --- iterate folds with tqdm ------------------------------------ #
    for f_idx, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc="Folds", unit="fold")
    ):
        # fit per‐window pipelines in parallel, showing a bar "Fold f_idx"
        est = MultiWindowEstimator(
            base_cls=base_estimator_cls,
            params_per_window=params_per_window,
        ).fit(
            X[train_idx],
            y[train_idx],
            n_jobs=n_jobs_windows,
            show_progress=True,
            desc=f"Fold {f_idx}",
        )

        proba = est.predict_proba(X[test_idx])[:, :, 1]  # class 1 == DD
        # compute ROC-AUC & PR-AUC per window
        for w in range(n_windows):
            fold_auc[f_idx, 0, w] = roc_auc_score(y[test_idx], proba[:, w])
            fold_auc[f_idx, 1, w] = average_precision_score(y[test_idx], proba[:, w])
        decision_scores[test_idx] = proba

    # --- full data fit for feature masks --------------------------- #
    full_est = MultiWindowEstimator(
        base_cls=base_estimator_cls,
        params_per_window=params_per_window,
    ).fit(X, y, n_jobs=n_jobs_windows, show_progress=False)

    selected_features = np.vstack(
        [pipe.named_steps["selector"].get_support() for pipe in full_est.estimators_]
    )

    return {
        "decision_scores": decision_scores,
        "labels": y,
        "fold_auc": fold_auc,
        "selected_features": selected_features,
        "params_per_window": params_per_window,
        "cv_indices": splits,
        "t_centres_ms": dataset.metadata.get("t_centres_ms"),
    }


# -----------------------------------------------------------------------------#
#                      cluster-based permutation testing                        #
# -----------------------------------------------------------------------------#
def permutation_test_decision_scores(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 5_000,
    tail: int = 1,
    seed: int | None = None,
    **mne_kwargs,
) -> Dict[str, Any]:
    """
    Run a Maris–Oostenveld cluster permutation on window-wise decision scores.

    Notes
    -----
    Forward any keyword arguments to
    :pyfunc:`mne.stats.permutation_cluster_test`.
    """
    dd_scores = decision_scores[labels == 1]
    ct_scores = decision_scores[labels == 0]

    if dd_scores.size == 0 or ct_scores.size == 0:  # pragma: no cover
        raise ValueError("Both classes need at least one subject.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        T_obs, clusters, p_vals, H0 = permutation_cluster_test(
            [dd_scores, ct_scores],
            tail=tail,
            n_permutations=n_permutations,
            n_jobs=1,
            seed=seed,
            **mne_kwargs,
        )

    return {"T_obs": T_obs, "clusters": clusters, "p_values": p_vals, "H0": H0}
