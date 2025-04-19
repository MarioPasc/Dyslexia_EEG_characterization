# -*- coding: utf-8 -*-
"""trainer.py
================
Low-level helpers for sliding-window classification of EEG-derived RQA
features and permutation statistics.

The main routine now exposes **raw per-fold, per-window AUC curves** so that
power-users can apply their own aggregation or statistical workflow.

Key public functions
--------------------
* :func:`classification_per_electrode` - returns
    * per-subject decision-score matrix *(subjects x windows)*
    * **per-fold AUC tensor** ``(n_folds, 2, n_windows)`` where the
      second dimension encodes ROC-AUC (index 0) and PR-AUC (index 1)
    * forward-model patterns *(windows x metrics)*
    * subject labels
* :func:`permutation_test_decision_scores` - cluster-based non-parametric
  comparison (DD vs CT) on the decision-score matrix.

Both functions are fully type-hinted and include extensive doctrings so
that they can double as user-level documentation.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np
import mne
from mne.decoding import Scaler, Vectorizer, SlidingEstimator, get_coef, cross_val_multiscore
from mne.stats import permutation_cluster_test
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset

__all__: Tuple[str, ...] = (
    "classification_per_electrode",
    "permutation_test_decision_scores",
)


def _make_epochs(elec: str, X: np.ndarray, sfreq: float) -> mne.EpochsArray:
    """Construct an :class:`~mne.EpochsArray` from the RQA feature matrix.

    Parameters
    ----------
    elec
        Electrode code (e.g. ``"Fz"``).
    X
        Feature tensor with shape ``(n_subjects, n_metrics, n_windows)``.
    sfreq
        Sampling frequency that links the window index to real time
        (usually ``20.`` Hz when windows are 50 ms apart).

    Returns
    -------
    epochs
        Single-epoch MNE container whose *channels* correspond to the
        15 RQA metrics.
    """
    info = mne.create_info(
        ch_names=[f"{elec}_{m}" for m in RQA_METRICS],
        sfreq=sfreq,
        ch_types="eeg",
    )
    epochs = mne.EpochsArray(X, info, tmin=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Missing channel location", RuntimeWarning, "mne"
        )
        epochs.set_montage("standard_1020", on_missing="ignore")
    return epochs


# -----------------------------------------------------------------------------
#                           CLASSIFICATION ROUTINE
# -----------------------------------------------------------------------------

def classification_per_electrode(
    elec: str,
    dataset: EEGDataset,
    model: BaseEstimator,
    *,
    pos_label: int = 1,
    n_jobs: int | None = -1,
) -> Dict[str, Any]:
    """Run a *window-wise* classifier on one electrode and expose raw AUC.

    The procedure fits **one classifier per time window** using
    :class:`mne.decoding.SlidingEstimator`. Two separate SlidingEstimators
    are created - one scored with ``roc_auc`` and another with
    ``average_precision`` - so we recover **fold-level** ROC-AUC and PR-AUC
    matrices.  These are stacked into a tensor ``(n_folds, 2, n_windows)``
    for maximum downstream flexibility.

    Parameters
    ----------
    elec, dataset, model, pos_label, n_jobs
        See full doc-string in previous revision; semantics unchanged.

    Returns
    -------
    results : dict
        Keys and shapes::

            {
              'decision_scores' : (n_subjects, n_windows),
              'labels'          : (n_subjects,),
              'fold_auc'        : (n_folds, 2, n_windows),
              'patterns'        : (n_windows, n_metrics)
            }

        ``fold_auc[:, 0, :]`` → ROC-AUC; ``fold_auc[:, 1, :]`` → PR-AUC.
    """
    # ------------------------------------------------------------------
    # 0. Concatenate tensors & build labels
    # ------------------------------------------------------------------
    dd, ct, cv = dataset.dd, dataset.ct, dataset.cv
    X = np.concatenate([dd, ct])                                 # (N, 15, T)
    y = np.concatenate([np.ones(len(dd)), np.zeros(len(ct))])    # (N,)
    groups = np.arange(len(y))                                   # unique ID per subject

    # ------------------------------------------------------------------
    # 1. Common preprocessing pipeline (scaling + vectorisation + model)
    # ------------------------------------------------------------------
    epochs = _make_epochs(elec, X, sfreq=float(dataset.metadata.get("sfreq", 1.0)))
    base = make_pipeline(
        Scaler(epochs.info, scalings="median"),
        Vectorizer(),
        model,
    )

    # ------------------------------------------------------------------
    # 2. Fold-wise decision scores (per subject) for advanced stats
    # ------------------------------------------------------------------
    est_dec = SlidingEstimator(base, scoring=None, n_jobs=n_jobs)
    proba = cross_val_predict(
        est_dec,
        X=epochs.get_data(),
        y=y,
        cv=cv,
        groups=groups,
        method="predict_proba",
        n_jobs=n_jobs,
    )  # → (N, 2, T)
    decision_scores = proba[:, pos_label, :]

    # ------------------------------------------------------------------
    # 3. Fold-level AUC tensors (ROC and PR)
    # ------------------------------------------------------------------
    est_roc = SlidingEstimator(base, scoring="roc_auc", n_jobs=n_jobs)
    est_pr = SlidingEstimator(base, scoring="average_precision", n_jobs=n_jobs)
    auc_roc_folds = cross_val_multiscore(
        est_roc, epochs.get_data(), y, cv=cv, groups=groups, n_jobs=n_jobs
    )  # (n_folds, n_windows)
    auc_pr_folds = cross_val_multiscore(
        est_pr, epochs.get_data(), y, cv=cv, groups=groups, n_jobs=n_jobs
    )  # (n_folds, n_windows)
    fold_auc = np.stack([auc_roc_folds, auc_pr_folds], axis=1)   # (folds, 2, T)

    # ------------------------------------------------------------------
    # 4. Patterns for interpretability (fit on full data)
    # ------------------------------------------------------------------
    est_dec.fit(epochs.get_data(), y)
    patterns = np.squeeze(get_coef(est_dec, "patterns_", inverse_transform=True))

    return {
        "decision_scores": decision_scores,
        "labels": y,
        "fold_auc": fold_auc,
        "patterns": patterns,
    }


# -----------------------------------------------------------------------------
#                            PERMUTATION TEST
# -----------------------------------------------------------------------------

def permutation_test_decision_scores(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 5000,
    tail: int = 1,
    threshold: float | dict | None = None,
    adjacency: np.ndarray | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Cluster-based permutation test on DD vs CT decision-score curves.

    Parameters
    ----------
    decision_scores
        Matrix with shape ``(n_subjects, n_windows)`` returned by
        :func:`compute_decision_scores_per_electrode`.
    labels
        Binary vector (``1`` = DD, ``0`` = CT) of length ``n_subjects``.
    n_permutations
        Number of label permutations used to build the null distribution
        (defaults to ``5000`` as recommended by Maris & Oostenveld, 2007).
    tail
        * ``1`` - test whether DD > CT.
        * ``0`` - two-sided.
        * ``-1`` - test whether DD < CT.
    threshold
        Cluster-forming threshold (e.g. ``dict(start=0.0, step=0.2)``) or
        a fixed float.  ``None`` lets MNE choose a default t-distribution
        threshold based on the sample size.
    adjacency
        Optional boolean adjacency matrix if you wish to use
        spatio-temporal clustering.  Pass ``None`` for purely temporal
        clustering.
    seed
        Seed for the permutation RNG to ensure reproducibility.

    Returns
    -------
    stats
        Dictionary with the keys::

            {
                'T_obs'    : ndarray (n_windows,),
                'clusters' : list[ndarray],
                'p_values' : ndarray (n_clusters,),
                'H0'       : ndarray (n_permutations,)
            }

        See the documentation of
        :func:`mne.stats.permutation_cluster_test` for details.
    """
    rng = np.random.default_rng(seed)

    dd_scores = decision_scores[labels == 1]
    ct_scores = decision_scores[labels == 0]

    if dd_scores.size == 0 or ct_scores.size == 0:
        raise ValueError("Both classes must contain at least one subject.")

    with warnings.catch_warnings():
        # MNE may emit a warning when no clusters are found; silence it so
        # users can decide based on the return values.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        T_obs, clusters, pvals, H0 = permutation_cluster_test(
            [dd_scores, ct_scores],
            adjacency=adjacency,
            n_permutations=n_permutations,
            tail=tail,
            threshold=threshold,
            n_jobs=1,
            out_type="mask",
            seed=rng,
        )

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "p_values": pvals,
        "H0": H0,
    }
