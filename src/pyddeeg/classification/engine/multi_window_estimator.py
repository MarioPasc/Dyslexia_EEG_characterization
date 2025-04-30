# -*- coding: utf-8 -*-
"""
multi_window_estimator.py
=========================

A *minimal*, sklearn-compatible replica of MNE-Python’s ``SlidingEstimator`` that
allows **one hyper-parameter configuration per time-window**.

``MultiWindowEstimator``

* expects input tensors ``X`` with shape *(n_samples, n_features, n_windows)*;
* trains an *independent* scikit-learn **Pipeline** for every window:
  ``StandardScaler → SelectKBest → base_estimator(**params)``;
* exposes :pyattr:`classes_`, :py:meth:`predict` and :py:meth:`predict_proba`
  so that it can be used with ``cross_val_predict``, grid-searches, etc.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple, Type, Union
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pyddeeg.classification.pipeline import build_pipeline, load_selector


from joblib import Parallel, delayed
from tqdm.auto import tqdm

__all__: Sequence[str] = ("MultiWindowEstimator",)


# -----------------------------------------------------------------------------#
#                               helper functions                               #
# -----------------------------------------------------------------------------#


def _validate_dims(
    X: np.ndarray, params_per_window: List[Dict[str, Any]]
) -> Tuple[int, int, int]:
    """
    Sanity-check ``X`` and return its dimensions.

    Raises
    ------
    ValueError
        If ``X`` does not have exactly three dimensions or the number of
        windows mismatches ``params_per_window``.
    """
    if X.ndim != 3:
        raise ValueError("`X` must be shaped (n_samples, n_features, n_windows).")

    n_samples, n_features, n_windows = X.shape
    if n_windows != len(params_per_window):
        raise ValueError(
            "Mismatch: X contains "
            f"{n_windows} windows but len(params_per_window) is "
            f"{len(params_per_window)}."
        )
    return n_samples, n_features, n_windows


# -----------------------------------------------------------------------------#
#                              public estimator                                #
# -----------------------------------------------------------------------------#
class MultiWindowEstimator(BaseEstimator, ClassifierMixin):
    """
    Train **one independent model per time-window**.

    Parameters
    ----------
    base_cls
        Estimator **class** that supports :py:meth:`predict_proba`.
        params_per_window
            One dict **per window**::
                {
                  "model_params":    {...},   # ← tuned clf hyper-params
                  "selector_params": {...},   # ← tuned feature selector attrs
                }

    Notes
    -----
    *The object is fully clonable* – it stores only the *class* of the base
    estimator and immutable parameter dictionaries until :py:meth:`fit` is
    called.
    """

    # ------------------------------------------------------------------ #
    #                           construction                             #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_cls: Type[BaseEstimator],
        params_per_window: List[Dict[str, Any]],
        *,
        selector_cfg_path: Union[str, Path],
    ) -> None:
        self.base_cls = base_cls
        self.params_per_window = params_per_window
        self.selector_cfg_path = selector_cfg_path

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_jobs: int = 1,
        show_progress: bool = False,
        desc: str = "Windows",
    ) -> "MultiWindowEstimator":
        """Fit each window-specific pipeline."""
        _, _, n_windows = _validate_dims(X, self.params_per_window)

        def _fit_one(w: int, params: Dict[str, Any]) -> Pipeline:
            # split the nested dict -------------------------------
            model_kwargs: Dict[str, Any] = params.get("model_params", {})
            sel_kwargs: Dict[str, Any] = params.get("selector_params", {})

            # load *template* selector from YAML, then overwrite with Optuna best
            sel_template, fixed = load_selector(self.selector_cfg_path, trial=None)
            selector_cls = type(sel_template)
            selector = selector_cls(**{**fixed, **sel_kwargs})

            model = self.base_cls().set_params(**model_kwargs)
            pipe = build_pipeline(selector=selector, model=model)
            pipe.fit(X[:, :, w], y)
            return pipe

        iterable = range(n_windows)
        if show_progress:
            iterable = tqdm(iterable, desc=desc, unit="win")  # type: ignore

        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(w, self.params_per_window[w]) for w in iterable
        )

        # expose classes_ for compatibility
        self.classes_ = self.estimators_[0].named_steps["model"].classes_
        return self

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        _validate_dims(X, self.estimators_)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        n_samples, _, n_windows = X.shape
        n_classes = len(self.classes_)
        proba = np.empty((n_samples, n_windows, n_classes), float)
        for w, est in enumerate(self.estimators_):
            proba[:, w, :] = est.predict_proba(X[:, :, w])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=-1)]
