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
from typing import Any, Dict, List, Sequence, Tuple, Type

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


from joblib import Parallel, delayed
from tqdm.auto import tqdm

__all__: Sequence[str] = ("MultiWindowEstimator",)


# -----------------------------------------------------------------------------#
#                               helper functions                               #
# -----------------------------------------------------------------------------#
def _build_pipeline(
    *, k: int, base_cls: Type[BaseEstimator], model_params: Dict[str, Any]
) -> Pipeline:
    """
    Construct the *per-window* preprocessing / modelling pipeline.

    Parameters
    ----------
    k
        Number of features to keep in the univariate ANOVA selector.
    base_cls
        Estimator *class* (e.g. ``HistGradientBoostingClassifier``).
    model_params
        Arguments forwarded to :py:meth:`base_cls.set_params`.

    Returns
    -------
    pipeline
        A fully initialised but **un-fitted** scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", base_cls().set_params(**model_params)),
        ]
    )


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
        ``List[Dict]`` where *each* dictionary contains:

        * ``"k"`` – number of top features to keep after ANOVA F-test.
        * any other key-value pairs forwarded to
          ``base_cls().set_params(**params)``.

        *Length must equal the number of windows in the input data.*

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
    ) -> None:
        self.base_cls = base_cls
        self.params_per_window = params_per_window

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

        def _fit_one(w: int, params: dict) -> Pipeline:
            cfg = copy.deepcopy(params)
            k = int(cfg.pop("k"))
            pipe = _build_pipeline(k=k, base_cls=self.base_cls, model_params=cfg)
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
