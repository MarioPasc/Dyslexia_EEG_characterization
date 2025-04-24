# window_estimator.py
# -*- coding: utf-8 -*-
"""
Window-specific model wrapper for MNE-Python SlidingEstimator
============================================================

`SlidingEstimator` instantiates **one clone of ``base_estimator`` per time
window** (`n_tasks`).  In our use–case we already *know* the best
hyper-parameters for each window and we want those to be baked-in during the
re-training / outer-CV stage.

`WindowParamEstimator` therefore acts as a tiny *factory*:

───────────────────────────────────
┌───────────────┐   clone() & fit
│               │◀──────────────┐  (w = 0)
│ SlidingEstimator ─────────────┤
│               │◀──────────────┘  (w = 1)
└───────────────┘
         │
         ▼
WindowParamEstimator
         │  chooses params_per_window[w]
         └─► builds a real scikit-learn Pipeline
               (SelectKBest → model(**params))
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


__all__: Sequence[str] = ("WindowParamEstimator",)


# A *module-level* counter survives sklearn.clone() calls
_WINDOW_COUNTER = itertools.count()


class WindowParamEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrapper that injects **window-specific hyper-parameters**.

    Parameters
    ----------
    base_cls
        *Class* (not an instance) of the underlying estimator
        (e.g. ``HistGradientBoostingClassifier``).  It **must** implement
        ``predict_proba`` so that SlidingEstimator can call it.
    params_per_window
        A list where ``params_per_window[w]`` is a dict of keyword
        arguments for ``base_cls.set_params`` *plus* a mandatory key
        ``"k"`` with the number of features to keep in the
        ``SelectKBest`` step.

        **Length must equal `n_windows`.**

    Notes
    -----
    *   The class is intentionally *stateless* after fitting; every call to
        :meth:`fit` overwrites ``self.fitted_`` with a brand-new pipeline.
    *   A *global* counter is used so that successive clones get the
        correct window index even though their ``__init__`` is called first.
    """

    #: attribute set by ``fit`` so that sliding-estimator can access it
    classes_: np.ndarray

    def __init__(
        self,
        base_cls: Type[BaseEstimator],
        params_per_window: List[Dict[str, Any]],
    ) -> None:
        self.base_cls = base_cls
        self.params_per_window = params_per_window

    # --------------------------------------------------------------------- #
    #                            scikit-learn API                           #
    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "WindowParamEstimator":
        """Train a **window-specific** pipeline on ``(X, y)``."""
        window_idx = next(_WINDOW_COUNTER)
        try:
            p = copy.deepcopy(self.params_per_window[window_idx])
        except IndexError as exc:  # pragma: no cover
            raise IndexError(
                f"Asked for window #{window_idx} but "
                f"`params_per_window` has only {len(self.params_per_window)} "
                "entries."
            ) from exc

        # Split out the univariate-feature-selection parameter
        k = p.pop("k")

        # Build the real model
        model = self.base_cls().set_params(**p)
        self.fitted_: Pipeline = Pipeline(
            steps=[
                ("selector", SelectKBest(f_classif, k=k)),
                ("model", model),
            ]
        ).fit(X, y)

        # expose .classes_  (required by MNE / cross_val_predict)
        if not hasattr(self.fitted_[-1], "classes_"):  # pragma: no cover
            raise AttributeError(
                f"{self.base_cls.__name__} has no `classes_` attribute – "
                "classification pipeline cannot be used with SlidingEstimator."
            )
        self.classes_ = getattr(self.fitted_[-1], "classes_")  # type: ignore[attr-defined]
        return self

    # ---------- prediction delegates ------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "fitted_")
        return self.fitted_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "fitted_")
        return self.fitted_.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:  # optional
        check_is_fitted(self, "fitted_")
        if hasattr(self.fitted_, "decision_function"):
            return self.fitted_.decision_function(X)  # type: ignore[attr-defined]
        raise AttributeError("Underlying model does not implement decision_function")

    # ---------- repr / param handling ------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"<WindowParamEstimator base={self.base_cls.__name__} "
            f"(windows={len(self.params_per_window)})>"
        )

    # scikit-learn clone-ability
    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # noqa: D401
        return {
            "base_cls": self.base_cls,
            "params_per_window": (
                copy.deepcopy(self.params_per_window)
                if deep
                else self.params_per_window
            ),
        }

    def set_params(self, **params: Any):  # noqa: D401
        for key, val in params.items():
            setattr(self, key, val)
        return self
