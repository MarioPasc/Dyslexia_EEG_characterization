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

# pyddeeg/classification/window_estimator.py
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Type

import copy
import itertools
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


# --------------------------------------------------------------------- #
#  Counter shared by all clones – don’t touch                           #
# --------------------------------------------------------------------- #
_WINDOW_COUNTER = itertools.count()


class WindowParamEstimator(BaseEstimator, ClassifierMixin):  # ★ add ClassifierMixin
    """Inject window-specific hyper-parameters into a SlidingEstimator."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        base_cls: Type[BaseEstimator],
        params_per_window: List[Dict[str, Any]],
    ) -> None:
        self.base_cls = base_cls
        self.params_per_window = params_per_window
        # make absolutely sure every clone *knows* it is a classifier
        self._estimator_type = "classifier"

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):  # type: ignore[override]
        w_idx = next(_WINDOW_COUNTER)
        params = copy.deepcopy(self.params_per_window[w_idx])
        k = params.pop("k")

        # build and **fit** the real model for this window
        inner = Pipeline(
            [
                ("selector", SelectKBest(f_classif, k=k)),
                ("model", self.base_cls().set_params(**params)),
            ]
        )
        self.fitted_ = inner.fit(X, y)  # ★ was missing
        self.classes_ = self.fitted_[-1].classes_
        return self

    # ------------------------------------------------------------------
    # scikit-learn delegates
    def predict(self, X):
        check_is_fitted(self, "fitted_")
        return self.fitted_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_")
        return self.fitted_.predict_proba(X)

    def decision_function(self, X):
        """Only delegate if the inner model implements it; otherwise fall back."""
        check_is_fitted(self, "fitted_")
        if hasattr(self.fitted_[-1], "decision_function"):
            return self.fitted_.decision_function(X)
        # Fallback lets roc-auc scorer continue with probabilities
        return self.predict_proba(X)[:, 1]

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
