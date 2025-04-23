from __future__ import annotations

"""OptunaEstimator
================================
Tuning wrapper that re-uses the outer cross-validator stored in an
:class:`EEGDataset` (no inner folds) and therefore avoids any kind of
nested-CV data leakage.  It is window-agnostic: you still call
:meth:`fit` with the data of ne window, but the splitter comes from
``dataset.cv``.
"""

from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import optuna

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from pyddeeg.classification.optimization.utils import suggest_from_config
from pyddeeg.classification.dataloaders import EEGDataset

__all__: Tuple[str, ...] = ("OptunaEstimator",)


class OptunaEstimator(BaseEstimator):
    """Optuna‑driven hyper‑parameter *and* feature‑selection optimisation.

    Parameters
    ----------
    base_estimator
        Any *instantiated* scikit‑learn estimator (e.g. ``LogisticRegression()``).
    hyperparameters
        Search‑space definition consumed by :pyfunc:`suggest_from_config`.
    dataset
        The *outer* dataset wrapper whose ``cv`` splitter will be re‑used.
    n_trials
        Number of Optuna trials per :meth:`fit` (default ``50``).
    random_state
        Seed for reproducibility.
    k_range
        Range for ``SelectKBest(k)``.
    study_name
        Optional name for the Optuna study. Useful for logging.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        hyperparameters: Dict[str, Dict[str, Any]],
        dataset: EEGDataset,
        *,
        n_trials: int = 50,
        random_state: Optional[int] = None,
        k_range: Tuple[int, int] = (5, 15),
        study_name: Optional[str] = None,
    ) -> None:
        # --- Basic type safety ------------------------------------------------
        if isinstance(base_estimator, type):
            raise TypeError("base_estimator must be an *instance*, not a class.")
        if not isinstance(base_estimator, BaseEstimator):
            raise TypeError("base_estimator must inherit from sklearn's BaseEstimator.")

        self.base_estimator = base_estimator
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.n_trials = int(n_trials)
        self.random_state = random_state
        self.k_range = k_range
        self.study_name = study_name

    # ---------------------------------------------------------------------
    #                   Optuna objective & helper methods
    # ---------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """A single Optuna trial."""
        # Suppress specific warnings within objective function
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant")
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )

            params = suggest_from_config(trial, self.hyperparameters)
            k = trial.suggest_int("k", self.k_range[0], self.k_range[1])

            selector = SelectKBest(f_classif, k=k)
            model = clone(self.base_estimator).set_params(**params)
            pipeline = Pipeline([("selector", selector), ("model", model)])

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=self.dataset.cv,
                scoring="roc_auc",
                n_jobs=-1,
                groups=np.arange(len(y)),
            )
            return float(np.mean(scores))

    # ------------------------------------------------------------------
    #                           Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401
        """Optimise hyper‑parameters on (X, y) and fit final pipeline."""
        from pyddeeg.classification import logger

        # Create study with name if provided
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            study_name=self.study_name,
        )

        if self.study_name:
            logger.info(f"Starting optimization for study: {self.study_name}")

        # Suppress warnings during optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant")
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )

            study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)

        best = study.best_params.copy()
        k = best.pop("k")
        selector = SelectKBest(f_classif, k=k)
        model = clone(self.base_estimator).set_params(**best)
        self.pipeline_ = Pipeline([("selector", selector), ("model", model)])
        self.pipeline_.fit(X, y)

        self.best_params_ = {"k": k, **best}
        self.study_ = study

        if self.study_name:
            logger.info(
                f"Completed optimization for {self.study_name}. Best score: {study.best_value:.4f}"
            )

        return self

    # scikit‑learn delegation wrappers
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        return self.pipeline_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        return self.pipeline_.predict(X)
