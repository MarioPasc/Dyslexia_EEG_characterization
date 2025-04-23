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

from pathlib import Path

import numpy as np
import optuna

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from pyddeeg.classification.optimization.utils import (
    suggest_from_config,
    load_optuna_config,
)
from pyddeeg.classification import EEGDataset

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
        config_yaml: str | Path | Dict[str, Any] | None = None,
        n_trials: int | None = None,
        random_state: int | None = None,
        k_range: Tuple[int, int] = (5, 15),
        study_name: str | None = None,
    ) -> None:

        # ------------------------------------------------------------------ #
        # Parse YAML and override defaults                                   #
        # ------------------------------------------------------------------ #
        cfg = load_optuna_config(config_yaml or {})  # empty dict if None

        self.n_trials = n_trials or cfg["n_trials"]
        self.random_state = random_state or cfg["random_state"]
        self.scoring = cfg["scoring"]

        self._sampler = cfg["sampler_obj"]
        self._pruner = cfg["pruner_obj"]

        self._storage_dir: Path | None = cfg["storage_dir"]
        self._single_db: bool = cfg["single_db"]

        # --- Basic type safety ------------------------------------------------
        if isinstance(base_estimator, type):
            raise TypeError("base_estimator must be an *instance*, not a class.")
        if not isinstance(base_estimator, BaseEstimator):
            raise TypeError("base_estimator must inherit from sklearn's BaseEstimator.")

        self.base_estimator = base_estimator
        self.hyperparameters = hyperparameters
        self.dataset = dataset
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
                scoring=self.scoring,
                n_jobs=-1,
                groups=np.arange(len(y)),
            )
            return float(np.mean(scores))

    def _get_storage(self, study_name: str) -> str | None:
        """
        Return an sqlite:/// URL or None depending on the YAML `save_trials`
        section.
        """
        if self._storage_dir is None:
            return None

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        if self._single_db:
            db_path = self._storage_dir / "optuna_studies.db"
        else:
            db_path = self._storage_dir / f"{study_name}.db"
        return f"sqlite:///{db_path}"

    # ------------------------------------------------------------------
    #                           Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401
        """Optimise hyper‑parameters on (X, y) and fit final pipeline."""

        from pyddeeg.classification import logger

        storage_url = self._get_storage(self.study_name or "study")

        study = optuna.create_study(
            direction="maximize",
            sampler=self._sampler,
            pruner=self._pruner,
            storage=storage_url,
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
