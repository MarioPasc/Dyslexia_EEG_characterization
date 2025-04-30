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

from contextlib import suppress
from filelock import FileLock
import sqlite3
from sqlalchemy import create_engine
from optuna.storages._rdb.models import BaseModel

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from pyddeeg.classification.optimization.utils import (
    suggest_from_config,
    load_optuna_config,
)
from pyddeeg.classification import EEGDataset
from pyddeeg.classification.pipeline import (
    build_pipeline,
    resolve_dotted,
    load_selector,
)

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
        selector_cfg_path: str | Path = "",
        storage_dir: str | Path | None = None,
        n_trials: int | None = None,
        random_state: int | None = None,
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

        yaml_storage = cfg["storage_dir"]
        self._storage_dir: Path | None = (
            Path(storage_dir).expanduser() if storage_dir else yaml_storage
        )
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
        self.study_name = study_name

        self.selector_cfg_path = selector_cfg_path

        self.cv = StratifiedGroupKFold(
            n_splits=self.dataset.cv.get_n_splits(),
            shuffle=True,
            random_state=self.random_state,
        )

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

            model_params = suggest_from_config(trial, self.hyperparameters)
            selector, selector_params = load_selector(self.selector_cfg_path, trial)

            model = clone(self.base_estimator).set_params(**model_params)
            pipeline = build_pipeline(selector=selector, model=model)

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                groups=np.arange(len(y)),
            )

            trial.set_user_attr("cv_scores", scores.tolist())
            trial.set_user_attr("mean", float(np.mean(scores)))
            trial.set_user_attr("median", float(np.median(scores)))
            trial.set_user_attr("std", float(np.std(scores)))

            return float(np.mean(scores))

    def _get_storage(self, study_name: str) -> str | None:
        """
        Build an sqlite:/// URL *and* make sure the schema exists exactly once,
        even when hundreds of processes start at the same time.
        """
        if self._storage_dir is None:
            return None

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._storage_dir / (
            "optuna_studies.db" if self._single_db else f"{study_name}.db"
        )
        url = f"sqlite:///{db_path}"

        # ---------- inter-process lock -----------------------------------------
        lockfile = db_path.with_suffix(".lock")
        with FileLock(lockfile, timeout=300):  # 5-min safety timeout
            if not db_path.exists() or db_path.stat().st_size == 0:
                eng = create_engine(
                    url,
                    connect_args={"check_same_thread": False, "timeout": 60},
                )
                # "IF NOT EXISTS" isn’t emitted by SQLAlchemy, so we still need
                # to swallow the race between 2 very unlucky processes.
                with suppress(sqlite3.OperationalError):
                    BaseModel.metadata.create_all(eng)

        return url

    # ------------------------------------------------------------------
    #                           Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401
        """Optimise hyper‑parameters on (X, y) and fit final pipeline."""

        from pyddeeg.classification import optuna_logger as logger

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

        # ---- 1) split the winning parameter set ------------------------- #
        best_all = study.best_params.copy()

        model_param_keys = set(self.hyperparameters.keys())
        model_params_best = {k: v for k, v in best_all.items() if k in model_param_keys}
        selector_params_best = {
            k: v for k, v in best_all.items() if k not in model_param_keys
        }

        # ---- 2) re-instantiate selector *with its own best params* ------- #
        selector_tpl, _ = load_selector(self.selector_cfg_path, trial=None)
        selector_best = type(selector_tpl)(**selector_params_best)

        model_best = clone(self.base_estimator).set_params(**model_params_best)
        self.pipeline_ = build_pipeline(selector=selector_best, model=model_best)
        self.pipeline_.fit(X, y)

        # ---- 3) expose as a tidy, two-level dict ------------------------ #
        self.best_params_ = {
            "model_params": model_params_best,
            "selector_params": selector_params_best,
        }

        self.study_ = study
        bt = study.best_trial.user_attrs
        self.best_trial_performance = {
            "mean": bt["mean"],
            "median": bt["median"],
            "std": bt["std"],
            "folds": bt["cv_scores"],
        }

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
