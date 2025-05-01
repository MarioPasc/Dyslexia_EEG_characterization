# -*- coding: utf-8 -*-
"""
optuna_estimator.py – *dataset-free* optimisation wrapper
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score
from pyddeeg.classification.pipeline import build_pipeline, load_selector
from pyddeeg.classification.optimization.utils import (
    load_optuna_config,
    suggest_from_config,
)

__all__: Sequence[str] = ("OptunaEstimator",)


class OptunaEstimator(BaseEstimator):
    """
    Hyper-parameter + feature-selection tuner that only needs:
    * a *cv splitter* (object or iterable of index tuples)
    * X, y, groups (supplied at ``fit``)
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        hyperparameters: Dict[str, Dict[str, Any]],
        *,
        selector_cfg_path: str | Path,
        cv: Any,  # can be int, CV object, or list[(train, val)]
        n_trials: int = 50,
        random_state: int = 42,
        study_name: str | None = None,
        storage_dir: str | Path | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.hyperparameters = hyperparameters
        self.selector_cfg_path = Path(selector_cfg_path)
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.study_name = study_name
        self.storage_dir = Path(storage_dir) if storage_dir else None

        # default Optuna config
        cfg = load_optuna_config({})
        self.scoring = cfg["scoring"]
        self._sampler, self._pruner = cfg["sampler_obj"], cfg["pruner_obj"]

    # -------------------------- private helpers ----------------------- #
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, groups):
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
            groups=groups,
            n_jobs=-1,
        )
        trial.set_user_attr("cv_scores", scores.tolist())
        return float(np.mean(scores))

    # ------------------------------ API ------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        study = optuna.create_study(
            direction="maximize",
            sampler=self._sampler,
            pruner=self._pruner,
            storage=(
                f"sqlite:///{self.storage_dir / 'optuna.db'}"
                if self.storage_dir
                else None
            ),
            study_name=self.study_name,
            load_if_exists=True,
        )
        study.optimize(
            lambda t: self._objective(t, X, y, groups), n_trials=self.n_trials
        )

        best = study.best_params
        selector_tpl, _ = load_selector(self.selector_cfg_path, trial=None)

        model_best = clone(self.base_estimator).set_params(
            **{k: v for k, v in best.items() if k in self.hyperparameters}
        )
        selector_best = type(selector_tpl)(
            **{k: v for k, v in best.items() if k not in self.hyperparameters}
        )
        self.pipeline_ = build_pipeline(selector=selector_best, model=model_best)
        self.pipeline_.fit(X, y)

        self.best_params_ = dict(
            model_params=model_best.get_params(),
            selector_params=selector_best.get_params(),
        )
        return self

    # proxies
    def predict_proba(self, X):  # noqa: D401
        return self.pipeline_.predict_proba(X)

    def predict(self, X):  # noqa: D401
        return self.pipeline_.predict(X)
