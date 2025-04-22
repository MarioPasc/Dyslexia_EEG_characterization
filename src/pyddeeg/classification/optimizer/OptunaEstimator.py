from typing import Any, Dict, Optional, Tuple
import optuna
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pyddeeg.classification.optimizer.utils import suggest_from_config


class OptunaEstimator(BaseEstimator):
    """
    Wrapper for Optuna-driven hyperparameter and feature-selection optimization.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Any scikit-learn estimator to tune.
    hyperparameters : Dict[str, Dict[str, Any]]
        Configuration of hyperparameter search space.
        See `suggest_from_config` for format.
    cv : StratifiedKFold
        Inner cross-validation splitter.
    n_trials : int, default=50
        Number of Optuna trials per `.fit`.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    k_range : Tuple[int, int], default=(5, 15)
        Range for SelectKBest `k`.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        hyperparameters: Dict[str, Dict[str, Any]],
        cv: StratifiedKFold,
        n_trials: int = 50,
        random_state: Optional[int] = None,
        k_range: Tuple[int, int] = (5, 15),
    ) -> None:
        # ensure base_estimator is instance, not class
        if isinstance(base_estimator, type):
            try:
                self.base_estimator = base_estimator()
            except Exception as e:
                raise TypeError(
                    "base_estimator class must be instantiable with no args; "
                    "or pass an estimator instance instead."
                )
        elif isinstance(base_estimator, BaseEstimator):
            self.base_estimator = base_estimator
        else:
            raise TypeError(
                "base_estimator must be a scikit-learn estimator instance "
                "or a class inheriting BaseEstimator."
            )
        self.hyperparameters = hyperparameters
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.k_range = k_range

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        # Suggest model hyperparameters
        params = suggest_from_config(trial, self.hyperparameters)
        # Separate feature-count
        k = trial.suggest_int("k", self.k_range[0], self.k_range[1])

        # Build pipeline: SelectKBest + cloned estimator
        selector = SelectKBest(f_classif, k=k)
        model = clone(self.base_estimator).set_params(**params)
        pipeline = Pipeline([("selector", selector), ("model", model)])

        # Evaluate with inner CV
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=self.cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptunaEstimator":
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)

        # Build final pipeline with best params
        best = study.best_params.copy()
        k = best.pop("k")
        selector = SelectKBest(f_classif, k=k)
        model = clone(self.base_estimator).set_params(**best)
        self.pipeline_ = Pipeline([("selector", selector), ("model", model)])
        self.pipeline_.fit(X, y)

        self.best_params_ = {"k": k, **best}
        self.study_ = study
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(X)
