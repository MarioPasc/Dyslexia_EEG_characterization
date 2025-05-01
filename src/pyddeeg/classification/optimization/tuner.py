# -*- coding: utf-8 -*-
"""
tuner.py – window-wise Optuna using *raw arrays* (no dataset clones)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold

from pyddeeg.classification import OptunaEstimator
from pyddeeg.classification.optimization import (
    OPTUNA_FALLBACK_PATH,
    SELECTOR_FALLBACK_PATH,
)


def _optuna_for_window(
    Xw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    base_estimator,
    inner_cv,
    metadata: Dict[str, Any],
    random_state: int,
    win_idx: int,
    optuna_cfg=OPTUNA_FALLBACK_PATH,
    selector_cfg=SELECTOR_FALLBACK_PATH,
    storage_dir: Path | None = None,
):
    study_name = f"{metadata['direction']}_{metadata['window_ms']}ms_win{win_idx:03d}"

    opt = OptunaEstimator(
        base_estimator=base_estimator,
        hyperparameters=hyperparam_cfg,
        selector_cfg_path=selector_cfg,
        cv=inner_cv,
        n_trials=optuna_cfg.get("n_trials", 50) if isinstance(optuna_cfg, dict) else 50,
        random_state=random_state + win_idx,
        study_name=study_name,
        storage_dir=storage_dir,
    )
    opt.fit(Xw, y, groups)
    return {"best_params": opt.best_params_}


# ------------------------------ public API ----------------------------- #
def tune_one_electrode_parallel(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_splits: List[tuple[np.ndarray, np.ndarray]],
    metadata: Dict[str, Any],
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    base_estimator,
    n_jobs: int | None = None,
    random_state: int = 42,
    optuna_cfg: str | Path | Dict[str, Any] | None = OPTUNA_FALLBACK_PATH,
    selector_cfg: str | Path = SELECTOR_FALLBACK_PATH,
    storage_dir: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Parallel window-wise tuning that only needs *arrays* and an *inner split list*.
    """

    n_windows = X.shape[-1]
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # turn list[(train, val)] into a splitter object that sklearn can iterate
    class _PredefinedCV:
        def get_n_splits(self):  # noqa: D401
            return len(inner_splits)

        def split(self, *_):
            yield from inner_splits

    inner_cv = _PredefinedCV()

    bar = tqdm(total=n_windows, desc=f"Tuning {metadata['elec']} ({n_jobs} cores)")
    with Parallel(n_jobs=n_jobs, backend="loky") as pool:
        results = pool(
            delayed(_optuna_for_window)(
                Xw=X[..., w],
                y=y,
                groups=groups,
                hyperparam_cfg=hyperparam_cfg,
                base_estimator=base_estimator,
                inner_cv=inner_cv,
                metadata=metadata,
                random_state=random_state,
                win_idx=w,
                optuna_cfg=optuna_cfg,
                selector_cfg=selector_cfg,
                storage_dir=Path(storage_dir) if storage_dir else None,
            )
            for w in range(n_windows)
        )
        bar.update(n_windows)
    bar.close()
    return results
