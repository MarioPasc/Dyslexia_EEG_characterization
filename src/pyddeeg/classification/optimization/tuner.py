from __future__ import annotations

"""tuner.py – sequential *and* parallel Optuna tuning
======================================================
Drop‑in replacement that shows a **live, reliable progress‑bar** when the
parallel version is used.  Uses joblib’s *loky* backend (multi‑process,
no GIL contention) and integrates with tqdm via the lightweight
`tqdm‑joblib` shim if it is available on the system.  When the shim is
missing we fall back to a tiny in‑place monkey‑patch of joblib’s
``BatchCompletionCallBack`` so that the progress bar still advances.
"""

from typing import Any, Dict, List
import warnings

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm

from pyddeeg.classification import EEGDataset, OptunaEstimator
from pyddeeg.classification.optimization import (
    OPTUNA_FALLBACK_PATH,
    SELECTOR_FALLBACK_PATH,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _optuna_for_window(
    Xw: np.ndarray,
    y: np.ndarray,
    dataset: EEGDataset,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    base_estimator,
    random_state: int,
    win_idx: int,
    optuna_configuration: str | Path | Dict[str, Any] | None = OPTUNA_FALLBACK_PATH,
    selector_configuration: str | Path = SELECTOR_FALLBACK_PATH,
    storage_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run Optuna on **one** time‑window and return its ``best_params_`` dict."""
    window_ms = dataset.metadata.get("window_ms", "?")
    study_name = f"{dataset.metadata.get('direction','dir')}_{window_ms}ms_win{win_idx}"

    opt = OptunaEstimator(
        base_estimator=base_estimator,
        hyperparameters=hyperparam_cfg,
        dataset=dataset,
        config_yaml=optuna_configuration,
        selector_cfg_path=selector_configuration,
        random_state=random_state + win_idx,  # perturb seed per window
        storage_dir=storage_dir,
        study_name=study_name,
    )

    results = opt.fit(Xw, y)
    return {
        "best_params": results.best_params_,
        "performance": results.best_trial_performance,
    }


# -----------------------------------------------------------------------------
# Public API – sequential version
# -----------------------------------------------------------------------------


def tune_one_electrode(
    elec: str,
    dataset: EEGDataset,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    *,
    base_estimator,
    optuna_configuration: str | Path | Dict[str, Any] | None = OPTUNA_FALLBACK_PATH,
    selector_configuration: str | Path = SELECTOR_FALLBACK_PATH,
    random_state: int = 42,
    storage_dir: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Sequential hyper‑parameter tuning with a simple tqdm progress bar."""
    warnings.filterwarnings("ignore", message="Features .* are constant")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
    )

    X = np.concatenate([dataset.dd, dataset.ct])
    y = np.concatenate([np.ones(len(dataset.dd)), np.zeros(len(dataset.ct))])
    n_windows = X.shape[-1]

    bar = tqdm(range(n_windows), desc=f"Tuning {elec} (seq)")
    params: List[Dict[str, Any]] = []
    for w in bar:
        params.append(
            _optuna_for_window(
                Xw=X[..., w],
                y=y,
                dataset=dataset,
                hyperparam_cfg=hyperparam_cfg,
                base_estimator=base_estimator,
                random_state=random_state,
                win_idx=w,
                storage_dir=storage_dir,
                optuna_configuration=optuna_configuration,
                selector_configuration=selector_configuration,
            )
        )
    bar.close()
    return params


# -----------------------------------------------------------------------------
# Public API – parallel version with robust progress‑bar
# -----------------------------------------------------------------------------

try:
    from tqdm_joblib import tqdm_joblib  # type: ignore

    _HAS_TQDM_JOBLIB = True
except ImportError:  # pragma: no cover – optional dep
    _HAS_TQDM_JOBLIB = False


def _patch_joblib_for_tqdm(bar: tqdm):  # pragma: no cover
    """Monkey‑patch joblib so that *every* completed batch ticks the bar."""
    from joblib import parallel

    class _TqdmCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):  # noqa: D401
            bar.update(len(self.batch))
            return super().__call__(*args, **kwargs)

    parallel.BatchCompletionCallBack = _TqdmCallback  # type: ignore[attr-defined]


def tune_one_electrode_parallel(
    elec: str,
    dataset: EEGDataset,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    *,
    base_estimator,
    optuna_configuration: str | Path | Dict[str, Any] | None = OPTUNA_FALLBACK_PATH,
    selector_configuration: str | Path | Dict[str, Any] | None = SELECTOR_FALLBACK_PATH,
    random_state: int = 42,
    storage_dir: str | Path | None = None,
    n_jobs: int | None = None,
) -> List[Dict[str, Any]]:
    """Parallel window‑wise Optuna with a live tqdm progress bar."""
    warnings.filterwarnings("ignore", message="Features .* are constant")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
    )
    X = np.concatenate([dataset.dd, dataset.ct])
    y = np.concatenate([np.ones(len(dataset.dd)), np.zeros(len(dataset.ct))])
    n_windows = X.shape[-1]

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    bar = tqdm(total=n_windows, desc=f"Tuning {elec} (par, {n_jobs} jobs)")

    # ---- progress integration ---------------------------------------------
    if _HAS_TQDM_JOBLIB:
        cm = tqdm_joblib(bar)
    else:
        _patch_joblib_for_tqdm(bar)
        cm = nullcontext()  # type: ignore

    # ---- dispatch ----------------------------------------------------------
    from contextlib import nullcontext

    with cm:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_optuna_for_window)(
                Xw=X[..., w],
                y=y,
                dataset=dataset,
                hyperparam_cfg=hyperparam_cfg,
                base_estimator=base_estimator,
                random_state=random_state,
                win_idx=w,
                storage_dir=storage_dir,
                optuna_configuration=optuna_configuration,
                selector_configuration=selector_configuration,
            )
            for w in range(n_windows)
        )

    bar.close()
    return results
