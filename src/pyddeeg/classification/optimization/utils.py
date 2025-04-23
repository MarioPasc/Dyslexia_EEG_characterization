from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

import optuna

__all__: Tuple[str, ...] = (
    "suggest_from_config",
    "load_optuna_config",
    "suggest_from_config",
)


def _instantiate(cls_path: str, params: Dict[str, Any]):
    """
    Import `cls_path` (e.g. "optuna.samplers.TPESampler") and return
    an *instance* with the given keyword arguments.
    """
    mod_path, _, cls_name = cls_path.rpartition(".")
    mod = __import__(mod_path, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls(**params)


def load_optuna_config(cfg: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a YAML (or already-parsed dict) with the following *optional* keys
    and return a dict ready to feed into `OptunaEstimator`.

    ```yaml
    n_trials: 120
    random_state: 42
    scoring: roc_auc
    sampler:
      TPESampler:
        n_startup_trials: 10
        n_ei_candidates: 24
    pruner:
      MedianPruner:
        n_warmup_steps: 5
    save_trials:
      output_dir: /tmp/optuna_runs
      single_db: true            # if false → one-file-per-study
    ```
    """
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(Path(cfg).read_text())

    out: Dict[str, Any] = {}
    out["n_trials"] = int(cfg.get("n_trials", 50))  # type: ignore
    out["random_state"] = cfg.get("random_state")  # type: ignore
    out["scoring"] = cfg.get("scoring", "roc_auc")  # type: ignore

    # Sampler
    sampler = cfg.get("sampler")  # type: ignore
    if sampler:
        ((name, params),) = sampler.items()
        full = f"optuna.samplers.{name}"
        out["sampler_obj"] = _instantiate(full, params or {})
    else:
        out["sampler_obj"] = optuna.samplers.TPESampler(seed=out["random_state"])

    # Pruner
    pruner = cfg.get("pruner")  # type: ignore
    if pruner:
        ((name, params),) = pruner.items()
        full = f"optuna.pruners.{name}"
        out["pruner_obj"] = _instantiate(full, params or {})
    else:
        out["pruner_obj"] = None

    # Storage
    save_cfg = cfg.get("save_trials")  # type: ignore
    if save_cfg:
        out["storage_dir"] = Path(save_cfg["output_dir"]).expanduser()
        out["single_db"] = bool(save_cfg.get("single_db", True))
    else:
        out["storage_dir"] = None
        out["single_db"] = True

    return out


def suggest_from_config(
    trial: optuna.Trial,
    hyperparameters: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Construct a dict of suggested parameter values from a configuration mapping.

    Parameters
    ----------
    trial
        An Optuna Trial object.
    hyperparameters
        Mapping of parameter names to their search definitions. Each entry
        must include:
          - "type": one of ("loguniform", "float", "integer", "categorical").
          - "min": lower bound (for float/loguniform/integer).
          - "max": upper bound (for float/loguniform/integer).
          - "choices": sequence of values (for categorical).

    Returns
    -------
    params
        Dict mapping parameter names to trial-suggested values.
    """
    params: Dict[str, Any] = {}
    for name, cfg in hyperparameters.items():
        t = cfg.get("type")
        if t == "loguniform":
            low = float(cfg["min"])
            high = float(cfg["max"])
            params[name] = trial.suggest_float(name, low, high, log=True)
        elif t == "float":
            low = float(cfg["min"])
            high = float(cfg["max"])
            params[name] = trial.suggest_float(name, low, high, log=False)
        elif t == "integer":
            low = int(cfg["min"])
            high = int(cfg["max"])
            step = int(cfg.get("step", 1))
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif t == "categorical":
            choices = cfg.get("choices")
            if choices is None:
                raise ValueError(
                    f"Categorical parameter '{name}' requires 'choices' list."
                )
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unsupported hyperparameter type '{t}' for '{name}'.")
    return params
