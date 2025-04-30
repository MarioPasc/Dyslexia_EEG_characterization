from __future__ import annotations

"""Utilities for reading the *selector* configuration YAML used by
both Optuna tuning and the final evaluation stage.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from sklearn.base import BaseEstimator
from importlib import import_module

import optuna


# --------------------------------------------------------------------------- #
# Helper: import "pkg.module.ClassName" → <class '...'>
# --------------------------------------------------------------------------- #
def _resolve_dotted(path: str) -> Any:
    mod, _, name = path.rpartition(".")
    return getattr(import_module(mod), name)


# --------------------------------------------------------------------------- #
# Helper: auto-import dotted strings used as parameter *values*
# --------------------------------------------------------------------------- #
def _maybe_import(value: Any) -> Any:
    """If *value* is a dotted-path string and can be imported, return the
    imported object; otherwise return *value* unchanged."""
    if isinstance(value, str) and "." in value:
        try:
            return _resolve_dotted(value)
        except (ImportError, AttributeError):
            pass
    return value


# --------------------------------------------------------------------------- #
# Helper: turn the YAML spec into (instance, search_space)
# --------------------------------------------------------------------------- #
def load_selector(
    cfg_path: Union[str, Path],
    trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Read ``cfg_path`` and build the feature selector.

    Parameters
    ----------
    cfg_path
        YAML file with keys ``class`` and ``params``.
    trial
        When given, *sample* the parameters from ``params`` using Optuna
        and return the search-space that was consumed.  When *None*, any
        nested dictionaries are treated as **fixed** values (use this path
        during re-training / evaluation).

    Returns
    -------
    selector
        Instantiated feature selector.
    search_space
        The (possibly empty) parameter dict associated with this selector.
        Returned mainly so that ``OptunaEstimator`` can record *all* best
        parameters in ``.best_params_``.
    """
    cfg_path = Path(cfg_path).expanduser()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    cls_path: str = cfg["class"]
    space: Dict[str, Any] = cfg.get("params", {})

    cls = _resolve_dotted(cls_path)

    # --- no tuning --------------------------------------------------------- #
    if trial is None or optuna is None:
        fixed: Dict[str, Any] = {}
        for k, v in space.items():
            if isinstance(v, dict):  # e.g. {"value": 10}
                v = v.get("value")
            fixed[k] = _maybe_import(v)
        return cls(**fixed), fixed

    # --- tuning with Optuna ------------------------------------------------- #
    # We support three primitive distributions:
    #   int   → {"low": 10, "high": 50}
    #   float → {"low": .1, "high": 1.5, "log": true}
    #   choice→ {"choices": [1, 5, 10]}
    #
    suggested: Dict[str, Any] = {}
    for name, spec in space.items():
        if not isinstance(spec, dict):  # constant
            suggested[name] = _maybe_import(spec)
            continue

        if "choices" in spec:  # categorical
            suggested[name] = trial.suggest_categorical(name, spec["choices"])
            choice = trial.suggest_categorical(name, spec["choices"])
            suggested[name] = _maybe_import(choice)
        elif all(k in spec for k in ("low", "high")):
            low, high = spec["low"], spec["high"]
            if isinstance(low, int) and isinstance(high, int):
                suggested[name] = trial.suggest_int(name, low, high)
            else:
                suggested[name] = trial.suggest_float(
                    name, float(low), float(high), log=spec.get("log", False)
                )
        else:  # pragma: no cover
            raise ValueError(f"Unknown selector param spec: {spec!r}")

    return cls(**suggested), suggested
