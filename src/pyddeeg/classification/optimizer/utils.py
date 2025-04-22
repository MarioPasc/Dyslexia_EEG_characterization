from typing import Any, Dict, Union
import optuna


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
            params[name] = trial.suggest_loguniform(name, low, high)
        elif t == "float":
            low = float(cfg["min"])
            high = float(cfg["max"])
            params[name] = trial.suggest_float(name, low, high)
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
