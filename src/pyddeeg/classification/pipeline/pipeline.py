from __future__ import annotations

"""Shared helpers for building sklearn pipelines."""

from typing import Sequence
from importlib import import_module

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


__all__: Sequence[str] = ("build_pipeline", "resolve_dotted")


def resolve_dotted(path: str) -> type:
    """
    Import a class from its dotted path (``"pkg.mod.ClassName"``).

    Raises
    ------
    (ImportError, AttributeError)
        If the module or attribute cannot be found.
    """
    module_path, _, cls_name = path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, cls_name)


def build_pipeline(
    *,
    selector: BaseEstimator,
    model: BaseEstimator,
    with_scaler: bool = True,
) -> Pipeline:
    """
    Assemble the preprocessing / modelling pipeline used throughout the project.

    Parameters
    ----------
    selector
        Any *instantiated* feature selector, e.g. ``SelectKBest(...)``.
    model
        Any *instantiated* classifier or regressor that supports ``fit``.
    with_scaler
        Prepend :class:`~sklearn.preprocessing.StandardScaler` when ``True``.
    """
    steps = []
    if with_scaler:
        steps.append(("scale", StandardScaler()))
    steps.extend(
        [
            ("selector", selector),
            ("model", model),
        ]
    )
    return Pipeline(steps)
