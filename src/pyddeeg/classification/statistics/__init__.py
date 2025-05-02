"""
Statistical tests for post-processing nested_evaluate results.
"""

from .cluster import cluster_permutation_test
from .window_perm import per_window_auc_test
from .crossval import cv_consistency_test
from .feature import (
    aggregate_selector_stats,
    feature_selection_binomial_test,
    feature_score_ttest,
)

__all__ = [
    "cluster_permutation_test",
    "per_window_auc_test",
    "cv_consistency_test",
    "aggregate_selector_stats",
    "feature_selection_binomial_test",
    "feature_score_ttest",
]
