#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 2 – Per-fold AUC distribution inside each significant cluster."""
from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")


def _cluster_labels(cluster_list: list[np.ndarray]) -> list[str]:
    return [f"c{i}" for i, _ in enumerate(cluster_list, 1)]


def _extract_cluster_auc(
    fold_auc: np.ndarray, clusters: list[np.ndarray], p_vals: np.ndarray, alpha: float
) -> pd.DataFrame:
    """Return long-form DF with columns: fold, cluster, auc."""
    rows: list[dict[str, float | int | str]] = []
    for label, idx, p in zip(_cluster_labels(clusters), clusters, p_vals):
        #if p >= alpha:
        #    continue
        for fold_idx, auc_vec in enumerate(fold_auc):
            rows.append(
                {"fold": fold_idx, "cluster": label, "auc": auc_vec[idx].mean()}
            )
    return pd.DataFrame(rows)


def plot_fold_box(cv_path: str | Path, stats_path: str | Path) -> plt.Figure:
    cv = load_npz(cv_path)
    st = load_npz(stats_path)

    fold_auc = cv["fold_auc"][:, 0]  # k × windows
    df = _extract_cluster_auc(
        fold_auc,
        st["cluster"]["clusters"],
        st["cluster"]["p_values"],
        st["cluster"]["alpha"],
    )
    if df.empty:
        raise RuntimeError("No significant clusters → Fig 2 not created.")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="cluster", y="auc", ax=ax)
    sns.swarmplot(data=df, x="cluster", y="auc", ax=ax, color=".25", size=4)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set(ylabel="Fold-mean ROC-AUC", xlabel="Significant cluster")
    sns.despine()
    return fig


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fig = plot_fold_box(args.cv, args.stats)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
