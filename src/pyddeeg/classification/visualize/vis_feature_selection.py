#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 4 – Feature selection frequency with significance dots."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")


def plot_feature_sel(stats_path: str | Path, cv_path: str | Path) -> plt.Figure:
    st = load_npz(stats_path)
    cv = load_npz(cv_path)
    sel_freq = st["feature_agg"]["selection_frequency"]  # windows × feat
    freq_sig = st["feature_binom"]["significant_mask"]  # feat
    feat_names = cv["selector_stats"]["names"]

    t = time_axis(sel_freq.shape[0], Path(cv_path).parent.name.split("_")[2])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        sel_freq.T,
        cmap="viridis",
        ax=ax,
        cbar_kws=dict(label="Selection frequency"),
        xticklabels=np.round(t, 1),
        yticklabels=feat_names,
    )
    # overlay dots
    for feat_idx, sig in enumerate(freq_sig):
        if sig:
            ax.scatter(
                [-0.5], [feat_idx + 0.5], marker="*", color="white", s=80, clip_on=False
            )  # legend hack
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RQA metric")
    ax.set_title("Feature selection frequency\n(★ row significant > chance)")
    plt.xticks(rotation=45, ha="right")
    return fig


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fig = plot_feature_sel(args.stats, args.cv)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
