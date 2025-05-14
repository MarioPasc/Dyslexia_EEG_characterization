#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 3 – Heat-map of mean(DD) − mean(CT) decision scores."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")


def plot_decision_heat(cv_path: str | Path) -> plt.Figure:
    cv = load_npz(cv_path)
    dec = cv["decision_scores"]  # subj × windows
    y = cv["labels"].astype(bool)
    diff = dec[y].mean(0) - dec[~y].mean(0)  # (windows,)

    t = time_axis(diff.size, Path(cv_path).parent.name.split("_")[2])

    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(
        diff[np.newaxis, :],
        cmap="coolwarm",
        center=0,
        cbar_kws=dict(label="Δ probability (DD − CT)"),
        ax=ax,
        xticklabels=np.round(t, 1),
        yticklabels=[],
    )
    ax.set_xlabel("Time (s)")
    ax.set_title("Effect size heat-map")
    plt.xticks(rotation=45, ha="right")
    return fig


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fig = plot_decision_heat(args.cv)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
