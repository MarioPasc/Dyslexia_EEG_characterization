#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 5 – Mean model coefficients across folds: heat-map with non-significant windows hatched."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")


def plot_coeff_heat(cv_path: str | Path, stats_path: str | Path) -> plt.Figure:
    cv = load_npz(cv_path)
    st = load_npz(stats_path)

    coefs = cv["coefficients"]  # folds × windows × feat
    mean_coef = np.nanmean(coefs, 0)  # windows × feat
    feat_names = cv["selector_stats"]["names"]

    t = time_axis(mean_coef.shape[0], Path(cv_path).parent.name.split("_")[2])

    # windows not individually sig
    nonsig = ~st["per_window"]["significant_mask"].astype(bool)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mean_coef.T,
        cmap="coolwarm",
        center=0,
        xticklabels=np.round(t, 1),
        yticklabels=feat_names,
        cbar_kws=dict(label="Mean coefficient (signed)"),
        ax=ax,
    )

    # hatch non-sig columns
    for idx in np.where(nonsig)[0]:
        ax.add_patch(
            plt.Rectangle(
                (idx, 0),
                1,
                mean_coef.shape[1],
                fill=False,
                hatch="////",
                edgecolor="black",
                lw=0.0,
            )
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RQA metric")
    ax.set_title("Model coefficients (hatched = AUC not sig.)")
    plt.xticks(rotation=45, ha="right")
    return fig


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fig = plot_coeff_heat(args.cv, args.stats)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
