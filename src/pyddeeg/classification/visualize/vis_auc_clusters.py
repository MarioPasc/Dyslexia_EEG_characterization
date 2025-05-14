#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 1 – AUC × time with cluster & per-window significance overlays."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")


def plot_auc_clusters(
    cv_path: str | Path, stats_path: str | Path, title: str | None = None
) -> plt.Figure:
    """Return a Matplotlib Figure object – no file I/O here."""
    cv = load_npz(cv_path)
    st = load_npz(stats_path)

    fold_auc = cv["fold_auc"][:, 0]  # (k × windows) – ROC only
    t = time_axis(fold_auc.shape[1], Path(cv_path).parent.name.split("_")[2])

    mean_auc = fold_auc.mean(0)
    se_auc = fold_auc.std(0, ddof=1) / np.sqrt(fold_auc.shape[0])

    # --- plotting ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_auc, lw=2, label="Mean ROC-AUC")
    ax.fill_between(t, mean_auc - se_auc, mean_auc + se_auc, alpha=0.3)

    # horizontal line at chance
    ax.axhline(0.5, ls="--", lw=1, color="grey")

    # per-window FDR-significant
    sig_win = st["per_window"]["significant_mask"].astype(bool)
    ax.bar(
        t[sig_win],
        0.03,
        bottom=0.98,
        width=t[1] - t[0],
        color="firebrick",
        alpha=0.7,
        label="Per-window q < 0.05",
    )

    # cluster permutation significant (may overlap)
    clusters = st["cluster"]["clusters"]
    p_vals = st["cluster"]["p_values"]
    alpha = st["cluster"]["alpha"]
    for idx, p in zip(clusters, p_vals):
        idx = np.asarray(idx)
        if p < alpha:
            ax.axvspan(
                t[idx.min()] - (t[1] - t[0]) / 2,
                t[idx.max()] + (t[1] - t[0]) / 2,
                color="gold",
                alpha=0.25,
                label="Cluster p < 0.05",
            )

    ax.set(xlabel="Time (s)", ylabel="ROC-AUC", ylim=(0.4, 1.05))
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate legend
    by_lbl = dict(zip(labels, handles))
    ax.legend(by_lbl.values(), by_lbl.keys(), loc="lower right", frameon=False)
    sns.despine()
    return fig


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True, help="Path to cv_results.npz")
    ap.add_argument("--stats", required=True, help="Path to stats.npz")
    ap.add_argument("--out", type=Path, help="Figure file (png/svg/pdf)")
    args = ap.parse_args()

    fig = plot_auc_clusters(
        args.cv,
        args.stats,
        title=f"AUC & significant windows – {Path(args.cv).parent.name}",
    )
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
