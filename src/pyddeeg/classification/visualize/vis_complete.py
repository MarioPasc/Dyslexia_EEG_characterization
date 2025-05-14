#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG leverage figure – 2-column version (labels on right)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddeeg.classification.utils.visualize_utils import load_npz, time_axis

sns.set_context("talk")

# ───────────────────────── helpers ─────────────────────────
def _scan_root(root: Path) -> Dict[str, Dict[str, Path]]:
    out = {}
    for d in root.iterdir():
        if d.is_dir():
            elec = d.name.split("_")[0]
            out[elec] = {"cv": d / "cv_results.npz", "stats": d / "stats.npz"}
    return out


def _step_arrays(values: np.ndarray, *, window_ms: int, hop_ms: int):
    n = values.size
    edges = np.arange(n + 1) * hop_ms
    x = np.repeat(edges, 2)[1:].astype(float)
    y = np.repeat(values, 2)
    x = np.append(x, edges[-1] + window_ms)
    y = np.append(y, [values[-1], values[-1]])
    return x / 1_000, y


def _cluster_bounds(idx_like):
    arr = np.asarray(idx_like).ravel()
    return int(arr.min()), int(arr.max())


# ──────────────────────── main builder ─────────────────────
def leverage_figure(root: Path, *, hop_ms: int | None = None,
                    electrode_order: List[str] | None = None) -> plt.Figure:
    info = _scan_root(root)
    if electrode_order is None:
        electrode_order = sorted(info)

    n_elec   = len(electrode_order)
    row_h    = 1.2                       # ← was 0.9
    fig_h    = n_elec * row_h * 1.5
    fig, axs = plt.subplots(
        n_elec, 2, figsize=(17, fig_h),
        sharex="col",
        gridspec_kw=dict(width_ratios=[1.4, 1.0], hspace=0.35, wspace=0.05),
    )

    window_ms = None
    dur_s     = None
    metrics   = None

    for r, elec in enumerate(electrode_order):
        ax_auc, ax_hm = axs[r]

        cv = load_npz(info[elec]["cv"])
        st = load_npz(info[elec]["stats"])

        if window_ms is None:
            window_ms = int(info[elec]["cv"].parent.name.split("_")[2])
        hop = hop_ms or window_ms // 2

        # ─── ROC-AUC panel ────────────────────────────────────────
        fold_auc = cv["fold_auc"][:, 0]
        mean_auc = fold_auc.mean(0)
        std_auc  = fold_auc.std(0, ddof=1)

        x, y  = _step_arrays(mean_auc, window_ms=window_ms, hop_ms=hop)
        _, lo = _step_arrays(mean_auc - std_auc, window_ms=window_ms, hop_ms=hop)
        _, hi = _step_arrays(mean_auc + std_auc, window_ms=window_ms, hop_ms=hop)

        ax_auc.fill_between(x, lo, hi, step="post", alpha=0.25)
        ax_auc.plot(x, y, drawstyle="steps-post", lw=1.5)

        ax_auc.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax_auc.set_ylim(0.0, 1.05)                  # full 0–1 range
        ax_auc.set_yticks(np.arange(0, 1.1, 0.2))
        ax_auc.set_ylabel(elec, rotation=0, ha="right", va="center",
                          labelpad=25, fontsize=9)
        ax_auc.tick_params(axis="x", length=0)

        clusters = st["cluster"]["clusters"]
        p_vals   = st["cluster"]["p_values"]
        alpha_th = st["cluster"].get("alpha", 0.05)
        for idx, p in zip(clusters, p_vals):
            s_i, e_i = _cluster_bounds(idx)
            start = (s_i * hop) / 1_000
            end   = (e_i * hop + window_ms) / 1_000
            ax_auc.axvspan(start, end, color=(0.6, 0.6, 0.6, 0.25), linewidth=0)
            if p < alpha_th:
                ax_auc.plot((start + end) / 2, 1.02, "*", color="black", ms=6, clip_on=False)

        if dur_s is None:
            dur_ms = (mean_auc.size - 1) * hop + window_ms
            dur_s  = dur_ms / 1_000

        # ─── heat-map panel ───────────────────────────────────────
        sel_freq = st["feature_agg"]["selection_frequency"]
        if metrics is None:
            metrics = cv["selector_stats"]["names"]

        sns.heatmap(
            sel_freq.T,
            vmin=0, vmax=1, cmap="rocket_r",
            ax=ax_hm, cbar=False,
            xticklabels=False,
            yticklabels=(metrics if r == 0 else False),
        )

        # put metric labels on the right
        ax_hm.yaxis.tick_right()
        ax_hm.yaxis.set_label_position("right")
        ax_hm.tick_params(axis="y", pad=8)
        ax_hm.set_yticks([])
        ax_hm.tick_params(axis="x", length=0)
        if r == 0:
            ax_hm.set_title("Selection frequency")
        if r == n_elec - 1:
            ax_auc.set_xlabel("Time (s)")
            ax_hm.set_xlabel("Time (s)")

    # ─── shared x-axis config ───────────────────────────────────────
    for ax_c in axs[:, 0]:
        ax_c.set_xlim(0, dur_s)
        ax_c.set_xticks(np.arange(0, dur_s + 0.1, 10.0))
        ax_c.set_xticklabels(ax_c.get_xticks().astype(int))

    axs[0, 0].set_title("ROC-AUC (mean ± SD)   –   shaded = cluster, ★ p < 0.05", pad=10)
    sns.despine(fig)
    plt.tight_layout()
    return fig


# ───────────────────────── CLI wrapper ──────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hop", type=int, default=None,
                    help="Hop ms (default = ½ window length)")
    args = ap.parse_args()

    fig = leverage_figure(args.root, hop_ms=args.hop)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    _cli()
