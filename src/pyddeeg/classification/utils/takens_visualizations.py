#!/usr/bin/env python3
"""visualize_rqa_takens.py

Generate a 2 × 3 error-bar visualization of tuned RQA parameters (τ, m, ε)
for CT and DD EEG-subject groups and export per-metric CSV summary tables.

Revision 4 – *Configurable central tendency*
-------------------------------------------
* New CLI flag **``--metric {mode,mean,median}``** chooses the statistic used for
  **τ** and **m** (ε always uses the mean):

  * ``mode``   → most-frequent value (previous default)
  * ``mean``   → arithmetic mean
  * ``median`` → sample median

* Implements helper functions ``_mean_1d`` and ``_median_1d``.
* Output CSV names now reflect the chosen statistic, e.g. ``tau_mode_std.csv`` or
  ``m_median_std.csv``.

Everything else (folder layout, plotting style, labels) is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_NAMES: Tuple[str, str, str] = ("tau", "m", "eps")
COLOURS: Dict[str, str] = {"CT": "tab:blue", "DD": "tab:orange"}

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def load_group_takens(
    data_root: Path,
    group: str,
    direction: str,
    window: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load window centres and Takens parameters for *group* (all subjects)."""

    folder = data_root / f"{group}_{direction}"
    files = sorted(folder.glob("*.npz"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need ≥2 *.npz files in {folder}; got {len(files)}.")

    metrics_npz = np.load(files[0], allow_pickle=True)
    takens_npz = np.load(files[1], allow_pickle=True)

    window_centers_ms = metrics_npz[f"{window}_centers"]
    takens = takens_npz[f"{window}_takens"]

    if takens.ndim != 3 or takens.shape[1] != 3:
        raise ValueError(
            f"Takens has shape {takens.shape}, expected (n_subjects, 3, n_windows)."
        )

    return window_centers_ms, takens


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------


def _mode_1d(arr: np.ndarray) -> float:
    """Return sample mode of a 1-D numeric array (tie-break: smallest value)."""

    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[np.argmax(counts)])


def _mean_1d(arr: np.ndarray) -> float:
    """Return arithmetic mean of a 1-D array."""

    return float(arr.mean())


def _median_1d(arr: np.ndarray) -> float:
    """Return median of a 1-D array."""

    return float(np.median(arr))


def central_and_std(
    takens: np.ndarray,
    central_fn: Callable[[np.ndarray], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute central-tendency and standard deviation arrays.

    *Indices 0 and 1* (τ, m) use *central_fn*; index 2 (ε) always uses mean.

    Parameters
    ----------
    takens : np.ndarray
        Shape (n_subjects, 3, n_windows).
    central_fn : Callable[[np.ndarray], float]
        Function returning a statistic for a 1-D array.

    Returns
    -------
    central : np.ndarray, shape (3, n_windows)
    std     : np.ndarray, shape (3, n_windows)
    """

    n_metrics, n_windows = takens.shape[1], takens.shape[2]
    central = np.empty((n_metrics, n_windows), dtype=float)
    std = np.empty_like(central)

    for idx in range(n_metrics):
        slice_ = takens[:, idx, :]  # (subjects, windows)
        if idx in (0, 1):  # tau or m
            central[idx] = np.apply_along_axis(central_fn, 0, slice_)
        else:  # eps → mean
            central[idx] = slice_.mean(axis=0)
        std[idx] = slice_.std(axis=0, ddof=1)

    return central, std


# -----------------------------------------------------------------------------
# DataFrame builder
# -----------------------------------------------------------------------------


def build_metric_dataframes(
    central: Dict[str, np.ndarray],
    std: Dict[str, np.ndarray],
    groups: List[str],
    central_label: str,
) -> Dict[str, pd.DataFrame]:
    """Return CSV-ready DataFrames of "central±std", row-indexed by group."""

    n_windows = next(iter(central.values())).shape[1]
    dfs: Dict[str, pd.DataFrame] = {}

    for metric_idx, metric in enumerate(METRIC_NAMES):
        label = central_label if metric in ("tau", "m") else "mean"
        rows = {
            g: [
                f"{central[g][metric_idx, w]:.3f}±{std[g][metric_idx, w]:.3f}"
                for w in range(n_windows)
            ]
            for g in groups
        }
        dfs[metric] = pd.DataFrame(rows, index=range(n_windows)).T
        dfs[metric].attrs["label"] = label  # store for naming later
    return dfs


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_metrics(
    window_centers_ms: np.ndarray,
    central: Dict[str, np.ndarray],
    std: Dict[str, np.ndarray],
    groups: List[str],
    output_path: Path,
) -> None:
    """Generate 2 × 3 error-bar figure adhering to the spec."""

    t_sec = window_centers_ms / 1000.0
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 6), sharex=True, constrained_layout=True
    )

    title_row0 = [r"$\tau$", "Embedding dimension (m)", r"$\varepsilon$"]

    for row, group in enumerate(groups):
        for col, metric in enumerate(METRIC_NAMES):
            ax = axes[row, col]
            colour = COLOURS.get(group, None)
            ax.errorbar(
                t_sec,
                central[group][col],
                yerr=std[group][col],
                fmt="o",
                markersize=3,
                capsize=2,
                elinewidth=0.8,
                color=colour,
                ecolor=colour,
                alpha=0.6,
            )
            if row == 0:
                ax.set_title(title_row0[col])
            if col == 0:
                ax.set_ylabel(group)
            if row == len(groups) - 1:
                ax.set_xlabel("Time (s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI parser
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise tuned RQA parameters with configurable central tendency and export tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="rqa_ch_* directory."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["CT", "DD"],
        help="Group folders to include (row order).",
    )
    parser.add_argument("--direction", default="UP", help="Folder suffix (e.g. 'UP').")
    parser.add_argument(
        "--window", default="w500", help="Window identifier inside the npz files."
    )
    parser.add_argument(
        "--metric",
        choices=["mode", "mean", "median"],
        default="mode",
        help="Statistic for τ and m (ε always mean).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for plots/ and csv/ sub-folders.",
    )
    parser.add_argument(
        "--fig-name", default="rqa_takens_over_time.png", help="Output figure name."
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:  # pragma: no-cover
    args = parse_args()

    if len(args.groups) != 2:
        raise ValueError("Exactly two groups expected for the 2 × 3 layout.")

    # Map metric string to function -----------------------------------------------------
    fn_map: Dict[str, Callable[[np.ndarray], float]] = {
        "mode": _mode_1d,
        "mean": _mean_1d,
        "median": _median_1d,
    }
    central_fn = fn_map[args.metric]

    central: Dict[str, np.ndarray] = {}
    std: Dict[str, np.ndarray] = {}
    window_centers_ms: np.ndarray | None = None

    # Load & compute -------------------------------------------------------------------
    for group in args.groups:
        centers, takens = load_group_takens(
            args.data_root, group, args.direction, args.window
        )
        if window_centers_ms is None:
            window_centers_ms = centers
        elif not np.array_equal(window_centers_ms, centers):
            raise ValueError("Window centres mismatch across groups.")

        central[group], std[group] = central_and_std(takens, central_fn)

    assert window_centers_ms is not None

    # Directories ----------------------------------------------------------------------
    plots_dir = args.output_dir / "plots"
    csv_dir = args.output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Plot -----------------------------------------------------------------------------
    plot_metrics(
        window_centers_ms, central, std, args.groups, plots_dir / args.fig_name
    )

    # CSV ------------------------------------------------------------------------------
    dfs = build_metric_dataframes(central, std, args.groups, args.metric)
    for metric, df in dfs.items():
        label = df.attrs.get("label", "mean")
        df.to_csv(csv_dir / f"{metric}_{label}_std.csv")

    print("✔ Outputs saved:")
    print(f"  Figure → {plots_dir / args.fig_name}")
    for metric in METRIC_NAMES:
        label = dfs[metric].attrs.get("label", "mean")
        print(f"  CSV    → {csv_dir / f'{metric}_{label}_std.csv'}")


if __name__ == "__main__":
    main()
