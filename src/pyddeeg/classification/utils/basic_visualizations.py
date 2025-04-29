import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification.utils.time_domain_parser import window_to_time_domain


def plot_auc_and_selected_features(
    results: Dict[str, Any],
    dataset: EEGDataset,
    metric: Literal["roc", "pr"] = "roc",
    time_resolution_ms: int = 1,
    figsize: Tuple[float, float] = (12, 7),
    auc_color: Optional[str] = None,
    auc_label: Optional[str] = None,
    heatmap_cmap: str = "Greens",
    show_colorbar: bool = True,
    fig: Optional[Figure] = None,
    axes: Optional[Tuple[Axes, Axes]] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plot mean AUC (ROC or PR) over time and a heatmap of selected RQA features per window.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from evaluate_frozen_models. Must contain 'fold_auc' and 'selected_features'.
    dataset : EEGDataset
        Dataset object containing metadata for time domain conversion.
    metric : Literal["roc", "pr"], optional
        Which performance metric to plot: "roc" for ROC-AUC or "pr" for PR-AUC. Default is "roc".
    time_resolution_ms : int, optional
        Resolution of time axis in milliseconds. Default is 1.
    figsize : Tuple[float, float], optional
        Figure size in inches. Default is (12, 7).
    auc_color : Optional[str], optional
        Line color for AUC plot. If None, uses "C0" for ROC and "C1" for PR.
    auc_label : Optional[str], optional
        Label for the AUC plot. If None, uses metric name.
    heatmap_cmap : str, optional
        Colormap for the selected features heatmap. Default is "Greens".
    show_colorbar : bool, optional
        Whether to show colorbar for the heatmap. Default is True.
    fig : Optional[Figure], optional
        Existing figure to plot on. If None, a new figure is created.
    axes : Optional[Tuple[Axes, Axes]], optional
        Existing axes to plot on. If None, new axes are created.

    Returns
    -------
    Tuple[Figure, Tuple[Axes, Axes]]
        The figure and axes objects (fig, (ax1, ax2)) for further customization.

    Raises
    ------
    KeyError
        If required keys are missing in the results dictionary.
    ValueError
        If an invalid metric is specified.

    Example
    -------
    >>> fig, (ax1, ax2) = plot_auc_and_selected_features(results, dataset, metric="roc")
    >>> plt.show()
    """
    # --- Validate inputs ---
    if metric not in ("roc", "pr"):
        raise ValueError("metric must be 'roc' or 'pr'")
    if "fold_auc" not in results or "selected_features" not in results:
        raise KeyError(
            "Results dictionary must contain 'fold_auc' and 'selected_features'."
        )

    # --- Extract and process AUC data ---
    fold_auc = results["fold_auc"]  # shape: (n_folds, 2, n_windows)
    metric_idx = 0 if metric == "roc" else 1
    auc = fold_auc[:, metric_idx, :]  # shape: (n_folds, n_windows)
    auc_mean = np.mean(auc, axis=0)
    auc_std = np.std(auc, axis=0)

    # --- Convert AUC to time domain ---
    auc_mean_time, time_axis_ms = window_to_time_domain(
        auc_mean, dataset, window_dim=0, time_resolution_ms=time_resolution_ms
    )
    auc_std_time, _ = window_to_time_domain(
        auc_std, dataset, window_dim=0, time_resolution_ms=time_resolution_ms
    )
    time_axis_s = time_axis_ms / 1000.0

    # --- Prepare selected features heatmap ---
    selected_features = results["selected_features"]  # shape: (n_windows, n_metrics)
    n_metrics = selected_features.shape[1]
    selected_features_time = []

    for i in range(n_metrics):
        # Interpolate each metric's selection mask over time
        metric_time, _ = window_to_time_domain(
            selected_features[:, i],  # shape: (n_windows,)
            dataset,
            window_dim=0,
            time_resolution_ms=time_resolution_ms,
        )
        selected_features_time.append(metric_time)

    selected_features_time = np.vstack(
        selected_features_time
    )  # shape: (n_metrics, n_timepoints)
    selected_features_time = selected_features_time.astype(float)  # for imshow

    # --- Setup figure and axes ---
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
    ax1, ax2 = axes

    # --- Plot AUC curve ---
    if auc_color is None:
        auc_color = "C0" if metric == "roc" else "C1"
    if auc_label is None:
        auc_label = "ROC-AUC" if metric == "roc" else "PR-AUC"

    ax1.plot(time_axis_s, auc_mean_time, label=auc_label, color=auc_color)
    ax1.fill_between(
        time_axis_s,
        auc_mean_time - auc_std_time,
        auc_mean_time + auc_std_time,
        color=auc_color,
        alpha=0.2,
    )
    ax1.set_ylabel("AUC")
    ax1.set_title(f"Fold-wise {auc_label} over time")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Plot selected features heatmap ---
    im = ax2.imshow(
        selected_features_time,
        aspect="auto",
        extent=[time_axis_s[0], time_axis_s[-1], 0, len(RQA_METRICS)],
        cmap=heatmap_cmap,
        interpolation="nearest",
        origin="lower",
    )
    ax2.set_yticks(np.arange(len(RQA_METRICS)) + 0.5)
    ax2.set_yticklabels(RQA_METRICS)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RQA Metric")
    ax2.set_title("Selected features per window (1=selected)")
    if show_colorbar:
        plt.colorbar(im, ax=ax2, orientation="vertical", label="Selected")

    plt.tight_layout()
    return fig, (ax1, ax2)
