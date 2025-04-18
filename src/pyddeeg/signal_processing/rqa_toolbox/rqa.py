#!/usr/bin/env python3

import os
from typing import Union, Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pyunicorn.timeseries import RecurrencePlot

def compute_rqa_metrics_for_window(
    window_signal: np.ndarray,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int = 2,
    min_vertical_line: int = 2,
    min_white_vertical_line: int = 1,
    metrics_to_use: Optional[List[str]] = None,
    cache_recurrence_plot: bool = False,
) -> Tuple[Dict[str, float], Optional[RecurrencePlot]]:
    """
    Compute requested RQA metrics from a window of EEG signal.

    Parameters:
    -----------
    window_signal : np.ndarray
        Signal window to analyze
    embedding_dim, time_delay, radius, etc. : Various parameters
        RQA computation parameters
    cache_recurrence_plot : bool
        If True, return the RecurrencePlot object for potential reuse

    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary of computed metrics
    rp : Optional[RecurrencePlot]
        RecurrencePlot object if cache_recurrence_plot is True, otherwise None
    """

    if distance_metric.lower() == "meandist":
        from scipy.spatial.distance import pdist

        distances = pdist(window_signal.reshape(-1, 1), metric="euclidean")
        mean_dist = np.mean(distances) if len(distances) > 0 else 1.0
        normalized_radius = radius / mean_dist
        actual_radius = normalized_radius
        distance_metric = "euclidean"
    else:
        actual_radius = radius

    rp = RecurrencePlot(
        time_series=window_signal,
        dim=embedding_dim,
        tau=time_delay,
        metric=distance_metric,
        threshold=actual_radius,
        silence_level=2,
    )

    # Define metric computation functions
    metric_functions = {
        "RR": lambda: rp.recurrence_rate(),
        "DET": lambda: rp.determinism(l_min=min_diagonal_line),
        "L_max": lambda: rp.max_diaglength(),
        "L_mean": lambda: rp.average_diaglength(l_min=min_diagonal_line),
        "ENT": lambda: rp.diag_entropy(l_min=min_diagonal_line),
        "LAM": lambda: rp.laminarity(v_min=min_vertical_line),
        "TT": lambda: rp.trapping_time(v_min=min_vertical_line),
        "V_max": lambda: rp.max_vertlength(),
        "V_mean": lambda: rp.average_vertlength(v_min=min_vertical_line),
        "V_ENT": lambda: rp.vert_entropy(v_min=min_vertical_line),
        "W_max": lambda: rp.max_white_vertlength(),
        "W_mean": lambda: rp.average_white_vertlength(w_min=min_white_vertical_line),
        "W_ENT": lambda: rp.white_vert_entropy(w_min=min_white_vertical_line),
        "CLEAR": lambda: (
            rp.complexity_entropy() if hasattr(rp, "complexity_entropy") else None
        ),
        "PERM_ENT": lambda: (
            rp.permutation_entropy() if hasattr(rp, "permutation_entropy") else None
        ),
    }

    if metrics_to_use is None:
        metrics_to_use = metric_functions.keys() # Use all!

    # Compute requested metrics
    metrics = {}
    for metric in metrics_to_use:
        if metric in metric_functions:
            try:
                metrics[metric] = metric_functions[metric]()
            except Exception:
                metrics[metric] = None
        else:
            metrics[metric] = None

    if cache_recurrence_plot:
        return metrics, rp
    else:
        return metrics, None

def plot_rqa(
    data: np.ndarray,
    time: np.ndarray,
    dim: int = 1,
    tau: int = 1,
    threshold: float = 0.5,
    metric: str = "euclidean",
    show: bool = True,
    save_path: Optional[Union[str, os.PathLike]] = None,
    save_format: str = "svg",
) -> Dict[str, float]:
    """
    Generate and visualize a Recurrence Plot of a time series using PyUnicorn,
    optionally display the figure, save it to disk, and return RQA metrics.

    This version removes x-axis labels from the RQA plot itself and places them
    on the top subplot. It also shares the y-axis ticks with the left subplot.

    Parameters
    ----------
    data : np.ndarray
        The 1D time-series data to analyze (e.g., EEG amplitude).
    time : np.ndarray
        The corresponding time points for `data`.
    dim : int, optional
        Embedding dimension for phase space reconstruction. Default is 1.
    tau : int, optional
        Time delay for phase space reconstruction. Default is 1.
    threshold : float, optional
        Distance threshold for determining recurrence. Default is 0.5.
    metric : str, optional
        Distance metric to use (e.g. 'euclidean'). Default is 'euclidean'.
    show : bool, optional
        If True, displays the plot. If False, the plot is not shown. Default is True.
    save_path : str or os.PathLike, optional
        If provided, the path where the figure is saved. Default is None (no saving).
    save_format : str, optional
        The format to use when saving (e.g., 'svg', 'png'). Default is 'svg'.

    Returns
    -------
    Dict[str, float]
        A dictionary containing common RQA metrics, e.g.:
        {
          "Recurrence Rate": ...,
          "Determinism": ...,
          "Laminarity": ...,
          ...
        }
    """
    # 1) Create the RecurrencePlot object
    rp = RecurrencePlot(
        data, dim=dim, tau=tau, threshold=threshold, metric=metric, normalize=False
    )
    rec_matrix = rp.recurrence_matrix()

    # 2) Set up the figure and GridSpec
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1, 4],
        height_ratios=[1, 4],
        wspace=0.05,
        hspace=0.05,
    )

    # We'll create the RQA axis first so the others can share x/y with it
    ax_rp = fig.add_subplot(gs[1, 1])

    # The top subplot shares the x-axis with the RQA (same time range)
    ax_signal_top = fig.add_subplot(gs[0, 1], sharex=ax_rp)
    # The left subplot shares the y-axis with the RQA (same time range)
    ax_signal_left = fig.add_subplot(gs[1, 0], sharey=ax_rp)

    # 2a) Plot the Recurrence Matrix in the center
    #     We'll define extent so x and y both range from time[0] to time[-1]
    im = ax_rp.imshow(
        rec_matrix,
        cmap="binary",
        origin="lower",
        extent=[time[0], time[-1], time[0], time[-1]],  # type: ignore
    )
    # Remove axis labels and numeric labels on the RQA
    ax_rp.set_xlabel(None)  # type: ignore
    ax_rp.set_ylabel(None)  # type: ignore
    # Hide numeric tick labels on both axes (but keep tick *lines* so they match up)
    ax_rp.tick_params(labelbottom=False, labelleft=False)

    # 2b) Plot the signal on the top axis (time vs. amplitude)
    #     Since ax_signal_top shares x with ax_rp, the x-range is time[0] to time[-1]
    ax_signal_top.plot(time, data, color="C0")
    # Let the top axis show the time ticks (matching the RQA’s x-range)
    ax_signal_top.set_xlim([time[0], time[-1]])  # type: ignore
    # We'll define some evenly spaced tick locations (adjust to preference)
    n_ticks = 7
    x_ticks = np.linspace(time[0], time[-1], n_ticks)

    ax_signal_top.set_xticks(x_ticks)
    ax_signal_top.set_xticklabels([f"{v:.1f}" for v in x_ticks])
    ax_signal_top.set_xlabel("Time")
    ax_signal_top.xaxis.set_label_position("top")

    ax_signal_top.set_ylabel("Signal")
    ax_signal_top.yaxis.set_label_position("right")
    ax_signal_top.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    # 2c) Plot the signal on the left axis (data vs. time),
    #     which shares the y-axis with ax_rp
    ax_signal_left.plot(data, time, color="C0")
    # We'll keep the y-range consistent with the RQA: [time[0], time[-1]]
    ax_signal_left.set_ylim([time[0], time[-1]])  # type: ignore
    y_ticks = np.linspace(time[0], time[-1], n_ticks)
    ax_signal_left.set_yticks(y_ticks)
    ax_signal_left.set_yticklabels([f"{v:.1f}" for v in y_ticks])
    ax_signal_left.set_ylabel("Time")

    # For the x-range (data), define ticks as you like
    ax_signal_left.set_xlabel("Signal")

    # Optionally invert the x-axis to mimic the original style
    # ax_signal_left.invert_xaxis()

    # 3) Compute some RQA metrics
    rqa_metrics = {
        "Recurrence Rate": rp.recurrence_rate(),
        "Determinism": rp.determinism(),
        "Laminarity": rp.laminarity(),
        "Trapping Time": rp.trapping_time(),
    }

    # 4) Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path, format=save_format, bbox_inches="tight")

    # 5) Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return rqa_metrics
