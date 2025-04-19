import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, Literal, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification.utils.time_domain_parser import window_to_time_domain
from pyddeeg.classification import logger

def plot_classification_results(
    results: Dict[str, Any],
    electrode: str,
    dataset: EEGDataset,
    metric: Literal["roc", "pr"] = "roc",
    time_in_ms: bool = True,
    time_resolution_ms: int = 1,
    fig: Optional[Figure] = None,
    axes: Optional[Tuple[Axes, Axes]] = None,
    **kwargs
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plot classification performance and patterns from electrode classification results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from classification_per_electrode function.
        Must contain 'fold_auc' tensor and 'patterns'.
    electrode : str
        Name of the electrode used for classification (e.g., 'Fz', 'T7').
    dataset : EEGDataset
        Dataset object containing metadata for time domain conversion.
    metric : Literal["roc", "pr"], optional
        Which performance metric to plot: "roc" for ROC-AUC or "pr" for Precision-Recall AUC.
        Default is "roc".
    time_in_ms : bool, optional
        If True, use millisecond time axis. If False, show window indices.
        Default is True.
    time_resolution_ms : int, optional
        Resolution of time axis in milliseconds when time_in_ms=True. Default is 1ms.
    fig : Optional[Figure], optional
        Existing figure to plot on. If None, a new figure is created.
    axes : Optional[Tuple[Axes, Axes]], optional
        Existing axes to plot on. If None, new axes are created.
    **kwargs
        Additional keyword arguments for styling and customization:
        - figsize : Tuple[float, float] - Figure size in inches (default: (14, 10))
        - cmap : str - Colormap for patterns heatmap (default: "RdBu_r")
        - title : str - Custom plot title
        - colorbar_kwargs : dict - Arguments for colorbar
        - performance_kwargs : dict - Styling arguments for performance plot
        - patterns_kwargs : dict - Styling arguments for patterns plot
        - Any other matplotlib parameters for Figure and Axes
    
    Returns
    -------
    Tuple[Figure, Tuple[Axes, Axes]]
        The figure and axes objects (fig, (ax1, ax2)) for further customization if needed.
    
    Raises
    ------
    KeyError
        If the required keys are not found in the results dictionary.
    ValueError
        If an invalid metric is specified.
        
    Examples
    --------
    >>> results = trainer.classification_per_electrode("T7", dataset, model)
    >>> fig, (ax1, ax2) = plot_classification_results(
    ...     results, 
    ...     "T7", 
    ...     dataset, 
    ...     metric="roc",
    ...     ylim=(0.4, 1.0),
    ...     figsize=(12, 8),
    ...     title="Custom title",
    ...     performance_kwargs={"color": "blue", "linewidth": 2},
    ...     patterns_kwargs={"cmap": "viridis"}
    ... )
    >>> plt.show()
    """
    # Validate inputs
    if metric not in ["roc", "pr"]:
        raise ValueError("metric must be 'roc' or 'pr'")
    
    if "fold_auc" not in results or "patterns" not in results:
        raise KeyError("Results dictionary missing required keys: 'fold_auc' or 'patterns'")
    
    # Extract the AUC values and compute mean/std
    metric_idx = 0 if metric == "roc" else 1  # 0 for ROC-AUC, 1 for PR-AUC
    fold_auc = results["fold_auc"]
    
    # Check if fold_auc has the expected structure
    if fold_auc.ndim != 3 or fold_auc.shape[1] != 2:
        raise ValueError(f"Expected fold_auc shape (n_folds, 2, n_windows), got {fold_auc.shape}")
        
    logger.debug(f"Computing {metric} metrics over {fold_auc.shape[0]} folds")
    
    # Calculate mean and std across folds
    mean_perf = np.mean(fold_auc[:, metric_idx, :], axis=0)
    std_perf = np.std(fold_auc[:, metric_idx, :], axis=0)
    
    # Extract common styling parameters from kwargs
    figsize = kwargs.pop("figsize", (14, 10))
    cmap = kwargs.pop("cmap", "RdBu_r")
    title = kwargs.pop("title", None)
    performance_kwargs = kwargs.pop("performance_kwargs", {})
    patterns_kwargs = kwargs.pop("patterns_kwargs", {})
    colorbar_kwargs = kwargs.pop("colorbar_kwargs", {})
    
    # Create figure and axes if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    ax1, ax2 = axes
    
    # Get patterns
    patterns = results["patterns"]
    
    # Determine x-axis values
    if time_in_ms:
        # Convert window-indexed data to time domain
        mean_perf_time, time_axis = window_to_time_domain(
            mean_perf, 
            dataset, 
            window_dim=0, 
            time_resolution_ms=time_resolution_ms
        )
        
        std_perf_time, _ = window_to_time_domain(
            std_perf, 
            dataset, 
            window_dim=0, 
            time_resolution_ms=time_resolution_ms
        )
        
        patterns_time, _ = window_to_time_domain(
            patterns, 
            dataset, 
            window_dim=1, 
            time_resolution_ms=time_resolution_ms
        )
        
        x_values = time_axis
        x_label = 'Time (ms)'
        plot_mean = mean_perf_time
        plot_std = std_perf_time
        plot_patterns = patterns_time
    else:
        x_values = np.arange(len(mean_perf))
        x_label = 'Window index'
        plot_mean = mean_perf
        plot_std = std_perf
        plot_patterns = patterns
    
    # Plot performance metric (ROC-AUC or PR-AUC)
    # Default plotting parameters
    default_perf_kwargs = {
        "color": "blue",
        "linewidth": 1.5,
        "label": None
    }
    # Update defaults with user-provided kwargs
    plot_kwargs = {**default_perf_kwargs, **performance_kwargs}
    
    line = ax1.plot(x_values, plot_mean, **plot_kwargs)
    
    # Default fill_between params
    fill_kwargs = {
        "alpha": 0.3,
        "color": plot_kwargs.get("color", "blue")
    }
    ax1.fill_between(x_values, plot_mean - plot_std, plot_mean + plot_std, **fill_kwargs)
    
    # Add baseline at 0.5
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    metric_label = "ROC AUC" if metric == "roc" else "Precision-Recall AUC"
    ax1.set_ylabel(metric_label)
    
    # Apply any additional ax1 stylings from kwargs
    for key, value in kwargs.items():
        if hasattr(ax1, f"set_{key}"):
            getattr(ax1, f"set_{key}")(value)
    
    # Set title
    if title is None:
        title = f'Classification performance and patterns for {electrode} electrode ({metric_label})'
    ax1.set_title(title)
    
    # Default patterns plot parameters
    default_patterns_kwargs = {
        "cmap": cmap,
        "aspect": "auto",
    }
    im_kwargs = {**default_patterns_kwargs, **patterns_kwargs}
    
    # Plot pattern heatmap
    im = ax2.imshow(plot_patterns, 
                   extent=[min(x_values), max(x_values), 15, 0],
                   **im_kwargs)
    
    ax2.set_yticks(np.arange(0.5, 15.5))
    ax2.set_yticklabels(RQA_METRICS)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('RQA Metrics')
    
    # Apply any additional ax2 stylings from kwargs
    for key, value in kwargs.items():
        if hasattr(ax2, f"set_{key}"):
            getattr(ax2, f"set_{key}")(value)
    
    # Default colorbar params
    default_cbar_kwargs = {
        "label": "Pattern weight",
        "orientation": "horizontal"
    }
    cbar_kwargs = {**default_cbar_kwargs, **colorbar_kwargs}
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, **cbar_kwargs)
    
    plt.tight_layout()
    return fig, (ax1, ax2)