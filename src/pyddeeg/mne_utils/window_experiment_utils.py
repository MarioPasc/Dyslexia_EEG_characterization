"""
MNE utilities for EEG data analysis and visualization.

This module provides utilities for converting RQA metric data into MNE objects
and creating standardized visualizations for EEG analysis.
"""

from typing import Dict, List, Optional, Union, Any
import os
import json
import numpy as np
import mne
from mne.channels import make_standard_montage
from pyddeeg import METRIC_NAME_TO_INDEX, EEG_CHANNELS
import matplotlib.pyplot as plt

def load_dataset_indexes(root_path: str) -> Dict[str, Any]:
    """
    Load dataset index file that maps window sizes, directions, and electrodes to file paths.
    
    Parameters
    ----------
    root_path : str
        Path to the root directory containing the processed RQA data.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping window size, direction, and electrode to file paths.
    
    Raises
    ------
    FileNotFoundError
        If the dataset index file cannot be found.
    """
    electrode_indexed_path = os.path.join(root_path, "electrode_indexed")
    index_file = os.path.join(electrode_indexed_path, "dataset_index.json")
    
    try:
        with open(index_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset index file not found at {index_file}")


def get_patient_evoked(
    window_size: str,
    direction: str,
    group: str,
    metric_name: str,
    patient_index: int,
    dataset_indexes: Dict[str, Any],
    sfreq: float = 1.0,
    tmin: float = 0.0,
    montage_name: str = "standard_1020"
) -> mne.Evoked:
    """
    Extract patient data for a specific metric across all electrodes and 
    convert it to an MNE Evoked object.

    Parameters
    ----------
    window_size : str
        Size of the window (e.g., 'window_50').
    direction : str
        Direction of stimulus ('up' or 'down').
    group : str
        Patient group ('CT' or 'DD').
    metric_name : str
        Name of the RQA metric.
    patient_index : int
        Index of the patient in the .npz file.
    dataset_indexes : Dict[str, Any]
        Dictionary mapping window size and direction to file paths.
    sfreq : float, optional
        Sampling frequency in Hz, defaults to 1.0 (one value per window).
    tmin : float, optional
        Start time of the data in seconds, defaults to 0.0.
    montage_name : str, optional
        Name of the standard montage used, defaults to "standard_1020".

    Returns
    -------
    mne.Evoked
        MNE Evoked object containing the specified patient's RQA data 
        for all channels across time windows.
    
    Raises
    ------
    ValueError
        If the specified parameters are invalid or no data is found.
    """
    # Validate metric
    if metric_name not in METRIC_NAME_TO_INDEX:
        raise ValueError(f"Invalid metric_name. Options: {list(METRIC_NAME_TO_INDEX.keys())}")
    
    metric_index = METRIC_NAME_TO_INDEX[metric_name]

    # Collect the data (channels x time_windows)
    electrode_data = []
    electrode_names = []

    # Validate window size and direction
    if window_size not in dataset_indexes:
        raise ValueError(f"Invalid window_size. Options: {list(dataset_indexes.keys())}")
    
    if direction not in dataset_indexes[window_size]:
        raise ValueError(f"Invalid direction. Options: {list(dataset_indexes[window_size].keys())}")

    electrodes = list(dataset_indexes[window_size][direction].keys())

    for electrode in electrodes:
        file_paths = dataset_indexes[window_size][direction][electrode]
        file_path = None

        # Look for the correct file in the specified group
        for path in file_paths:
            filename = os.path.basename(path)
            if f"{electrode}_{group}_{direction.upper()}" in filename:
                file_path = path
                break
        
        # Skip if no file found
        if file_path is None:
            print(f"Warning: No data found for electrode {electrode} in group {group}")
            continue

        # Load the data
        try:
            npz_file = np.load(file_path)
            metrics_data = npz_file['metrics']

            # Check if patient index is valid
            if patient_index >= metrics_data.shape[0]:
                print(f"Warning: Patient index {patient_index} out of range for {electrode} in group {group}")
                continue

            # Extract data for the specified patient and metric
            data = metrics_data[patient_index, metric_index, :]

            # Append to main list
            electrode_data.append(data)
            electrode_names.append(electrode)

        except Exception as e:
            print(f"Error loading data for electrode {electrode}: {e}")

    if not electrode_data:
        raise ValueError("No valid data found for the specified parameters")

    # Convert list to (n_channels, n_times)
    data_tensor = np.array(electrode_data)

    # Create MNE Info
    ch_types = ['eeg'] * len(electrode_names)
    info = mne.create_info(ch_names=electrode_names, sfreq=sfreq, ch_types=ch_types)
    
    # Build an EvokedArray from the data
    evoked = mne.EvokedArray(data_tensor, info, tmin=tmin)

    # Set montage (electrode locations)
    montage = make_standard_montage(montage_name)
    evoked.set_montage(montage)
    
    return evoked


def create_joint_plot(
    evoked: mne.Evoked,
    metric_name: str,
    title: Optional[str] = None,
    times: Union[str, List[float]] = "peaks",
    picks: str = "eeg",
    exclude: str = "bads",
    show: bool = False
) -> plt.figure:
    """
    Create a joint plot with time series and topomaps using MNE.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked object containing the data to plot.
    metric_name : str
        Name of the RQA metric being displayed.
    title : str, optional
        Title for the plot. If None, no title is displayed.
    times : str or list of float, optional
        Times to plot. Can be "peaks" or list of time points.
        Default is "peaks".
    picks : str, optional
        Channels to plot. Default is "eeg".
    exclude : str, optional
        Channels to exclude. Default is "bads".
    show : bool, optional
        Whether to show the plot immediately. Default is False.
    
    Returns
    -------
    mne.viz.Figure
        The created figure object.
    """
    # Configure plot arguments
    ts_args = dict(
        spatial_colors=True,
        zorder='std',
        units=dict(eeg=metric_name),
        scalings=dict(eeg=1),
    )
    
    topomap_args = dict(
        outlines='head',
        contours=0
    )
    
    # Temporarily modify evoked attributes to prevent auto-appended titles
    old_nave = evoked.nave
    old_comment = evoked.comment
    
    evoked.nave = None  # Prevents "Nave=1" from being appended
    evoked.comment = None  # Removes default comment from the auto-title
    
    # Create the plot
    fig = evoked.plot_joint(
        times=times,
        title=title,
        picks=picks,
        exclude=exclude,
        show=show,
        ts_args=ts_args,
        topomap_args=topomap_args
    )
    
    # Restore original attributes
    evoked.nave = old_nave
    evoked.comment = old_comment
    
    # Adjust axis labels
    time_ax = fig.axes[-3]  # the main time-series axis
    time_ax.set_xlabel("Window index")
    cbar_ax = fig.axes[-1]
    cbar_ax.set_xlabel("Metric value")
    cbar_ax.yaxis.get_offset_text().set_visible(False)
    
    return fig


def compare_patients(
    window_size: str, 
    direction: str,
    metric_name: str,
    patient_indices: List[int],
    groups: List[str],
    dataset_indexes: Dict[str, Any],
    sfreq: float = 1.0,
    tmin: float = 0.0,
    montage_name: str = "standard_1020"
) -> List[mne.Evoked]:
    """
    Compare multiple patients by creating Evoked objects for each.
    
    Parameters
    ----------
    window_size : str
        Size of the window (e.g., 'window_50').
    direction : str
        Direction of stimulus ('up' or 'down').
    metric_name : str
        Name of the RQA metric.
    patient_indices : List[int]
        List of patient indices to compare.
    groups : List[str]
        List of patient groups corresponding to patient_indices.
    dataset_indexes : Dict[str, Any]
        Dictionary mapping window size and direction to file paths.
    sfreq : float, optional
        Sampling frequency in Hz, defaults to 1.0.
    tmin : float, optional
        Start time of the data in seconds, defaults to 0.0.
    montage_name : str, optional
        Name of the standard montage used.
    
    Returns
    -------
    List[mne.Evoked]
        List of Evoked objects for each patient.
        
    Raises
    ------
    ValueError
        If patient_indices and groups have different lengths.
    """
    if len(patient_indices) != len(groups):
        raise ValueError("patient_indices and groups must have the same length")
    
    evoked_list = []
    
    for idx, group in zip(patient_indices, groups):
        try:
            evoked = get_patient_evoked(
                window_size=window_size,
                direction=direction,
                group=group,
                metric_name=metric_name,
                patient_index=idx,
                dataset_indexes=dataset_indexes,
                sfreq=sfreq,
                tmin=tmin,
                montage_name=montage_name
            )
            evoked_list.append(evoked)
        except ValueError as e:
            print(f"Could not load patient {idx} from group {group}: {e}")
    
    return evoked_list

# ...existing code...

def create_group_comparison_plot(
    patient_index: int,
    window_size: str,
    direction: str,
    metric_name: str,
    dataset_indexes: Dict[str, Any],
    figsize: tuple = (20, 8),
    sfreq: float = 1.0,
    tmin: float = 0.0,
    times: Union[str, List[float]] = "peaks",
    show: bool = False,
) -> plt.Figure:
    """
    Create side-by-side joint plots comparing the same patient's data between CT and DD groups.
    
    Parameters
    ----------
    patient_index : int
        Index of the patient to analyze.
    window_size : str
        Size of the window (e.g., 'window_50').
    direction : str
        Direction of stimulus ('up' or 'down').
    metric_name : str
        Name of the RQA metric.
    dataset_indexes : Dict[str, Any]
        Dictionary mapping window size and direction to file paths.
    figsize : tuple, optional
        Size of the figure (width, height), defaults to (20, 8).
    sfreq : float, optional
        Sampling frequency in Hz, defaults to 1.0.
    tmin : float, optional
        Start time of the data in seconds, defaults to 0.0.
    times : str or list of float, optional
        Times to plot. Can be "peaks" or list of time points.
    show : bool, optional
        Whether to show the plot immediately. Default is False.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure containing the side-by-side joint plots.
    
    Raises
    ------
    ValueError
        If data for either group cannot be found.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Create figure with a grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    
    # Get evoked objects for CT and DD groups
    try:
        ct_evoked = get_patient_evoked(
            window_size=window_size,
            direction=direction,
            group="CT",
            metric_name=metric_name,
            patient_index=patient_index,
            dataset_indexes=dataset_indexes,
            sfreq=sfreq,
            tmin=tmin
        )
    except ValueError as e:
        raise ValueError(f"Could not load CT data: {e}")
    
    try:
        dd_evoked = get_patient_evoked(
            window_size=window_size,
            direction=direction,
            group="DD",
            metric_name=metric_name,
            patient_index=patient_index,
            dataset_indexes=dataset_indexes,
            sfreq=sfreq,
            tmin=tmin
        )
    except ValueError as e:
        raise ValueError(f"Could not load DD data: {e}")
    
    # Create joint plots for each group
    # We'll create them on separate figures first
    ct_fig = create_joint_plot(
        evoked=ct_evoked,
        metric_name=metric_name,
        title=f"{metric_name} - Control Group - Patient {patient_index}",
        times=times,
        show=False
    )
    
    dd_fig = create_joint_plot(
        evoked=dd_evoked,
        metric_name=metric_name,
        title=f"{metric_name} - Dyslexia Group - Patient {patient_index}",
        times=times,
        show=False
    )
    
    # Close the original figures to avoid displaying them
    plt.close(ct_fig)
    plt.close(dd_fig)
    
    # Create new axes in our grid for the copied figures
    ax_ct = fig.add_subplot(gs[0, 0])
    ax_dd = fig.add_subplot(gs[0, 1])
    
    # Copy figures to our new axes using their saved images
    # This approach uses matplotlib's figure canvas to convert MNE figures to images
    ct_canvas = ct_fig.canvas
    dd_canvas = dd_fig.canvas
    ct_canvas.draw()
    dd_canvas.draw()
    
    # Get the RGB buffer from the canvas
    ct_img = np.array(ct_canvas.buffer_rgba())
    dd_img = np.array(dd_canvas.buffer_rgba())
    
    # Display the images
    ax_ct.imshow(ct_img)
    ax_dd.imshow(dd_img)
    
    # Remove axes ticks since we're displaying images
    ax_ct.axis('off')
    ax_dd.axis('off')
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig

def create_grand_average(
    evoked_list: List[mne.Evoked],
    metric_name: str
) -> mne.Evoked:
    """
    Create a grand average from multiple evoked objects.
    
    Parameters
    ----------
    evoked_list : List[mne.Evoked]
        List of Evoked objects to average.
    metric_name : str
        Name of the RQA metric for naming the grand average.
    
    Returns
    -------
    mne.Evoked
        Grand average evoked object.
        
    Raises
    ------
    ValueError
        If an empty list is provided.
    """
    if not evoked_list:
        raise ValueError("Empty evoked list provided")
        
    # Create grand average
    grand_avg = mne.grand_average(evoked_list)
    
    # Set a descriptive name
    grand_avg.comment = f"Grand Average - {metric_name}"
    
    return grand_avg


def plot_topomaps_over_time(
    evoked: mne.Evoked,
    metric_name: str,
    times: Optional[List[float]] = None,
    n_times: int = 5,
    title: Optional[str] = None,
    show: bool = False
) -> plt.figure:
    """
    Plot topographic maps at selected time points.
    
    Parameters
    ----------
    evoked : mne.Evoked
        Evoked object containing the data to plot.
    metric_name : str
        Name of the RQA metric for labeling.
    times : List[float], optional
        Specific time points to plot. If None, equally spaced points are used.
    n_times : int, optional
        Number of time points to plot if times is None. Default is 5.
    title : str, optional
        Title for the figure. If None, a default title is created.
    show : bool, optional
        Whether to show the plot immediately. Default is False.
    
    Returns
    -------
    mne.viz.Figure
        The created figure object.
    """
    # If times not specified, create equally spaced time points
    if times is None:
        start = evoked.times[0]
        end = evoked.times[-1]
        times = np.linspace(start, end, n_times)
    
    # Create default title if none provided
    if title is None:
        title = f"{metric_name} Topographic Maps"
    
    # Create the topoplot
    fig = evoked.plot_topomap(
        times=times,
        ch_type='eeg',
        title=title,
        show=show,
        colorbar=True,
        outlines='head',
        contours=0
    )
    
    return fig


def save_evoked_to_fif(
    evoked: mne.Evoked,
    output_dir: str,
    filename: str,
    overwrite: bool = False
) -> str:
    """
    Save an Evoked object to a .fif file.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The Evoked object to save.
    output_dir : str
        Directory where the file will be saved.
    filename : str
        Filename without extension.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is False.
    
    Returns
    -------
    str
        Path to the saved file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Add .fif extension if not present
    if not filename.endswith('.fif'):
        filename += '.fif'
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    # Save the evoked object
    evoked.save(file_path, overwrite=overwrite)
    
    return file_path


def load_evoked_from_fif(file_path: str) -> mne.Evoked:
    """
    Load an Evoked object from a .fif file.
    
    Parameters
    ----------
    file_path : str
        Path to the .fif file.
    
    Returns
    -------
    mne.Evoked
        The loaded Evoked object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return mne.read_evokeds(file_path)[0]