#!/usr/bin/env python3
"""
rqa_multiwindow_band_processor.py - HPC Version

Process EEG datasets with Recurrence Quantification Analysis (RQA) metrics
for specific channels, a specific frequency band, and multiple window sizes.
"""

import os
import time
from datetime import datetime
import yaml

from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import logging
from pathlib import Path
import argparse

from contextlib import contextmanager
import gc

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

import threading

from pyddeeg.signal_processing.rqa_toolbox.rqe_parallelizable import (
    process_single_channel_band,
)

# Define EEG channel names
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "Cz",
]


#### Context Managers ####
@contextmanager
def log_task_progress(client, logger, update_interval=300):
    """
    Context manager that logs Dask task progress at regular intervals.
    """
    stop_event = threading.Event()

    def _log_progress():
        last_completed = 0
        start_time = time.time()

        while not stop_event.is_set():
            try:
                # Get scheduler info
                info = client.scheduler_info()

                # Extract task statistics
                workers = info.get("workers", {})
                total_workers = len(workers)

                tasks = info.get("tasks", {})
                n_tasks = len(tasks)
                completed = sum(1 for t in tasks.values() if t.get("state") == "memory")
                processing = sum(
                    1 for t in tasks.values() if t.get("state") == "processing"
                )
                pending = sum(1 for t in tasks.values() if t.get("state") == "pending")

                # Calculate completion rate
                elapsed = time.time() - start_time
                tasks_per_second = completed / max(elapsed, 1)

                # Log progress (simplified)
                if n_tasks > 0:
                    percent_complete = (completed / n_tasks) * 100
                    logger.info(
                        f"Progress: {completed}/{n_tasks} tasks ({percent_complete:.1f}%) "
                        f"Rate: {tasks_per_second:.2f} tasks/sec"
                    )

                    # Only log memory info if we're at a concerning level
                    mem_usage = [
                        w.get("memory", {}).get("memory_usage", 0)
                        / w.get("memory", {}).get("memory_limit", 1)
                        for w in workers.values()
                        if w.get("memory", {}).get("memory_limit", 0) > 0
                    ]

                    if mem_usage and max(mem_usage) > 0.8:
                        logger.warning(
                            f"High memory usage detected: {max(mem_usage)*100:.1f}% of limit"
                        )

            except Exception as e:
                logger.warning(f"Error getting progress information: {str(e)}")

            # Sleep until next update
            stop_event.wait(update_interval)

    # Start progress monitoring thread
    progress_thread = threading.Thread(target=_log_progress, daemon=True)
    progress_thread.start()

    try:
        yield
    finally:
        # Stop the monitoring thread
        stop_event.set()
        progress_thread.join(timeout=5.0)


def write_metadata_file(
    output_dir: Path,
    dataset_name: str,
    target_channel: str,
    target_band: int,
    window_sizes: List[int],
    rqa_params: Dict[str, Any],
    normalize_metrics: bool,
    channel_metadata: Dict[str, Any],
) -> Path:
    """
    Write metadata to a text file including channel, band, window sizes, and metrics information.
    """
    if target_channel.lower() == "all":
        metadata_filename = (
            f"{dataset_name}_band{target_band}_all_channels_rqa_metadata.txt"
        )
    else:
        metadata_filename = (
            f"{dataset_name}_band{target_band}_{target_channel}_rqa_metadata.txt"
        )

    metadata_file = output_dir / metadata_filename

    with open(metadata_file, "w") as f:
        # Write header
        f.write(f"RQA Metadata for {dataset_name}\n")
        f.write("=" * 50 + "\n\n")

        # Write processing details
        f.write("PROCESSING DETAILS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Target channel: {target_channel}\n")
        f.write(f"Target frequency band: {target_band}\n")
        f.write(f"Window sizes: {window_sizes} ms\n")
        f.write(f"Normalize metrics: {normalize_metrics}\n")
        f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Write channel information
        f.write("CHANNEL INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Channel indices: {channel_metadata['channel_indices']}\n")
        f.write(f"Channel names: {channel_metadata['channel_names']}\n\n")

        # Write RQA parameters
        f.write("RQA PARAMETERS\n")
        f.write("-" * 30 + "\n")
        for key, value in sorted(rqa_params.items()):
            if (
                key != "raw_signal_window_size"
            ):  # Skip window size as we list them separately
                f.write(f"{key}: {value}\n")
        f.write("\n")

        # Write metric indices mapping
        f.write("METRICS MAPPING\n")
        f.write("-" * 30 + "\n")
        f.write("The tensor's last dimension contains the following metrics:\n")
        for i, metric in enumerate(rqa_params.get("metrics_to_use", [])):
            f.write(f"Index {i}: {metric}\n")

    return metadata_file


def log_memory_usage(logger):
    """Log current memory usage of the process."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)  # Convert to GB
        logger.info(f"Process memory usage: {memory_gb:.2f} GB")
    except ImportError:
        logger.warning("psutil not available, memory usage reporting disabled")
    except Exception as e:
        logger.warning(f"Error getting memory information: {str(e)}")


def update_status_file(status_dir, stage, message):
    """Write status update to a file for external monitoring."""
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, "status.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(status_file, "a") as f:
        f.write(f"[{timestamp}] {stage}: {message}\n")

    # Only write to stage-specific files for major stages
    major_stages = ["init", "error", "fatal", "complete"]
    if stage in major_stages:
        stage_file = os.path.join(status_dir, f"{stage}.txt")
        with open(stage_file, "w") as f:
            f.write(f"[{timestamp}] {message}\n")


def setup_logging(log_dir: str = "./logs", suffix: str = "") -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Add suffix to log filename if provided
    if suffix:
        log_file = log_path / f"rqa_processing_{suffix}.log"
    else:
        log_file = log_path / f"rqa_processing.log"

    logger = logging.getLogger("EEG_RQA_Processor")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create file handler with WARNING level
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.WARNING)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


#### Actual processing functions ####


@delayed
def process_channel_band_with_gc(
    signal: np.ndarray,
    rqa_params: Dict[str, Any],
    normalize_metrics: bool,
) -> np.ndarray:
    """
    Process a single channel/band with explicit garbage collection.
    """
    # Force garbage collection before starting
    gc.collect()

    # Process the data - always set return_rqe to False
    result = process_single_channel_band(
        signal=signal,
        rqa_params=rqa_params,
        normalize_metrics=normalize_metrics,
        return_rqe=False,
    )

    # Force garbage collection after processing
    gc.collect()

    # We only need the RQA metrics array, ignore RQE and correlation values
    rqa_metrics = result[0]
    return rqa_metrics


def get_channel_indices(target_channel: str) -> List[int]:
    """
    Get list of channel indices to process based on target_channel.
    """
    if target_channel.lower() == "all":
        # Return all channels except Cz (which is the last channel, index 31)
        return list(range(len(EEG_CHANNELS) - 1))
    else:
        # Find the index of the specified channel
        try:
            channel_idx = EEG_CHANNELS.index(target_channel)
            return [channel_idx]
        except ValueError:
            raise ValueError(
                f"Channel '{target_channel}' not found. Available channels: {', '.join(EEG_CHANNELS)}"
            )


def process_dataset_for_window_size(
    data: np.ndarray,
    rqa_params: Dict[str, Any],
    target_channel: str,
    target_band: int,
    window_size: int,
    normalize_metrics: bool = False,
    dataset_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    status_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process specific channel(s) of an EEG dataset with RQA analysis for a single window size.

    Parameters:
    -----------
    data : np.ndarray
        EEG data for specific band with shape (n_patients, n_channels, n_samples)
    rqa_params : Dict[str, Any]
        Parameters for RQA computation
    target_channel : str
        Channel name or "All" to process all channels except Cz
    window_size : int
        Window size for RQA computation in ms
    normalize_metrics : bool
        Whether to normalize metrics
    dataset_name : str
        Name of the dataset for logging
    logger : Optional[logging.Logger]
        Logger instance
    status_dir : Optional[str]
        Directory for status files

    Returns:
    --------
    Tuple[np.ndarray, Dict[str, Any]]
        RQA metrics and channel metadata
    """
    if logger is None:
        logger = logging.getLogger("EEG_RQA_Processor")

    n_patients, n_channels, n_samples = data.shape

    # Get the channel indices to process
    try:
        channel_indices = get_channel_indices(target_channel)
        channel_names = [EEG_CHANNELS[idx] for idx in channel_indices]

        if target_channel.lower() == "all":
            channel_description = (
                f"all channels except Cz ({len(channel_indices)} channels)"
            )
        else:
            channel_description = (
                f"channel {target_channel} (index {channel_indices[0]})"
            )

        logger.info(f"Processing dataset {dataset_name}: {data.shape}")
        logger.info(
            f"Processing {channel_description} with window size {window_size}ms"
        )

        if status_dir:
            update_status_file(
                status_dir,
                "process_details",
                f"Processing {dataset_name} with shape {data.shape} - {channel_description} - Window {window_size}ms",
            )
    except ValueError as e:
        logger.error(f"Channel selection error: {str(e)}")
        raise

    # Create a copy of RQA parameters and update the window size
    current_rqa_params = rqa_params.copy()
    current_rqa_params["raw_signal_window_size"] = window_size

    # Extract parameters for output shape calculation
    stride = current_rqa_params.get("stride", 1)
    metrics_to_use = current_rqa_params.get(
        "metrics_to_use", ["RR", "DET", "ENT", "TT"]
    )

    # Calculate output shapes
    num_windows = (n_samples - window_size) // stride + 1
    num_metrics = len(metrics_to_use)

    logger.info(
        f"Calculated dimensions: num_windows={num_windows}, num_metrics={num_metrics}"
    )

    # Initialize output array - only for selected channels
    n_selected_channels = len(channel_indices)
    rqa_metrics_array = np.full(
        (n_patients, n_selected_channels, num_windows, num_metrics), np.nan
    )

    # Create Dask task graph for selected channels
    tasks = {}
    task_indices = []

    # Create a progress counter
    total_tasks = n_patients * n_selected_channels
    logger.info(f"Creating {total_tasks} tasks for parallel processing")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Creating {total_tasks} tasks for {dataset_name} - Window {window_size}ms",
        )

    start_time = time.time()
    task_creation_start = time.time()

    # Create tasks only for selected channels
    for patient_idx in range(n_patients):
        for local_ch_idx, global_ch_idx in enumerate(channel_indices):
            # Create a unique key for this task
            task_key = (patient_idx, local_ch_idx)
            task_indices.append(task_key)

            # Extract signal for this patient, channel combination
            # Use the global channel index to access data but the local index for output arrays
            signal = data[patient_idx, global_ch_idx, :].copy()

            # Create a delayed task with memory management
            tasks[task_key] = process_channel_band_with_gc(
                signal=signal,
                rqa_params=current_rqa_params,
                normalize_metrics=normalize_metrics,
            )

    task_creation_time = time.time() - task_creation_start
    logger.info(f"Task graph creation completed in {task_creation_time:.2f} seconds")
    logger.info(f"Total tasks created: {len(tasks)}")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Task graph created for {dataset_name}: {len(tasks)} tasks - Window {window_size}ms",
        )

    # Compute all tasks at once
    compute_start = time.time()
    logger.info(f"Starting Dask computation for window size {window_size}ms")

    if status_dir:
        update_status_file(
            status_dir,
            "compute",
            f"Starting computation for {dataset_name}: {len(tasks)} tasks - Window {window_size}ms",
        )

    # Process the tasks
    results = dask.compute(tasks)[0]

    compute_time = time.time() - compute_start
    logger.info(f"Dask computation completed in {compute_time:.2f} seconds")

    if status_dir:
        update_status_file(
            status_dir,
            "compute",
            f"Computation completed for {dataset_name} - Window {window_size}ms in {compute_time:.2f} seconds",
        )

    # Populate the output array
    collection_start = time.time()
    logger.info(f"Collecting results into output array")

    successful_tasks = 0
    empty_results = 0

    for task_key in task_indices:
        patient_idx, local_ch_idx = task_key
        rqa_matrix = results[task_key]

        # Store RQA metrics - using local channel index
        if rqa_matrix.size > 0:
            actual_windows = min(rqa_matrix.shape[0], num_windows)
            rqa_metrics_array[patient_idx, local_ch_idx, :actual_windows, :] = (
                rqa_matrix
            )
            successful_tasks += 1
        else:
            empty_results += 1

    collection_time = time.time() - collection_start
    logger.info(
        f"Result collection completed in {collection_time:.2f} seconds: "
        f"{successful_tasks} successful tasks, {empty_results} empty results"
    )

    # Store channel information in metadata
    channel_metadata = {
        "channel_indices": channel_indices,
        "channel_names": channel_names,
        "target_channel": target_channel,
    }

    # Clean up and force garbage collection
    del tasks, results
    gc.collect()

    total_elapsed = time.time() - start_time
    logger.info(
        f"Window size {window_size}ms processing complete in {total_elapsed:.2f} seconds"
    )

    return rqa_metrics_array, channel_metadata


def process_dataset_channels_multi_windows(
    data: np.ndarray,
    rqa_params: Dict[str, Any],
    target_channel: str,
    target_band: int,
    window_sizes: List[int],
    normalize_metrics: bool = False,
    dataset_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    status_dir: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """
    Process specific channel(s) of an EEG dataset with RQA metrics for multiple window sizes.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_channels, n_samples, n_bands)
    rqa_params : Dict[str, Any]
        Parameters for RQA computation
    target_channel : str
        Channel name or "All" to process all channels except Cz
    target_band : int
        Index of the frequency band to analyze
    window_sizes : List[int]
        List of window sizes to use for RQA computation
    normalize_metrics : bool
        Whether to normalize metrics
    dataset_name : str
        Name of the dataset for logging
    logger : Optional[logging.Logger]
        Logger instance
    status_dir : Optional[str]
        Directory for status files

    Returns:
    --------
    Tuple[Dict[int, np.ndarray], Dict[str, Any]]
        Dictionary mapping window size -> RQA metrics array, and channel metadata
    """
    if logger is None:
        logger = logging.getLogger("EEG_RQA_Processor")

    # Extract the target band from data
    n_patients, n_channels, n_samples, n_bands = data.shape

    if target_band >= n_bands:
        error_msg = (
            f"Target band {target_band} out of range (only {n_bands} bands available)"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract the target band data
    logger.info(f"Extracting band {target_band} from data")
    band_data = data[:, :, :, target_band]

    # Store results for each window size
    results_dict = {}
    channel_metadata = None  # We'll capture this from the first window size processing

    # Process each window size
    for window_size in window_sizes:
        logger.info(f"Processing window size {window_size}ms")

        rqa_metrics, metadata = process_dataset_for_window_size(
            data=band_data,
            rqa_params=rqa_params,
            target_channel=target_channel,
            target_band=target_band,
            window_size=window_size,
            normalize_metrics=normalize_metrics,
            dataset_name=dataset_name,
            logger=logger,
            status_dir=status_dir,
        )

        # Store results
        results_dict[window_size] = rqa_metrics

        # Store metadata from first processing (channel info will be the same for all window sizes)
        if channel_metadata is None:
            channel_metadata = metadata

    return results_dict, channel_metadata  # type: ignore


#### Helper functions ####


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


#### Main function ####


def main():
    """Main function to process EEG datasets for specific channel(s)."""
    parser = argparse.ArgumentParser(
        description="Process EEG datasets with RQA analysis for specific channels, band and window sizes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of cores to use (default: use value from SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="Memory limit per worker (default: use value from config or 75% of available)",
    )
    parser.add_argument(
        "--status-dir",
        type=str,
        default="./status",
        help="Directory to write status files for external monitoring",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Interval in seconds between progress updates",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Specific channel to process (overrides config file setting)",
    )
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Get target channel - command line overrides config file
    target_channel = (
        args.channel if args.channel else config.get("target_channel", "All")
    )

    # Get target band and window sizes from config
    target_band = config.get("target_bandwidth", 0)
    window_sizes = config.get("window_sizes", [100])

    # Set up logging with channel and band information in the filename
    log_dir = config.get("logging", {}).get("directory", "./logs")
    logger = setup_logging(
        log_dir, suffix=f"{target_channel.lower()}_band{target_band}"
    )
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Target channel: {target_channel}")
    logger.info(f"Target band: {target_band}")
    logger.info(f"Window sizes: {window_sizes}")

    # Create status directory
    status_dir = args.status_dir
    os.makedirs(status_dir, exist_ok=True)
    update_status_file(
        status_dir,
        "init",
        f"Starting RQA processing for channel: {target_channel}, band: {target_band}",
    )

    # Create output directory if it doesn't exist
    output_dir = Path(config.get("output_directory", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get resources from SLURM environment if available, otherwise use config/args
    if os.environ.get("SLURM_JOB_ID"):
        logger.info("Running in SLURM environment")
        update_status_file(status_dir, "env", "Running in SLURM environment")

        # Get number of CPUs from SLURM if not specified in args
        if args.cores is None:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                n_workers = int(slurm_cpus)
                logger.info(f"Using SLURM_CPUS_PER_TASK: {n_workers}")
            else:
                slurm_ntasks = os.environ.get("SLURM_NTASKS")
                if slurm_ntasks:
                    n_workers = int(slurm_ntasks)
                    logger.info(f"Using SLURM_NTASKS: {n_workers}")
                else:
                    n_workers = config.get("dask", {}).get("n_workers", -1)
                    logger.info(f"Using config n_workers: {n_workers}")
        else:
            n_workers = args.cores
            logger.info(f"Using command line cores: {n_workers}")

        # If n_workers is still -1, use a reasonable default
        if n_workers == -1:
            n_workers = 4
            logger.info(f"Using default n_workers: {n_workers}")

        # Get memory limit
        if args.memory_limit:
            memory_limit = args.memory_limit
        else:
            memory_limit = config.get("dask", {}).get("memory_limit", None)

            if not memory_limit or memory_limit == "auto":
                # Try to get memory from SLURM
                slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
                if slurm_mem:
                    # Convert to bytes (SLURM uses MB)
                    mem_mb = int(slurm_mem)
                    # Use 90% of available memory since we're on a supercomputer
                    memory_limit = f"{int(mem_mb * 0.9)}MB"
                    logger.info(f"Using 90% of SLURM_MEM_PER_NODE: {memory_limit}")
                else:
                    # Default to a safe value if nothing else is available
                    memory_limit = "4GB"
                    logger.info(f"Using default memory limit: {memory_limit}")
    else:
        logger.info("Running outside SLURM environment")
        n_workers = (
            args.cores
            if args.cores is not None
            else config.get("dask", {}).get("n_workers", 4)
        )
        memory_limit = args.memory_limit or config.get("dask", {}).get(
            "memory_limit", "4GB"
        )

    # Optimization for number of workers
    threads_per_worker = config.get("dask", {}).get("threads_per_worker", 1)

    # When processing a single channel, we can use more workers with less memory
    if target_channel.lower() != "all":
        # For single channel, we can use more workers
        if n_workers > 32:
            # Still limit for stability
            actual_workers = min(n_workers, 64)
            logger.info(
                f"Single channel mode: using {actual_workers} workers for better parallelism"
            )
            n_workers = actual_workers
    else:
        # For all channels, use the original optimization
        if n_workers > 16:
            actual_workers = max(8, n_workers // 4)
            logger.info(
                f"All-channel mode: using {actual_workers} workers instead of {n_workers}"
            )
            n_workers = actual_workers

    # Set up LocalCluster for HPC use
    logger.info(f"Setting up LocalCluster with {n_workers} workers")
    update_status_file(
        status_dir, "cluster", f"Setting up LocalCluster with {n_workers} workers"
    )

    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,  # Use processes for better isolation
            silence_logs=logging.INFO,  # Reduce log noise
            local_directory=os.environ.get(
                "MYLOCALSCRATCH", None
            ),  # Use local scratch if available
        )

        client = Client(cluster)
        logger.info(f"Dask LocalCluster started with {n_workers} workers")
        logger.info(
            f"Worker configuration: threads_per_worker={threads_per_worker}, memory_limit={memory_limit}"
        )
        update_status_file(
            status_dir, "cluster", f"Dask cluster ready with {n_workers} workers"
        )

        # Log initial memory usage
        log_memory_usage(logger)

        # Get RQA parameters from config
        rqa_params = config.get("rqa_parameters", {})
        normalize_metrics = config.get("normalize_metrics", False)

        logger.info(f"RQA parameters: {rqa_params}")
        logger.info(f"Normalize metrics: {normalize_metrics}")

        # Get input directory and dataset filenames
        input_dir = config.get("input_directory", "./data")
        datasets = config.get("datasets", {})

        # Start progress monitoring
        with log_task_progress(client, logger, update_interval=args.progress_interval):
            # Process each dataset
            for dataset_name, filename in datasets.items():
                file_path = os.path.join(input_dir, filename)
                logger.info(f"Processing dataset {dataset_name} from file {file_path}")
                update_status_file(status_dir, "dataset", f"Starting {dataset_name}")

                # Load data - do this inside the loop to free memory between datasets
                try:
                    logger.info(f"Loading data from {file_path}")
                    update_status_file(status_dir, "load", f"Loading {dataset_name}")

                    data = np.load(file_path)["data"]
                    logger.info(f"Loaded {dataset_name} with shape {data.shape}")
                    update_status_file(
                        status_dir,
                        "load",
                        f"Loaded {dataset_name} with shape {data.shape}",
                    )

                    # Log memory after loading
                    log_memory_usage(logger)

                    # Process the dataset with selected channel(s), band, and window sizes
                    update_status_file(
                        status_dir,
                        "process",
                        f"Processing {dataset_name} - {target_channel} - band {target_band}",
                    )

                    rqa_results_dict, channel_metadata = (
                        process_dataset_channels_multi_windows(
                            data=data,
                            rqa_params=rqa_params,
                            target_channel=target_channel,
                            target_band=target_band,
                            window_sizes=window_sizes,
                            normalize_metrics=normalize_metrics,
                            dataset_name=dataset_name,
                            logger=logger,
                            status_dir=status_dir,
                        )
                    )

                    # Log memory after processing
                    log_memory_usage(logger)

                    # Save results - include channel and band information in filename
                    if target_channel.lower() == "all":
                        output_filename = f"{dataset_name}_band{target_band}_all_channels_rqa_results.npz"
                    else:
                        output_filename = f"{dataset_name}_band{target_band}_{target_channel}_rqa_results.npz"

                    update_status_file(
                        status_dir,
                        "save",
                        f"Saving results for {dataset_name} - {target_channel} - band {target_band}",
                    )
                    output_file = output_dir / output_filename
                    logger.info(f"Saving results to {output_file}")

                    # Save data for all window sizes
                    np_save_dict = {
                        f"window_{size}": arr for size, arr in rqa_results_dict.items()
                    }
                    np_save_dict["window_sizes"] = np.array(window_sizes)

                    np.savez_compressed(output_file, **np_save_dict)

                    # Write metadata to a separate text file
                    metadata_file = write_metadata_file(
                        output_dir=output_dir,
                        dataset_name=dataset_name,
                        target_channel=target_channel,
                        target_band=target_band,
                        window_sizes=window_sizes,
                        rqa_params=rqa_params,
                        normalize_metrics=normalize_metrics,
                        channel_metadata=channel_metadata,
                    )

                    logger.info(f"Metadata saved to {metadata_file}")
                    update_status_file(
                        status_dir,
                        "save",
                        f"Results and metadata saved for {dataset_name} - {target_channel} - band {target_band}",
                    )

                    # Clean up to free memory
                    del data, rqa_results_dict, channel_metadata
                    gc.collect()
                    log_memory_usage(logger)

                except Exception as e:
                    logger.error(
                        f"Error processing dataset {dataset_name}: {str(e)}",
                        exc_info=True,
                    )
                    update_status_file(
                        status_dir, "error", f"Error on {dataset_name}: {str(e)}"
                    )

        # Shut down the client and cluster
        logger.info("Processing complete. Closing Dask client and cluster.")
        update_status_file(status_dir, "shutdown", "Closing Dask client and cluster")
        client.close()
        cluster.close()
        logger.info("Dask client and cluster closed.")
        update_status_file(
            status_dir,
            "complete",
            f"Processing complete for {target_channel} - band {target_band}",
        )

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        update_status_file(status_dir, "fatal", f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
