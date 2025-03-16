#!/usr/bin/env python3
"""
eeg_rqe_processor.py - HPC Version

Process multiple EEG datasets with Recurrence Quantification Analysis (RQA) and
Recurrence Quantification Entropy (RQE) metrics using Dask for parallelization.

Optimized for execution on high-memory HPC clusters with no internet connectivity.
This version processes all tasks at once instead of batching by patients.
"""

import os
import time
from datetime import datetime
import yaml

from typing import Dict, Tuple, Optional, Any

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

#### Self-contained script, includes the functions from rqe_parallelizable ####

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import scipy.stats as stats
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
    if metrics_to_use is None:
        metrics_to_use = ["RR", "DET", "L_max", "L_mean", "ENT", "LAM", "TT"]

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


def compute_rqa_batch(
    signal: np.ndarray,
    raw_signal_window_size: int,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int,
    metrics_to_use: List[str],
    stride: int = 1,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute RQA metrics for a batch of sliding windows.
    Optimized version for Dask parallel processing.

    Parameters:
    -----------
    signal : np.ndarray
        1D array containing the signal to analyze
    raw_signal_window_size, embedding_dim, etc. : Various parameters
        RQA computation parameters
    stride : int
        Step size between consecutive windows
    batch_size : Optional[int]
        Number of windows to process in one batch (None = all)

    Returns:
    --------
    rqa_matrix : np.ndarray
        2D array with shape (num_windows, len(metrics_to_use))
    """
    n = len(signal)
    num_windows = (n - raw_signal_window_size) // stride + 1

    if batch_size is None or batch_size >= num_windows:
        batch_size = num_windows

    # Pre-allocate results array
    rqa_results = np.full((batch_size, len(metrics_to_use)), np.nan)

    for i in range(batch_size):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + raw_signal_window_size]

        metrics, _ = compute_rqa_metrics_for_window(
            window_signal=window_data,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line,
            metrics_to_use=metrics_to_use,
        )

        # Extract metrics in the same order as metrics_to_use
        for j, key in enumerate(metrics_to_use):
            rqa_results[i, j] = metrics.get(key, np.nan)

    return rqa_results


def compute_rqe_batch(
    rqa_matrix: np.ndarray, rqa_space_window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RQE and correlation for a batch of RQA metrics.
    Optimized version for Dask parallel processing.

    Parameters:
    -----------
    rqa_matrix : np.ndarray
        2D array of shape (num_windows, num_metrics)
    rqa_space_window_size : int
        Size of window for computing RQE correlation in the RQA space

    Returns:
    --------
    rqe_values : np.ndarray
        Array of RQE values
    corr_values : np.ndarray
        Array of correlation values
    """
    T, L = rqa_matrix.shape
    Q = max(1, T - rqa_space_window_size + 1)

    rqe_list = np.zeros(Q)
    corr_list = np.zeros(Q)

    for start_idx in range(Q):
        sub_block = rqa_matrix[start_idx : start_idx + rqa_space_window_size, :]
        # Transpose to get each metric as a row
        sub_block_T = sub_block.T

        # Collect pairwise correlations
        pairwise_rho = []
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                # Skip if either series has NaN values
                if (
                    np.isnan(sub_block_T[l1, :]).any()
                    or np.isnan(sub_block_T[l2, :]).any()
                ):
                    continue

                try:
                    rho, _ = stats.spearmanr(sub_block_T[l1, :], sub_block_T[l2, :])
                    if not np.isnan(rho):
                        pairwise_rho.append(abs(rho))
                except Exception:
                    # Handle potential errors in correlation computation
                    pass

        if len(pairwise_rho) == 0:
            rqe_list[start_idx] = 1.0
            corr_list[start_idx] = 0.0
        else:
            # RQE = product(1 + |rho|)
            product_val = 1.0
            for c in pairwise_rho:
                product_val *= 1.0 + c
            rqe_list[start_idx] = product_val

            # Corr = mean(|rho|)
            corr_list[start_idx] = np.mean(pairwise_rho)

    return rqe_list, corr_list


def process_single_channel_band(
    signal: np.ndarray, rqa_params: Dict, normalize_metrics: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single channel/band combination.
    This function is designed to be the unit of parallelization.

    Parameters:
    -----------
    signal : np.ndarray
        1D array representing a single channel/band
    rqa_params : Dict
        Dictionary of RQA parameters
    normalize_metrics : bool
        Whether to normalize metrics before RQE computation

    Returns:
    --------
    rqa_matrix : np.ndarray
        Matrix of RQA metrics for each time window
    rqe_values : np.ndarray
        Array of RQE values
    corr_values : np.ndarray
        Array of correlation values
    """
    # Extract parameters
    embedding_dim = rqa_params.get("embedding_dim", 10)
    radius = rqa_params.get("radius", 0.8)
    time_delay = rqa_params.get("time_delay", 1)
    raw_signal_window_size = rqa_params.get("raw_signal_window_size", 100)
    rqa_space_window_size = rqa_params.get("rqa_space_window_size", 25)
    min_diagonal_line = rqa_params.get("min_diagonal_line", 5)
    min_vertical_line = rqa_params.get("min_vertical_line", 1)
    min_white_vertical_line = rqa_params.get("min_white_vertical_line", 1)
    metrics_to_use = rqa_params.get("metrics_to_use", ["RR", "DET", "ENT", "TT"])
    stride = rqa_params.get("stride", 1)

    # Compute RQA metrics
    rqa_matrix = compute_rqa_batch(
        signal=signal,
        raw_signal_window_size=raw_signal_window_size,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        radius=radius,
        distance_metric="euclidean",
        min_diagonal_line=min_diagonal_line,
        min_vertical_line=min_vertical_line,
        min_white_vertical_line=min_white_vertical_line,
        metrics_to_use=metrics_to_use,
        stride=stride,
    )

    # Normalize metrics if requested
    if normalize_metrics:
        for j in range(rqa_matrix.shape[1]):
            col = rqa_matrix[:, j]
            if np.all(np.isnan(col)):
                continue

            valid_values = col[~np.isnan(col)]
            if len(valid_values) > 0:
                col_min = np.min(valid_values)
                col_max = np.max(valid_values)

                if col_max > col_min:
                    col[~np.isnan(col)] = (col[~np.isnan(col)] - col_min) / (
                        col_max - col_min
                    )
                    rqa_matrix[:, j] = col

    # Check if we have enough windows for RQE computation
    if rqa_matrix.shape[0] <= rqa_space_window_size:
        # Return empty arrays if not enough windows
        return rqa_matrix, np.array([]), np.array([])

    # Compute RQE and correlation
    rqe_values, corr_values = compute_rqe_batch(
        rqa_matrix, rqa_space_window_size=rqa_space_window_size
    )

    return rqa_matrix, rqe_values, corr_values


#### Context Managers ####


@contextmanager
def log_task_progress(client, logger, update_interval=60):
    """
    Context manager that logs Dask task progress at regular intervals.

    Parameters:
    -----------
    client : dask.distributed.Client
        The Dask client
    logger : logging.Logger
        Logger to use for progress updates
    update_interval : int
        How often to log updates (in seconds)
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
                active_workers = sum(
                    1 for w in workers.values() if w.get("active", False)
                )

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
                tasks_since_last = completed - last_completed
                last_completed = completed

                # Log progress
                if n_tasks > 0:
                    percent_complete = (completed / n_tasks) * 100
                    logger.info(
                        f"Progress: {completed}/{n_tasks} tasks completed ({percent_complete:.1f}%) "
                        f"- {processing} processing, {pending} pending - "
                        f"Rate: {tasks_per_second:.2f} tasks/sec - "
                        f"Workers: {active_workers}/{total_workers} active"
                    )

                    # Log worker-specific information (memory usage, CPU)
                    if (
                        total_workers > 0 and total_workers < 10
                    ):  # Only log individual workers if there aren't too many
                        for worker_id, worker_info in workers.items():
                            memory = worker_info.get("memory", {})
                            mem_used = memory.get("memory_usage", 0) / (
                                1024**3
                            )  # Convert to GB
                            mem_limit = memory.get("memory_limit", 0) / (
                                1024**3
                            )  # Convert to GB
                            logger.info(
                                f"Worker {worker_id}: "
                                f"Memory: {mem_used:.2f}/{mem_limit:.2f} GB "
                                f"({(mem_used/mem_limit)*100 if mem_limit > 0 else 0:.1f}%) - "
                                f"Tasks: {worker_info.get('processing', []).__len__()} processing"
                            )
                    elif (
                        total_workers >= 10
                    ):  # For many workers, just log summary stats
                        mem_usage = [
                            w.get("memory", {}).get("memory_usage", 0) / (1024**3)
                            for w in workers.values()
                        ]
                        if mem_usage:
                            avg_mem = sum(mem_usage) / len(mem_usage)
                            max_mem = max(mem_usage)
                            logger.info(
                                f"Worker Memory: Avg: {avg_mem:.2f} GB, Max: {max_mem:.2f} GB"
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
        progress_thread.join(timeout=5.0)  # Wait for the thread to terminate


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

    # Also write to a stage-specific file for easy grepping
    stage_file = os.path.join(status_dir, f"{stage}.txt")
    with open(stage_file, "w") as f:
        f.write(f"[{timestamp}] {message}\n")


def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = (
        log_path / f"rqe_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("EEG_RQE_Processor")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


#### Actual processing functions ####


@delayed
def process_channel_band_with_gc(
    signal: np.ndarray, rqa_params: Dict[str, Any], normalize_metrics: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single channel/band with explicit garbage collection.

    This wrapper adds memory management around the process_single_channel_band function.
    """
    # Force garbage collection before starting
    gc.collect()

    # Process the data
    result = process_single_channel_band(
        signal=signal, rqa_params=rqa_params, normalize_metrics=normalize_metrics
    )

    # Force garbage collection after processing
    gc.collect()

    return result


def process_dataset_parallel(
    data: np.ndarray,
    rqa_params: Dict[str, Any],
    normalize_metrics: bool = False,
    dataset_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    status_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process an entire EEG dataset with RQA/RQE analysis using Dask for HPC.

    This version processes all tasks at once, optimized for high-memory supercomputers.
    """
    if logger is None:
        logger = logging.getLogger("EEG_RQE_Processor")

    n_patients, n_channels, n_samples, n_bands = data.shape
    logger.info(f"Processing dataset {dataset_name}: {data.shape}")
    logger.info(f"Processing all tasks in parallel (no batching)")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Processing {dataset_name} with shape {data.shape}",
        )

    # Extract parameters for output shape calculation
    stride = rqa_params.get("stride", 1)
    raw_signal_window_size = rqa_params.get("raw_signal_window_size", 100)
    rqa_space_window_size = rqa_params.get("rqa_space_window_size", 25)
    metrics_to_use = rqa_params.get("metrics_to_use", ["RR", "DET", "ENT", "TT"])

    # Calculate output shapes
    num_windows = (n_samples - raw_signal_window_size) // stride + 1
    num_rqe_windows = max(1, num_windows - rqa_space_window_size + 1)
    num_metrics = len(metrics_to_use)

    logger.info(
        f"Calculated dimensions: num_windows={num_windows}, num_rqe_windows={num_rqe_windows}, num_metrics={num_metrics}"
    )

    # Initialize output arrays
    rqa_metrics_array = np.full(
        (n_patients, n_channels, n_bands, num_windows, num_metrics), np.nan
    )
    rqe_values_array = np.full(
        (n_patients, n_channels, n_bands, num_rqe_windows), np.nan
    )
    corr_values_array = np.full(
        (n_patients, n_channels, n_bands, num_rqe_windows), np.nan
    )

    # Create Dask task graph for all tasks
    tasks = {}
    task_indices = []

    # Create a progress counter
    total_tasks = n_patients * n_channels * n_bands
    logger.info(f"Creating {total_tasks} tasks for parallel processing")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Creating {total_tasks} tasks for {dataset_name}",
        )

    start_time = time.time()
    task_creation_start = time.time()

    # Create all tasks at once
    for patient_idx in range(n_patients):
        for channel_idx in range(n_channels):
            for band_idx in range(n_bands):
                # Create a unique key for this task
                task_key = (patient_idx, channel_idx, band_idx)
                task_indices.append(task_key)

                # Extract signal for this patient, channel, band combination
                signal = data[patient_idx, channel_idx, :, band_idx].copy()

                # Create a delayed task with memory management
                tasks[task_key] = process_channel_band_with_gc(
                    signal=signal,
                    rqa_params=rqa_params,
                    normalize_metrics=normalize_metrics,
                )

    task_creation_time = time.time() - task_creation_start
    logger.info(f"Task graph creation completed in {task_creation_time:.2f} seconds")
    logger.info(f"Total tasks created: {len(tasks)}")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Task graph created for {dataset_name}: {len(tasks)} tasks",
        )

    # Compute all tasks at once
    compute_start = time.time()
    logger.info(f"Starting Dask computation of all tasks")

    if status_dir:
        update_status_file(
            status_dir,
            "compute",
            f"Starting computation for {dataset_name}: {len(tasks)} tasks",
        )

    # Process the tasks
    results = dask.compute(tasks)[0]

    compute_time = time.time() - compute_start
    logger.info(f"Dask computation completed in {compute_time:.2f} seconds")

    if status_dir:
        update_status_file(
            status_dir,
            "compute",
            f"Computation completed for {dataset_name} in {compute_time:.2f} seconds",
        )

    # Populate the output arrays
    collection_start = time.time()
    logger.info(f"Collecting results into output arrays")

    successful_tasks = 0
    empty_results = 0

    for task_key in task_indices:
        patient_idx, channel_idx, band_idx = task_key
        rqa_matrix, rqe_values, corr_values = results[task_key]

        # Store RQA metrics
        if rqa_matrix.size > 0:
            actual_windows = min(rqa_matrix.shape[0], num_windows)
            rqa_metrics_array[
                patient_idx, channel_idx, band_idx, :actual_windows, :
            ] = rqa_matrix
            successful_tasks += 1
        else:
            empty_results += 1

        # Store RQE values
        if rqe_values.size > 0:
            actual_rqe_windows = min(rqe_values.shape[0], num_rqe_windows)
            rqe_values_array[
                patient_idx, channel_idx, band_idx, :actual_rqe_windows
            ] = rqe_values
            corr_values_array[
                patient_idx, channel_idx, band_idx, :actual_rqe_windows
            ] = corr_values

    collection_time = time.time() - collection_start
    logger.info(
        f"Result collection completed in {collection_time:.2f} seconds: "
        f"{successful_tasks} successful tasks, {empty_results} empty results"
    )

    # Clean up and force garbage collection
    del tasks, results
    gc.collect()

    total_elapsed = time.time() - start_time
    logger.info(
        f"Dataset {dataset_name} processing complete in {total_elapsed:.2f} seconds"
    )

    return rqa_metrics_array, rqe_values_array, corr_values_array


#### Helper functions ####


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


#### Main function ####


def main():
    """Main function to process all EEG datasets."""
    parser = argparse.ArgumentParser(
        description="Process EEG datasets with RQA/RQE analysis using Dask."
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
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Set up logging
    log_dir = config.get("logging", {}).get("directory", "./logs")
    logger = setup_logging(log_dir)
    logger.info(f"Configuration loaded from {args.config}")

    # Create status directory
    status_dir = args.status_dir
    os.makedirs(status_dir, exist_ok=True)
    update_status_file(status_dir, "init", "Starting RQE processing")

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

    # For supercomputer optimization, we use a higher memory limit per worker
    # but fewer workers to avoid memory issues
    threads_per_worker = config.get("dask", {}).get("threads_per_worker", 1)

    # Calculate optimal worker configuration for supercomputer
    # On a supercomputer, having fewer workers with more memory is often better
    # than many workers with less memory each
    if n_workers > 16:
        # Adjust worker count to be reasonable for large machines
        actual_workers = max(8, n_workers // 4)
        logger.info(
            f"Optimizing for supercomputer: using {actual_workers} workers instead of {n_workers}"
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

                    # Process the dataset with all tasks at once
                    update_status_file(
                        status_dir, "process", f"Processing {dataset_name}"
                    )
                    rqa_metrics, rqe_values, corr_values = process_dataset_parallel(
                        data=data,
                        rqa_params=rqa_params,
                        normalize_metrics=normalize_metrics,
                        dataset_name=dataset_name,
                        logger=logger,
                        status_dir=status_dir,
                    )

                    # Log memory after processing
                    log_memory_usage(logger)

                    # Save results
                    update_status_file(
                        status_dir, "save", f"Saving results for {dataset_name}"
                    )
                    output_file = output_dir / f"{dataset_name}_rqe_results.npz"
                    logger.info(f"Saving results to {output_file}")
                    np.savez_compressed(
                        output_file,
                        rqa_metrics=rqa_metrics,
                        rqe_values=rqe_values,
                        corr_values=corr_values,
                        rqa_params=rqa_params,
                        normalize_metrics=normalize_metrics,
                    )
                    logger.info(f"Results saved successfully")
                    update_status_file(
                        status_dir, "save", f"Results saved for {dataset_name}"
                    )

                    # Clean up to free memory
                    del data, rqa_metrics, rqe_values, corr_values
                    gc.collect()
                    log_memory_usage(logger)  # Log memory after cleanup

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
        update_status_file(status_dir, "complete", "Processing complete")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        update_status_file(status_dir, "fatal", f"Fatal error: {str(e)}")
        raise
