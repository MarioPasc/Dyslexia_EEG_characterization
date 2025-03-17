#!/usr/bin/env python3
"""
eeg_rqe_channel_processor.py - HPC Version

Process EEG datasets with Recurrence Quantification Analysis (RQA) and
Recurrence Quantification Entropy (RQE) metrics for specific channels using Dask.

This version allows processing either a single channel or all channels (except Cz).
Optimized for execution on high-memory HPC clusters with no internet connectivity.
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


def setup_logging(log_dir: str = "./logs", suffix: str = "") -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Add suffix to log filename if provided
    if suffix:
        log_file = (
            log_path
            / f"rqe_processing_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
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


def get_channel_indices(target_channel: str) -> List[int]:
    """
    Get list of channel indices to process based on target_channel.

    Parameters:
    -----------
    target_channel : str
        Channel name or "All" to process all channels except Cz

    Returns:
    --------
    List[int]
        List of channel indices to process
    """
    if target_channel.lower() == "all":
        # Return all channels except Cz (which is the last channel, index 31)
        return list(range(len(EEG_CHANNELS) - 1))  # All except the last one (Cz)
    else:
        # Find the index of the specified channel
        try:
            channel_idx = EEG_CHANNELS.index(target_channel)
            return [channel_idx]
        except ValueError:
            raise ValueError(
                f"Channel '{target_channel}' not found. Available channels: {', '.join(EEG_CHANNELS)}"
            )


def process_dataset_channels(
    data: np.ndarray,
    rqa_params: Dict[str, Any],
    target_channel: str,
    normalize_metrics: bool = False,
    dataset_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    status_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process specific channel(s) of an EEG dataset with RQA/RQE analysis using Dask.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_channels, n_samples, n_bands)
    rqa_params : Dict[str, Any]
        Parameters for RQA computation
    target_channel : str
        Channel name or "All" to process all channels except Cz
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
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        RQA metrics, RQE values, and correlation values
    """
    if logger is None:
        logger = logging.getLogger("EEG_RQE_Processor")

    n_patients, n_channels, n_samples, n_bands = data.shape

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
        logger.info(f"Processing {channel_description}")

        if status_dir:
            update_status_file(
                status_dir,
                "process_details",
                f"Processing {dataset_name} with shape {data.shape} - {channel_description}",
            )
    except ValueError as e:
        logger.error(f"Channel selection error: {str(e)}")
        raise

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

    # Initialize output arrays - only for selected channels
    n_selected_channels = len(channel_indices)
    rqa_metrics_array = np.full(
        (n_patients, n_selected_channels, n_bands, num_windows, num_metrics), np.nan
    )
    rqe_values_array = np.full(
        (n_patients, n_selected_channels, n_bands, num_rqe_windows), np.nan
    )
    corr_values_array = np.full(
        (n_patients, n_selected_channels, n_bands, num_rqe_windows), np.nan
    )

    # Create Dask task graph for selected channels
    tasks = {}
    task_indices = []

    # Create a progress counter
    total_tasks = n_patients * n_selected_channels * n_bands
    logger.info(f"Creating {total_tasks} tasks for parallel processing")

    if status_dir:
        update_status_file(
            status_dir,
            "process_details",
            f"Creating {total_tasks} tasks for {dataset_name} - {channel_description}",
        )

    start_time = time.time()
    task_creation_start = time.time()

    # Create tasks only for selected channels
    for patient_idx in range(n_patients):
        for local_ch_idx, global_ch_idx in enumerate(channel_indices):
            for band_idx in range(n_bands):
                # Create a unique key for this task
                task_key = (patient_idx, local_ch_idx, band_idx)
                task_indices.append(task_key)

                # Extract signal for this patient, channel, band combination
                # Use the global channel index to access data but the local index for output arrays
                signal = data[patient_idx, global_ch_idx, :, band_idx].copy()

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
    logger.info(f"Starting Dask computation of all tasks for {channel_description}")

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
        patient_idx, local_ch_idx, band_idx = task_key
        rqa_matrix, rqe_values, corr_values = results[task_key]

        # Store RQA metrics - using local channel index
        if rqa_matrix.size > 0:
            actual_windows = min(rqa_matrix.shape[0], num_windows)
            rqa_metrics_array[
                patient_idx, local_ch_idx, band_idx, :actual_windows, :
            ] = rqa_matrix
            successful_tasks += 1
        else:
            empty_results += 1

        # Store RQE values - using local channel index
        if rqe_values.size > 0:
            actual_rqe_windows = min(rqe_values.shape[0], num_rqe_windows)
            rqe_values_array[
                patient_idx, local_ch_idx, band_idx, :actual_rqe_windows
            ] = rqe_values
            corr_values_array[
                patient_idx, local_ch_idx, band_idx, :actual_rqe_windows
            ] = corr_values

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
        f"Dataset {dataset_name} processing complete in {total_elapsed:.2f} seconds"
    )

    return rqa_metrics_array, rqe_values_array, corr_values_array, channel_metadata


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
        description="Process EEG datasets with RQA/RQE analysis for specific channels using Dask."
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

    # Set up logging with channel information in the filename
    log_dir = config.get("logging", {}).get("directory", "./logs")
    logger = setup_logging(log_dir, suffix=target_channel.lower())
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Target channel: {target_channel}")

    # Create status directory
    status_dir = args.status_dir
    os.makedirs(status_dir, exist_ok=True)
    update_status_file(
        status_dir, "init", f"Starting RQE processing for channel: {target_channel}"
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

    # For supercomputer optimization, we use a higher memory limit per worker
    # but fewer workers to avoid memory issues
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

                    # Process the dataset with selected channel(s)
                    update_status_file(
                        status_dir,
                        "process",
                        f"Processing {dataset_name} - {target_channel}",
                    )
                    rqa_metrics, rqe_values, corr_values, channel_metadata = (
                        process_dataset_channels(
                            data=data,
                            rqa_params=rqa_params,
                            target_channel=target_channel,
                            normalize_metrics=normalize_metrics,
                            dataset_name=dataset_name,
                            logger=logger,
                            status_dir=status_dir,
                        )
                    )

                    # Log memory after processing
                    log_memory_usage(logger)

                    # Save results - include channel information in filename
                    if target_channel.lower() == "all":
                        output_filename = f"{dataset_name}_all_channels_rqe_results.npz"
                    else:
                        output_filename = (
                            f"{dataset_name}_{target_channel}_rqe_results.npz"
                        )

                    update_status_file(
                        status_dir,
                        "save",
                        f"Saving results for {dataset_name} - {target_channel}",
                    )
                    output_file = output_dir / output_filename
                    logger.info(f"Saving results to {output_file}")
                    np.savez_compressed(
                        output_file,
                        rqa_metrics=rqa_metrics,
                        rqe_values=rqe_values,
                        corr_values=corr_values,
                        rqa_params=rqa_params,
                        normalize_metrics=normalize_metrics,
                        channel_metadata=channel_metadata,
                    )
                    logger.info(f"Results saved successfully")
                    update_status_file(
                        status_dir,
                        "save",
                        f"Results saved for {dataset_name} - {target_channel}",
                    )

                    # Clean up to free memory
                    del data, rqa_metrics, rqe_values, corr_values, channel_metadata
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
        update_status_file(
            status_dir, "complete", f"Processing complete for {target_channel}"
        )

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        update_status_file(status_dir, "fatal", f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
