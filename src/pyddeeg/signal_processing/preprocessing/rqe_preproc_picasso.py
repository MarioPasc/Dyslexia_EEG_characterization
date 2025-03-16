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
import gc
import numpy as np
import yaml
from typing import Dict, Tuple, List, Optional, Union, Any
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Import the parallelizable RQE functions
from pyddeeg.signal_processing.rqa_toolbox.rqe_parallelizable import (
    process_single_channel_band,
)


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

    # Compute all tasks at once
    compute_start = time.time()
    logger.info(f"Starting Dask computation of all tasks")

    # Process the tasks
    results = dask.compute(tasks)[0]

    compute_time = time.time() - compute_start
    logger.info(f"Dask computation completed in {compute_time:.2f} seconds")

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


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


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
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Set up logging
    log_dir = config.get("logging", {}).get("directory", "./logs")
    logger = setup_logging(log_dir)
    logger.info(f"Configuration loaded from {args.config}")

    # Create output directory if it doesn't exist
    output_dir = Path(config.get("output_directory", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get resources from SLURM environment if available, otherwise use config/args
    if os.environ.get("SLURM_JOB_ID"):
        logger.info("Running in SLURM environment")

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

    # Get RQA parameters from config
    rqa_params = config.get("rqa_parameters", {})
    normalize_metrics = config.get("normalize_metrics", False)

    logger.info(f"RQA parameters: {rqa_params}")
    logger.info(f"Normalize metrics: {normalize_metrics}")

    # Get input directory and dataset filenames
    input_dir = config.get("input_directory", "./data")
    datasets = config.get("datasets", {})

    # Process each dataset
    for dataset_name, filename in datasets.items():
        file_path = os.path.join(input_dir, filename)
        logger.info(f"Processing dataset {dataset_name} from file {file_path}")

        # Load data - do this inside the loop to free memory between datasets
        try:
            logger.info(f"Loading data from {file_path}")
            data = np.load(file_path)["data"]
            logger.info(f"Loaded {dataset_name} with shape {data.shape}")

            # Process the dataset with all tasks at once
            rqa_metrics, rqe_values, corr_values = process_dataset_parallel(
                data=data,
                rqa_params=rqa_params,
                normalize_metrics=normalize_metrics,
                dataset_name=dataset_name,
                logger=logger,
            )

            # Save results
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

            # Clean up to free memory
            del data, rqa_metrics, rqe_values, corr_values
            gc.collect()

        except Exception as e:
            logger.error(
                f"Error processing dataset {dataset_name}: {str(e)}", exc_info=True
            )

    # Shut down the client and cluster
    client.close()
    cluster.close()
    logger.info("Processing complete. Dask client and cluster closed.")


if __name__ == "__main__":
    main()
