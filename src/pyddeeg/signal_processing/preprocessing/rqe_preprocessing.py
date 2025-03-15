#!/usr/bin/env python3
"""
eeg_rqe_processor.py

Process multiple EEG datasets with Recurrence Quantification Analysis (RQA) and
Recurrence Quantification Entropy (RQE) metrics using Dask for parallelization.

The script processes multiple EEG datasets, computing RQA metrics for sliding windows
in the time domain, and then computing RQE and correlation metrics across those RQA metrics.

Configuration is loaded from a YAML file, allowing easy adjustment of all parameters.
"""

import os
import time
import numpy as np
import yaml
from typing import Dict, Tuple, List, Optional, Union, Any
import dask
from dask.distributed import Client, progress, wait
from dask import delayed
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Import the parallelizable RQE functions
from pyddeeg.signal_processing.rqa_toolbox.rqe_parallelizable import (
    process_single_channel_band,
)


# Set up logging
def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Set up logging with file and console handlers."""
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create a unique log filename with timestamp
    log_file = (
        log_path / f"rqe_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure logging
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


def process_dataset(
    data: np.ndarray,
    rqa_params: Dict[str, Any],
    normalize_metrics: bool = False,
    dataset_name: str = "unknown",
    n_workers: int = -1,
    chunk_size: Union[int, str] = "auto",
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process an entire EEG dataset with RQA/RQE analysis using Dask parallelization.

    Parameters:
    -----------
    data : np.ndarray
        EEG data with shape (n_patients, n_channels, n_samples, n_bands)
    rqa_params : Dict[str, Any]
        Parameters for RQA/RQE analysis
    normalize_metrics : bool
        Whether to normalize metrics before RQE computation
    dataset_name : str
        Name of the dataset for logging purposes
    n_workers : int
        Number of parallel jobs to use (-1 = all cores)
    chunk_size : Union[int, str]
        Size of chunks for processing ('auto' or number of patients per chunk)
    logger : Optional[logging.Logger]
        Logger for output messages

    Returns:
    --------
    rqa_metrics_array : np.ndarray
        Array of RQA metrics with shape determined by parameters
    rqe_values_array : np.ndarray
        Array of RQE values
    corr_values_array : np.ndarray
        Array of correlation values
    """
    if logger is None:
        logger = logging.getLogger("EEG_RQE_Processor")

    n_patients, n_channels, n_samples, n_bands = data.shape
    logger.info(f"Processing dataset {dataset_name}: {data.shape}")

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
        f"Calculated dimensions: num_windows={num_windows}, num_rqe_windows={num_rqe_windows}"
    )

    # Create Dask task graph
    tasks = {}
    task_indices = []

    # Use Dask's delayed function to create a task graph
    for patient_idx in range(n_patients):
        for channel_idx in range(n_channels):
            for band_idx in range(n_bands):
                # Create a unique key for this task
                task_key = (patient_idx, channel_idx, band_idx)
                task_indices.append(task_key)

                # Extract signal for this patient, channel, band combination
                signal = data[patient_idx, channel_idx, :, band_idx]

                # Create a delayed task
                tasks[task_key] = delayed(process_single_channel_band)(
                    signal=signal,
                    rqa_params=rqa_params,
                    normalize_metrics=normalize_metrics,
                )

    logger.info(f"Created {len(tasks)} tasks for dataset {dataset_name}")

    # Compute all tasks
    logger.info(f"Computing tasks for dataset {dataset_name}...")
    start_time = time.time()

    # Convert the dictionary of delayed objects to a list and compute
    task_values = dask.compute(tasks)
    results = task_values[0]  # Extract from tuple

    elapsed = time.time() - start_time
    logger.info(f"Computation completed in {elapsed:.2f} seconds")

    # Initialize output arrays
    # RQA metrics array: shape = (n_patients, n_channels, n_bands, n_windows, n_metrics)
    rqa_metrics_array = np.full(
        (n_patients, n_channels, n_bands, num_windows, num_metrics), np.nan
    )

    # RQE values array: shape = (n_patients, n_channels, n_bands, n_rqe_windows)
    rqe_values_array = np.full(
        (n_patients, n_channels, n_bands, num_rqe_windows), np.nan
    )

    # Correlation values array: shape = (n_patients, n_channels, n_bands, n_rqe_windows)
    corr_values_array = np.full(
        (n_patients, n_channels, n_bands, num_rqe_windows), np.nan
    )

    # Populate the output arrays
    logger.info("Assembling results...")
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

    logger.info(
        f"Results assembled: {successful_tasks} successful tasks, {empty_results} empty results"
    )
    logger.info(f"Dataset {dataset_name} processing complete")

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

    # Set up Dask client according to config
    dask_config = config.get("dask", {})
    n_workers = dask_config.get("n_workers", -1)
    threads_per_worker = dask_config.get("threads_per_worker", 1)
    memory_limit = dask_config.get("memory_limit", "auto")

    if n_workers == -1:
        # Let Dask decide based on available resources
        client = Client(
            threads_per_worker=threads_per_worker, memory_limit=memory_limit
        )
    else:
        # Use the specified number of workers
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        )

    logger.info(f"Dask client started with dashboard at {client.dashboard_link}")
    logger.info(
        f"Worker configuration: n_workers={n_workers}, threads_per_worker={threads_per_worker}, memory_limit={memory_limit}"
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

        # Load data
        try:
            data = np.load(file_path)["data"]
            logger.info(f"Loaded {dataset_name} with shape {data.shape}")

            # Process the dataset
            rqa_metrics, rqe_values, corr_values = process_dataset(
                data=data,
                rqa_params=rqa_params,
                normalize_metrics=normalize_metrics,
                dataset_name=dataset_name,
                n_workers=n_workers,
                logger=logger,
            )

            # Save results
            output_file = output_dir / f"{dataset_name}_rqe_results.npz"
            np.savez_compressed(
                output_file,
                rqa_metrics=rqa_metrics,
                rqe_values=rqe_values,
                corr_values=corr_values,
                rqa_params=rqa_params,
                normalize_metrics=normalize_metrics,
            )
            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(
                f"Error processing dataset {dataset_name}: {str(e)}", exc_info=True
            )

    # Shut down the client
    client.close()
    logger.info("Processing complete. Dask client closed.")


if __name__ == "__main__":
    main()
