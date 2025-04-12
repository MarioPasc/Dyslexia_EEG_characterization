#!/usr/bin/env python3
"""
RQA Analysis Script for EEG Signal Processing.

This script performs Recurrence Quantification Analysis (RQA) on EEG data
with multiple window sizes across multiple datasets. Configuration is loaded from a YAML file.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable
from pyddeeg.signal_processing.rqa_toolbox.utils import extract_signal_windows
from pyddeeg.signal_processing.rqa_toolbox.rqa import compute_rqa_metrics_for_window
from pyddeeg.signal_processing.preprocessing.pipelines import CHANNEL_NAME_TO_INDEX

# Add Dask imports
from dask import delayed
from dask.distributed import Client, LocalCluster, progress

# Logging
def setup_logging(config_dict: Dict):
    """
    Set up logging configuration based on YAML settings.
    
    Parameters:
        config_dict: Dictionary with logging configuration
    """
    log_dir = config_dict.get('logging', {}).get('directory', 'logs')
    log_file = config_dict.get('logging', {}).get('filename', 'rqa_windows.log')
    log_level_name = config_dict.get('logging', {}).get('level', 'INFO')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Map string log level to logging constant
    log_level = getattr(logging, log_level_name.upper())
    
    # Configure logging
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("rqa_windows")
    logger.info(f"Logging initialized at {log_path} with level {log_level_name}")
    
    return logger

# Dask setup
def setup_dask_client(n_workers: int, threads_per_worker: int, memory_limit: str, logger: logging.Logger) -> Client:
    """
    Set up a Dask client for parallel processing.
    
    Parameters:
    -----------
    n_workers : int
        Number of worker processes to use
    threads_per_worker : int
        Number of threads per worker process
    memory_limit : str
        Memory limit per worker (e.g., "4GB")
    logger : logging.Logger
        Logger object for recording progress
        
    Returns:
    --------
    client : dask.distributed.Client
        Configured Dask client
    """
    logger.info(f"Setting up Dask cluster with {n_workers} workers, "
                f"{threads_per_worker} threads per worker, "
                f"{memory_limit} memory limit per worker")
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    
    return client

# Data structures
@dataclass
class RQAConfig:
    """Configuration for RQA analysis.
    
    Attributes:
        input_directory: Base directory for input NPZ files
        output_directory: Base directory for output files
        datasets: Dictionary mapping dataset names to filenames
        target_channel: Target channel index to analyze
        target_bandwidth: Target frequency band index to analyze
        embedding_dim: Embedding dimension for phase space reconstruction
        time_delay: Time delay for phase space reconstruction
        radius: Threshold radius for recurrence detection
        distance_metric: Distance metric for recurrence calculation
        metrics_to_use: List of RQA metrics to compute
        window_sizes: List of window sizes to analyze (in samples)
        min_diagonal_line: Minimum diagonal line length
        min_vertical_line: Minimum vertical line length
        min_white_vertical_line: Minimum white vertical line length
        save_results: Whether to save statistics as CSV
        file_prefix: Prefix for output filenames
        verbose: Whether to print detailed statistics
        dask_n_workers: Number of Dask worker processes
        dask_threads_per_worker: Number of threads per Dask worker
        dask_memory_limit: Memory limit per Dask worker
        use_dask: Whether to use Dask for parallel processing
    """
    # Directory and dataset attributes
    input_directory: str
    output_directory: str
    datasets: Dict[str, str]
    
    # Target attributes
    target_channel: int
    target_bandwidth: int
    
    # RQA parameters
    embedding_dim: int
    time_delay: int
    radius: float
    distance_metric: str
    metrics_to_use: List[str]
    min_diagonal_line: int
    min_vertical_line: int
    min_white_vertical_line: int
    
    # Window parameters
    window_sizes: List[int]
    
    # Output parameters
    save_results: bool
    file_prefix: str
    verbose: bool
    
    # Dask parameters
    use_dask: bool = True
    dask_n_workers: int = 4
    dask_threads_per_worker: int = 1
    dask_memory_limit: str = "4GB"


def load_config(yaml_path: str) -> RQAConfig:
    """
    Read and parse RQA configuration from YAML file.
    
    Parameters:
        yaml_path: Path to the YAML configuration file
    
    Returns:
        Configuration object containing all parameters
    
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        KeyError: If required configuration keys are missing
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Get dask configuration with defaults if not present
    dask_config = config_dict.get('dask', {})
    use_dask = dask_config.get('use_dask', True)
    dask_n_workers = dask_config.get('n_workers', 4)
    dask_threads_per_worker = dask_config.get('threads_per_worker', 1)
    dask_memory_limit = dask_config.get('memory_limit', "4GB")
    
    # Handle channel specification (name or index)
    target_channel = config_dict['target_channel']
    if isinstance(target_channel, str) and target_channel in CHANNEL_NAME_TO_INDEX:
        target_channel = CHANNEL_NAME_TO_INDEX[target_channel]
    else:
        # Convert string indices to integers if needed
        target_channel = int(target_channel) if not isinstance(target_channel, int) else target_channel
        if target_channel not in CHANNEL_NAME_TO_INDEX.values():
            raise ValueError(f"Invalid target channel: {target_channel}. Must be a valid index or channel name.")

    # Create unified configuration object
    return RQAConfig(
        # Directory and dataset attributes
        input_directory=config_dict['input_directory'],
        output_directory=config_dict['output_directory'],
        datasets=config_dict['datasets'],
        
        # Target attributes
        target_channel=target_channel,
        target_bandwidth=config_dict['target_bandwidth'],
        
        # RQA parameters
        embedding_dim=config_dict['rqa_parameters']['embedding_dim'],
        time_delay=config_dict['rqa_parameters']['time_delay'],
        radius=config_dict['rqa_parameters']['radius'],
        distance_metric='euclidean',  # Default - could be added to YAML if needed
        metrics_to_use=config_dict['rqa_parameters']['metrics_to_use'],
        min_diagonal_line=config_dict['rqa_parameters']['min_diagonal_line'],
        min_vertical_line=config_dict['rqa_parameters']['min_vertical_line'],
        min_white_vertical_line=config_dict['rqa_parameters']['min_white_vertical_line'],
        
        # Window parameters
        window_sizes=config_dict['window_sizes'],
        
        # Output parameters
        save_results=config_dict.get('save_results', True),
        file_prefix=config_dict.get('file_prefix', 'rqa_analysis'),
        verbose=config_dict.get('verbose', True),
        
        # Dask parameters
        use_dask=use_dask,
        dask_n_workers=dask_n_workers,
        dask_threads_per_worker=dask_threads_per_worker,
        dask_memory_limit=dask_memory_limit
    )

# Code
def process_single_patient(
    patient_idx: int,
    patient_signal: np.ndarray,
    window_sizes: List[int],
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    metrics_to_use: List[str],
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int
) -> Dict[int, np.ndarray]:
    """
    Process a single patient's EEG signal for all window sizes.
    
    Parameters:
    -----------
    patient_idx : int
        Patient index (for logging only)
    patient_signal : np.ndarray
        1D array containing the patient's EEG signal
    window_sizes : List[int]
        List of window sizes to process
    embedding_dim : int
        Embedding dimension for phase space reconstruction
    time_delay : int
        Time delay for phase space reconstruction
    radius : float
        Threshold radius for recurrence detection
    distance_metric : str
        Distance metric to use for recurrence calculation
    metrics_to_use : List[str]
        List of specific RQA metrics to compute
    min_diagonal_line : int
        Minimum diagonal line length for determinism calculation
    min_vertical_line : int
        Minimum vertical line length for laminarity calculation
    min_white_vertical_line : int
        Minimum white vertical line length
    
    Returns:
    --------
    results : Dict[int, np.ndarray]
        Dictionary mapping window sizes to metrics tensors
    """
    results = {}
    
    for window_size in window_sizes:
        stride = window_size // 2
        metrics_tensor, _ = compute_rqa_metrics_for_signal(
            signal=patient_signal,
            window_size=window_size,
            stride=stride,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            metrics_to_use=metrics_to_use,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line
        )
        
        results[window_size] = metrics_tensor
    
    return results

def compute_rqa_metrics_for_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    embedding_dim: int = 10,
    time_delay: int = 1,
    radius: float = 0.8,
    distance_metric: str = "euclidean",
    metrics_to_use: list = None,
    min_diagonal_line: int = 2,
    min_vertical_line: int = 2,
    min_white_vertical_line: int = 2
) -> tuple[np.ndarray, list]:
    """
    Compute RQA metrics for a signal using sliding windows and return as a tensor.
    
    Parameters:
    -----------
    signal : np.ndarray
        1D array containing the signal to analyze
    window_size : int
        Size of each window in samples
    stride : int
        Step size between consecutive windows in samples
    embedding_dim : int
        Embedding dimension for phase space reconstruction
    time_delay : int
        Time delay for phase space reconstruction
    radius : float
        Threshold radius for recurrence detection
    distance_metric : str
        Distance metric to use for recurrence calculation
    metrics_to_use : list
        List of specific RQA metrics to compute
    min_diagonal_line : int
        Minimum diagonal line length for determinism calculation
    min_vertical_line : int
        Minimum vertical line length for laminarity calculation
    min_white_vertical_line : int
        Minimum white vertical line length
        
    Returns:
    --------
    metrics_tensor : np.ndarray
        Tensor with shape [metrics, window_points] containing RQA metrics for each window
    metric_names : list
        List of metric names in the same order as in the tensor
    """
    # Extract windows from signal
    windows = extract_signal_windows(signal, window_size, stride)
    
    # Initialize list to store metrics for each window
    all_metrics = []
    
    # Process each window
    for window_signal in windows:
        metrics, _ = compute_rqa_metrics_for_window(
            window_signal=window_signal,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            metrics_to_use=metrics_to_use,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line
        )
        all_metrics.append(list(metrics.values()))
    
    # Convert to numpy array and transpose to get [metrics, window_points] shape
    metrics_tensor = np.array(all_metrics).T if all_metrics else np.array([])
    
    # Extract metric names for reference
    metric_names = list(metrics.keys()) if all_metrics else []
    
    return metrics_tensor, metric_names

def process_dataset(
    dataset_name: str,
    dataset_path: str, 
    config: RQAConfig,
    logger: logging.Logger,
    client: Client = None
) -> Dict[str, Any]:
    """
    Process a single dataset, computing RQA metrics for each patient.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being processed
    dataset_path : str
        Path to the NPZ file
    config : RQAConfig
        Configuration object with processing parameters
    logger : logging.Logger
        Logger object for recording progress
    client : dask.distributed.Client, optional
        Dask client for parallel processing
    
    Returns:
    --------
    results : Dict[str, Any]
        Dictionary containing the processing results
    """
    logger.info(f"Processing dataset: {dataset_name} ({dataset_path})")
    
    try:
        # Load dataset
        data = np.load(dataset_path)["data"]
        num_patients, num_electrodes, num_points, num_bands = data.shape
        
        logger.info(f"Dataset shape: {data.shape} (patients, electrodes, points, bands)")
        logger.info(f"Number of patients: {num_patients}, Number of electrodes: {num_electrodes}, Number of points: {num_points}, Number of bands: {num_bands}")
        logger.info(f"Target channel: {config.target_channel}, Target bandwidth: {config.target_bandwidth}")
        # Check if channel and band indices are valid
        if config.target_channel >= num_electrodes:
            raise ValueError(f"Target channel index {config.target_channel} is out of bounds (max: {num_electrodes-1})")
        if config.target_bandwidth >= num_bands:
            raise ValueError(f"Target bandwidth index {config.target_bandwidth} is out of bounds (max: {num_bands-1})")
        
        logger.info(f"Using channel {config.target_channel}, bandwidth {config.target_bandwidth}")
        
        # Initialize result tensors
        results_by_window = {}
        metric_names = None
        
        # Process first patient to get dimensions
        logger.info(f"Processing patient 0 to determine dimensions...")
        patient_signal = data[0, config.target_channel, :, config.target_bandwidth]
        
        # For each window size, compute metrics for the first patient
        for window_size in config.window_sizes:
            stride = window_size // 2  # 50% overlap
            metrics_tensor, names = compute_rqa_metrics_for_signal(
                signal=patient_signal,
                window_size=window_size,
                stride=stride,
                embedding_dim=config.embedding_dim,
                time_delay=config.time_delay,
                radius=config.radius,
                distance_metric=config.distance_metric,
                metrics_to_use=config.metrics_to_use,
                min_diagonal_line=config.min_diagonal_line,
                min_vertical_line=config.min_vertical_line,
                min_white_vertical_line=config.min_white_vertical_line
            )
            
            if metric_names is None:
                metric_names = names
            
            # For this window size, record the number of windows
            num_metrics, num_windows = metrics_tensor.shape
            results_by_window[window_size] = {
                'stride': stride,
                'num_windows': num_windows,
                'results_tensor': np.zeros((num_patients, num_metrics, num_windows)),
                'window_centers': np.array([stride * i + window_size // 2 for i in range(num_windows)])
            }
            
            # Store the first patient's results
            results_by_window[window_size]['results_tensor'][0, :, :] = metrics_tensor
        
        # Now process the rest of the patients in parallel using Dask
        if num_patients > 1:
            logger.info(f"Processing remaining {num_patients-1} patients in parallel...")
            
            # Create a list of delayed tasks for each patient
            if client is not None:
                tasks = []
                for patient_idx in range(1, num_patients):
                    patient_signal = data[patient_idx, config.target_channel, :, config.target_bandwidth]
                    
                    # Create a delayed task for this patient
                    task = delayed(process_single_patient)(
                        patient_idx=patient_idx,
                        patient_signal=patient_signal,
                        window_sizes=config.window_sizes,
                        embedding_dim=config.embedding_dim,
                        time_delay=config.time_delay,
                        radius=config.radius,
                        distance_metric=config.distance_metric,
                        metrics_to_use=config.metrics_to_use,
                        min_diagonal_line=config.min_diagonal_line,
                        min_vertical_line=config.min_vertical_line,
                        min_white_vertical_line=config.min_white_vertical_line
                    )
                    tasks.append((patient_idx, task))
                
                # Compute all tasks in parallel
                logger.info(f"Submitting {len(tasks)} tasks to Dask cluster")
                results_futures = client.compute([task for _, task in tasks])
                progress(results_futures)
                
                # Get results and store in the results tensor
                results_list = client.gather(results_futures)
                
                # Put results in the right place
                for idx, patient_results in enumerate(results_list):
                    patient_idx = idx + 1  # Adjust index (first patient was processed separately)
                    
                    for window_size, metrics_tensor in patient_results.items():
                        results_by_window[window_size]['results_tensor'][patient_idx, :, :] = metrics_tensor
                
                logger.info("Parallel processing completed")
            else:
                # Fall back to sequential processing if no client provided
                logger.warning("No Dask client provided, falling back to sequential processing")
                for patient_idx in range(1, num_patients):
                    if patient_idx % 5 == 0:  # Log every 5 patients
                        logger.info(f"Processing patient {patient_idx}/{num_patients}...")
                        
                    patient_signal = data[patient_idx, config.target_channel, :, config.target_bandwidth]
                    
                    # Process sequentially
                    patient_results = process_single_patient(
                        patient_idx=patient_idx,
                        patient_signal=patient_signal,
                        window_sizes=config.window_sizes,
                        embedding_dim=config.embedding_dim,
                        time_delay=config.time_delay,
                        radius=config.radius,
                        distance_metric=config.distance_metric,
                        metrics_to_use=config.metrics_to_use,
                        min_diagonal_line=config.min_diagonal_line,
                        min_vertical_line=config.min_vertical_line,
                        min_white_vertical_line=config.min_white_vertical_line
                    )
                    
                    for window_size, metrics_tensor in patient_results.items():
                        results_by_window[window_size]['results_tensor'][patient_idx, :, :] = metrics_tensor
        
        # Create summary statistics
        dataset_results = {
            'dataset_name': dataset_name,
            'num_patients': num_patients,
            'metric_names': metric_names,
            'window_sizes': config.window_sizes,
            'results_by_window': results_by_window
        }
        
        logger.info(f"Dataset {dataset_name} processed successfully.")
        return dataset_results
    
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
        raise

def process_all_datasets(config: RQAConfig, logger: logging.Logger, client: Client = None) -> Dict[str, Any]:
    """
    Process all datasets specified in the configuration.
    
    Parameters:
    -----------
    config : RQAConfig
        Configuration object with processing parameters
    logger : logging.Logger
        Logger object for recording progress
    client : dask.distributed.Client, optional
        Dask client for parallel processing
    
    Returns:
    --------
    all_results : Dict[str, Any]
        Dictionary containing results for all datasets
    """
    logger.info("Starting processing of all datasets")
    
    all_results = {}
    
    # Process each dataset
    for dataset_name, dataset_filename in config.datasets.items():
        dataset_path = os.path.join(config.input_directory, dataset_filename)
        logger.info(f"Processing dataset {dataset_name} from {dataset_path}")
        
        try:
            # Process this dataset with the Dask client
            dataset_results = process_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                config=config,
                logger=logger,
                client=client
            )
            
            all_results[dataset_name] = dataset_results
            
            save_dataset_results(
                dataset_name=dataset_name,
                results=dataset_results,
                config=config,
                logger=logger
            )
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            # Continue with other datasets even if one fails
            continue
    
    logger.info(f"Processed {len(all_results)} datasets successfully")
    return all_results

def save_dataset_results(
    dataset_name: str,
    results: Dict[str, Any],
    config: RQAConfig,
    logger: logging.Logger
) -> None:
    """
    Save processing results for a single dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    results : Dict[str, Any]
        Processing results for the dataset
    config : RQAConfig
        Configuration object with output parameters
    logger : logging.Logger
        Logger object for recording progress
    """
    # Create output directory
    dataset_output_dir = os.path.join(config.output_directory, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Save NPZ file with all metrics
    output_npz = os.path.join(dataset_output_dir, f"{config.file_prefix}_{dataset_name}_metrics.npz")
    logger.info(f"Saving metrics to {output_npz}")
    
    # Prepare data for NPZ file
    npz_data = {
        'metric_names': results['metric_names'],
        'window_sizes': config.window_sizes,
    }
    
    # Add window-specific data
    for window_size, window_data in results['results_by_window'].items():
        npz_data[f'w{window_size}_metrics'] = window_data['results_tensor']
        npz_data[f'w{window_size}_centers'] = window_data['window_centers']
        npz_data[f'w{window_size}_stride'] = window_data['stride']
    
    np.savez_compressed(output_npz, **npz_data)
    logger.info(f"Results for dataset {dataset_name} saved successfully")

# Main
def main():
    """Main function to process YAML config and run RQA analysis."""
    parser = argparse.ArgumentParser(description="Perform RQA analysis on EEG data across multiple datasets")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")    
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity")
    parser.add_argument("--channel", "-c", type=int, help="Override target channel index in config")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    args = parser.parse_args()
    
    try:
        # Load raw config dictionary for logging setup
        with open(args.config, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Set up logging
        logger = setup_logging(config_dict)
        logger.info(f"Starting RQA windows analysis with config: {args.config}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Override target channel if specified via command line
        if args.channel is not None:
            logger.info(f"Overriding target channel from config ({config.target_channel}) with command line value: {args.channel}")
            config.target_channel = args.channel
            
        # Override verbosity if specified
        if args.verbose:
            config.verbose = True
            
        # Override parallel processing if specified
        if args.no_parallel:
            logger.info("Parallel processing disabled via command line")
            config.use_dask = False
            
        # Create main output directory if it doesn't exist
        os.makedirs(config.output_directory, exist_ok=True)
        
        # Log configuration details
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  Input directory: {config.input_directory}")
        logger.info(f"  Output directory: {config.output_directory}")
        logger.info(f"  Target channel: {config.target_channel}")
        logger.info(f"  Target bandwidth: {config.target_bandwidth}")
        logger.info(f"  Window sizes: {config.window_sizes}")
        logger.info(f"  RQA metrics: {config.metrics_to_use}")
        logger.info(f"  Datasets to process: {list(config.datasets.keys())}")
        
        # Setup Dask client if enabled
        client = None
        if config.use_dask:
            logger.info("Setting up Dask client for parallel processing")
            client = setup_dask_client(
                n_workers=config.dask_n_workers,
                threads_per_worker=config.dask_threads_per_worker,
                memory_limit=config.dask_memory_limit,
                logger=logger
            )
        
        # Process all datasets
        start_time = time.time()
        results = process_all_datasets(config=config, logger=logger, client=client)
        end_time = time.time()
        
        # Shut down Dask client if it was created
        if client is not None:
            logger.info("Shutting down Dask client")
            client.close()
        
        # Summary statistics
        total_datasets = len(results)
        total_patients = sum(results[dataset]['num_patients'] for dataset in results) if results else 0
        processing_time = end_time - start_time
        
        logger.info(f"RQA analysis completed successfully in {processing_time:.2f} seconds")
        logger.info(f"Processed {total_datasets} datasets with {total_patients} total patients")
        logger.info(f"Results saved to: {config.output_directory}")
        
        return 0  # Success exit code
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return 1  # Error exit code
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1  # Error exit code
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if 'config_dict' in locals() and config_dict.get('logging', {}).get('level', '').upper() == 'DEBUG':
            logger.exception("Detailed traceback:")
        return 1  # Error exit code


if __name__ == "__main__":
    # Add time module for execution timing
    import time
    sys.exit(main())