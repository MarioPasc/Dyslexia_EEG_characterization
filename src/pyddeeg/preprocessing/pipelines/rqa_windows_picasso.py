#!/usr/bin/env python3
"""
RQA Analysis Script for EEG Signal Processing.

This script performs Recurrence Quantification Analysis (RQA) on EEG data
with multiple window sizes across multiple datasets. Configuration is loaded from a YAML file.
"""

import os
import sys

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
)

import argparse
import logging
import time

import yaml
from pathlib import Path

import numpy as np

from typing import List, Dict, Optional, Any, Tuple

from dataclasses import dataclass

from pyddeeg.preprocessing.tools.rqa_toolbox.utils import iter_signal_windows
from pyddeeg.preprocessing.tools.rqa_toolbox.rqa import compute_rqa_metrics_for_window
from pyddeeg.preprocessing.pipelines import CHANNEL_NAME_TO_INDEX
from pyddeeg.preprocessing.tools.rqa_toolbox.optimization.tuner import tune_window

# Add Dask imports
from dask import delayed
from dask.distributed import Client, LocalCluster, progress

import gc


# Logging
def setup_logging(config_dict: Dict):
    """
    Set up logging configuration based on YAML settings.

    Parameters:
        config_dict: Dictionary with logging configuration
    """
    log_dir = config_dict.get("logging", {}).get("directory", "logs")
    log_file = config_dict.get("logging", {}).get("filename", "rqa_windows.log")
    log_level_name = config_dict.get("logging", {}).get("level", "INFO")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Map string log level to logging constant
    log_level = getattr(logging, log_level_name.upper())

    # Configure logging
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logger = logging.getLogger("rqa_windows")
    logger.info(f"Logging initialized at {log_path} with level {log_level_name}")

    return logger


# Dask setup
def setup_dask_client(
    n_workers: int, threads_per_worker: int, memory_limit: str, logger: logging.Logger
) -> Client:
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
    logger.info(
        f"Setting up Dask cluster with {n_workers} workers, "
        f"{threads_per_worker} threads per worker, "
        f"{memory_limit} memory limit per worker"
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
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

    # Optimization parameters
    optimise_takens: bool = True
    tuning_max_lag: int = 100
    tuning_max_dim: int = 10
    tuning_rec_rate: float = 0.15

    # Dask parameters
    use_dask: bool = True
    dask_n_workers: int = 4
    dask_threads_per_worker: int = 1
    dask_memory_limit: str = "4GB"


def load_config(yaml_path: str) -> RQAConfig:
    """
    Read and parse the YAML configuration into an :class:`RQAConfig`.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # ---------- Dask ----------
    dask_cfg = cfg.get("dask", {})
    use_dask = dask_cfg.get("use_dask", True)
    dask_n_workers = dask_cfg.get("n_workers", 4)
    dask_threads_worker = dask_cfg.get("threads_per_worker", 1)
    dask_mem_limit = dask_cfg.get("memory_limit", "4GB")

    # ---------- Channel name ↔ index ----------
    tgt_ch = cfg["target_channel"]
    if isinstance(tgt_ch, str) and tgt_ch in CHANNEL_NAME_TO_INDEX:
        tgt_ch = CHANNEL_NAME_TO_INDEX[tgt_ch]
    else:
        tgt_ch = int(tgt_ch)
        if tgt_ch not in CHANNEL_NAME_TO_INDEX.values():
            raise ValueError(f"Invalid target channel: {tgt_ch}")

    # ---------- Build dataclass ----------
    return RQAConfig(
        input_directory=cfg["input_directory"],
        output_directory=cfg["output_directory"],
        datasets=cfg["datasets"],
        target_channel=tgt_ch,
        target_bandwidth=int(cfg["target_bandwidth"]),
        # base (fallback) Takens params
        embedding_dim=cfg["rqa_parameters"]["embedding_dim"],
        time_delay=cfg["rqa_parameters"]["time_delay"],
        radius=cfg["rqa_parameters"]["radius"],
        distance_metric="euclidean",
        metrics_to_use=cfg["rqa_parameters"]["metrics_to_use"],
        min_diagonal_line=cfg["rqa_parameters"]["min_diagonal_line"],
        min_vertical_line=cfg["rqa_parameters"]["min_vertical_line"],
        min_white_vertical_line=cfg["rqa_parameters"]["min_white_vertical_line"],
        window_sizes=cfg["window_sizes"],
        # ---- NEW: tuning flags --------------------------------------
        optimise_takens=cfg.get("optimise_takens", False),
        tuning_max_lag=cfg.get("tuning_max_lag", 100),
        tuning_max_dim=cfg.get("tuning_max_dim", 10),
        tuning_rec_rate=cfg.get("tuning_rec_rate", 0.15),
        save_results=cfg.get("save_results", True),
        file_prefix=cfg.get("file_prefix", "rqa_analysis"),
        verbose=cfg.get("verbose", True),
        use_dask=use_dask,
        dask_n_workers=dask_n_workers,
        dask_threads_per_worker=dask_threads_worker,
        dask_memory_limit=dask_mem_limit,
    )


# Code
def process_single_patient(  # unchanged signature
    patient_idx: int,
    patient_signal: np.ndarray,
    window_sizes: List[int],
    *,
    optimise_takens: bool,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    metrics_to_use: List[str],
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int,
    tuning_max_lag: int,
    tuning_max_dim: int,
    tuning_rec_rate: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the (metrics × windows) tensor **streamingly** so that at most
    *one* EEG window lives in memory.

    Returns
    -------
    Dict[window_size, Tuple[metrics[ n_m , n_w ], takens[3 , n_w]]]
    """
    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    n_metrics = len(metrics_to_use)

    # ensure the raw signal is float64 & contiguous for Numba / pyRQA
    sig = np.ascontiguousarray(patient_signal, dtype=np.float64)

    for w_size in window_sizes:
        stride = w_size // 2
        n_win = (sig.size - w_size) // stride + 1
        if n_win <= 0:
            continue

        metrics_arr = np.empty((n_metrics, n_win), dtype=np.float32)
        takens_arr = np.empty((3, n_win), dtype=np.float32)

        for idx, w in iter_signal_windows(sig, w_size, stride):
            # contiguous float64 slice view
            w64 = np.ascontiguousarray(w, dtype=np.float64)

            if optimise_takens:
                tau, m, eps = tune_window(
                    w64,
                    max_lag=tuning_max_lag,
                    max_dim=tuning_max_dim,
                    rec_rate=tuning_rec_rate,
                )
            else:
                m, tau, eps = embedding_dim, time_delay, radius

            takens_arr[:, idx] = (tau, m, eps)

            metric_dict, _ = compute_rqa_metrics_for_window(
                window_signal=w64,
                embedding_dim=m,
                time_delay=tau,
                radius=eps,
                distance_metric=distance_metric,
                metrics_to_use=metrics_to_use,
                min_diagonal_line=min_diagonal_line,
                min_vertical_line=min_vertical_line,
                min_white_vertical_line=min_white_vertical_line,
            )
            metrics_arr[:, idx] = list(metric_dict.values())

        results[w_size] = (metrics_arr, takens_arr)

        # keep RSS low between different window sizes
        gc.collect()

    return results


def process_dataset(
    dataset_name: str,
    dataset_path: str,
    config: RQAConfig,
    logger: logging.Logger,
    client: Client | None = None,
) -> Dict[str, Any]:
    """
    Compute RQA + tuned Takens parameters for every patient in *dataset*.
    The first patient is processed directly to size the tensors; the rest
    are dispatched (optionally) to Dask.
    """
    logger.info(f"Processing dataset: {dataset_name} ({dataset_path})")

    data = np.load(dataset_path)["data"]
    n_pat, n_chan, _, n_band = data.shape

    # --- containers ---------------------------------------------------
    results_by_window: Dict[int, dict] = {}
    takens_by_window: Dict[int, np.ndarray] = {}

    # ---------- first patient (dims & names) --------------------------
    first = process_single_patient(
        patient_idx=0,
        patient_signal=data[0, config.target_channel, :, config.target_bandwidth],
        window_sizes=config.window_sizes,
        optimise_takens=config.optimise_takens,
        embedding_dim=config.embedding_dim,
        time_delay=config.time_delay,
        radius=config.radius,
        distance_metric=config.distance_metric,
        metrics_to_use=config.metrics_to_use,
        min_diagonal_line=config.min_diagonal_line,
        min_vertical_line=config.min_vertical_line,
        min_white_vertical_line=config.min_white_vertical_line,
        tuning_max_lag=config.tuning_max_lag,
        tuning_max_dim=config.tuning_max_dim,
        tuning_rec_rate=config.tuning_rec_rate,
    )

    for wsize, (met, tak) in first.items():
        stride = wsize // 2
        n_met, n_win = met.shape
        results_by_window[wsize] = {
            "stride": stride,
            "num_windows": n_win,
            "results_tensor": np.zeros((n_pat, n_met, n_win)),
            "window_centers": np.arange(n_win) * stride + wsize // 2,
        }
        takens_by_window[wsize] = np.zeros((n_pat, 3, n_win))
        results_by_window[wsize]["results_tensor"][0] = met
        takens_by_window[wsize][0] = tak

    # ---------- remaining patients (optional Dask) --------------------
    def _patient_task(
        idx: int,
        dataset_path: str,
        target_ch: int,
        target_bw: int,
        cfg: RQAConfig,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Runs in a worker process – reads ONE patient and returns numpy
        results.  All large arrays are released before returning."""
        import gc

        with np.load(dataset_path, mmap_mode="r") as npz_file:
            sig = np.asarray(
                npz_file["data"][idx, target_ch, :, target_bw], dtype=np.float64
            )

        out = process_single_patient(
            patient_idx=idx,
            patient_signal=sig,
            window_sizes=cfg.window_sizes,
            optimise_takens=cfg.optimise_takens,
            embedding_dim=cfg.embedding_dim,
            time_delay=cfg.time_delay,
            radius=cfg.radius,
            distance_metric=cfg.distance_metric,
            metrics_to_use=cfg.metrics_to_use,
            min_diagonal_line=cfg.min_diagonal_line,
            min_vertical_line=cfg.min_vertical_line,
            min_white_vertical_line=cfg.min_white_vertical_line,
            tuning_max_lag=cfg.tuning_max_lag,
            tuning_max_dim=cfg.tuning_max_dim,
            tuning_rec_rate=cfg.tuning_rec_rate,
        )

        # explicit cleanup to keep the nanny happy
        del sig
        gc.collect()
        from pyddeeg.preprocessing.tools.rqa_toolbox.optimization.tuner import (
            _estimate_tau,
        )

        _estimate_tau.cache_clear()
        return out

    if n_pat > 1:
        indices = list(range(1, n_pat))
        if client:
            futures = client.map(
                _patient_task,
                indices,
                dataset_path=dataset_path,
                target_ch=config.target_channel,
                target_bw=config.target_bandwidth,
                cfg=config,
            )
            for idx, pat_dict in zip(indices, client.gather(futures)):
                for wsize, (met, tak) in pat_dict.items():
                    results_by_window[wsize]["results_tensor"][idx] = met
                    takens_by_window[wsize][idx] = tak
                # --- Logging every X patients ---
                if idx % 10 == 0:
                    for wsize, tak in pat_dict.items():
                        tak_arr = tak[1]  # shape [3, n_win]
                        n_win = tak_arr.shape[1]
                        if n_win >= 3:
                            mid_idxs = (
                                [n_win // 2 - 1, n_win // 2, n_win // 2 + 1]
                                if n_win > 3
                                else list(range(n_win))
                            )
                        else:
                            mid_idxs = list(range(n_win))
                        vals = tak_arr[:, mid_idxs]
                        logger.info(
                            f"[Patient {idx}] Window {wsize}: tau={vals[0]}, m={vals[1]}, eps={vals[2]}"
                        )
        else:
            for idx in indices:
                pat_dict = _patient_task(idx)
                for wsize, (met, tak) in pat_dict.items():
                    results_by_window[wsize]["results_tensor"][idx] = met
                    takens_by_window[wsize][idx] = tak
                # --- Logging every X patients ---
                if idx % 10 == 0:
                    for wsize, (met, tak_arr) in pat_dict.items():
                        n_win = tak_arr.shape[1]
                        if n_win >= 3:
                            mid_idxs = (
                                [n_win // 2 - 1, n_win // 2, n_win // 2 + 1]
                                if n_win > 3
                                else list(range(n_win))
                            )
                        else:
                            mid_idxs = list(range(n_win))
                        vals = tak_arr[:, mid_idxs]
                        logger.info(
                            f"[Patient {idx}] Window {wsize}: tau={vals[0]}, m={vals[1]}, eps={vals[2]}"
                        )
    # ---------- return ------------------------------------------------
    return {
        "dataset_name": dataset_name,
        "num_patients": n_pat,
        "window_sizes": config.window_sizes,
        "metric_names": config.metrics_to_use,
        "results_by_window": results_by_window,
        "takens_by_window": takens_by_window,  # ← NEW
    }


def process_all_datasets(
    config: RQAConfig, logger: logging.Logger, client: Client = None
) -> Dict[str, Any]:
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
                client=client,
            )

            all_results[dataset_name] = dataset_results

            save_dataset_results(
                dataset_name=dataset_name,
                results=dataset_results,
                config=config,
                logger=logger,
            )

            # Force garbage collection
            import gc

            gc.collect()

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
    logger: logging.Logger,
) -> None:
    """Save metrics and Takens parameters in separate NPZ files."""
    out_dir = os.path.join(config.output_directory, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ----------- metrics -----------
    metrics_file = os.path.join(
        out_dir, f"{config.file_prefix}_{dataset_name}_metrics.npz"
    )
    mdata = {
        "metric_names": results["metric_names"],
        "window_sizes": results["window_sizes"],
    }

    for w, win_dict in results["results_by_window"].items():
        mdata[f"w{w}_metrics"] = win_dict["results_tensor"]
        mdata[f"w{w}_centers"] = win_dict["window_centers"]
        mdata[f"w{w}_stride"] = win_dict["stride"]

    np.savez_compressed(metrics_file, **mdata)
    logger.info(f"Metrics saved to {metrics_file}")

    # ----------- Takens ------------
    takens_file = os.path.join(
        out_dir, f"{config.file_prefix}_{dataset_name}_takens.npz"
    )
    tdata = {"window_sizes": results["window_sizes"]}
    for w, arr in results["takens_by_window"].items():
        tdata[f"w{w}_takens"] = arr  # shape [patients, 3, windows]
    np.savez_compressed(takens_file, **tdata)
    logger.info(f"Takens parameters saved to {takens_file}")


# Main
def main():
    """Main function to process YAML config and run RQA analysis."""
    parser = argparse.ArgumentParser(
        description="Perform RQA analysis on EEG data across multiple datasets"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "--channel", "-c", type=int, help="Override target channel index in config"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    args = parser.parse_args()

    try:
        # Load raw config dictionary for logging setup
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)

        # Set up logging
        logger = setup_logging(config_dict)
        logger.info(f"Starting RQA windows analysis with config: {args.config}")

        # Load configuration
        config = load_config(args.config)

        # Override target channel if specified via command line
        if args.channel is not None:
            logger.info(
                f"Overriding target channel from config ({config.target_channel}) with command line value: {args.channel}"
            )
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
                logger=logger,
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
        total_patients = (
            sum(results[dataset]["num_patients"] for dataset in results)
            if results
            else 0
        )
        processing_time = end_time - start_time

        logger.info(
            f"RQA analysis completed successfully in {processing_time:.2f} seconds"
        )
        logger.info(
            f"Processed {total_datasets} datasets with {total_patients} total patients"
        )
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
        if (
            "config_dict" in locals()
            and config_dict.get("logging", {}).get("level", "").upper() == "DEBUG"
        ):
            logger.exception("Detailed traceback:")
        return 1  # Error exit code


def run_from_config(
    cfg_path: Path | str, *, channel: str | int, no_parallel: bool = False
) -> None:
    argv = ["--config", str(cfg_path), "--channel", str(channel)]
    if no_parallel:
        argv.append("--no-parallel")
    sys.argv = ["rqa_windows_picasso.py", *argv]
    sys.exit(main())  # re-use existing entry point


if __name__ == "__main__":
    # Add time module for execution timing
    import time

    sys.exit(main())
