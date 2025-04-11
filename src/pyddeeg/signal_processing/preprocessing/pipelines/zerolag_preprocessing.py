# -*- coding: utf-8 -*-
"""
EEG Preprocessing Pipeline

This script performs preprocessing on EEG time series data, including:
- Loading data from specified directories
- Filtering with zero-lag bandpass FIR filter
- Standardizing signals
- Saving preprocessed data in separate files based on condition

Author: Improved version based on original code
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

import gc

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pyddeeg.signal_processing.preprocessing.tools import zerolag_bpfir2

# Optional imports for accelerated processing
try:
    import dask.array as da
    from dask.diagnostics import ProgressBar

    ACCELERATED_LIBRARIES_AVAILABLE = True
except ImportError:
    ACCELERATED_LIBRARIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eeg_preprocessing")


@dataclass
class EEGConfig:
    """Configuration parameters for EEG preprocessing."""

    data_dir: str
    output_dir: str
    age: str
    stim: str
    sampling_freq: int
    nframes: int
    speed: str
    eeg_bands: Dict[str, Tuple[float, float]]
    ch_names: List[str]
    use_multiprocessing: bool = False
    show_filter_response: bool = False


def load_config(config_path: str) -> EEGConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        EEGConfig object with all configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Create output directory if it doesn't exist
    Path(config_data["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Set multiprocessing based on speed setting and available libraries
    use_multiprocessing = (
        config_data["speed"] == "fast" and ACCELERATED_LIBRARIES_AVAILABLE
    )

    if config_data["speed"] == "fast" and not ACCELERATED_LIBRARIES_AVAILABLE:
        logger.warning(
            "Fast processing requested but required libraries not available. "
            "Install dask for accelerated processing."
        )

    return EEGConfig(
        data_dir=config_data["data_dir"],
        output_dir=config_data["output_dir"],
        age=config_data["age"],
        stim=config_data["stim"],
        sampling_freq=config_data["sampling_freq"],
        nframes=config_data["nframes"],
        speed=config_data["speed"],
        eeg_bands=config_data["eeg_bands"],
        ch_names=config_data["ch_names"],
        use_multiprocessing=use_multiprocessing,
        show_filter_response=config_data.get("show_filter_response", False),
    )


def process_single_sample(
    sample: np.ndarray,
    band_info: Tuple[str, Tuple[float, float]],
    scaler: StandardScaler,
    config: EEGConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single sample for a specific band.

    Args:
        sample: Single sample data
        band_info: Tuple containing (band_name, (low_freq, high_freq))
        scaler: StandardScaler instance
        config: Configuration parameters

    Returns:
        Tuple of (scaled data, filtered data)
    """
    band_name, (low_freq, high_freq) = band_info
    fc = (high_freq - low_freq) / 2

    # Scale the data
    scaled_data = scaler.fit_transform(sample)

    # Initialize filtered data for this band
    filtered_data = np.zeros_like(scaled_data)

    # Apply filter to each channel
    for c in range(scaled_data.shape[0]):
        filtered_data[c, :] = zerolag_bpfir2(
            scaled_data[c, :],
            fc,
            low_freq,
            high_freq,
            fs=config.sampling_freq,
            show_response=(c == 0 and config.show_filter_response),
        )

    return scaled_data, filtered_data


def data_preprocessing(data_x: np.ndarray, config: EEGConfig) -> np.ndarray:
    """
    Preprocess EEG data with scaling and bandpass filtering.

    Args:
        data_x: Input data of shape (samples, channels, time)
        config: Configuration parameters

    Returns:
        Filtered data of shape (samples, channels, time, bands)
    """
    logger.info("Starting data preprocessing")
    bands = list(config.eeg_bands.keys())
    data_x_f = np.zeros((data_x.shape[0], data_x.shape[1], data_x.shape[2], len(bands)))

    if config.use_multiprocessing and ACCELERATED_LIBRARIES_AVAILABLE:
        logger.info("Using accelerated processing with dask")
        return _accelerated_preprocessing(data_x, config)

    # Standard sequential processing
    for s in tqdm(range(data_x.shape[0]), desc="Processing samples"):
        scaler = StandardScaler()
        # Scale the entire sample at once
        data_x[s, :, :] = scaler.fit_transform(data_x[s, :, :])

        # Process each band
        for band_idx, (band, freq_range) in enumerate(config.eeg_bands.items()):
            fc = (freq_range[1] - freq_range[0]) / 2

            # data_x.shape[1] - 1 to exlucde the Cz channel from being bandfiltered
            for c in range(data_x.shape[1] - 1):
                data_x_f[s, c, :, band_idx] = zerolag_bpfir2(
                    data_x[s, c, :],
                    fc,
                    freq_range[0],
                    freq_range[1],
                    fs=config.sampling_freq,
                    show_response=(
                        s == 0
                        and c == 0
                        and band_idx == 0
                        and config.show_filter_response
                    ),
                )

    logger.info("Preprocessing completed")
    return data_x_f


def _accelerated_preprocessing(data_x: np.ndarray, config: EEGConfig) -> np.ndarray:
    """
    Accelerated preprocessing using dask for parallel computation.

    Args:
        data_x: Input data of shape (samples, channels, time)
        config: Configuration parameters

    Returns:
        Filtered data of shape (samples, channels, time, bands)
    """
    bands = list(config.eeg_bands.keys())
    data_x_f = np.zeros((data_x.shape[0], data_x.shape[1], data_x.shape[2], len(bands)))

    # Convert to dask array for parallel processing
    dask_data = da.from_array(data_x, chunks=(1, data_x.shape[1], data_x.shape[2]))

    def process_chunk(chunk_data):
        result = np.zeros(
            (
                chunk_data.shape[0],
                chunk_data.shape[1],
                chunk_data.shape[2],
                len(config.eeg_bands),
            )
        )

        for s in range(chunk_data.shape[0]):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(chunk_data[s])

            for band_idx, (band, freq_range) in enumerate(config.eeg_bands.items()):
                fc = (freq_range[1] - freq_range[0]) / 2

                # Same change as the slow mode, ignore the Cz channel when doing the bandpass filter
                for c in range(chunk_data.shape[1] - 1):
                    result[s, c, :, band_idx] = zerolag_bpfir2(
                        scaled_data[c, :],
                        fc,
                        freq_range[0],
                        freq_range[1],
                        fs=config.sampling_freq,
                    )

        return result

    # Apply function to chunks in parallel
    result = dask_data.map_blocks(
        process_chunk,
        chunks=(
            dask_data.chunks[0],
            dask_data.chunks[1],
            dask_data.chunks[2],
            len(bands),
        ),
        new_axis=3,
    )

    # Compute results with progress bar
    with ProgressBar():
        data_x_f = result.compute()

    return data_x_f


def load_data(data_dir: str, age: str, stim: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data from files.

    Args:
        data_dir: Directory containing data files
        age: Age group
        stim: Stimulus type

    Returns:
        Tuple of (control_data, dyslexia_data)
    """
    datafile = f"data_timeseries_{age}_{stim}.npz"
    file_path = os.path.join(data_dir, datafile)

    logger.info(f"Loading data from {file_path}")

    try:
        data = np.load(file_path)
        return data["datamatrix_C"], data["datamatrix_D"]
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def main(config_path: str) -> None:
    """
    Main function to run the EEG preprocessing pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    output_dir = config.output_dir

    logger.info(
        f"Starting EEG preprocessing pipeline for age={config.age}, stim={config.stim}"
    )
    logger.info(f"Processing mode: {config.speed}")

    # Load data
    data_C, data_D = load_data(config.data_dir, config.age, config.stim)

    # Process Control UP data
    logger.info("Processing Control UP data")
    if os.path.exists(os.path.join(output_dir, f"CT_UP_preprocess_{config.stim}.npz")):
        logger.warning(
            f"Preprocessed Control UP data already exists. Skipping this preprocessing stage."
        )
    else:
        data_c_up = data_C[:, :, :, 0]
        data_c_up_f = data_preprocessing(data_c_up, config)
        np.savez(
            os.path.join(output_dir, f"CT_UP_preprocess_{config.stim}.npz"),
            data=data_c_up_f,
        )
        logger.info("Saved preprocessed Control UP data")
        del data_c_up, data_c_up_f
        gc.collect()

    # Process Control DOWN data
    logger.info("Processing Control DOWN data")
    if os.path.exists(
        os.path.join(output_dir, f"CT_DOWN_preprocess_{config.stim}.npz")
    ):
        logger.warning(
            f"Preprocessed Control UP data already exists. Skipping this preprocessing stage."
        )
    else:
        data_c_down = data_C[:, :, :, 1]
        data_c_down_f = data_preprocessing(data_c_down, config)
        np.savez(
            os.path.join(output_dir, f"CT_DOWN_preprocess_{config.stim}.npz"),
            data=data_c_down_f,
        )
        logger.info("Saved preprocessed Control DOWN data")
        del data_c_down, data_c_down_f, data_C
        gc.collect()

    # Process Dyslexia UP data
    logger.info("Processing Dyslexia UP data")
    if os.path.exists(os.path.join(output_dir, f"DD_UP_preprocess_{config.stim}.npz")):
        logger.warning(
            f"Preprocessed Control UP data already exists. Skipping this preprocessing stage."
        )
    else:
        data_d_up = data_D[:, :, :, 0]
        data_d_up_f = data_preprocessing(data_d_up, config)
        np.savez(
            os.path.join(output_dir, f"DD_UP_preprocess_{config.stim}.npz"),
            data=data_d_up_f,
        )
        logger.info("Saved preprocessed Dyslexia UP data")
        del data_d_up, data_d_up_f
        gc.collect()

    # Process Dyslexia DOWN data
    logger.info("Processing Dyslexia DOWN data")
    if os.path.exists(
        os.path.join(output_dir, f"DD_DOWN_preprocess_{config.stim}.npz")
    ):
        logger.warning(
            f"Preprocessed Control UP data already exists. Skipping this preprocessing stage."
        )
    else:
        data_d_down = data_D[:, :, :, 1]
        data_d_down_f = data_preprocessing(data_d_down, config)
        np.savez(
            os.path.join(output_dir, f"DD_DOWN_preprocess_{config.stim}.npz"),
            data=data_d_down_f,
        )
        logger.info("Saved preprocessed Dyslexia DOWN data")
        del data_d_down, data_d_down_f, data_D
        gc.collect()

    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Preprocessing Pipeline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    main(args.config)
