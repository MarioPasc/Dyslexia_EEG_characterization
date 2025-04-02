#!/usr/bin/env python3
# filepath: /home/mariopasc/Python/Projects/Dyslexia_EEG_characterization/src/pyddeeg/utils/postprocessing/reorganize_zerolag_results.py

"""
Script to create a structured dataset from EEG data files.

This script organizes EEG data into a hierarchical folder structure by processing
one stimulus at a time to optimize memory usage:
- Root: leeduca_eeg
  - Stimuli type (syllable_2hz, word_8hz, phonem_20hz)
    - Direction (up, down)
      - Group (ct: control, dd: developmental dyslexia)
        - Patient (P1, P2, ...)
          - bands/
            - delta/ (band 0)
            - theta/ (band 1)
            - alpha/ (band 2)
            - beta/  (band 3)
            - gamma/ (band 4)
"""

import os
import re
import logging
import gc
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Generator
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
STIMULI_MAPPING = {"2": "syllable_2hz", "8": "word_8hz", "20": "phonem_20hz"}

BAND_MAPPING = {0: "delta", 1: "theta", 2: "alpha", 3: "beta", 4: "gamma"}

# Source data path
ROOT_PROCESSED = "/home/mariopasc/Python/Datasets/EEG/timeseries/processed/zerolag"
# Target directory for new structure
OUTPUT_DIR = "/home/mariopasc/Python/Datasets/EEG/leeduca_eeg"


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse the filename to extract group, direction, and stimuli information.

    Args:
        filename: The filename to parse

    Returns:
        Tuple containing (group, direction, stimuli)
    """
    # Extract pattern from filename (e.g., DD_UP_preprocess_8.npz)
    pattern = r"(DD|CT)_(UP|DOWN)_preprocess_(\d+)\.npz"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern")

    group = match.group(1).lower()  # 'dd' or 'ct'
    direction = match.group(2).lower()  # 'up' or 'down'
    stimuli = match.group(3)  # '2', '8', or '20'

    return group, direction, stimuli


def create_folder_structure(output_dir: Union[str, Path]) -> None:
    """
    Create the base folder structure for the dataset.

    Args:
        output_dir: Path to the output directory
    """
    output_dir = Path(output_dir)

    # Create root directory if it doesn't exist
    if output_dir.exists():
        logger.warning(
            f"Output directory {output_dir} already exists. Files may be overwritten."
        )
    else:
        output_dir.mkdir(parents=True)
        logger.info(f"Created root directory: {output_dir}")

    # Create the hierarchical structure
    for stimuli_code, stimuli_name in STIMULI_MAPPING.items():
        for direction in ["up", "down"]:
            for group in ["ct", "dd"]:
                # Determine number of patients based on the data shapes
                num_patients = 34 if group == "ct" else 15

                for patient_idx in range(1, num_patients + 1):
                    patient_dir = (
                        output_dir
                        / stimuli_name
                        / direction
                        / group
                        / f"P{patient_idx}"
                        / "bands"
                    )

                    # Create band directories
                    for _, band_name in BAND_MAPPING.items():
                        band_dir = patient_dir / band_name
                        band_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Folder structure created successfully.")


def find_stimulus_files(stimuli_code: str) -> List[Tuple[str, str]]:
    """
    Find all files for a specific stimulus.

    Args:
        stimuli_code: The stimulus code to filter by ("2", "8", or "20")

    Returns:
        List of tuples containing (filename, filepath)
    """
    stimulus_files = []

    for root, _, files in os.walk(ROOT_PROCESSED):
        for file in files:
            if file.endswith(f"_{stimuli_code}.npz"):
                filepath = os.path.join(root, file)
                stimulus_files.append((file, filepath))

    return stimulus_files


def process_stimulus_files(stimuli_code: str, output_dir: Union[str, Path]) -> None:
    """
    Process all files for a specific stimulus.

    Args:
        stimuli_code: The stimulus code to process ("2", "8", or "20")
        output_dir: Path to the output directory
    """
    output_dir = Path(output_dir)
    stimuli_name = STIMULI_MAPPING[stimuli_code]

    logger.info(f"Processing {stimuli_name} files...")

    # Find all files for this stimulus
    stimulus_files = find_stimulus_files(stimuli_code)

    if not stimulus_files:
        logger.warning(f"No files found for stimulus {stimuli_code}")
        return

    logger.info(f"Found {len(stimulus_files)} files for {stimuli_name}")

    # Process each file one by one
    for filename, filepath in tqdm(
        stimulus_files, desc=f"Processing {stimuli_name} files"
    ):
        try:
            # Parse filename
            group, direction, _ = parse_filename(filename)

            # Load data
            logger.debug(f"Loading file: {filename}")
            data_array = np.load(filepath)["data"]
            logger.debug(f"Loaded {filename} with shape: {data_array.shape}")

            # Get patient count from the first dimension of the array
            num_patients = data_array.shape[0]

            # For each patient
            for patient_idx in range(num_patients):
                patient_data = data_array[patient_idx]  # Shape: (32, 68000, 5)

                # For each frequency band
                for band_idx, band_name in BAND_MAPPING.items():
                    # Extract data for this band: shape (32, 68000)
                    band_data = patient_data[:, :, band_idx]

                    # Construct path for saving
                    save_path = (
                        output_dir
                        / stimuli_name
                        / direction
                        / group
                        / f"P{patient_idx+1}"
                        / "bands"
                        / band_name
                    )
                    save_file = save_path / f"{band_name}_data.npz"

                    # Save the band data
                    np.savez_compressed(save_file, data=band_data)

            logger.info(
                f"Processed {filename} - created data for {num_patients} patients"
            )

            # Explicitly release memory
            del data_array
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")


def main() -> None:
    """Main function to execute the dataset structure creation."""
    logger.info("Starting dataset structure creation")

    # Create the folder structure
    create_folder_structure(OUTPUT_DIR)

    # Process one stimulus at a time
    for stimuli_code in STIMULI_MAPPING.keys():
        logger.info(
            f"Starting processing for stimulus {stimuli_code} ({STIMULI_MAPPING[stimuli_code]})"
        )
        process_stimulus_files(stimuli_code, OUTPUT_DIR)

        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Completed processing for stimulus {stimuli_code}")

    logger.info(f"Dataset structure creation completed. Data stored in {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
