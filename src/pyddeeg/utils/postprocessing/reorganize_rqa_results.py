#!/usr/bin/env python3
"""
Reorganize RQA Results

This script processes RQA data from all electrodes into combined tensors.
It takes the path to a directory containing folders for each electrode,
where each folder contains four types of files: CT_DOWN, CT_UP, DD_DOWN, and DD_UP.
The script reads these files, extracts the RQA metrics, and combines them into a single tensor for each type.
The combined tensors are then saved as compressed .npz files.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

# Define the order of electrodes
ELECTRODE_ORDER: List[str] = [
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

METRIC_ORDER: List[str] = [
    "RR",
    "DET",
    "L_max",
    "L_mean",
    "ENT",
    "LAM",
    "TT",
    "V_max",
    "V_mean",
    "V_ENT",
    "W_max",
    "W_mean",
    "W_ENT",
    "CLEAR",
    "PERM_ENT",
]


def reorganize_rqa_results(
    path: str,
    stimuli: str,
    output_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Process RQA data from all electrodes into combined tensors.

    This function reads RQA data files from individual electrode folders,
    combines them into consolidated tensors for each condition type
    (CT_DOWN, CT_UP, DD_DOWN, DD_UP), and saves them as compressed .npz files.

    The shapes of the output tensors vary by type:
    - CT_DOWN and CT_UP: (34, num_electrodes, 5, 9701, 15)
    - DD_DOWN and DD_UP: (15, num_electrodes, 5, 9701, 15)

    Args:
        path (str): Path to the directory containing electrode folders
        stimuli: (str): Stimuli type from "slow_2hz", "syllable_8hz", or "phonem_20hz"
        output_path (Optional[str]): Path where processed files will be saved.
                                    If None, uses the input path.

    Returns:
        Dict[str, str]: Dictionary mapping file types to their saved file paths
    """
    # If no output path is specified, use the input path
    output_path = output_path or path

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Get all electrode folders
    folders = [
        os.path.join(path, folder)
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]

    # Dictionary to store data for each file type
    data_by_type: Dict[str, List[np.ndarray]] = {
        "CT_DOWN": [],
        "CT_UP": [],
        "DD_DOWN": [],
        "DD_UP": [],
    }

    # Dictionary to track processed electrodes
    processed_electrodes: set = set()

    # Dictionary to store output file paths
    result_files: Dict[str, str] = {}

    # Dictionary to track expected shapes for each file type
    expected_shapes: Dict[str, tuple] = {
        "CT_DOWN": (34, 1, 5, 9701, 15),
        "CT_UP": (34, 1, 5, 9701, 15),
        "DD_DOWN": (15, 1, 5, 9701, 15),
        "DD_UP": (15, 1, 5, 9701, 15),
    }

    # Process electrodes in the specified order
    for electrode in tqdm(ELECTRODE_ORDER, desc="Processing electrodes"):
        electrode_found = False

        for folder in folders:
            channel = os.path.basename(folder).split("_")[2]

            if channel == electrode:
                electrode_found = True
                processed_electrodes.add(channel)

                # Look for the four file types in this electrode folder
                for file_type in data_by_type.keys():
                    file_name = None
                    for file in os.listdir(folder):
                        if file.startswith(file_type) and file.endswith(".npz"):
                            file_name = file
                            break

                    if file_name:
                        file_path = os.path.join(folder, file_name)
                        data = np.load(file_path)
                        rqa_metrics = data["rqa_metrics"]

                        # If this is the first electrode, update the expected shape
                        if not data_by_type[file_type]:
                            expected_shapes[file_type] = rqa_metrics.shape

                        data_by_type[file_type].append(rqa_metrics)
                    else:
                        print(
                            f"Warning: File type {file_type} not found for electrode {electrode}"
                        )
                        # Add zeros with the appropriate shape for this file type
                        data_by_type[file_type].append(
                            np.zeros(expected_shapes[file_type])
                        )

                break

        if not electrode_found:
            print(f"Warning: Electrode {electrode} not found in the dataset")
            # Add zeros for missing electrodes
            for file_type in data_by_type.keys():
                # We might not have seen any examples yet, so check if we have data
                if data_by_type[file_type]:
                    # Use the shape from the first electrode's data
                    shape = data_by_type[file_type][0].shape
                else:
                    # Use the default expected shape
                    shape = expected_shapes[file_type]

                data_by_type[file_type].append(np.zeros(shape))

    # Check if all electrodes in the folders were processed
    for folder in folders:
        channel = os.path.basename(folder).split("_")[2]
        if channel not in processed_electrodes:
            print(
                f"Warning: Electrode {channel} was found in the dataset but not in the predefined order"
            )

    # Combine data for each type and save
    for file_type, data_list in tqdm(
        data_by_type.items(), desc="Saving combined data files"
    ):
        if data_list:
            # Stack along the second dimension (electrode dimension)
            combined_data = np.concatenate(data_list, axis=1)

            # Save the combined data
            output_file = os.path.join(
                output_path, f"{file_type}_{stimuli}_rqa_processed.npz"
            )
            np.savez_compressed(output_file, rqa_metrics=combined_data)
            print(
                f"Saved {file_type} data with shape {combined_data.shape} to {output_file}"
            )
            # Store the file path in the results dictionary
            result_files[file_type] = output_file
        else:
            print(f"No data found for {file_type}")

    return result_files
