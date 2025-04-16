#!/usr/bin/env python
"""
EEG Data Reorganizer

This script reorganizes EEG RQA data from electrode-indexed to window-size-indexed format.
"""

import os
import numpy as np
import argparse
import json  

def load_data(raw_dir):
    """
    Load and prepare the data dictionary mapping electrodes to file paths.
    
    Parameters:
    -----------
    raw_dir : str
        Path to the directory containing raw data folders
        
    Returns:
    --------
    data : dict
        Dictionary with electrode channels as keys and lists of file paths as values
    window_sizes : numpy.ndarray
        Array of window sizes
    """
    data = {f.split("_")[2]: os.path.join(raw_dir, f) for f in os.listdir(raw_dir)}
    folders = ['DD_DOWN', 'CT_UP', 'DD_UP', 'CT_DOWN']
    
    for key, value in data.items():
        data[key] = [os.path.join(value, folders[i], f"rqa_analysis_{folders[i]}_metrics.npz") for i in range(4)]
    
    # Get window sizes from the first file
    window_sizes = np.load(data[list(data.keys())[0]][0], allow_pickle=True)["window_sizes"]
    
    return data, window_sizes


def reorganize_data(data, window_sizes, electrode_indexed_dir):
    """
    Reorganize EEG data by window size
    
    Parameters:
    -----------
    data : dict
        Dictionary with electrode channels as keys and lists of file paths as values
    window_sizes : list
        List of window sizes
    electrode_indexed_dir : str
        Root directory for electrode-indexed output files
    """
    # Process each window size
    for w in window_sizes:
        print(f"Processing window size {w}...")
        # Create window size directory if it doesn't exist
        window_dir = os.path.join(electrode_indexed_dir, f"window_{w}")
        up_dir = os.path.join(window_dir, "up")
        down_dir = os.path.join(window_dir, "down")
        os.makedirs(up_dir, exist_ok=True)
        os.makedirs(down_dir, exist_ok=True)
        
        # Keys for this window size
        metric_key = f"w{w}_metrics"
        centers_key = f"w{w}_centers"
        stride_key = f"w{w}_stride"
        
        # Get common data (from first electrode, first condition file)
        first_electrode = list(data.keys())[0]
        first_file = data[first_electrode][0]
        common_data = np.load(first_file, allow_pickle=True)
        
        # Save metadata (common fields)
        metadata = {
            "metric_names": common_data["metric_names"],
            "centers": common_data[centers_key],
            "stride": common_data[stride_key]
        }
        np.savez(os.path.join(window_dir, "metadata.npz"), **metadata)
        
        # Process each electrode and condition
        for electrode, files in data.items():
            for file_path in files:
                # Get condition from file path
                condition = file_path.split("/")[-2]  # DD_DOWN, CT_UP, etc.
                
                # Load data
                npz_data = np.load(file_path, allow_pickle=True)
                
                # Extract only metrics for this window size
                metrics = {
                    "metrics": npz_data[metric_key]
                }
                
                # Determine output directory based on condition
                # More precise check to distinguish UP vs DOWN conditions
                if condition.endswith("_UP"):
                    output_dir = up_dir
                elif condition.endswith("_DOWN"):
                    output_dir = down_dir
                else:
                    print(f"Warning: Unrecognized condition format: {condition}")
                    continue
                output_file = os.path.join(output_dir, f"{electrode}_{condition}.npz")
                np.savez(output_file, **metrics)
        
    print(f"Data reorganization complete. Files saved to {electrode_indexed_dir}")

def build_dataset_index(electrode_indexed_dir, output_file=None):
    """
    Build a dictionary indexing all the data files by window, direction (up/down), and electrode.
    Save this index as a JSON file.
    
    Parameters:
    -----------
    electrode_indexed_dir : str
        Root directory for electrode-indexed data
    output_file : str, optional
        Path to save the JSON index. If None, saves to 'dataset_index.json' in electrode_indexed_dir
        
    Returns:
    --------
    dataset : dict
        Dictionary with structure: {window: {up/down: {electrode: [file_paths]}}}
    """
    if output_file is None:
        output_file = os.path.join(electrode_indexed_dir, "dataset_index.json")
        
    dataset = {}
    for window in os.listdir(electrode_indexed_dir):
        window_path = os.path.join(electrode_indexed_dir, window)
        if not os.path.isdir(window_path):
            continue
            
        dataset[window] = {}
        for updown in os.listdir(window_path):
            updown_path = os.path.join(window_path, updown)
            if not os.path.isdir(updown_path):
                continue
                
            dataset[window][updown] = {}
            for electrode_file in os.listdir(updown_path):
                electrode = electrode_file.split("_")[0]
                if electrode not in dataset[window][updown]:
                    dataset[window][updown][electrode] = []
                    
                electrode_path = os.path.join(updown_path, electrode_file)
                if os.path.isfile(electrode_path) and electrode_path not in dataset[window][updown][electrode]:
                    dataset[window][updown][electrode].append(electrode_path)
    """
    # Convert to relative paths for better portability
    for window in dataset:
        for updown in dataset[window]:
            for electrode in dataset[window][updown]:
                dataset[window][updown][electrode] = [
                    os.path.relpath(path, start=electrode_indexed_dir)
                    for path in dataset[window][updown][electrode]
                ]
    """
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset index saved to {output_file}")
    return dataset

def main(root_dir=None, raw_dir=None, output_dir=None):
    """
    Main function that orchestrates the data reorganization process.
    
    Parameters:
    -----------
    root_dir : str, optional
        Root directory for all data
    raw_dir : str, optional
        Directory containing raw data
    output_dir : str, optional
        Directory for output
    """
    # Set default paths if not provided
    if root_dir is None:
        root_dir = "/home/mario/Python/Datasets/EEG/timeseries/processed/rqa_windows/"
    
    if raw_dir is None:
        raw_dir = os.path.join(root_dir, "raw")
    
    if output_dir is None:
        output_dir = os.path.join(root_dir, "dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and get window sizes
    data, window_sizes = load_data(raw_dir)
    
    # Reorganize data
    reorganize_data(data, window_sizes, output_dir)

    # Build dataset index
    build_dataset_index(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize EEG RQA data by window size.")
    parser.add_argument("--root-dir", default="/home/mario/Python/Datasets/EEG/timeseries/processed/rqa_windows/",
                        help="Root directory for all data")
    parser.add_argument("--raw-dir", help="Directory containing raw data")
    parser.add_argument("--output-dir", help="Directory for output")
    
    args = parser.parse_args()
    
    main(
        root_dir=args.root_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir
    )