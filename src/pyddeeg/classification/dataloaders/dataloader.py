import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from dataclasses import dataclass

import numpy as np


@dataclass
class EEGDataset:
    """
    Dataclass containing training and testing data for EEG classification.
    
    Attributes:
        train_data: Original training data without labels (n_samples, n_metrics, n_timepoints)
        test_data: Original testing data without labels (n_samples, n_metrics, n_timepoints)
        train_with_labels: Training data with labels as additional dimension (n_samples, n_metrics+1, n_timepoints)
        test_with_labels: Testing data with labels as additional dimension (n_samples, n_metrics+1, n_timepoints)
        train_labels: Training labels (n_samples,)
        test_labels: Testing labels (n_samples,)
        dd_train_indices: Indices of dyslexia patients in training set
        ct_train_indices: Indices of control patients in training set
        dd_test_indices: Indices of dyslexia patients in test set
        ct_test_indices: Indices of control patients in test set
        metadata: Dictionary containing additional information about the dataset
    """
    train_data: np.ndarray
    test_data: np.ndarray
    train_with_labels: np.ndarray
    test_with_labels: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    dd_train_indices: List[int]
    ct_train_indices: List[int]
    dd_test_indices: List[int]
    ct_test_indices: List[int]
    metadata: Dict[str, Any]
    
    def summary(self) -> None:
        """Print a summary of the dataset."""
        print(f"EEG Dataset Summary:")
        print(f"  Original train data shape: {self.train_data.shape}")
        print(f"  Train data with labels shape: {self.train_with_labels.shape}")
        print(f"  Original test data shape: {self.test_data.shape}")
        print(f"  Test data with labels shape: {self.test_with_labels.shape}")
        print(f"  Training samples: {len(self.train_labels)} ({sum(self.train_labels == 1)} dyslexia, {sum(self.train_labels == 0)} control)")
        print(f"  Testing samples: {len(self.test_labels)} ({sum(self.test_labels == 1)} dyslexia, {sum(self.test_labels == 0)} control)")
        if self.metadata:
            print("  Metadata:")
            for key, value in self.metadata.items():
                print(f"    {key}: {value}")


def create_labeled_dataset(
    dataset_root: Union[str, Path],
    window: str,
    direction: str,
    electrode: str,
    fold_info: Dict[str, Any],
    fold_index: int = 0
) -> EEGDataset:
    """
    Create training and testing datasets with labels incorporated as an additional dimension.
    
    Args:
        dataset_root: Root directory containing the EEG dataset
        window: Window size identifier (e.g., "window_200")
        direction: Direction identifier (e.g., "up")
        electrode: Electrode identifier (e.g., "T7")
        fold_info: Dictionary containing fold information from stratified_kfold
        fold_index: Which fold to use (default: 0)
        
    Returns:
        EEGDataset object containing training and testing data with labels
    """
    # Load dataset indexes
    dataset_index_path = os.path.join(dataset_root, "dataset_index.json")
    with open(dataset_index_path, "r") as f:
        dataset_indexes = json.load(f)
    
    # Get file paths for dyslexia and control data
    dd_path, ct_path = dataset_indexes[window][direction][electrode]
    
    # Load data
    dd_data = np.load(dd_path)["metrics"]
    ct_data = np.load(ct_path)["metrics"]
    
    # Get fold information
    fold = fold_info.get("folds", [])[fold_index]
    dd_train_index = np.array(fold.get("dd_train", []), dtype=np.int32)
    ct_train_index = np.array(fold.get("ct_train", []), dtype=np.int32)
    dd_test_index = np.array(fold.get("dd_test", []), dtype=np.int32)
    ct_test_index = np.array(fold.get("ct_test", []), dtype=np.int32)
    
    # Create labels
    dd_labels = np.ones(dd_data.shape[0])
    ct_labels = np.zeros(ct_data.shape[0])
    
    # Split data based on fold indices
    dd_train = dd_data[dd_train_index, :, :]
    dd_test = dd_data[dd_test_index, :, :]
    ct_train = ct_data[ct_train_index, :, :]
    ct_test = ct_data[ct_test_index, :, :]
    
    # Split labels based on fold indices
    dd_train_labels = dd_labels[dd_train_index]
    dd_test_labels = dd_labels[dd_test_index]
    ct_train_labels = ct_labels[ct_train_index]
    ct_test_labels = ct_labels[ct_test_index]
    
    # Concatenate data
    train_data = np.concatenate((dd_train, ct_train), axis=0)
    test_data = np.concatenate((dd_test, ct_test), axis=0)
    
    # Concatenate labels
    train_labels = np.concatenate((dd_train_labels, ct_train_labels))
    test_labels = np.concatenate((dd_test_labels, ct_test_labels))
    
    # Incorporate labels as an additional dimension
    train_labels_expanded = train_labels[:, np.newaxis, np.newaxis]
    test_labels_expanded = test_labels[:, np.newaxis, np.newaxis]
    
    # Broadcast labels to match the time points dimension (window-level labels)
    train_labels_broadcast = np.broadcast_to(
        train_labels_expanded, 
        (train_data.shape[0], 1, train_data.shape[2])
    )
    test_labels_broadcast = np.broadcast_to(
        test_labels_expanded, 
        (test_data.shape[0], 1, test_data.shape[2])
    )
    
    # Concatenate along metrics dimension (axis=1)
    train_with_labels = np.concatenate((train_data, train_labels_broadcast), axis=1)
    test_with_labels = np.concatenate((test_data, test_labels_broadcast), axis=1)
    
    # Create metadata
    metadata = {
        "window": window,
        "direction": direction,
        "electrode": electrode,
        "fold_index": fold_index,
        "dd_shape": dd_data.shape,
        "ct_shape": ct_data.shape
    }
    
    # Return the EEGDataset object
    return EEGDataset(
        train_data=train_data,
        test_data=test_data,
        train_with_labels=train_with_labels,
        test_with_labels=test_with_labels,
        train_labels=train_labels,
        test_labels=test_labels,
        dd_train_indices=dd_train_index.tolist(),
        ct_train_indices=ct_train_index.tolist(),
        dd_test_indices=dd_test_index.tolist(),
        ct_test_indices=ct_test_index.tolist(),
        metadata=metadata
    )