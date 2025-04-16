import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from dataclasses import dataclass

import numpy as np


@dataclass
class EEGDataset:
    """
    Dataclass containing training and testing data for EEG classification,
    ready for use with scikit-learn models.
    
    Attributes:
        X_train: Training data reshaped for scikit-learn (n_patients*n_timepoints, n_metrics)
        X_test: Testing data reshaped for scikit-learn (n_patients*n_timepoints, n_metrics)
        y_train: Training labels (one per window/timepoint)
        y_test: Testing labels (one per window/timepoint)
        patient_indices_train: Maps each sample in X_train to its patient index
        patient_indices_test: Maps each sample in X_test to its patient index
        metadata: Dictionary containing additional information about the dataset
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    patient_indices_train: np.ndarray
    patient_indices_test: np.ndarray
    dd_train_indices: List[int]
    ct_train_indices: List[int]
    dd_test_indices: List[int]
    ct_test_indices: List[int]
    metadata: Dict[str, Any]
    
    def summary(self) -> None:
        """Print a summary of the dataset."""
        print(f"EEG Dataset Summary (Scikit-Learn Ready):")
        print(f"  Training data shape: {self.X_train.shape}")
        print(f"  Testing data shape: {self.X_test.shape}")
        print(f"  Training labels shape: {self.y_train.shape}")
        print(f"  Testing labels shape: {self.y_test.shape}")
        print(f"  Unique training labels: {np.unique(self.y_train, return_counts=True)}")
        print(f"  Unique testing labels: {np.unique(self.y_test, return_counts=True)}")
        if self.metadata:
            print("  Metadata:")
            for key, value in self.metadata.items():
                print(f"    {key}: {value}")
    
    def get_patient_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Reshape window-level predictions back to patient-level structure.
        
        Args:
            y_pred: Predictions from sklearn model with shape (n_patients*n_timepoints,)
            
        Returns:
            Predictions reshaped to (n_patients, n_timepoints)
        """
        num_test_patients = len(self.dd_test_indices) + len(self.ct_test_indices)
        n_timepoints = self.metadata.get("n_timepoints", len(y_pred) // num_test_patients)
        
        predictions_by_patient = np.zeros((num_test_patients, n_timepoints))
        for i in range(num_test_patients):
            start_idx = i * n_timepoints
            end_idx = (i+1) * n_timepoints
            predictions_by_patient[i, :] = y_pred[start_idx:end_idx]
            
        return predictions_by_patient


def create_labeled_dataset(
    dataset_root: Union[str, Path],
    window: str,
    direction: str,
    electrode: str,
    fold_info: Dict[str, Any],
    fold_index: int = 0
) -> EEGDataset:
    """
    Create training and testing datasets with scikit-learn compatible format.
    
    Args:
        dataset_root: Root directory containing the EEG dataset
        window: Window size identifier (e.g., "window_200")
        direction: Direction identifier (e.g., "up")
        electrode: Electrode identifier (e.g., "T7")
        fold_info: Dictionary containing fold information from stratified_kfold
        fold_index: Which fold to use (default: 0)
        
    Returns:
        EEGDataset object containing scikit-learn ready data
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
    
    # Create patient indices for tracking
    train_patient_count = len(train_labels)
    test_patient_count = len(test_labels)
    
    # Reshape data for sklearn (samples × features)
    # Original shape: (patients, metrics, timepoints)
    # New shape: (patients × timepoints, metrics)
    n_train_patients = train_data.shape[0]
    n_test_patients = test_data.shape[0]
    n_metrics = train_data.shape[1]
    n_timepoints = train_data.shape[2]
    
    # Initialize sklearn-ready data arrays
    X_train = np.zeros((n_train_patients * n_timepoints, n_metrics))
    X_test = np.zeros((n_test_patients * n_timepoints, n_metrics))
    
    # Create arrays to track which patient each sample belongs to
    patient_indices_train = np.zeros(n_train_patients * n_timepoints, dtype=np.int32)
    patient_indices_test = np.zeros(n_test_patients * n_timepoints, dtype=np.int32)
    
    # Reshape training data
    for i in range(n_train_patients):
        start_idx = i * n_timepoints
        end_idx = (i+1) * n_timepoints
        X_train[start_idx:end_idx, :] = train_data[i, :, :].T
        patient_indices_train[start_idx:end_idx] = i
    
    # Reshape testing data
    for i in range(n_test_patients):
        start_idx = i * n_timepoints
        end_idx = (i+1) * n_timepoints
        X_test[start_idx:end_idx, :] = test_data[i, :, :].T
        patient_indices_test[start_idx:end_idx] = i
    
    # Create window-level labels by repeating patient labels
    y_train = np.repeat(train_labels, n_timepoints)
    y_test = np.repeat(test_labels, n_timepoints)
    
    # Create metadata
    metadata = {
        "window": window,
        "direction": direction,
        "electrode": electrode,
        "fold_index": fold_index,
        "dd_shape": dd_data.shape,
        "ct_shape": ct_data.shape,
        "n_metrics": n_metrics,
        "n_timepoints": n_timepoints,
        "n_train_patients": n_train_patients,
        "n_test_patients": n_test_patients
    }
    
    # Return the EEGDataset object with sklearn-ready data
    return EEGDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        patient_indices_train=patient_indices_train,
        patient_indices_test=patient_indices_test,
        dd_train_indices=dd_train_index.tolist(),
        ct_train_indices=ct_train_index.tolist(),
        dd_test_indices=dd_test_index.tolist(),
        ct_test_indices=ct_test_index.tolist(),
        metadata=metadata
    )