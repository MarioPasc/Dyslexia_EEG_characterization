import os
import json
from pathlib import Path
from typing import Dict, Union, Optional, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_kfold(
    root_path: Union[str, Path],
    window: str = "window_200",
    direction: str = "up",
    electrode: str = "T7",
    n_splits: int = 5,
    random_state: int = 42,
    output_file: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create stratified k-fold cross-validation splits for EEG data and save to JSON. The given
    window, direction, and electrode are only used to access the data, not filtering, the folds 
    are created on patient-level, and are stratified by the labels (dyslexia vs control), being 
    useful for all electrodes, directions, and windows, since they all share the same patients.
    
    Args:
        root_path: Root path to the processed EEG data json "dataset_index.json". 
                   Generated using pyddeeg/utils/postprocessig_reorganize_per_window_results.py
        window: Window size to use (e.g., "window_200")
        direction: Direction to use (e.g., "up")
        electrode: Electrode to use (e.g., "T7")
        n_splits: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        output_file: Path to save the JSON output (if None, won't save)
        
    Returns:
        Dictionary containing fold information and metadata
    """
    # Prepare paths
    root_path = Path(root_path)
    dataset_indexes_path = os.path.join(root_path, "dataset_index.json")
    
    # Load dataset indexes
    with open(dataset_indexes_path, "r") as f:
        dataset_indexes = json.load(f)
    
    # Get file paths for dyslexia and control data
    dd_path, ct_path = dataset_indexes[window][direction][electrode]
    
    # Load data
    dd = np.load(dd_path)["metrics"]
    ct = np.load(ct_path)["metrics"]
    
    # Create patient indices
    dd_indices = np.arange(dd.shape[0])
    ct_indices = np.arange(ct.shape[0])
    
    # Create labels (1 for dyslexia, 0 for control)
    dd_labels = np.ones(len(dd_indices))
    ct_labels = np.zeros(len(ct_indices))
    
    # Combine indices and labels
    all_patient_indices = np.concatenate([dd_indices, ct_indices])
    all_labels = np.concatenate([dd_labels, ct_labels])
    
    # Apply StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store fold information
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_patient_indices, all_labels)):
        # Get the patient indices for this fold
        train_patients = all_patient_indices[train_idx]
        test_patients = all_patient_indices[test_idx]
        
        # Separate by class (DD vs CT)
        dd_train = train_patients[train_patients < len(dd_indices)].tolist()
        ct_train = (train_patients[train_patients >= len(dd_indices)] - len(dd_indices)).tolist()
        
        dd_test = test_patients[test_patients < len(dd_indices)].tolist()
        ct_test = (test_patients[test_patients >= len(dd_indices)] - len(dd_indices)).tolist()
        
        folds.append({
            'fold_number': fold_idx,
            'dd_train': dd_train,
            'ct_train': ct_train,
            'dd_test': dd_test,
            'ct_test': ct_test,
            'train_counts': {'dd': len(dd_train), 'ct': len(ct_train)},
            'test_counts': {'dd': len(dd_test), 'ct': len(ct_test)}
        })
    
    # Create the complete fold information dictionary
    fold_info = {
        "folds": folds,
        "metadata": {
            "window": window,
            "direction": direction,
            "electrode": electrode,
            "n_splits": n_splits,
            "random_state": random_state,
            "dd_count": len(dd_indices),
            "ct_count": len(ct_indices),
            "dd_shape": dd.shape,
            "ct_shape": ct.shape
        }
    }
    
    # Save to JSON if output_file is provided
    if output_file:
        output_path = Path(os.path.join(output_file, f"stratified_kfold_indexes.json"))
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(fold_info, f, indent=2)
    
    return fold_info


def load_stratified_kfolds(
    json_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load stratified k-fold cross-validation splits from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing fold information
        
    Returns:
        Dictionary containing fold information and metadata with numpy arrays for indices
    """
    # Load JSON data
    with open(json_path, "r") as f:
        fold_info = json.load(f)
    
    # Convert lists back to numpy arrays for indices
    for i, fold in enumerate(fold_info["folds"]):
        fold_info["folds"][i]["dd_train"] = np.array(fold["dd_train"], dtype=np.int32)
        fold_info["folds"][i]["ct_train"] = np.array(fold["ct_train"], dtype=np.int32)
        fold_info["folds"][i]["dd_test"] = np.array(fold["dd_test"], dtype=np.int32)
        fold_info["folds"][i]["ct_test"] = np.array(fold["ct_test"], dtype=np.int32)
    
    return fold_info

def create_labeled_dataset(
    dd_data: np.ndarray,
    ct_data: np.ndarray,
    fold_info: dict,
    fold_index: int = 0
) -> tuple:
    """
    Create training and testing datasets with labels incorporated as an additional dimension.
    
    Args:
        dd_data: Dyslexia data with shape (n_dd_patients, n_metrics, n_timepoints)
        ct_data: Control data with shape (n_ct_patients, n_metrics, n_timepoints)
        fold_info: Dictionary containing fold information from stratified_kfold
        fold_index: Which fold to use (default: 0)
        
    Returns:
        Tuple containing:
        - train_data: Original training data
        - test_data: Original testing data
        - train_with_labels: Training data with labels as additional dimension
        - test_with_labels: Testing data with labels as additional dimension
        - train_labels: Training labels
        - test_labels: Testing labels
    """
    # Get fold information
    fold = fold_info.get("folds", [])[fold_index]
    dd_train_index = fold.get("dd_train", [])
    ct_train_index = fold.get("ct_train", [])
    dd_test_index = fold.get("dd_test", [])
    ct_test_index = fold.get("ct_test", [])
    
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
    
    # Reshape labels to be compatible with data for concatenation
    train_labels_expanded = train_labels[:, np.newaxis, np.newaxis]  # Shape becomes (n_samples, 1, 1)
    test_labels_expanded = test_labels[:, np.newaxis, np.newaxis]    # Shape becomes (n_samples, 1, 1)
    
    # Broadcast labels to match the time points dimension (window-level labels)
    train_labels_broadcast = np.broadcast_to(
        train_labels_expanded, 
        (train_data.shape[0], 1, train_data.shape[2])  # (n_samples, 1, n_timepoints)
    )
    test_labels_broadcast = np.broadcast_to(
        test_labels_expanded, 
        (test_data.shape[0], 1, test_data.shape[2])    # (n_samples, 1, n_timepoints)
    )
    
    # Concatenate along metrics dimension (axis=1)
    train_with_labels = np.concatenate((train_data, train_labels_broadcast), axis=1)
    test_with_labels = np.concatenate((test_data, test_labels_broadcast), axis=1)
    
    return train_data, test_data, train_with_labels, test_with_labels, train_labels, test_labels