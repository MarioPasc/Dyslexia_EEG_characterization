import os
import logging
from typing import Dict, Any, Union, Optional, Tuple, Literal
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from pyddeeg.classification.models.window_model import EEGClassifier
from pyddeeg.classification.dataloaders import EEGDataset


def validate_model(
    model: EEGClassifier,
    dataset: EEGDataset,
    threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    save_metrics: bool = False,
    split: Literal["train", "val"] = "val",
    save_predprobs: bool = False
) -> Dict[str, Any]:
    """
    Validate a trained model on the provided dataset, calculating AUC values per window.
    
    Args:
        model: Trained EEGModel to validate
        dataset: EEGDataset containing validation data
        threshold: Classification threshold (if None, uses model's default)
        output_dir: Directory to save validation results
        save_metrics: Whether to save metrics as tensor
        split: Whether to use training or validation split ("train" or "val")
        
    Returns:
        Dictionary containing validation metrics
    """
    logging.info(f"Validating model on {split} split...")
    
    # Select data based on split parameter
    if split == "train":
        X_data = dataset.X_train
        y_data = dataset.y_train
        patient_indices = dataset.dd_train_indices + dataset.ct_train_indices
        dd_indices = dataset.dd_train_indices
        ct_indices = dataset.ct_train_indices
    else:  # val is default
        X_data = dataset.X_test
        y_data = dataset.y_test
        patient_indices = dataset.dd_test_indices + dataset.ct_test_indices
        dd_indices = dataset.dd_test_indices
        ct_indices = dataset.ct_test_indices
    
    # Get prediction probabilities
    data_prob = model.predict_proba(X_data)[:, 1]
    
    # Get predictions using custom threshold if provided
    if threshold is not None and threshold != model.config.threshold:
        threshold_to_use = threshold
    else:
        threshold_to_use = model.config.threshold
    
    data_pred = model.predict_with_threshold(X_data, threshold_to_use)
    
    # Get patient-level probabilities
    if split == "train":
        probs_by_patient = dataset.get_patient_predictions(data_prob, is_training=True)
    else:
        probs_by_patient = dataset.get_patient_predictions(data_prob)
    
    # Calculate AUC per window index
    window_auc_roc, window_auc_pr = calculate_window_auc(dataset, probs_by_patient, 
                                                         dd_indices=dd_indices, ct_indices=ct_indices)
    
    # Create metrics tensor if requested
    metrics_tensor = None
    if save_metrics:
        num_patients = len(dd_indices) + len(ct_indices)
        num_windows = probs_by_patient.shape[1]
        
        # Shape: (val_patients, 1, 2, num_windows)
        metrics_tensor = np.zeros((num_patients, 1, 2, num_windows))
        
        # Create patient labels array
        patient_labels = np.zeros(num_patients)
        patient_labels[:len(dd_indices)] = 1
        
        # Fill metrics (same AUC value for every patient at each window)
        for p_idx in range(num_patients):
            metrics_tensor[p_idx, 0, 0, :] = window_auc_roc  # AUC-ROC for each window
            metrics_tensor[p_idx, 0, 1, :] = window_auc_pr   # AUC-PR for each window
        
        # Save metrics tensor if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            window = dataset.metadata.get("window", "")
            direction = dataset.metadata.get("direction", "")
            electrode = dataset.metadata.get("electrode", "")
            fold_index = dataset.metadata.get("fold_index", "")
            metrics_filename = f"{window}_{direction}_{electrode}_fold{fold_index}_{split}_metrics.npz"
            metrics_path = os.path.join(output_dir, metrics_filename)
            
            # Save metrics
            np.savez(
                metrics_path, 
                metrics=metrics_tensor,
                patient_labels=patient_labels,
                patient_indices=patient_indices
            )
            
            if save_predprobs:
                # Also save the patient-level predictions for further analysis
                patient_predictions_filename = f"{window}_{direction}_{electrode}_fold{fold_index}_{split}_predprobs.npz"
                patient_predictions_path = os.path.join(output_dir, patient_predictions_filename)
                np.savez(
                    patient_predictions_path, 
                    predictions=probs_by_patient,
                    patient_indices={
                        "dd_indices": dd_indices,
                        "ct_indices": ct_indices
                    }
                )
            logging.info(f"Metrics saved to {metrics_path}")
    
    # Collect metrics
    metrics = {
        "window_auc_roc": window_auc_roc,
        "window_auc_pr": window_auc_pr,
        "patient_predictions": probs_by_patient,
        "metrics_tensor": metrics_tensor,
        "threshold": threshold_to_use,
        "split": split
    }
    
    return metrics


def calculate_window_auc(
    dataset: EEGDataset, 
    probabilities_by_patient: np.ndarray,
    dd_indices: list,
    ct_indices: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate AUC values for each window across all patients.
    
    Args:
        dataset: EEGDataset containing patient information
        probabilities_by_patient: Prediction probabilities organized by patient and window
        dd_indices: Indices of dyslexia patients
        ct_indices: Indices of control patients
        
    Returns:
        Tuple of (AUC-ROC values, AUC-PR values) arrays with shape (windows,)
    """
    num_windows = probabilities_by_patient.shape[1]
    
    window_auc_roc = np.zeros(num_windows)
    window_auc_pr = np.zeros(num_windows)
    
    # Create true labels for each patient (1 for dyslexia, 0 for control)
    true_labels = np.zeros(probabilities_by_patient.shape[0])
    true_labels[:len(dd_indices)] = 1
    
    # For each window index
    for w in range(num_windows):
        # Get probabilities for this window across all patients
        window_probs = probabilities_by_patient[:, w]
        
        # Calculate ROC AUC for this window
        try:
            fpr, tpr, _ = roc_curve(true_labels, window_probs)
            window_auc_roc[w] = auc(fpr, tpr)
        except Exception as e:
            logging.warning(f"Failed to calculate ROC AUC for window {w}: {e}")
            window_auc_roc[w] = 0.5  # Default value if calculation fails
        
        # Calculate PR AUC for this window
        try:
            window_auc_pr[w] = average_precision_score(true_labels, window_probs)
        except Exception as e:
            logging.warning(f"Failed to calculate PR AUC for window {w}: {e}")
            window_auc_pr[w] = 0.5  # Default value if calculation fails
    
    return window_auc_roc, window_auc_pr