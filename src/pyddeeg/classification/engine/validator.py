import os
import logging
from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from pyddeeg.classification.models.EEGWindowModel import EEGClassifier
from pyddeeg.classification.dataloaders import create_labeled_dataset, EEGDataset


def validate_model(
    model: EEGClassifier,
    dataset: EEGDataset,
    threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    plot_results: bool = True,
    electrode_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a trained model on the provided dataset.
    
    Args:
        model: Trained EEGModel to validate
        dataset: EEGDataset containing validation data
        threshold: Classification threshold (if None, uses model's default)
        output_dir: Directory to save validation results (if None, won't save)
        plot_results: Whether to generate and display validation result plots
        electrode_name: Name of electrode for plot titles (if None, uses "Unknown")
        
    Returns:
        Dictionary containing validation metrics
    """
    logging.info("Validating model...")
    
    # Get predictions using standard threshold
    test_pred = model.predict(dataset.X_test)
    test_prob = model.predict_proba(dataset.X_test)[:, 1]
    
    # Get predictions using custom threshold if provided
    if threshold is not None and threshold != model.config.threshold:
        test_pred_threshold = model.predict_with_threshold(dataset.X_test, threshold)
        logging.info(f"Using custom threshold: {threshold}")
    else:
        test_pred_threshold = test_pred
        threshold = model.config.threshold
        logging.info(f"Using model threshold: {threshold}")
    
    # Calculate accuracy
    test_accuracy = accuracy_score(dataset.y_test, test_pred_threshold)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(dataset.y_test, test_pred_threshold, 
                                  target_names=["Control", "Dyslexia"], output_dict=True)
    print(classification_report(dataset.y_test, test_pred_threshold, 
                               target_names=["Control", "Dyslexia"]))
    
    # Get patient-level predictions
    test_predictions_by_patient = dataset.get_patient_predictions(test_pred_threshold)
    
    # Calculate stability by patient
    test_stability_by_patient = calculate_patient_stability(
        dataset, test_predictions_by_patient
    )
    
    # Collect metrics
    metrics = {
        "accuracy": test_accuracy,
        "classification_report": report,
        "predictions": test_pred_threshold,
        "probabilities": test_prob,
        "predictions_by_patient": test_predictions_by_patient,
        "stability_by_patient": test_stability_by_patient,
        "threshold": threshold,
    }
    
    # Plot results if requested
    if plot_results:
        if not electrode_name:
            electrode_name = dataset.metadata.get("electrode", "Unknown")
        
        plot_validation_results(dataset, metrics, electrode_name)
    
    # Save results if output directory is provided
    if output_dir:
        save_validation_results(metrics, output_dir, electrode_name)
    
    return metrics


def calculate_patient_stability(
    dataset: EEGDataset, 
    predictions_by_patient: np.ndarray
) -> np.ndarray:
    """
    Calculate prediction stability for each patient.
    
    Args:
        dataset: EEGDataset containing patient information
        predictions_by_patient: Predictions organized by patient
        
    Returns:
        Array of stability values for each patient
    """
    stability_by_patient = np.zeros(predictions_by_patient.shape[0])
    
    for i in range(predictions_by_patient.shape[0]):
        true_label = 1 if i < len(dataset.dd_test_indices) else 0
        correct_windows = np.sum(predictions_by_patient[i, :] == true_label)
        stability_by_patient[i] = correct_windows / predictions_by_patient.shape[1]
    
    return stability_by_patient


def plot_validation_results(
    dataset: EEGDataset,
    metrics: Dict[str, Any],
    electrode_name: str
) -> None:
    """
    Generate and display validation result plots.
    
    Args:
        dataset: EEGDataset used for validation
        metrics: Dictionary of validation metrics
        electrode_name: Name of electrode for plot titles
    """
    # Create confusion matrix
    cm = confusion_matrix(dataset.y_test, metrics["predictions"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=["Control", "Dyslexia"], 
               yticklabels=["Control", "Dyslexia"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Electrode {electrode_name}')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(dataset.y_test, metrics["probabilities"])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Electrode {electrode_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(dataset.y_test, metrics["probabilities"])
    avg_precision = average_precision_score(dataset.y_test, metrics["probabilities"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='purple', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Electrode {electrode_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
    
    # Plot predictions for first few patients
    predictions_by_patient = metrics["predictions_by_patient"]
    stability_by_patient = metrics["stability_by_patient"]
    
    plt.figure(figsize=(15, 10))
    for i in range(min(5, predictions_by_patient.shape[0])):  # Plot first 5 patients
        # Get the true label for this patient
        true_label = 1 if i < len(dataset.dd_test_indices) else 0
        patient_class = "Dyslexia" if true_label == 1 else "Control"
        patient_stability = stability_by_patient[i] * 100
        
        plt.subplot(5, 1, i+1)
        plt.plot(predictions_by_patient[i, :], 'o-', markersize=2, 
                 label=f'Stability: {patient_stability:.1f}%')
        plt.axhline(y=true_label, color='r', linestyle='--', label='True Label')
        plt.title(f"Patient {i} ({patient_class}) - Stability: {patient_stability:.1f}%")
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ['Control', 'Dyslexia'])
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot stability distribution
    dd_stability = stability_by_patient[:len(dataset.dd_test_indices)]
    ct_stability = stability_by_patient[len(dataset.dd_test_indices):]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([dd_stability*100, ct_stability*100], labels=['Dyslexia', 'Control'])
    plt.ylabel('Prediction Stability (%)')
    plt.title(f'Prediction Stability by Patient Group - Electrode {electrode_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Feature importance if available
    feature_importances = metrics.get("feature_importances")
    if feature_importances is not None:
        sorted_idx = np.argsort(feature_importances)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'RQA Features Importance - Electrode {electrode_name}')
        plt.tight_layout()
        plt.show()


def save_validation_results(
    metrics: Dict[str, Any],
    output_dir: str,
    electrode_name: str
) -> None:
    """
    Save validation results to files.
    
    Args:
        metrics: Dictionary of validation metrics
        output_dir: Directory to save results
        electrode_name: Name of electrode for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as numpy arrays
    results_path = os.path.join(output_dir, f"{electrode_name}_validation_results.npz")
    np.savez(
        results_path,
        predictions=metrics["predictions"],
        probabilities=metrics["probabilities"],
        predictions_by_patient=metrics["predictions_by_patient"],
        stability_by_patient=metrics["stability_by_patient"],
    )
    
    # Save classification report as text
    report_path = os.path.join(output_dir, f"{electrode_name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        for class_name, class_metrics in metrics["classification_report"].items():
            if isinstance(class_metrics, dict):
                f.write(f"\n{class_name}:\n")
                for metric_name, value in class_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"{class_name}: {class_metrics:.4f}\n")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate EEG classification model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model file')
    parser.add_argument('--dataset_root', type=str, required=True, 
                        help='Root directory containing the EEG dataset')
    parser.add_argument('--window', type=str, default="window_200", 
                        help='Window size identifier')
    parser.add_argument('--direction', type=str, default="up", 
                        help='Direction identifier')
    parser.add_argument('--electrode', type=str, default="T7", 
                        help='Electrode identifier')
    parser.add_argument('--fold_index', type=int, default=0, 
                        help='Which fold to use for cross-validation')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Custom threshold for classification (if None, uses model default)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save validation results')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load model
    model = EEGClassifier.load(args.model_path)
    logging.info(f"Model loaded from {args.model_path}")
    
    # Create folds or load existing
    from pyddeeg.classification.utils.strat_kfold import stratified_kfold
    folds = stratified_kfold(root_path=args.dataset_root, n_splits=5, 
                            random_state=42, output_file=None)
    
    # Create dataset
    dataset = create_labeled_dataset(
        dataset_root=args.dataset_root,
        window=args.window,
        direction=args.direction,
        electrode=args.electrode,
        fold_info=folds,
        fold_index=args.fold_index
    )
    
    # Validate model
    metrics = validate_model(
        model=model,
        dataset=dataset,
        threshold=args.threshold,
        output_dir=args.output_dir,
        plot_results=True,
        electrode_name=args.electrode
    )