import os
import logging
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from pyddeeg.classification.models.EEGWindowModel import EEGClassifier, ModelConfig, create_model_config
from pyddeeg.classification.dataloaders import create_labeled_dataset, EEGDataset
from pyddeeg.classification.utils.strat_kfold import stratified_kfold

def train_model(
    dataset: EEGDataset,
    model_config: ModelConfig,
    output_dir: Optional[str] = None,
    plot_training_metrics: bool = True
) -> Tuple[EEGClassifier, Dict[str, Any]]:
    """
    Train a model using the specified dataset and configuration.
    
    Args:
        dataset: EEGDataset containing training and testing data
        model_config: Configuration for the model to train
        output_dir: Directory to save model and metrics (if None, won't save)
        plot_training_metrics: Whether to generate and display training metrics plots
        
    Returns:
        Tuple of (trained EEGModel, dictionary of metrics)
    """
    logging.info(f"Training {model_config.model_name} model...")
    
    # Create and train the model
    model = EEGClassifier(model_config)
    model.fit(dataset.X_train, dataset.y_train)
    
    # Evaluate on training set
    train_pred = model.predict(dataset.X_train)
    train_prob = model.predict_proba(dataset.X_train)[:, 1]
    train_accuracy = accuracy_score(dataset.y_train, train_pred)
    
    # Evaluate on test set
    test_pred = model.predict(dataset.X_test)
    test_prob = model.predict_proba(dataset.X_test)[:, 1]
    test_accuracy = accuracy_score(dataset.y_test, test_pred)
    
    # Calculate overfitting gap
    accuracy_gap = train_accuracy - test_accuracy
    
    # Collect metrics
    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": accuracy_gap,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
        "train_probabilities": train_prob,
        "test_probabilities": test_prob,
    }
    
    # Get patient-level predictions
    test_predictions_by_patient = dataset.get_patient_predictions(test_pred)
    train_predictions_by_patient = dataset.get_patient_predictions(train_pred)
    
    metrics["train_predictions_by_patient"] = train_predictions_by_patient
    metrics["test_predictions_by_patient"] = test_predictions_by_patient
    
    logging.info(f"Training accuracy: {train_accuracy:.4f}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    logging.info(f"Accuracy gap (overfitting measure): {accuracy_gap:.4f}")
    
    # Print classification reports
    print("\nTraining Classification Report:")
    print(classification_report(dataset.y_train, train_pred, target_names=["Control", "Dyslexia"]))
    
    print("\nTest Classification Report:")
    print(classification_report(dataset.y_test, test_pred, target_names=["Control", "Dyslexia"]))
    
    # Plot metrics if requested
    if plot_training_metrics:
        plot_training_results(dataset, metrics)
    
    # Save model if output directory is provided
    if output_dir:
        model_path = model.save(output_dir)
        logging.info(f"Model saved to {model_path}")
    
    return model, metrics


def plot_training_results(
    dataset: EEGDataset,
    metrics: Dict[str, Any]
) -> None:
    """
    Generate and display plots for model training results.
    
    Args:
        dataset: EEGDataset used for training/testing
        metrics: Dictionary containing metrics from training
    """
    # Plot ROC curves for both sets
    fpr_train, tpr_train, _ = roc_curve(dataset.y_train, metrics["train_probabilities"])
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(dataset.y_test, metrics["test_probabilities"])
    roc_auc_test = auc(fpr_test, tpr_test)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training ROC (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Overfitting Check')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # Calculate prediction stability by patient
    train_predictions_by_patient = metrics["train_predictions_by_patient"]
    test_predictions_by_patient = metrics["test_predictions_by_patient"]
    
    train_stability_by_patient = np.zeros(train_predictions_by_patient.shape[0])
    test_stability_by_patient = np.zeros(test_predictions_by_patient.shape[0])
    
    # Calculate training stability
    for i in range(train_predictions_by_patient.shape[0]):
        true_label = 1 if i < len(dataset.dd_train_indices) else 0
        correct_windows = np.sum(train_predictions_by_patient[i, :] == true_label)
        train_stability_by_patient[i] = correct_windows / train_predictions_by_patient.shape[1]
    
    # Calculate testing stability
    for i in range(test_predictions_by_patient.shape[0]):
        true_label = 1 if i < len(dataset.dd_test_indices) else 0
        correct_windows = np.sum(test_predictions_by_patient[i, :] == true_label)
        test_stability_by_patient[i] = correct_windows / test_predictions_by_patient.shape[1]
    
    # Plot stability comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot average stability
    ax1.bar(['Training', 'Testing'], 
            [np.mean(train_stability_by_patient)*100, np.mean(test_stability_by_patient)*100], 
            color=['blue', 'red'])
    ax1.set_ylabel('Average Patient Stability (%)')
    ax1.set_title('Average Prediction Stability - Training vs Testing')
    ax1.set_ylim(0, 100)
    
    # Plot individual patient stabilities
    train_dd_stability = train_stability_by_patient[:len(dataset.dd_train_indices)]
    train_ct_stability = train_stability_by_patient[len(dataset.dd_train_indices):]
    test_dd_stability = test_stability_by_patient[:len(dataset.dd_test_indices)]
    test_ct_stability = test_stability_by_patient[len(dataset.dd_test_indices):]
    
    # Box plot comparing stability distributions
    box_data = [train_dd_stability*100, train_ct_stability*100, 
                test_dd_stability*100, test_ct_stability*100]
    ax2.boxplot(box_data)
    ax2.set_xticklabels(['Train DD', 'Train CT', 'Test DD', 'Test CT'])
    ax2.set_ylabel('Stability (%)')
    ax2.set_title('Patient Prediction Stability Distribution')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EEG classification model')
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
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save model and results')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create folds or load existing
    folds = stratified_kfold(root_path=args.dataset_root, n_splits=5, 
                            random_state=42, output_file=args.output_dir)
    
    # Create dataset
    dataset = create_labeled_dataset(
        dataset_root=args.dataset_root,
        window=args.window,
        direction=args.direction,
        electrode=args.electrode,
        fold_info=folds,
        fold_index=args.fold_index
    )
    
    # Create model configuration
    model_config = create_model_config(
        model_type="histogram_gbm",
        learning_rate=0.1,
        max_depth=10,
        max_iter=100,
        random_state=42,
        model_name=f"eeg_gbm_{args.electrode}_{args.window}_{args.direction}_fold{args.fold_index}"
    )
    
    # Train model
    model, metrics = train_model(
        dataset=dataset,
        model_config=model_config,
        output_dir=args.output_dir,
        plot_training_metrics=True
    )