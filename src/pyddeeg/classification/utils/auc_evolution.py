import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional

from pyddeeg.classification.dataloaders import EEGDataset

def plot_auc_evolution(
    dataset: EEGDataset,
    metrics_dir: str,
    auc_type: Literal["ROC", "PR"] = "ROC",
    figsize: tuple = (15, 6),
    save_fig: bool = False,
    output_dir: Optional[str] = None
):
    """
    Plot the evolution of AUC values per window for CT and DD patients.
    
    Args:
        dataset: EEGDataset object with metadata
        metrics_dir: Directory containing the metrics NPZ files
        auc_type: Type of AUC to plot, either "ROC" or "PR"
        figsize: Figure size as a tuple (width, height)
        save_fig: Whether to save the figure
        output_dir: Directory to save the figure if save_fig is True
    
    Returns:
        Matplotlib figure
    """
    # Extract dataset metadata
    window = dataset.metadata.get("window", "")
    direction = dataset.metadata.get("direction", "")
    electrode = dataset.metadata.get("electrode", "")
    fold_index = dataset.metadata.get("fold_index", "")
    
    # Define file paths for metrics data
    train_metrics_file = f"{window}_{direction}_{electrode}_fold{fold_index}_train_metrics.npz"
    val_metrics_file = f"{window}_{direction}_{electrode}_fold{fold_index}_val_metrics.npz"
    
    train_metrics_path = os.path.join(metrics_dir, train_metrics_file)
    val_metrics_path = os.path.join(metrics_dir, val_metrics_file)
    
    # Load metrics data
    train_data = np.load(train_metrics_path)
    val_data = np.load(val_metrics_path)
    
    # Get metrics tensors
    train_metrics_tensor = train_data["metrics"]
    val_metrics_tensor = val_data["metrics"]
    
    # Get patient labels
    train_patient_labels = train_data["patient_labels"]
    val_patient_labels = val_data["patient_labels"]
    
    # Get AUC index (0 for ROC, 1 for PR)
    auc_idx = 0 if auc_type == "ROC" else 1
    
    # Extract AUC values for each patient group
    train_dd_indices = np.where(train_patient_labels == 1)[0]
    train_ct_indices = np.where(train_patient_labels == 0)[0]
    val_dd_indices = np.where(val_patient_labels == 1)[0]
    val_ct_indices = np.where(val_patient_labels == 0)[0]
    
    # Extract AUC values per window
    num_windows = train_metrics_tensor.shape[3]
    
    # For CT patients (mean and std across patients)
    train_ct_aucs = train_metrics_tensor[train_ct_indices, 0, auc_idx, :]
    train_ct_mean = np.mean(train_ct_aucs, axis=0)
    train_ct_std = np.std(train_ct_aucs, axis=0)
    
    val_ct_aucs = val_metrics_tensor[val_ct_indices, 0, auc_idx, :]
    val_ct_mean = np.mean(val_ct_aucs, axis=0)
    val_ct_std = np.std(val_ct_aucs, axis=0)
    
    # For DD patients (mean and std across patients)
    train_dd_aucs = train_metrics_tensor[train_dd_indices, 0, auc_idx, :]
    train_dd_mean = np.mean(train_dd_aucs, axis=0)
    train_dd_std = np.std(train_dd_aucs, axis=0)
    
    val_dd_aucs = val_metrics_tensor[val_dd_indices, 0, auc_idx, :]
    val_dd_mean = np.mean(val_dd_aucs, axis=0)
    val_dd_std = np.std(val_dd_aucs, axis=0)
    
    # Create window indices for x-axis
    window_indices = np.arange(num_windows)
    
    # Create figure with 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Plot CT data
    axs[0].errorbar(window_indices, train_ct_mean, yerr=train_ct_std, 
                   label=f'Train AUC-{auc_type}', marker='o', linestyle='-', color='blue', capsize=3)
    axs[0].errorbar(window_indices, val_ct_mean, yerr=val_ct_std, 
                   label=f'Val AUC-{auc_type}', marker='s', linestyle='--', color='darkblue', capsize=3)
    axs[0].set_title('Control Patients (CT)')
    axs[0].set_xlabel('Window Index')
    axs[0].set_ylabel(f'AUC-{auc_type}')
    axs[0].set_ylim(0.0, 1.0)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot DD data
    axs[1].errorbar(window_indices, train_dd_mean, yerr=train_dd_std, 
                   label=f'Train AUC-{auc_type}', marker='o', linestyle='-', color='red', capsize=3)
    axs[1].errorbar(window_indices, val_dd_mean, yerr=val_dd_std, 
                   label=f'Val AUC-{auc_type}', marker='s', linestyle='--', color='darkred', capsize=3)
    axs[1].set_title('Developmental Dyslexia Patients (DD)')
    axs[1].set_xlabel('Window Index')
    axs[1].set_ylim(0.0, 1.0)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle(f'AUC-{auc_type} Evolution by Window ({window} {direction} {electrode})', y=1.05)
    
    # Save figure if requested
    if save_fig and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"auc_{auc_type.lower()}_evolution_{window}_{direction}_{electrode}_fold{fold_index}.png")
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {fig_path}")
    
    return fig