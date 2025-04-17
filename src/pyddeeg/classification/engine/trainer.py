import logging
from typing import Optional

from pyddeeg.classification.models.window_model import EEGClassifier, ModelConfig
from pyddeeg.classification.dataloaders import EEGDataset

def train_model(
    dataset: EEGDataset,
    model_config: ModelConfig,
    save_model: bool = False,
    output_dir: Optional[str] = None
) -> EEGClassifier:
    """
    Train a model using the specified dataset and configuration.
    
    Args:
        dataset: EEGDataset containing training data
        model_config: Configuration for the model to train
        save_model: Whether to save the trained model
        output_dir: Directory to save model if save_model is True (required if save_model=True)
        
    Returns:
        Trained EEGClassifier model
    """
    logging.info(f"Training {model_config.model_name} model...")
    
    # Create and train the model
    model = EEGClassifier(model_config)
    model.fit(dataset.X_train, dataset.y_train)
    
    # Save model if requested
    if save_model:
        if output_dir is None:
            raise ValueError("output_dir must be specified when save_model is True")
        model_path = model.save(output_dir)
        logging.info(f"Model saved to {model_path}")
    
    return model