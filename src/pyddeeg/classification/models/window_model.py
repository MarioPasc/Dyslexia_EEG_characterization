import os
from typing import Dict, Any, Union, Optional, Type, Callable
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.base import BaseEstimator

from pyddeeg.classification.models import MODEL_CONFIGS

@dataclass
class ModelConfig:
    """Configuration for a machine learning model.
    
    Attributes:
        model_type: The scikit-learn model class
        hyperparameters: Dictionary of hyperparameters for the model
        threshold: Optional threshold for binary classification probabilities
        model_name: Name identifier for the model
    """
    model_type: Type[BaseEstimator]
    hyperparameters: Dict[str, Any]
    threshold: Optional[float] = 0.5
    model_name: str = "eeg_classifier"
    

class EEGClassifier:
    """
    Wrapper class for scikit-learn models used in EEG classification.
    
    This class provides a common interface for creating, training, saving
    and loading models for EEG time series classification.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model with the given configuration.
        
        Args:
            config: Configuration containing model type and hyperparameters
        """
        self.config = config
        self.model = config.model_type(**config.hyperparameters)
        self._trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EEGClassifier':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self._trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Class predictions
        """
        if not self._trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Features to predict probabilities for
            
        Returns:
            Array of probability estimates
        """
        if not self._trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def predict_with_threshold(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Make predictions using a custom probability threshold for class assignment.
        
        Args:
            X: Features to predict
            threshold: Probability threshold (defaults to config threshold)
            
        Returns:
            Binary predictions based on the threshold
        """
        if threshold is None:
            threshold = self.config.threshold
            
        proba = self.predict_proba(X)[:, 1]  # Probability of class 1
        return (proba >= threshold).astype(int)
    
    def save(self, directory: str) -> str:
        """
        Save the trained model to the specified directory.
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model file
        """
        if not self._trained:
            raise ValueError("Cannot save an untrained model")
            
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{self.config.model_name}.joblib")
        joblib.dump(self, model_path)
        return model_path
    
    @classmethod
    def load(cls, model_path: str) -> 'EEGClassifier':
        """
        Load a trained model from the specified path.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded EEGModel instance
        """
        return joblib.load(model_path)
    
def create_model_config(
    model_type: str = "histogram_gbm",
    hyperparameters: Dict[str, Any] = None,
) -> ModelConfig:
    """
    Create a model configuration based on the specified model type and hyperparameters.
    
    Args:
        model_type: Type of model to create (e.g., 'histogram_gbm')
        hyperparameters: Dictionary containing model hyperparameters and optional settings
                        (threshold, model_name, etc.)
        
    Returns:
        ModelConfig object with appropriate settings
    """
    if hyperparameters is None:
        hyperparameters = {}
    
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Get model class and defaults
    model_config = MODEL_CONFIGS[model_type]
    model_class = model_config["class"]
    
    # Extract ModelConfig-specific parameters
    threshold = hyperparameters.pop("threshold", 0.5)
    model_name = hyperparameters.pop("model_name", model_config["default_name"])
    
    # Create model hyperparameters by starting with defaults and updating with provided values
    final_hyperparameters = model_config["defaults"].copy()
    final_hyperparameters.update(hyperparameters)
    
    return ModelConfig(
        model_type=model_class,
        hyperparameters=final_hyperparameters,
        threshold=threshold,
        model_name=model_name
    )