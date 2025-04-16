# pyddeeg/classification/models/__init__.py

from sklearn.ensemble import HistGradientBoostingClassifier

# Map string model type to actual sklearn model class and default hyperparameters
MODEL_CONFIGS = {
    "histogram_gbm": {
        "class": HistGradientBoostingClassifier,
        "defaults": {
            "loss": 'log_loss',
            "learning_rate": 0.1,
            "max_depth": 10,
            "max_iter": 100,
            "random_state": 42,
        },
        "default_name": "eeg_histogram_gbm"
    },
    # Add more models here as needed
}