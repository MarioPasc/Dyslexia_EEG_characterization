import warnings
import logging
from pyddeeg.classification.dataloaders import EEGDataset
from pyddeeg.classification.optimization import OptunaEstimator
from pyddeeg.classification.engine.multi_window_estimator import MultiWindowEstimator

# Set up module-level logger
logger = logging.getLogger("pyddeeg.classification")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Filter specific warnings globally
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)

# Optuna logger settings - make it less verbose
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)

__all__ = ["logger", "EEGDataset", "OptunaEstimator", "MultiWindowEstimator"]
