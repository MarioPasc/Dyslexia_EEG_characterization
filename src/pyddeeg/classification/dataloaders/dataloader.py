from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

@dataclass
class EEGDataset:
    """
    Dataclass for storing EEG dataset information for classification tasks.

    Attributes
    ----------
    dd : np.ndarray
        Array of RQA metrics for the dyslexia (DD) group. Shape: (n_subjects, n_metrics, n_windows).
    ct : np.ndarray
        Array of RQA metrics for the control (CT) group. Shape: (n_subjects, n_metrics, n_windows).
    cv : object
        Cross-validation splitter object (e.g., StratifiedGroupKFold or GroupKFold).
    metadata : dict
        Dictionary containing dataset metadata such as paths, window parameters, and routes.
    """
    dd: np.ndarray
    ct: np.ndarray
    cv: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(
        dataset_root: Path,
        window: str,
        direction: str,
        random_state: int = 42
    ) -> "EEGDataset":
        """
        Load EEG RQA metric data and metadata for a given window and direction.

        Parameters
        ----------
        dataset_root : Path
            Root directory of the processed EEG dataset.
        window : str
            Window identifier (e.g., "window_200" for 200 ms window).
        direction : str
            Direction of the RQA computation (e.g., "up", "down").
        random_state : int, optional
            Random seed for reproducibility in cross-validation splitting (default: 42).

        Returns
        -------
        EEGDataset
            An EEGDataset instance containing loaded data, cross-validator, and metadata.
        """
        meta = np.load(dataset_root / window / "metadata.npz")
        centres_ms = meta["centers"].astype(int)
        stride_ms = int(meta["stride"])
        window_ms = int(window.split("_")[-1])
        total_ms = centres_ms[-1] + window_ms // 2

        import json
        routes = json.loads((dataset_root / "dataset_index.json").read_text())[window][direction]

        try:
            cv = StratifiedGroupKFold(n_splits=5, random_state=random_state, shuffle=True)
            cv_type = "StratifiedGroupKFold"
        except ImportError:
            cv = GroupKFold(n_splits=5)
            cv_type = "GroupKFold"

        # Load the first electrode's data as an example
        first_elec = next(iter(routes))
        pA, pB = routes[first_elec]
        dd_path, ct_path = (pA, pB) if "DD" in Path(pA).stem.upper() else (pB, pA)
        dd = np.load(dd_path)["metrics"].astype(np.float32)
        ct = np.load(ct_path)["metrics"].astype(np.float32)

        metadata = dict(
            dataset_root=dataset_root,
            window=window,
            direction=direction,
            centres_ms=centres_ms,
            stride_ms=stride_ms,
            window_ms=window_ms,
            total_ms=total_ms,
            routes=routes,
            cv_type=cv_type
        )
        return EEGDataset(dd=dd, ct=ct, cv=cv, metadata=metadata)