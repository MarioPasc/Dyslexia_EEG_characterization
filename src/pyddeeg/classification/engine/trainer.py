import numpy as np
import mne
from mne.decoding import Scaler, Vectorizer, SlidingEstimator, cross_val_multiscore, get_coef
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from typing import Any, Dict
from pyddeeg import RQA_METRICS
from pyddeeg.classification.dataloaders import EEGDataset

def compute_auc_per_electrode(
    elec: str,
    dataset: EEGDataset,  
    model: BaseEstimator
) -> Dict[str, Any]:
    """
    Compute AUC and pattern statistics for a given electrode using a specified model.

    Parameters
    ----------
    elec : str
        Electrode name (e.g., 'Fz').
    dataset : EEGDataset
        EEGDataset object containing dd, ct, cv, and metadata.
    model : BaseEstimator
        The estimator to use at the end of the pipeline (e.g., LinearModel(LogisticRegression(...))).

    Returns
    -------
    results : dict
        Dictionary with mean/std ROC and PR scores, time axis, window size, and evoked patterns.
    """
    dd = dataset.dd
    ct = dataset.ct
    cv = dataset.cv

    X = np.concatenate([dd, ct])  # (patients, 15, windows)
    y = np.concatenate([np.ones(len(dd)), np.zeros(len(ct))]).astype(int)
    groups = np.arange(len(y))

    # We are dealing with window indexes, therefore we are going to use a samplig
    # frequency of 1 Hz, this would indicate that we have 1 sample per second, when, in
    # fact, we have 1 sample (15 RQA metrics) per window, but we parallelize window index
    # with second as base unit.
    # A later function will take care of matching each window index with its corresponding
    # time in milliseconds. This is only a workaround to make the code work with
    # mne.EpochsArray.
    info = mne.create_info([f"{elec}_{m}" for m in RQA_METRICS], 1, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, tmin=0)
    epochs.set_montage("standard_1020", on_missing="ignore")

    base = make_pipeline(
        Scaler(epochs.info, scalings="median"),
        Vectorizer(),
        model  
    )
    est_roc = SlidingEstimator(base, scoring="roc_auc", n_jobs=-1)
    est_pr  = SlidingEstimator(base, scoring="average_precision", n_jobs=-1)

    roc = cross_val_multiscore(est_roc, epochs.get_data(), y, cv=cv, groups=groups, n_jobs=-1)
    pr  = cross_val_multiscore(est_pr , epochs.get_data(), y, cv=cv, groups=groups, n_jobs=-1)

    est_roc.fit(epochs.get_data(), y)
    patterns = np.squeeze(get_coef(est_roc, "patterns_", inverse_transform=True))


    return dict(
        mean_roc=roc.mean(0), std_roc=roc.std(0),
        mean_pr=pr.mean(0), std_pr=pr.std(0),
        evoked_patterns=patterns,
    )