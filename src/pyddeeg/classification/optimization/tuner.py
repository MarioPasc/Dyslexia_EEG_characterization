import numpy as np

from pyddeeg.classification import OptunaEstimator
from pyddeeg.classification import EEGDataset

from sklearn.linear_model import LogisticRegression

from typing import Any, Dict


def tune_one_electrode(
    elec: str,
    dataset: EEGDataset,
    hyperparam_cfg: Dict[str, Dict[str, Any]],
    base_estimator=LogisticRegression(),
    n_trials: int = 50,
    random_state: int = 42,
):
    """
    Returns
    -------
    best_params : list[dict]
        len(best_params) == n_windows.  Each dict can be fed straight back
        into `base_estimator.set_params(**params)`.
    """
    X = np.concatenate([dataset.dd, dataset.ct])  # (N, 15, T)
    y = np.concatenate([np.ones(len(dataset.dd)), np.zeros(len(dataset.ct))])

    n_windows = X.shape[-1]
    best_params: list[dict] = []

    # ⇣⇣ loop over windows, slice (N, 15) ⇣⇣
    for w in range(n_windows):
        Xw = X[..., w]  # (N, 15)

        opt = OptunaEstimator(
            base_estimator=base_estimator,
            hyperparameters=hyperparam_cfg,
            dataset=dataset,
            n_trials=n_trials,
            random_state=random_state,
        )
        opt.fit(Xw, y)
        best_params.append(opt.best_params_)  # <<<<<<<<
    return best_params
