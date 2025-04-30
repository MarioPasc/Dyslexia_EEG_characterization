#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline for **one EEG electrode**.

Reads three YAML files:
• settings.yaml   - global paths, window length, etc.
• model.yaml      - base estimator & hyper-parameter search space
• optuna.yaml     - sampler / pruner / persistence options

The script is meant for slurm array jobs, where the environment variable
SLURM_ARRAY_TASK_ID maps to CHANNELS[idx] in exactly the same order you
listed in your submission header.
"""
from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
from sklearn.base import BaseEstimator
from importlib import import_module

from pyddeeg.classification import EEGDataset
from pyddeeg.classification import logger
from pyddeeg.utils.postprocessing.reorganize_per_window_results import (
    reorganization_pipeline,
)
from pyddeeg.classification.optimization.tuner import tune_one_electrode_parallel
from pyddeeg.classification.engine.trainer import (
    evaluate_frozen_models,
    permutation_test_decision_scores,
)
from pyddeeg.classification.utils.report import (
    print_welcome_banner,
    print_dataset_summary,
    print_tuning_summary,
    print_cv_results_summary,
    print_permutation_test_summary,
)

# -----------------------------------------------------------------------------#
#                              helpers                                          #
# -----------------------------------------------------------------------------#
CHANNELS: List[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "Cz",
]


def _resolve_base_estimator(spec: str) -> BaseEstimator:
    """
    Turn a dotted string like ``sklearn.ensemble.HistGradientBoostingClassifier``
    into an *instance* with no arguments.
    """
    mod_path, _, cls_name = spec.rpartition(".")
    cls = getattr(import_module(mod_path), cls_name)
    return cls()


def _cli() -> list:
    """Parse `--electrode` or infer from SLURM_ARRAY_TASK_ID."""
    parser = ArgumentParser()
    parser.add_argument("--electrode", help="Electrode name (e.g. Fz)")
    parser.add_argument("--settings", help="Setting yaml file", default="settings.yaml")
    parser.add_argument("--threads", help="Number of threads", default=1, type=int)

    args = parser.parse_args()

    if not args.electrode:
        # --- slurm array fall-back ---------------------------------------------
        try:
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError as exc:
            raise RuntimeError(
                "No --electrode given and SLURM_ARRAY_TASK_ID not set."
            ) from exc
        return [CHANNELS[idx], args.settings, args.threads]
    else:
        return [args.electrode, args.settings, args.threads]


# -----------------------------------------------------------------------------#
#                                main                                          #
# -----------------------------------------------------------------------------#
def main() -> None:  # pragma: no cover

    argparser_settings = _cli()
    elec = argparser_settings[0]
    settings_file = argparser_settings[1]
    threads = argparser_settings[2]
    print_welcome_banner(package_version="0.1")

    logger.info(f"Electrode: {elec}")
    logger.info(f"Settings file: {settings_file}")
    logger.info(f"Threads: {threads}")

    # ----------------------- load settings.yaml -----------------------------
    cfg: Dict[str, Any] = yaml.safe_load(Path(settings_file).read_text())
    ds_root = Path(cfg["dataset_root"]).expanduser()

    window = cfg["window"]
    direction = cfg["direction"]

    # Check if ds_root / dataset_index.json exists
    dataset_index_path = ds_root / "dataset_index.json"
    if not dataset_index_path.exists():
        logger.info(
            f"Dataset index not found at {dataset_index_path}. Running the data reorganization script first."
        )

        raw_dir = cfg["raw_data_results"]
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(
                f"Raw data folder not found at {ds_root / 'raw'} after reorganization."
            )
        # Run the reorganization pipeline
        reorganization_pipeline(raw_dir=raw_dir, output_dir=ds_root)

        # Basic checking of the dataset index and folders
        if not dataset_index_path.exists():
            raise FileNotFoundError(
                f"Dataset index not found at {dataset_index_path} after reorganization."
            )
        target_folder_experiment = ds_root / window / direction
        if not target_folder_experiment.exists():
            raise FileNotFoundError(
                f"Target folder {target_folder_experiment} not found after reorganization."
            )
        logger.info(
            f"Dataset index created at {dataset_index_path}. Continuing script execution."
        )

    seed = int(cfg.get("seed", 42))
    out_dir_root = Path(cfg["output_dir"]).expanduser()

    # ----------------------- load model.yaml --------------------------------
    mdl_cfg: Dict[str, Any] = yaml.safe_load(Path(cfg["model_cfg"]).read_text())
    base_estimator = _resolve_base_estimator(mdl_cfg["base_estimator"])
    hyperparams = mdl_cfg["hyperparameters"]

    # ----------------------- load optuna.yaml -------------------------------
    optuna_cfg = cfg["optuna_cfg"]  # just pass the path; OptunaEstimator parses

    # ----------------------- load selector.yaml -------------------------------
    selector_cfg = cfg["selector_cfg"]  # just pass the path; OptunaEstimator parses

    # ----------------------- electrode-specific out dir ---------------------
    run_dir = out_dir_root / f"{elec}_{window}_{direction}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------- step 1 – load dataset --------------------------
    dataset = EEGDataset.load(ds_root, window, direction, elec, random_state=seed)
    print_dataset_summary(dataset)

    # ----------------------- step 2 – hyper-parameter tuning ---------------
    # Contains two entries per window, the first one being the "best_params",
    # and the second one, being the "best_trial_performance".
    results_dir = tune_one_electrode_parallel(
        elec=elec,
        dataset=dataset,
        hyperparam_cfg=hyperparams,
        base_estimator=base_estimator,
        optuna_configuration=optuna_cfg,
        selector_configuration=selector_cfg,
        random_state=seed,
        n_jobs=int(threads),
        storage_dir=run_dir / "optuna",
    )
    # save best params to disk
    (Path(run_dir) / "best_params.json").write_text(
        json.dumps(
            [p["best_params"] for p in results_dir],
            indent=2,
        )
    )
    # save best performance of the best trial per window to disk
    (Path(run_dir) / "perf.json").write_text(
        json.dumps(
            [p["performance"] for p in results_dir],
            indent=2,
        )
    )
    logger.info("🔧 Tuning complete & parameters saved.")
    print_tuning_summary(results_dir)
    # ----------------------- step 3 – outer-CV evaluation -------------------

    results = evaluate_frozen_models(
        dataset=dataset,
        params_per_window=[
            p["best_params"] for p in results_dir
        ],  # ← list from the tuner
        base_estimator_cls=type(base_estimator),
        selector_cfg_path=selector_cfg,
        n_jobs_windows=cfg.get("cv_jobs", -1),
    )
    results["cv_indices"] = np.array(results["cv_indices"], dtype=object)
    np.savez_compressed(run_dir / "cv_results.npz", **results)
    print_cv_results_summary(results)

    """ To load these results:
    with np.load(run_dir / "cv_results.npz", allow_pickle=True) as data:
    cv_indices = data["cv_indices"].tolist()  # back to Python list of (train_idx, test_idx)
    """

    # ----------------------- step 4 – cluster-based permutation test ---------
    perm_results = permutation_test_decision_scores(
        decision_scores=results["decision_scores"],
        labels=results["labels"],
        n_permutations=10000,
        tail=1,
        seed=seed,
    )
    # save permutation-test outputs (T_obs, clusters, p_values, H0)
    np.savez_compressed(run_dir / "permutation_results.npz", **perm_results)
    print_permutation_test_summary(perm_results)

    logger.info("🎉 Job done.")


if __name__ == "__main__":
    main()
