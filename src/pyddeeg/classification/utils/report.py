# pyddeeg/utils/reporting.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility helpers that *summarise* key objects created by the pyddeeg
classification pipeline.

The functions emit pretty, human-readable output **and** write the same
information to the root *pyddeeg* logger so it ends up in your SLURM or
systemd logfiles.

Public API
----------
print_welcome_banner
print_dataset_summary
print_tuning_summary
print_cv_results_summary
print_permutation_test_summary
"""
from __future__ import annotations

from pathlib import Path
from textwrap import indent
from typing import Any, Dict, List, Sequence

import json
import logging

import numpy as np
from pyddeeg.classification import logger  # root logger
from pyddeeg.classification.dataloaders import EEGDataset

__all__ = [
    "print_welcome_banner",
    "print_dataset_summary",
    "print_tuning_summary",
    "print_cv_results_summary",
    "print_permutation_test_summary",
]

# --------------------------------------------------------------------------- #
#                                ASCII BRAIN                                  #
# --------------------------------------------------------------------------- #
_BRAIN_ART = r"""                                                                                                                         
                      ≠÷∞         √√≈∞                     
                  π≈√  π   ≠π π   √∞   ∞≠π                 
                ≈≠  ≠≠√√√ ππ≠√πππ√√√ππ√   ≠√               
               ÷  ≠√ π√ππ  π√=≈∞ πππ π ππ   =              
             ∞   √√√  π   √√√×√ππ         ≈  √≈            
            ≠  =√=ππ     π∞√√≈√√√       πππ√   =           
           ≠   √π           √∞√π           π√   ≈          
          √   √π≈∞√∞ππ   ≠   ππ  π  πππ∞π  π≈    √         
              π√π√√≠√√   ππ     ππ  π√=≈∞≈≠∞√    =         
         ÷    ≈√π√∞√√ππππ ππ-≈=ππ  ππππ√ππ√≠×    π         
         π   ≈≈√√π√√√ππππ  π+∞+∞  πππ √∞∞π√ππ≠             
            ≠√π√√√≈√√πππ     =π      ππ√√∞√√√π≠            
           √π√ππ π≈√ππ       ∞      π π√÷≠    π√           
           ππ  ππ π÷√ π π π π∞ππ    π π√π  √√  π  π        
         π √√π   √ πππ   πππ≠+∞πππ   ππ        π           
         ≠ √≈≈√   π    ππ√  π   πππ         π√√∞           
         ≠  ππ  π     =≠   π- ≈π   ≠√        π√  ≈         
          π √ππ          πππππ√     √    π π÷π√            
          ÷  √π π√π   π√≈πππ√√π∞√√π     πππππ∞  =          
           -  √≈π π√  π√  ππ π√ππ ππ   π π√ππ  ×∞          
           √×  √π≈√    ππππππππ√      √ √ππ√  √=           
            π-  ≈π       πππ∞ππ      π√√π≈   ∞÷            
              ×÷  √ππππ√    ≈π√ ∞√  √ π√√   ÷=             
               π≈÷   ≈ππ  √     π√π√π    =÷≈               
                  √÷   π          ππ   ÷=                  
                     π∞==÷=××÷×-××××≠π                                                              
"""


# --------------------------------------------------------------------------- #
#                               PUBLIC HELPERS                                #
# --------------------------------------------------------------------------- #
def print_welcome_banner(*, package_version: str | None = None) -> None:
    """
    Print a nice banner to the terminal and log that *pyddeeg* started.

    Parameters
    ----------
    package_version
        Optional version string to include next to *pyddeeg*.
    """
    title = f"pyddeeg {package_version}" if package_version else "pyddeeg"
    subtitle = (
        "Time-resolved EEG classification with non-linear recurrence chaos metrics\n"
        "Mario Pascual González & Dr. Ignacio Rodríguez Rodríguez  •  BioSiP-Lab, University of Málaga"
    )

    banner = f"""
{_BRAIN_ART}
{title.center(69)}
{subtitle}
"""
    print(banner)
    logger.info("🐍  %s initialised – welcome!", title)


# --------------------------------------------------------------------------- #
def print_dataset_summary(dataset: EEGDataset) -> None:
    """
    Summarise an :class:`~pyddeeg.classification.dataloaders.EEGDataset`.

    The helper drills down into the fields defined in *dataloader.py*:

    * Separate DD / CT sample counts
    * n_metrics • n_windows
    * CV splitter type & #splits
    * Key metadata (window length, stride, etc.)
    """
    # ---- shapes -----------------------------------------------------------
    n_dd, n_metrics, n_windows = dataset.dd.shape
    n_ct = dataset.ct.shape[0]
    n_subjects = n_dd + n_ct

    # ---- metadata ---------------------------------------------------------
    md = dataset.metadata
    window_ms = md.get("window_ms", "—")
    stride_ms = md.get("stride_ms", "—")
    direction = md.get("direction", "—")
    cv_type = md.get("cv_type", type(dataset.cv).__name__)
    n_splits = getattr(
        dataset.cv, "n_splits", getattr(dataset.cv, "get_n_splits", lambda: "?")()
    )

    # ----------------------------------------------------------------------
    lines = [
        "📊  EEGDataset summary",
        f"      • Subjects ............ {n_subjects}  (DD ={n_dd}, CT ={n_ct})",
        f"      • RQA metrics ......... {n_metrics}",
        f"      • Time-windows ........ {n_windows}  (window ={window_ms} ms, stride ={stride_ms} ms)",
        f"      • Direction ........... {direction}",
        f"      • CV splitter ......... {cv_type}  (n_splits ={n_splits})",
        f"      • Dataset-root ........ {Path(md.get('dataset_root','?')).expanduser()}",
    ]

    print("\n".join(lines))
    logger.info(
        "EEGDataset – %s", "; ".join(l.split("…")[-1].strip() for l in lines[1:])
    )


# --------------------------------------------------------------------------- #
def print_tuning_summary(tuning_results: Sequence[Dict[str, Any]]) -> None:
    """
    Summarise Optuna search results produced by ``tune_one_electrode(_parallel)``.
    """
    if len(tuning_results) == 0:
        print("🔧  Tuning results empty.")
        logger.warning("Tuning summary: empty results list.")
        return

    header = "🔧  Hyper-parameter tuning"
    print(header)
    logger.info(header)

    row_fmt = "      • Window #{idx:<2}: ROC-AUC = {auc:6.4f}   best_params = {params}"
    aucs = []
    for idx, win in enumerate(tuning_results, start=1):
        auc = float(win.get("performance", np.nan))
        aucs.append(auc)
        params = json.dumps(win.get("best_params", {}), separators=(",", ":"))
        print(row_fmt.format(idx=idx, auc=auc, params=params))
        logger.info("Window %s – ROC-AUC %.4f – params %s", idx, auc, params)

    print(f"      ↳ mean ROC-AUC across windows = {np.nanmean(aucs):.4f}")


# --------------------------------------------------------------------------- #
def print_cv_results_summary(cv_results: Dict[str, Any]) -> None:
    """
    Summarise nested CV output from ``evaluate_frozen_models`` (trainer.py):

    Expected keys
    -------------
    fold_auc : ndarray  (n_folds × 2 × n_windows)  – [ROC, PR] AUC
    decision_scores : ndarray (subjects × n_windows)
    labels : ndarray (subjects,)
    selected_features : ndarray (n_windows × n_features)
    """
    if "fold_auc" not in cv_results:
        print("🏁  CV summary: 'fold_auc' missing.")
        logger.warning("CV summary: key 'fold_auc' not present.")
        return

    fold_auc = np.asarray(cv_results["fold_auc"])
    roc_by_win = fold_auc[:, 0, :]  # (folds, windows)
    pr_by_win = fold_auc[:, 1, :]

    mean_roc_per_win = np.nanmean(roc_by_win, axis=0)
    mean_pr_per_win = np.nanmean(pr_by_win, axis=0)

    overall_roc = np.nanmean(mean_roc_per_win)
    overall_pr = np.nanmean(mean_pr_per_win)

    best_idx = int(np.nanargmax(mean_roc_per_win))
    best_auc = mean_roc_per_win[best_idx]

    print("🏁  Outer-CV results")
    print(f"      • Mean ROC-AUC across folds × windows .... {overall_roc:.4f}")
    print(f"      • Mean PR-AUC  across folds × windows .... {overall_pr:.4f}")
    print(
        f"      • Best window (by ROC-AUC) ............... #{best_idx:02}  (ROC = {best_auc:.4f})"
    )

    logger.info(
        "CV: mean ROC-AUC %.4f (best window %s → %.4f) – mean PR-AUC %.4f",
        overall_roc,
        best_idx,
        best_auc,
        overall_pr,
    )

    # ---- optional extras --------------------------------------------------
    n_features = cv_results.get("selected_features", np.empty((0, 0))).shape[-1]
    if n_features:
        mean_selected = cv_results["selected_features"].mean(axis=0)
        prop_sel = 100 * np.mean(mean_selected)
        print(
            f"      • Feature-selection ................. {prop_sel:.1f}% metrics kept (avg.)"
        )
        logger.info("Feature selection: %.2f %% metrics kept (averaged)", prop_sel)


# --------------------------------------------------------------------------- #
def print_permutation_test_summary(
    perm_results: Dict[str, Any],
    *,
    alpha: float = 0.05,
) -> None:
    """
    Summarise cluster-based permutation statistics returned by
    ``permutation_test_decision_scores`` (trainer.py).
    """
    p_vals = np.asarray(perm_results.get("p_values", []))
    if p_vals.size == 0:
        print("⚠️  Permutation test: no p-values found.")
        logger.warning("Permutation summary: p_values empty/missing.")
        return

    n_sig = int(np.sum(p_vals < alpha))
    first_sig = np.flatnonzero(p_vals < alpha).tolist()[:5]  # short preview

    msg = (
        f"🔬  Permutation test: {n_sig}/{len(p_vals)} clusters "
        f"significant at α = {alpha}."
    )
    if n_sig:
        msg += f" First significant clusters: {first_sig}"

    print(msg)
    logger.info("Permutation test: %s", msg)
