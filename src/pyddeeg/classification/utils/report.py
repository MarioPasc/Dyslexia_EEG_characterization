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
                                       :::-                                      
                                      -:--==                                     
                                ....  -====  .....                               
                          ..                         ..                          
                      ..                                 ..                      
              ....:..          ..:::---   :-==::.           ....:.               
              ::::::        .----=###===::=*#=*#*===        ::::::               
              .::::       .-----#####===:===######=====      :::: .              
             .           :-----####%=:::::::=*####======:           .            
           .            :----+#####===:::::-==*#####=====:           .           
          .           .:-----=######=========#######=====+-            .         
        .            -------##########======#########=====++            .        
       .             ---==+#%#########################%*+=+=             .       
      .              :::-------####*#########*####+=======.               .      
     .               :---------*###=:::===:::=####========+-               .     
  =--=+             -----------####===:::::-==####========++++            .--=+  
  +++**-            ::---------#####===::::==#####%=======++::            +++*** 
  ***##            -----------%######=:::::=+######=======++++:           .**##  
   .             --::--------=########:::::-########======+++:-+             .   
                 ------------##########====##########=====++++++=             .  
  .              -----------###########===###########=====++++++:             .  
  .              ::---------###########===###########=====+++++::             .  
  .            -::::---------#########==:==#########+=====+++-:::=.           .  
  .            ---:::--------########==:::===#######======++:::+++=           .  
  .            :::---------#######=====::::====+%#####====+=++++::            .  
  .            ------------%###:::::==#####+==::::=###%===++++++++=           .  
   .          .:--############:::::###########+::::=############++:           .  
   .      -#   ----+#########*:::-######**%#####::::##########+++++          .   
       :  .:   --:-+#---#############=-::::=*############=++##++.:           .   
    .   *  @    :---------###########==::::==##########*==++++++     **  :  .    
     .   %% .%   ---------=#####+-==#===-:==+====######===+++++=     .: %        
         =# =.     --##=--######--======::=======%######==##++     :# *-   .     
      .   .:  %     -=################==::==####*###########.     %   #   .      
       .   -.  #      --::=#*-%########=:-#########*+#%:::+     .= #.    .       
         .   +  = :    -::-#-:-########=:=########+=:##=::.     #       .        
          .   =. =.  .  .----:--####+##=:==#*####+=::====      .# =   .          
            .   + = =      .  .:---=====:======+==.  =     %   .*    .           
             .     = =  =.       :::         :::            :%=    .             
               .   --+.  =                                       .               
                 ..     .           %          .#+  =:        ..                 
                    ..             +.      #%%#+%   #.      .                    
                       ..           %#-:. #.:+#-        ..                       
                            ..                     ...                           
                                   ............


 /$$$$$$$  /$$            /$$$$$$  /$$                 /$$                 /$$      
| $$__  $$|__/           /$$__  $$|__/                | $$                | $$      
| $$  \ $$ /$$  /$$$$$$ | $$  \__/ /$$  /$$$$$$       | $$        /$$$$$$ | $$$$$$$ 
| $$$$$$$ | $$ /$$__  $$|  $$$$$$ | $$ /$$__  $$      | $$       |____  $$| $$__  $$
| $$__  $$| $$| $$  \ $$ \____  $$| $$| $$  \ $$      | $$        /$$$$$$$| $$  \ $$
| $$  \ $$| $$| $$  | $$ /$$  \ $$| $$| $$  | $$      | $$       /$$__  $$| $$  | $$
| $$$$$$$/| $$|  $$$$$$/|  $$$$$$/| $$| $$$$$$$/      | $$$$$$$$|  $$$$$$$| $$$$$$$/
|_______/ |__/ \______/  \______/ |__/| $$____/       |________/ \_______/|_______/ 
                                      | $$                                          
                                      | $$                                          
                                      |__/                                          
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
        "Time-resolved EEG classification with windowed non-linear recurrence chaos metrics for \n"
        "neural adaptation in developmental dyslexia research.\n"
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
    Summarize an EEGDataset with nested CV and per-fold label distributions.

    Prints:
    - Total subjects and class counts (DD / CT)
    - Number of RQA metrics and time-windows (with window/stride in ms)
    - Stimulus direction and electrode name
    - Nested CV scheme (outer × inner folds)
    - For each outer fold: train/test label counts

    Parameters
    ----------
    dataset : EEGDataset
        The dataset to summarize.
    """
    # ---- basic shapes and counts -----------------------------------------
    n_dd, n_metrics, n_windows = dataset.dd.shape
    n_ct = dataset.ct.shape[0]
    n_subjects = n_dd + n_ct

    # ---- nested-CV scheme -----------------------------------------------
    k_outer = len(dataset.outer_splits)
    # assume every fold has the same number of inner splits
    h_inner = len(dataset.inner_splits[0]) if dataset.inner_splits else 0

    # ---- metadata --------------------------------------------------------
    md = dataset.metadata
    window_ms = md.get("window_ms", "—")
    stride_ms = md.get("stride_ms", "—")
    direction = md.get("direction", "—")
    elec = md.get("elec", "—")

    # ---- compose lines ---------------------------------------------------
    lines = [
        "📊  EEGDataset summary",
        f"    • Subjects ............ {n_subjects}  (DD = {n_dd}, CT = {n_ct})",
        f"    • RQA metrics ......... {n_metrics}",
        f"    • Time-windows ........ {n_windows}  "
        f"(window = {window_ms} ms, stride = {stride_ms} ms)",
        f"    • Direction ........... {direction}",
        f"    • Electrode ........... {elec}",
        f"    • Nested CV ........... {k_outer} outer × {h_inner} inner folds",
        "    • Label distribution per fold:",
    ]

    for idx, dist in enumerate(dataset.fold_label_dist):
        tr = dist["train"]
        te = dist["test"]
        lines.append(
            f"        — Fold {idx}: "
            f"train (DD = {tr['DD']}, CT = {tr['CT']}), "
            f"test  (DD = {te['DD']}, CT = {te['CT']})"
        )

    print("\n".join(lines))

    # ---- concise logger info --------------------------------------------
    logger.info(
        "EEGDataset: %d subjects (DD=%d,CT=%d); %d metrics; %d windows; "
        "nested CV %dx%d; electrode=%s; direction=%s",
        n_subjects,
        n_dd,
        n_ct,
        n_metrics,
        n_windows,
        k_outer,
        h_inner,
        elec,
        direction,
    )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def print_tuning_summary(tuning_results: Sequence[Dict[str, Any]]) -> None:
    """
    Summarise Optuna search results.

    Each entry in *tuning_results* must contain
    ``"best_params"``  – dict of hyper-parameters
    ``"performance"``  – either a float **or** a dict with keys
                         {"mean", "std", "median", "folds"}.
    """
    if not tuning_results:
        print("🔧  Tuning results empty.")
        logger.warning("Tuning summary: empty results list.")
        return

    header = "🔧  Hyper-parameter tuning"
    print(header)
    logger.info(header)

    # Helper to pull a sensible scalar + formatted string out of "performance"
    def _parse_perf(perf: Any) -> tuple[float, str]:
        if isinstance(perf, dict):
            mean = float(perf.get("mean", np.nan))
            std = float(perf.get("std", np.nan))
            return mean, f"{mean:6.4f} ± {std:5.4f}"
        # fall back to “it’s probably a float already”
        mean = float(perf)
        return mean, f"{mean:6.4f}"

    row_fmt = "      • Window #{idx:<2}: ROC-AUC = {perf:<18}   best_params = {params}"

    means: list[float] = []
    for idx, win in enumerate(tuning_results, start=1):
        mean_auc, perf_str = _parse_perf(win.get("performance", np.nan))
        means.append(mean_auc)

        # pretty-print the params without new-lines and trim if too long
        params_txt = json.dumps(win.get("best_params", {}), separators=(",", ":"))
        params_txt = (params_txt[:80] + "…") if len(params_txt) > 80 else params_txt

        print(row_fmt.format(idx=idx, perf=perf_str, params=params_txt))
        logger.info(
            "Window %s – ROC-AUC %s – params %s", idx, perf_str.strip(), params_txt
        )

    print(f"      ↳ mean ROC-AUC across windows = {np.nanmean(means):.4f}")


# --------------------------------------------------------------------------- #
def print_cv_results_summary(cv_results: Dict[str, Any]) -> None:
    """
    Summarise nested CV output from `nested_evaluate`.

    Expected keys
    -------------
    decision_scores    : ndarray, shape (n_subjects, n_windows)
    labels             : ndarray, shape (n_subjects,)
    fold_auc           : ndarray, shape (n_folds, 2, n_windows)  # [ROC, PR]
    params_per_outer   : list of length n_folds, each a list of length n_windows of dicts
    selector_stats     : dict with keys
                         - names: ndarray (n_features,)
                         - scores: ndarray (n_folds, n_windows, n_features)  # selection freq
                         - mask  : ndarray (n_folds, n_windows, n_features)  # boolean
    perm_test          : dict passed to `print_permutation_test_summary`
    outer_indices      : list of (train_idx, test_idx)
    """
    # 1) basic shapes
    ds, ls = cv_results["decision_scores"], cv_results["labels"]
    fa = np.asarray(cv_results["fold_auc"])
    n_subjects, n_windows = ds.shape
    n_folds = fa.shape[0]
    print(f"🏁  CV results summary")
    print(f"      • Subjects  : {n_subjects}")
    print(f"      • Windows   : {n_windows}")
    print(f"      • Folds     : {n_folds}")

    # 2) overall ROC & PR
    roc = fa[:, 0, :]  # (folds, windows)
    pr = fa[:, 1, :]
    mean_roc_w = np.nanmean(roc, axis=0)
    std_roc_w = np.nanstd(roc, axis=0)
    mean_pr_w = np.nanmean(pr, axis=0)
    std_pr_w = np.nanstd(pr, axis=0)
    global_roc = np.nanmean(mean_roc_w)
    global_pr = np.nanmean(mean_pr_w)

    print(f"      • Global ROC-AUC : {global_roc:.4f}")
    print(f"      • Global PR-AUC  : {global_pr:.4f}")

    # best windows
    best_roc_idx = int(np.nanargmax(mean_roc_w))
    best_pr_idx = int(np.nanargmax(mean_pr_w))
    print(
        f"      • Best window by ROC : #{best_roc_idx:02}  ({mean_roc_w[best_roc_idx]:.4f} ± {std_roc_w[best_roc_idx]:.4f})"
    )
    print(
        f"      • Best window by PR  : #{best_pr_idx:02}  ({mean_pr_w[best_pr_idx]:.4f} ± {std_pr_w[best_pr_idx]:.4f})"
    )

    # 3) per-window summary table
    print("\n      • Per-window performance:")
    for w in range(n_windows):
        print(
            f"         - Win #{w:02}: ROC = {mean_roc_w[w]:.4f} ± {std_roc_w[w]:.4f}, "
            f"PR = {mean_pr_w[w]:.4f} ± {std_pr_w[w]:.4f}"
        )

    # logging
    logger.info(
        "CV global ROC = %.4f, PR = %.4f; best ROC win=%s (%.4f±%.4f), best PR win=%s (%.4f±%.4f)",
        global_roc,
        global_pr,
        best_roc_idx,
        mean_roc_w[best_roc_idx],
        std_roc_w[best_roc_idx],
        best_pr_idx,
        mean_pr_w[best_pr_idx],
        std_pr_w[best_pr_idx],
    )

    # 4) hyper‐parameter stability
    params_per_outer = cv_results.get(
        "params_per_outer", []
    )  # list[f][windows] of dicts
    if len(params_per_outer) > 0:

        def _sig(d: dict) -> tuple:
            # turn each {k: v} into a sorted tuple of (k, repr(v))
            return tuple(sorted((k, repr(v)) for k, v in d.items()))

        # for each window, count how many unique signatures across folds
        n_unique = [
            len({_sig(p) for p in win_params}) for win_params in zip(*params_per_outer)
        ]
        avg = float(np.mean(n_unique))
        print(f"\n      • Hyper-parameter stability:")
        print(
            f"         ↳ Avg. unique param-sets/window = {avg:.1f} (min={min(n_unique)}, max={max(n_unique)})"
        )
        logger.info("Param stability per window: %s", n_unique)

    # 5) selector diagnostics
    sel = cv_results.get("selector_stats", {})
    if sel:
        names = list(sel.get("names", []))
        scores = np.asarray(sel.get("scores", []))  # (folds, windows, feats)
        mask = np.asarray(sel.get("mask", []))  # same shape

        # average fraction of times each metric was kept
        frac_kept = scores.mean(axis=(0, 1))  # over folds & windows
        mean_frac = 100 * np.mean(frac_kept)
        top5_idx = np.argsort(frac_kept)[-5:][::-1]
        top5 = [(names[i], frac_kept[i] * 100) for i in top5_idx]

        print(f"\n      • Feature‐selection:")
        print(f"         ↳ Avg. metrics kept = {mean_frac:.1f}%")
        print(f"         ↳ Top-5 most stable metrics:")
        for name, pct in top5:
            print(f"            • {name:<20} {pct:5.1f}%")

        logger.info("Selector avg keep rate %.3f; top5 %s", mean_frac / 100, top5)

    # 6) permutation‐test summary
    perm = cv_results.get("perm_test", {})
    if perm:
        print("\n      • Permutation‐test clusters:")
        # delegate to your existing summary helper
        print_permutation_test_summary(perm)
    print("")  # trailing newline


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
