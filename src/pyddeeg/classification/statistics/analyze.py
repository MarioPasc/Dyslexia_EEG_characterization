#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run every statistical test defined in pyddeeg.statistics on a saved
*nested_evaluate* results file.

Example
-------
$ python analyze_results.py path/to/cv_results.npz --out stats_elecFz.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import numpy as np

from pyddeeg.classification.statistics import (
    cluster_permutation_test,
    per_window_auc_test,
    cv_consistency_test,
    aggregate_selector_stats,
    feature_selection_binomial_test,
    feature_score_ttest,
)


def _load(path: Path):
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        out = {}
        for k in data.files:
            v = data[k]
            out[k] = v.tolist() if v.dtype == "O" else v
        return out
    else:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def analyze_results(
    results: Path, out: Path, n_perm: int, alpha: float, threads: int
) -> None:
    """Load results and run statistical tests."""
    res = _load(results)
    dec = res["decision_scores"]
    y = res["labels"]
    outer = res["outer_indices"]

    stats = dict(
        cluster=cluster_permutation_test(
            dec, y, n_perm=n_perm, alpha=alpha, n_jobs=threads
        ),
        per_window=per_window_auc_test(
            dec, y, n_perm=max(1_000, n_perm // 10), alpha=alpha
        ),
        cv_consistency=cv_consistency_test(dec, y, outer, alpha=alpha),
    )

    if "selector_stats" in res:
        sel = res["selector_stats"]
        stats.update(
            feature_agg=aggregate_selector_stats(sel),
            feature_binom=feature_selection_binomial_test(sel, alpha=alpha),
            feature_score=feature_score_ttest(sel, alpha=alpha),
        )

    # analyze.py – just before np.savez_compressed
    for k, v in list(stats.items()):
        if not isinstance(v, np.ndarray):                 # dict, list, …
            stats[k] = np.asarray(v, dtype=object)

    np.savez_compressed(out, **stats)  # type: ignore
    print(f"✔  Statistical report written to {out}")


def main() -> None:
    """Parse arguments and run analysis."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results", type=Path, help="cv_results.npz from tune_hyperparams_electrode"
    )
    ap.add_argument("--out", type=Path, default="stats.npz", help="Output file")
    ap.add_argument("--n_perm", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    analyze_results(
        results=args.results,
        out=args.out,
        n_perm=args.n_perm,
        alpha=args.alpha,
        threads=args.threads,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
