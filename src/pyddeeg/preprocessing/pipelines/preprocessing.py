#!/usr/bin/env python3
"""
Unified EEG preprocessing pipeline.

Runs (optionally) the zero-lag filter, multi-window RQA and the
post-hoc reorganiser in a single command.

Usage
-----
    python preprocessing.py --config pipeline_config.yaml \
                            --stim 20 \
                            --channel Fp1 \
                            --do-zerolag \
                            --do-rqa \
                            --do-reorg
Typical Slurm array job passes only --channel; other stages are skipped
automatically when their outputs already exist.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import subprocess
import yaml
from typing import Any, Dict, List

# ── Stage helpers (import after PYTHONPATH is set in Slurm) ─────────────
from pyddeeg.preprocessing.pipelines.zerolag_preprocessing import (
    run as run_zerolag,
)  # stage-1
from pyddeeg.preprocessing.pipelines.rqa_windows_picasso import (
    run_from_config,
)  # stage-2
from pyddeeg.utils.postprocessing.reorganize_per_window_results import (
    reorganization_pipeline,
)  # stage-3


# ── Logging ─────────────────────────────────────────────────────────────
def setup_logger(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("pipeline")


# ── Orchestrator ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="EEG preprocessing pipeline")
    parser.add_argument(
        "--config", required=True, type=Path, help="Master YAML configuration"
    )
    parser.add_argument("--stim", required=True, help="Stimulus code (2, 8, 20)")
    parser.add_argument(
        "--channel", required=True, help="Electrode name or index (passed to RQA job)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable Dask inside RQA stage"
    )
    parser.add_argument(
        "--do-zerolag", action="store_true", help="Force rerun zero-lag stage"
    )
    parser.add_argument(
        "--do-rqa",
        action="store_true",
        help="Force rerun RQA stage even if results exist",
    )
    parser.add_argument(
        "--do-reorg",
        action="store_true",
        help="Run re-organisation (only after all electrodes)",
    )

    args = parser.parse_args()

    # ── Load YAML once ──────────────────────────────────────────────────
    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text())
    paths = cfg["paths"]
    log_dir = Path(paths["rqa_out_root"]) / "_logs"
    logger = setup_logger(log_dir)

    logger.info("===== EEG unified pipeline started =====")
    logger.info("Stimulus %s | Channel %s", args.stim, args.channel)

    # ─────────────────────────────────────────────────── Stage 1: Zero-lag
    zl_out_dir = Path(paths["zerolag_out_dir"])
    zl_file = zl_out_dir / f"CT_UP_preprocess_{args.stim}.npz"  # one sentinel
    if args.do_zerolag or not zl_file.exists():
        logger.info("[1/3] Running zero-lag preprocessing")
        zerolag_cfg_path = _build_zerolag_cfg(cfg, args.stim, zl_out_dir)
        run_zerolag(zerolag_cfg_path)
    else:
        logger.info("[1/3] Zero-lag outputs already present; skipping")

    # ─────────────────────────────────────────────────── Stage 2: RQA
    rqa_out_dir = Path(paths["rqa_out_root"]) / args.channel
    sentinel = rqa_out_dir / "CT_UP" / "rqa_analysis_CT_UP_metrics.npz"
    if args.do_rqa or not sentinel.exists():
        logger.info("[2/3] Running RQA for electrode %s", args.channel)
        rqa_cfg_path = _build_rqa_cfg(
            cfg, args.stim, args.channel, zl_out_dir, rqa_out_dir
        )
        run_from_config(
            rqa_cfg_path, channel=args.channel, no_parallel=args.no_parallel
        )
    else:
        logger.info("[2/3] RQA results for %s already present; skipping", args.channel)

    # ─────────────────────────────────────────────────── Stage 3: Re-org
    if args.do_reorg:
        logger.info("[3/3] Re-organising per-window datasets")
        out_dataset = Path(paths["dataset_out_dir"])
        reorganization_pipeline(raw_dir=paths["rqa_out_root"], output_dir=out_dataset)

    logger.info("Pipeline finished successfully")


# ── Internal helpers to materialise per-stage YAML ----------------------
def _build_zerolag_cfg(master: Dict[str, Any], stim: str, out_dir: Path) -> Path:
    """Write a minimal YAML for stage-1 and return its path."""
    cfg = {
        "data_dir": master["paths"]["raw_timeseries_dir"],
        "output_dir": str(out_dir),
        "age": "children",
        "stim": stim,
        **master["zerolag"],
        "nframes": 68000,
        "ch_names": [
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
        ],
    }
    path = out_dir / f"_zl_cfg_{stim}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _build_rqa_cfg(
    master: Dict[str, Any], stim: str, channel: str | int, in_dir: Path, out_dir: Path
) -> Path:
    datasets = {
        "CT_UP": f"CT_UP_preprocess_{stim}.npz",
        "DD_UP": f"DD_UP_preprocess_{stim}.npz",
        "CT_DOWN": f"CT_DOWN_preprocess_{stim}.npz",
        "DD_DOWN": f"DD_DOWN_preprocess_{stim}.npz",
    }
    rqa_cfg = {
        "input_directory": str(in_dir),
        "output_directory": str(out_dir),
        "datasets": datasets,
        "target_channel": channel,
        **master["rqa"],
        # keep logging inside electrode folder
        "logging": {
            "directory": str(out_dir / "_logs"),
            "filename": "rqa_windows.log",
            "level": "INFO",
        },
    }
    path = out_dir / f"_rqa_cfg_{channel}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rqa_cfg))
    return path


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
