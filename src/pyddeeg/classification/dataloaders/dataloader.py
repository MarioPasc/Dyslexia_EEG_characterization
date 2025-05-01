# -*- coding: utf-8 -*-
"""
dataloader.py – single-source of truth for data *and* split statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class EEGDataset:
    """
    RQA tensors + nested CV splits + per-fold label histograms.
    """

    dd: np.ndarray
    ct: np.ndarray
    outer_splits: List[Tuple[np.ndarray, np.ndarray]]
    inner_splits: List[List[Tuple[np.ndarray, np.ndarray]]]
    fold_label_dist: List[Dict[str, Dict[str, int]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------- factory ---------------------------- #
    @staticmethod
    def load(
        dataset_root: Path,
        window: str,
        direction: str,
        elec: str,
        *,
        cv_scheme: Tuple[int, int] = (5, 2),
        random_state: int = 42,
    ) -> "EEGDataset":
        """
        Load tensors and create k × h nested StratifiedGroupKFold splits.

        A ``fold_label_dist`` entry looks like
        ``{'train': {'DD': 12, 'CT': 25}, 'test': {'DD': 3, 'CT': 7}}``.
        """
        import json

        meta_file = dataset_root / window / "metadata.npz"
        meta_npz = np.load(meta_file)
        centres_ms = meta_npz["centers"].astype(int)
        window_ms = int(window.split("_")[-1])
        stride_ms = int(meta_npz["stride"])
        total_ms = centres_ms[-1] + window_ms // 2

        routes = json.loads((dataset_root / "dataset_index.json").read_text())[window][
            direction
        ]
        if elec not in routes:
            raise KeyError(f"Electrode {elec!r} not found in dataset index.")

        pA, pB = routes[elec]
        dd_path, ct_path = (pA, pB) if "DD" in Path(pA).stem.upper() else (pB, pA)
        dd, ct = np.load(dd_path)["metrics"], np.load(ct_path)["metrics"]

        # -------------------------- nested CV --------------------------- #
        k, h = cv_scheme
        y = np.r_[np.ones(len(dd)), np.zeros(len(ct))].astype(int)
        groups = np.arange(len(y))  # 1 group == 1 subject

        outer_cv = StratifiedGroupKFold(
            n_splits=k, shuffle=True, random_state=random_state
        )

        outer_splits, inner_splits, fold_label_dist = [], [], []

        for tr_g, te_g in outer_cv.split(y, y, groups):
            outer_splits.append((tr_g, te_g))

            # label histogram
            def _count(lbl_idx: np.ndarray) -> Dict[str, int]:
                return {
                    "DD": int(y[lbl_idx].sum()),
                    "CT": int(len(lbl_idx) - y[lbl_idx].sum()),
                }

            fold_label_dist.append({"train": _count(tr_g), "test": _count(te_g)})

            # inner splits inside outer-train
            inner_cv = StratifiedGroupKFold(
                n_splits=h, shuffle=True, random_state=random_state
            )
            inner_fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
            # remap outer-train → 0…|T|-1
            idx_map = {g: i for i, g in enumerate(tr_g)}
            for in_tr, in_te in inner_cv.split(y[tr_g], y[tr_g], groups[tr_g]):
                inner_fold_indices.append(
                    (tr_g[in_tr], tr_g[in_te])  # store **global** indices
                )
            inner_splits.append(inner_fold_indices)

        metadata = dict(
            centres_ms=centres_ms,
            window_ms=window_ms,
            stride_ms=stride_ms,
            total_ms=total_ms,
            direction=direction,
            elec=elec,
        )
        return EEGDataset(
            dd=dd.astype(np.float32),
            ct=ct.astype(np.float32),
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            fold_label_dist=fold_label_dist,
            metadata=metadata,
        )
