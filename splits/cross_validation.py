from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_group: str | None = None


def pooled_group_balanced_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 20260920):
    """Deterministically distribute each source group across folds.

    This avoids requiring sklearn/scipy merely to materialize the manifest.
    Each source is shuffled with a fixed seed and round-robin assigned, so
    every fold receives near-equal counts from every source.
    """
    groups = np.asarray(groups)
    if n_splits < 2 or n_splits > len(groups):
        raise ValueError("n_splits must be between 2 and the sample count")
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(groups), dtype=int)
    for group in np.unique(groups.astype(str)):
        indices = np.flatnonzero(groups.astype(str) == group)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % n_splits
    return [
        Fold(i, np.flatnonzero(fold_ids != i), np.flatnonzero(fold_ids == i))
        for i in range(n_splits)
    ]


def leave_one_group_out_folds(groups: np.ndarray):
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    return [
        Fold(i, np.flatnonzero(groups != held_out), np.flatnonzero(groups == held_out), held_out)
        for i, held_out in enumerate(unique)
    ]
