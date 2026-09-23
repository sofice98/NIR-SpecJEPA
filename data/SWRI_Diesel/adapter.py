"""Adapter and missing-target cross-validation policies for SWRI Diesel."""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd

from data.dataset_adapters import CanonicalDataset

DATASET_DIR = Path(__file__).resolve().parent
TARGET_COLUMNS = ("BP50", "CN", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")
MissingTargetStrategy = Literal["drop_missing", "all_train_spectra"]


def load_dataset(root: str | Path = DATASET_DIR) -> CanonicalDataset:
    root = Path(root)
    prop = pd.read_csv(root / "diesel_prop.csv", encoding="utf-8-sig")
    spec = pd.read_csv(root / "diesel_spec.csv", encoding="utf-8-sig")
    if len(prop) != len(spec):
        raise ValueError("Property and spectrum row counts differ")
    sample_id = prop.iloc[:, 0].astype(str).to_numpy()
    if not np.array_equal(sample_id, spec.iloc[:, 0].astype(str).to_numpy()):
        raise ValueError("Property and spectrum sample IDs are not aligned")
    targets = prop.loc[:, list(TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    spectra = spec.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(spectra).all():
        raise ValueError("Spectra contain non-finite values")
    wavelengths = pd.to_numeric(spec.columns[1:], errors="coerce").to_numpy(float)
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
        raise ValueError("Spectrum headers must be positive wavelengths")
    order = np.argsort(wavelengths)
    dataset = CanonicalDataset("SWRI_Diesel", sample_id, wavelengths[order], spectra[:, order], targets, np.full(len(sample_id), "SWRI_Diesel"), metadata={"missing_targets": targets.isna().sum().to_dict(), "source_spectral_axis": "wavelength_nm"})
    dataset.validate()
    return dataset


def target_valid_mask(dataset: CanonicalDataset, target: str) -> np.ndarray:
    if target not in dataset.targets.columns:
        raise KeyError(f"Unknown target {target!r}; choose from {dataset.targets.columns.tolist()}")
    return dataset.targets[target].notna().to_numpy()


def target_fold_indices(dataset: CanonicalDataset, target: str, train_idx: np.ndarray, valid_idx: np.ndarray, strategy: MissingTargetStrategy = "drop_missing") -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation indices for one target.

    ``drop_missing`` removes missing labels from both sides. ``all_train_spectra``
    keeps every training spectrum (for representation/pretraining) while only
    returning validation samples whose target is observed.
    """
    valid = target_valid_mask(dataset, target)
    train_idx = np.asarray(train_idx, dtype=int)
    valid_idx = np.asarray(valid_idx, dtype=int)
    if strategy == "drop_missing":
        return train_idx[valid[train_idx]], valid_idx[valid[valid_idx]]
    if strategy == "all_train_spectra":
        return train_idx, valid_idx[valid[valid_idx]]
    raise ValueError("strategy must be 'drop_missing' or 'all_train_spectra'")


def make_target_folds(dataset: CanonicalDataset, folds, target: str, strategy: MissingTargetStrategy = "drop_missing"):
    """Adapt generic Fold objects to a target-aware CV evaluation."""
    return [target_fold_indices(dataset, target, fold.train_idx, fold.test_idx, strategy) for fold in folds]
