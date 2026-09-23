"""Adapter for the crude_oil_1 CSV pair."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data.dataset_adapters import CanonicalDataset

DATASET_DIR = Path(__file__).resolve().parent
TARGET_COLUMNS = {"density": "density", "sulfur": "sulfur", "residual_carbon": "residual_carbon"}


def _wavelengths(columns: pd.Index) -> np.ndarray:
    wn = pd.to_numeric(pd.Index(columns), errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(wn)) or np.any(wn <= 0):
        raise ValueError("Spectrum headers must be positive wavenumbers")
    return 10_000_000.0 / wn


def load_dataset(root: str | Path = DATASET_DIR) -> CanonicalDataset:
    root = Path(root)
    prop = pd.read_csv(root / "crude_oil_1_prop.csv", encoding="utf-8-sig")
    spec = pd.read_csv(root / "crude_oil_1_spec.csv", encoding="utf-8-sig")
    if len(prop) != len(spec):
        raise ValueError("Property and spectrum row counts differ")
    sample_id = prop.iloc[:, 0].astype(str).to_numpy()
    if not np.array_equal(sample_id, spec.iloc[:, 0].astype(str).to_numpy()):
        raise ValueError("Property and spectrum sample IDs are not aligned")
    targets = prop.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    targets.columns = list(TARGET_COLUMNS)
    spectra = spec.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(spectra).all():
        raise ValueError("Spectra contain non-finite values")
    wavelengths = _wavelengths(spec.columns[1:])
    order = np.argsort(wavelengths)
    dataset = CanonicalDataset("crude_oil_1", sample_id, wavelengths[order], spectra[:, order], targets.reset_index(drop=True), np.full(len(sample_id), "crude_oil_1"))
    dataset.validate()
    return dataset
