"""Adapter for the wide LUCAS Soil corrected spectra CSV.

The source file stores one sample per row.  Columns named ``spc.<nm>`` are
reflectance values at wavelengths in nanometres; the remaining columns contain
sample metadata and soil laboratory measurements.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from data.dataset_adapters import CanonicalDataset

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = "LUCAS.SOIL_corr.csv"

# Numeric soil properties commonly used as supervised targets in LUCAS.
TARGET_COLUMNS = (
    "coarse",
    "clay",
    "silt",
    "sand",
    "pH.in.CaCl2",
    "pH.in.H2O",
    "OC",
    "CaCO3",
    "N",
    "P",
    "K",
    "CEC",
)
_SPECTRUM_RE = re.compile(r"^spc\.(\d+(?:\.\d+)?)$")


def _resolve_source(root: str | Path) -> Path:
    path = Path(root)
    if path.is_file():
        return path
    return path / DEFAULT_FILE


def _spectral_columns(columns: pd.Index) -> tuple[list[str], np.ndarray]:
    pairs = []
    for column in columns:
        match = _SPECTRUM_RE.match(str(column))
        if match:
            pairs.append((column, float(match.group(1))))
    if not pairs:
        raise ValueError("No LUCAS spectral columns matching 'spc.<wavelength>' were found")
    pairs.sort(key=lambda item: item[1])
    names = [item[0] for item in pairs]
    wavelengths = np.asarray([item[1] for item in pairs], dtype=float)
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("LUCAS spectral wavelengths must be strictly increasing")
    return names, wavelengths


def load_dataset(
    root: str | Path = DATASET_DIR,
    *,
    nrows: int | None = None,
) -> CanonicalDataset:
    """Load LUCAS Soil into the project's canonical spectral representation.

    ``root`` may be the dataset directory or the CSV path itself. ``nrows`` is
    provided for quick smoke tests and should be left as ``None`` for training.
    Missing soil measurements remain as ``NaN`` targets and can be handled by
    the target-aware CV policy used by the experiment scripts.
    """

    source = _resolve_source(root)
    if not source.exists():
        raise FileNotFoundError(f"LUCAS source file not found: {source}")

    # Read the header first so the 4,200 spectral columns can be selected
    # explicitly instead of loading unrelated metadata twice.
    header = pd.read_csv(source, nrows=0)
    spectral_columns, wavelengths = _spectral_columns(header.columns)
    required = ["sample.ID", "country", *TARGET_COLUMNS]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(f"LUCAS source is missing required columns: {missing}")

    usecols = ["sample.ID", "country", *TARGET_COLUMNS, *spectral_columns]
    frame = pd.read_csv(source, usecols=usecols, nrows=nrows, low_memory=False)
    sample_id = frame["sample.ID"].astype("string").fillna("").to_numpy(dtype=str)
    if not np.all(sample_id):
        raise ValueError("LUCAS sample.ID contains empty values")

    spectra = frame[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(spectra).all():
        raise ValueError("LUCAS spectra contain non-finite values")

    targets = frame.loc[:, list(TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    targets = targets.reset_index(drop=True)
    groups = frame["country"].astype("string").fillna("UNKNOWN").to_numpy(dtype=str)
    dataset = CanonicalDataset(
        name="LUCAS_SOIL",
        sample_id=sample_id,
        wavelengths_nm=wavelengths,
        spectra=spectra,
        targets=targets,
        groups=groups,
        metadata={
            "source_file": str(source),
            "source_spectral_axis": "wavelength_nm",
            "spectral_column_pattern": "spc.<nm>",
            "n_spectral_channels": len(spectral_columns),
            "missing_targets": targets.isna().sum().to_dict(),
            "group_column": "country",
        },
    )
    dataset.validate()
    return dataset

