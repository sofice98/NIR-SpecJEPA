"""Dataset-specific adapter for the private crude-oil spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset_adapters import CanonicalDataset, merge_datasets


DATASET_DIR = Path(__file__).resolve().parent
TARGET_COLUMNS = {
    "density": "密度（20℃）",
    "sulfur": "硫含量（质量分数）",
    "residual_carbon": "残炭（质量分数）",
}
TARGET_UNITS = {
    "density": "g/cm3",
    "sulfur": "mass_fraction",
    "residual_carbon": "mass_fraction",
}


def wavenumbers_cm_to_wavelengths_nm(wavenumbers_cm: np.ndarray) -> np.ndarray:
    """Convert positive wavenumbers in cm^-1 to wavelengths in nm."""
    values = np.asarray(wavenumbers_cm, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Wavenumbers must be a finite, positive one-dimensional array")
    return 10_000_000.0 / values


def _read_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read this dataset's row-oriented CSV and return an increasing nm axis."""
    frame = pd.read_csv(path, header=None, encoding="utf-8-sig")
    if frame.shape[0] < 2 or frame.shape[1] < 3:
        raise ValueError(f"Invalid spectrum CSV: {path}")
    wavenumbers_cm = pd.to_numeric(frame.iloc[0, 1:], errors="coerce").to_numpy(float)
    sample_ids = frame.iloc[1:, 0].astype(str).str.strip().to_numpy()
    spectra = frame.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(wavenumbers_cm).any() or np.isnan(spectra).any():
        raise ValueError(f"Non-numeric values found in {path}")
    wavelengths_nm = wavenumbers_cm_to_wavelengths_nm(wavenumbers_cm)
    order = np.argsort(wavelengths_nm)
    return sample_ids, wavelengths_nm[order], spectra[:, order]


def _read_targets_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = list(TARGET_COLUMNS.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing target columns {missing}")
    targets = frame[required].apply(pd.to_numeric, errors="coerce")
    targets.columns = list(TARGET_COLUMNS)
    if targets.isna().any().any():
        raise ValueError(f"{path}: target values contain NaN")
    return targets.reset_index(drop=True)


def _load_oil_group(group: str, spectrum_file: Path, target_file: Path) -> CanonicalDataset:
    sample_ids, wavelengths_nm, spectra = _read_spectrum_csv(spectrum_file)
    targets = _read_targets_csv(target_file)
    if len(sample_ids) != len(targets):
        raise ValueError(f"{group}: {len(sample_ids)} spectra vs {len(targets)} targets")
    canonical_ids = np.asarray([f"{group}_{int(float(value)):02d}" for value in sample_ids])
    dataset = CanonicalDataset(
        name=group,
        sample_id=canonical_ids,
        wavelengths_nm=wavelengths_nm,
        spectra=spectra,
        targets=targets,
        groups=np.full(len(canonical_ids), group),
    )
    dataset.validate()
    return dataset


def load_dataset(root: str | Path = DATASET_DIR) -> CanonicalDataset:
    """Load both oil groups and merge them on a common wavelength grid in nm."""
    root = Path(root)
    parts = [
        _load_oil_group("oil1", root / "原油1_光谱.csv", root / "原油1_属性.csv"),
        _load_oil_group("oil2", root / "原油2_光谱.csv", root / "原油2_属性.csv"),
    ]
    return merge_datasets(
        parts,
        name="crude_oil_private",
        metadata={
            "dataset": "crude_oil_private",
            "spectral_axis": "wavelength_nm",
            "source_spectral_axis": "wavenumber_cm-1",
            "target_units": TARGET_UNITS,
            "resampling": "linear interpolation to the coarsest native wavelength grid over overlap",
        },
    )
