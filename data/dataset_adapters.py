"""Dataset-independent canonical representation and merge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class CanonicalDataset:
    """A spectral dataset expressed on an increasing wavelength axis in nm."""

    name: str
    sample_id: np.ndarray
    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    targets: pd.DataFrame
    groups: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        sample_count = len(self.sample_id)
        if self.spectra.ndim != 2:
            raise ValueError(f"{self.name}: spectra must be 2D")
        if self.spectra.shape[0] != sample_count:
            raise ValueError(f"{self.name}: sample/spectrum count mismatch")
        if self.spectra.shape[1] != len(self.wavelengths_nm):
            raise ValueError(f"{self.name}: wavelength/spectrum width mismatch")
        if len(self.targets) != sample_count or len(self.groups) != sample_count:
            raise ValueError(f"{self.name}: sample/target/group count mismatch")
        if not np.all(np.isfinite(self.wavelengths_nm)) or not np.all(
            np.isfinite(self.spectra)
        ):
            raise ValueError(f"{self.name}: spectral data contain non-finite values")
        if np.any(self.wavelengths_nm <= 0) or not np.all(np.diff(self.wavelengths_nm) > 0):
            raise ValueError(f"{self.name}: wavelengths_nm must be positive and increasing")
        if self.targets.index.tolist() != list(range(sample_count)):
            raise ValueError(f"{self.name}: target row index is not canonical")


def merge_datasets(
    datasets: Iterable[CanonicalDataset],
    *,
    name: str = "pooled",
    grid_nm: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalDataset:
    """Resample compatible subsets to one wavelength grid and concatenate them."""
    parts = list(datasets)
    if not parts:
        raise ValueError("At least one dataset is required")
    for dataset in parts:
        dataset.validate()
    target_columns = parts[0].targets.columns.tolist()
    if any(dataset.targets.columns.tolist() != target_columns for dataset in parts[1:]):
        raise ValueError("Datasets being merged must have identical target columns")

    from preprocessing.spectral import resample_to_common_grid

    wavelengths_nm, resampled = resample_to_common_grid(parts, grid_nm=grid_nm)
    merged = CanonicalDataset(
        name=name,
        sample_id=np.concatenate([dataset.sample_id for dataset in parts]),
        wavelengths_nm=wavelengths_nm,
        spectra=np.concatenate(resampled, axis=0),
        targets=pd.concat([dataset.targets for dataset in parts], ignore_index=True),
        groups=np.concatenate([dataset.groups for dataset in parts]),
        metadata=dict(metadata or {}),
    )
    merged.validate()
    return merged
