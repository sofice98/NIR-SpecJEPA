"""Common-grid resampling and train-fold-only spectral transforms."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def resample_to_common_grid(datasets: Sequence, grid_nm: np.ndarray | None = None):
    """Resample all datasets to the coarsest wavelength grid on the shared interval.

    The common grid is selected from the first/coarsest native axis unless an
    explicit grid is supplied. Linear interpolation is used and extrapolation
    is forbidden.
    """
    low = max(float(dataset.wavelengths_nm.min()) for dataset in datasets)
    high = min(float(dataset.wavelengths_nm.max()) for dataset in datasets)
    if low >= high:
        raise ValueError("Spectral datasets have no overlapping wavelength interval")
    if grid_nm is None:
        reference = max(datasets, key=lambda d: np.median(np.diff(d.wavelengths_nm)))
        grid_nm = reference.wavelengths_nm[
            (reference.wavelengths_nm >= low) & (reference.wavelengths_nm <= high)
        ]
    grid_nm = np.asarray(grid_nm, dtype=float)
    if grid_nm.ndim != 1 or len(grid_nm) < 2 or np.any(np.diff(grid_nm) <= 0):
        raise ValueError("Common grid must be strictly increasing and contain >=2 points")
    if grid_nm[0] < low - 1e-6 or grid_nm[-1] > high + 1e-6:
        raise ValueError("Common grid requires extrapolation")
    arrays = [
        np.vstack([np.interp(grid_nm, dataset.wavelengths_nm, row) for row in dataset.spectra])
        for dataset in datasets
    ]
    return grid_nm, arrays


class FoldSpectralStandardizer:
    """Per-channel center/scale fitted only on a training fold."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, spectra: np.ndarray) -> "FoldSpectralStandardizer":
        values = np.asarray(spectra, dtype=float)
        if values.ndim != 2:
            raise ValueError("spectra must be 2D")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Fit standardizer before transform")
        return (np.asarray(spectra, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        return self.fit(spectra).transform(spectra)


def snv(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(spectra, dtype=float)
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return (values - mean) / np.where(scale < eps, 1.0, scale)


def msc(spectra: np.ndarray, reference: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(spectra, dtype=float)
    ref = values.mean(axis=0) if reference is None else np.asarray(reference, dtype=float)
    design = np.column_stack([np.ones(len(ref)), ref])
    corrected = np.empty_like(values)
    for i, row in enumerate(values):
        coeff, *_ = np.linalg.lstsq(design, row, rcond=None)
        corrected[i] = (row - coeff[0]) / (coeff[1] if abs(coeff[1]) > eps else 1.0)
    return corrected


def preprocess_pair(
    train_spectra: np.ndarray,
    test_spectra: np.ndarray,
    method: str = "R",
    sg_window: int = 11,
    sg_polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one selectable spectral transform to a train/test fold.

    ``MSC`` estimates its reference spectrum from the training fold only;
    all other transforms are sample-wise and can be applied independently.
    The returned arrays are not globally standardized; fold-wise centering and
    scaling should be applied afterwards when required by the experiment.
    """
    name = method.upper()
    if name not in {"R", "FD", "SD", "SNV", "MSC", "SG"}:
        raise ValueError(f"Unsupported preprocessing method: {method}")
    train = np.asarray(train_spectra, dtype=float)
    test = np.asarray(test_spectra, dtype=float)
    if name == "R":
        return train, test
    if name == "FD":
        deriv = 1
    elif name == "SD":
        deriv = 2
    elif name == "SNV":
        return snv(train), snv(test)
    elif name == "MSC":
        reference = train.mean(axis=0)
        return msc(train, reference=reference), msc(test, reference=reference)
    else:  # SG smoothing
        deriv = 0
    return (
        savitzky_golay(train, window=sg_window, polyorder=sg_polyorder, deriv=deriv),
        savitzky_golay(test, window=sg_window, polyorder=sg_polyorder, deriv=deriv),
    )


def savitzky_golay(spectra: np.ndarray, window: int = 11, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
    """Small dependency-free Savitzky-Golay implementation for baselines."""
    from scipy.signal import savgol_filter

    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and greater than polyorder")
    return savgol_filter(np.asarray(spectra, dtype=float), window, polyorder, deriv=deriv, axis=1)
