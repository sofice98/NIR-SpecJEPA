"""Adaptive iteratively reweighted penalized least-squares baseline correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirPLSConfig:
    """Numerical parameters for airPLS baseline estimation."""

    lam: float = 1.0e5
    diff_order: int = 2
    max_iter: int = 50
    asymmetry: float = 0.001
    tolerance: float = 1.0e-3


def _difference_penalty(length: int, order: int):
    """构造一阶或二阶差分矩阵作为平滑惩罚。"""
    from scipy.sparse import diags

    if length < 2:
        raise ValueError("airPLS requires at least two spectral points")
    if order not in (1, 2):
        raise ValueError("airPLS diff_order must be 1 or 2")
    if order == 1:
        diagonals = [np.ones(length - 1), -np.ones(length - 1)]
        return diags(diagonals, [0, 1], shape=(length - 1, length), format="csc")
    diagonals = [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)]
    return diags(diagonals, [0, 1, 2], shape=(length - 2, length), format="csc")


def airpls_baseline(spectrum: np.ndarray, config: AirPLSConfig | None = None) -> np.ndarray:
    """Estimate a smooth lower baseline for one finite 1-D spectrum."""
    cfg = AirPLSConfig() if config is None else config
    y = np.asarray(spectrum, dtype=float)
    if y.ndim != 1:
        raise ValueError("spectrum must be one-dimensional")
    if not np.isfinite(y).all():
        raise ValueError("spectrum must contain only finite values")
    if cfg.lam <= 0 or cfg.max_iter < 1 or not 0 < cfg.asymmetry < 1:
        raise ValueError("invalid airPLS parameters")

    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    length = len(y)
    if length < 3:
        return np.full_like(y, y.min())
    difference = _difference_penalty(length, cfg.diff_order)
    penalty = cfg.lam * (difference.T @ difference)
    weights = np.ones(length, dtype=float)
    baseline = np.zeros_like(y)
    scale = max(float(np.abs(y).sum()), np.finfo(float).eps)
    # 交替求解平滑基线并提高负残差权重，直至权重或残差收敛。
    for iteration in range(1, cfg.max_iter + 1):
        system = diags(weights, 0, shape=(length, length), format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * y), dtype=float)
        residual = y - baseline
        negative = residual[residual < 0]
        if negative.size == 0:
            break
        weight_scale = max(float(np.abs(negative).sum()), np.finfo(float).eps)
        updated = np.full(length, cfg.asymmetry, dtype=float)
        updated[residual < 0] = np.exp(iteration * np.abs(residual[residual < 0]) / weight_scale)
        updated = np.clip(updated, cfg.asymmetry, 1.0e12)
        if np.linalg.norm(updated - weights) / max(np.linalg.norm(weights), 1.0) < cfg.tolerance:
            weights = updated
            break
        weights = updated
        if float(np.abs(residual).sum()) / scale < cfg.tolerance:
            break
    return baseline


def airpls_correct(spectrum: np.ndarray, config: AirPLSConfig | None = None) -> np.ndarray:
    """Subtract the airPLS baseline from one spectrum."""
    y = np.asarray(spectrum, dtype=float)
    return y - airpls_baseline(y, config)


def airpls_correct_batch(spectra: np.ndarray, config: AirPLSConfig | None = None) -> np.ndarray:
    """Apply airPLS independently to every row of a ``[samples, points]`` array."""
    values = np.asarray(spectra, dtype=float)
    if values.ndim != 2:
        raise ValueError("spectra must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("spectra must contain only finite values")
    return np.vstack([airpls_correct(row, config) for row in values])
