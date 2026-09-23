"""Common frozen-representation regressors."""

from __future__ import annotations

import numpy as np
try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # classical baselines do not require PyTorch
    torch = None
    class _NN:
        class Module:
            pass
    nn = _NN()


class RidgeRegressor:
    """Multi-target ridge with an unpenalized intercept."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.mean_x_: np.ndarray | None = None
        self.mean_y_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x, y = np.asarray(x, float), np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        self.mean_x_, self.mean_y_ = x.mean(0), y.mean(0)
        xc, yc = x - self.mean_x_, y - self.mean_y_
        gram = xc.T @ xc + self.alpha * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, xc.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_x_ is None or self.mean_y_ is None:
            raise RuntimeError("Fit regressor before predict")
        return (np.asarray(x, float) - self.mean_x_) @ self.coef_ + self.mean_y_


class EncoderMLPRegressor(nn.Module):
    """End-to-end spectral regressor with an unfrozen encoder and two linear layers."""

    def __init__(
        self,
        encoder: nn.Module,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for EncoderMLPRegressor")
        super().__init__()
        if output_dim < 1 or hidden_dim < 1:
            raise ValueError("output_dim and hidden_dim must be positive")
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        _, pooled = self.encoder(spectra, None)
        return self.head(pooled)
