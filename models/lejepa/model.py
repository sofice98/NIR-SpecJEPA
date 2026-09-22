"""LeJEPA multi-view latent alignment for spectra."""

from __future__ import annotations

import torch
from torch import nn

from ..shared.backbone import SpectralPatchTransformer, SpectralUNet
from .sigreg import SIGReg


class SpectralLeJEPA(nn.Module):
    def __init__(
        self,
        encoder: SpectralPatchTransformer | SpectralUNet,
        projection_dim: int = 16,
        projector_hidden_dim: int = 128,
        sigreg_weight: float = 0.02,
        sigreg_knots: int = 17,
        sigreg_projections: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # LeJEPA predictor: the projector maps encoder representations to the
        # latent space on which invariance and SIGReg are applied.
        self.projector = nn.Sequential(
            nn.Linear(encoder.embedding_dim, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, projection_dim),
        )
        self.sigreg = SIGReg(knots=sigreg_knots, projections=sigreg_projections)
        self.sigreg_weight = sigreg_weight

    def forward(self, spectra: torch.Tensor, view_masks: torch.Tensor):
        if view_masks.ndim != 3:
            raise ValueError("view_masks must have shape [views, batch, n_patches]")
        embeddings = []
        projections = []
        for view_index, mask in enumerate(view_masks):
            view = spectra[view_index] if spectra.ndim == 3 else spectra
            _, pooled = self.encoder(view, mask)
            embeddings.append(pooled)
            projections.append(self.projector(pooled))
        projected = torch.stack(projections, dim=0)
        invariance = (projected.mean(dim=0, keepdim=True) - projected).square().mean()
        sigreg = self.sigreg(projected)
        loss = (1.0 - self.sigreg_weight) * invariance + self.sigreg_weight * sigreg
        return {
            "loss": loss,
            "invariance_mse": invariance,
            "sigreg": sigreg,
            "embedding": torch.stack(embeddings).mean(dim=0),
        }
