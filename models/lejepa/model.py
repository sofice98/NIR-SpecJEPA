"""LeJEPA alignment for masked and clean spectral views."""

from __future__ import annotations

import torch
from torch import nn

from ..shared.backbone import SpectralPatchTransformer, SpectralUNet
from ..shared.masking import add_mask_indicator, apply_point_mask
from .sigreg import SIGReg


class SpectralLeJEPA(nn.Module):
    """对齐掩码与完整光谱的投影表示，并施加 SIGReg 正则。"""

    def __init__(
        self,
        encoder: SpectralPatchTransformer | SpectralUNet,
        projection_dim: int = 16,
        projector_hidden_dim: int = 128,
        sigreg_weight: float = 0.02,
        sigreg_knots: int = 17,
        sigreg_projections: int = 256,
        mask_mode: str = "point",
    ) -> None:
        """包装共享编码器并配置投影头与正则权重。"""
        super().__init__()
        self.encoder = encoder
        if mask_mode not in {"point", "patch"}:
            raise ValueError("mask_mode must be 'point' or 'patch'")
        if mask_mode == "patch" and (
            not isinstance(encoder, SpectralPatchTransformer)
            or encoder.input_channels != 1
        ):
            raise ValueError(
                "patch masking requires a single-channel SpectralPatchTransformer"
            )
        self.mask_mode = mask_mode
        # Match the reference MLP's Linear/BatchNorm/ReLU hidden blocks.
        self.projector = nn.Sequential(
            nn.Linear(encoder.embedding_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projection_dim),
        )
        self.sigreg = SIGReg(knots=sigreg_knots, projections=sigreg_projections)
        self.sigreg_weight = sigreg_weight

    def _encode_patch_views(
        self, spectra: torch.Tensor, patch_mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace patch tokens for the masked view; keep the clean view intact."""
        expected_shape = (len(spectra), self.encoder.n_patches)
        if patch_mask.shape != expected_shape or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be bool with shape [batch, n_patches]")
        views = torch.cat([spectra, spectra], dim=0)
        view_masks = torch.cat([patch_mask, torch.zeros_like(patch_mask)], dim=0)
        return self.encoder(views, view_masks)[1]

    def _encode_point_views(
        self, spectra: torch.Tensor, point_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build masked input values and optional indicator channels."""
        if point_mask.ndim == 3:
            point_mask = point_mask[0]
        if point_mask.ndim == 2 and point_mask.shape[1] == self.encoder.n_patches:
            point_mask = point_mask.repeat_interleave(self.encoder.patch_size, dim=1)[
                :, : spectra.shape[1]
            ]
        if point_mask.shape != spectra.shape:
            raise ValueError("spectra and point_mask must have shape [batch, points]")
        if getattr(self.encoder, "input_channels", 1) == 2:
            masked = add_mask_indicator(spectra, point_mask)
        else:
            masked = apply_point_mask(spectra, point_mask)
        clean = spectra
        if getattr(self.encoder, "input_channels", 1) == 2:
            clean = add_mask_indicator(spectra, torch.zeros_like(point_mask))
        return self.encoder(torch.cat([masked, clean], dim=0), None)[1]

    def forward(self, spectra: torch.Tensor, point_mask: torch.Tensor):
        """Align masked and clean views; both contribute encoder gradients."""
        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [batch, points]")
        if self.mask_mode == "patch":
            flat_embeddings = self._encode_patch_views(spectra, point_mask)
        else:
            flat_embeddings = self._encode_point_views(spectra, point_mask)
        # Encode/project all views together, including shared BatchNorm statistics.
        embeddings = flat_embeddings.reshape(2, len(spectra), -1)
        projected = self.projector(flat_embeddings).reshape(2, len(spectra), -1)
        invariance = (projected.mean(dim=0) - projected).square().mean()
        sigreg = self.sigreg(projected)
        loss = (1.0 - self.sigreg_weight) * invariance + self.sigreg_weight * sigreg
        # Diagnostics for detecting projector collapse.  These statistics are
        # detached so they never add a gradient path or change optimization.
        z = projected.detach().reshape(-1, projected.shape[-1])
        projection_mean = z.mean()
        projection_std = z.std(dim=0).mean()
        projection_norm = z.norm(dim=1).mean()
        return {
            "loss": loss,
            "invariance_mse": invariance,
            "sigreg": sigreg,
            "projection_mean": projection_mean,
            "projection_std": projection_std,
            "projection_norm": projection_norm,
            "embedding": embeddings[0],
            "view_embeddings": embeddings,
            "projections": projected,
        }
