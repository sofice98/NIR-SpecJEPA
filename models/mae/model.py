"""Matched masked-autoencoder objective using point-zeroed inputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..shared.backbone import SpectralPatchTransformer, SpectralUNet
from ..shared.masking import add_mask_indicator, apply_point_mask


class SpectralMAE(nn.Module):
    """通过点级掩码重建完整光谱的自监督模型。"""

    def __init__(self, encoder: SpectralPatchTransformer | SpectralUNet, decoder_hidden_dim: int = 64) -> None:
        """包装共享编码器，并在需要时建立补丁解码器。"""
        super().__init__()
        self.encoder = encoder
        # Use the encoder's reconstruction path whenever it provides one.
        # This supports compatible custom U-Net encoders (for example the
        # exact dense U-Net used by E1-a) without requiring them to inherit the
        # repository's SpectralUNet class.
        self.decoder = None if callable(getattr(encoder, "reconstruct", None)) else nn.Sequential(
            nn.Linear(encoder.embedding_dim, decoder_hidden_dim), nn.GELU(),
            nn.Linear(decoder_hidden_dim, encoder.patch_size),
        )

    def reconstruct(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct patch values through either the backbone or MAE decoder."""
        if callable(getattr(self.encoder, "reconstruct", None)):
            return self.encoder.reconstruct(spectra, patch_mask)
        encoded, _ = self.encoder(spectra, patch_mask)
        return self.decoder(encoded)

    def forward(self, spectra: torch.Tensor, point_mask: torch.Tensor):
        """Mask points with zeros and reconstruct the complete clean curve."""
        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [batch, points]")
        if point_mask.ndim == 2 and point_mask.shape[1] == self.encoder.n_patches:
            point_mask = point_mask.repeat_interleave(self.encoder.patch_size, dim=1)[:, : spectra.shape[1]]
        if point_mask.shape != spectra.shape:
            raise ValueError("spectra and point_mask must have shape [batch, points]")
        if getattr(self.encoder, "input_channels", 1) == 2:
            masked_spectra = add_mask_indicator(spectra, point_mask)
        else:
            masked_spectra = apply_point_mask(spectra, point_mask)
        _, pooled = self.encoder(masked_spectra, None)
        reconstructed = self.reconstruct(masked_spectra, None)
        reconstructed_curve = reconstructed.reshape(reconstructed.shape[0], -1)[:, : spectra.shape[1]]
        if reconstructed_curve.shape != spectra.shape:
            raise ValueError(
                "MAE reconstruction shape mismatch: "
                f"got {tuple(reconstructed_curve.shape)}, expected {tuple(spectra.shape)}"
            )
        loss = F.mse_loss(reconstructed_curve, spectra)
        return {
            "loss": loss,
            "reconstruction_mse": loss,
            "embedding": pooled,
        }
