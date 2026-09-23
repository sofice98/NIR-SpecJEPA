"""Matched masked-autoencoder objective using the shared encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..shared.backbone import SpectralPatchTransformer, SpectralUNet


class SpectralMAE(nn.Module):
    def __init__(self, encoder: SpectralPatchTransformer | SpectralUNet, decoder_hidden_dim: int = 64) -> None:
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

    def forward(self, spectra: torch.Tensor, patch_mask: torch.Tensor):
        masks = patch_mask.unsqueeze(0) if patch_mask.ndim == 2 else patch_mask
        if masks.ndim != 3:
            raise ValueError("patch_mask must have shape [batch, patches] or [views, batch, patches]")
        if spectra.ndim == 3:
            targets = [self.encoder.patchify(view) for view in spectra]
        else:
            targets = [self.encoder.patchify(spectra)] * len(masks)
        losses, embeddings = [], []
        for view_index, mask in enumerate(masks):
            view = spectra[view_index] if spectra.ndim == 3 else spectra
            encoded, pooled = self.encoder(view, mask)
            if callable(getattr(self.encoder, "reconstruct", None)):
                # The mask is [batch, n_patches], so reconstruction must use
                # the corresponding 2-D view [batch, points], not the stacked
                # multi-view tensor [views, batch, points].
                reconstructed = self.encoder.reconstruct(view, mask)
            else:
                reconstructed = self.decoder(encoded)
            if reconstructed.shape != targets[view_index].shape:
                raise ValueError(
                    "MAE reconstruction shape mismatch: "
                    f"got {tuple(reconstructed.shape)}, "
                    f"expected {tuple(targets[view_index].shape)}"
                )
            losses.append(F.mse_loss(reconstructed[mask], targets[view_index][mask]))
            embeddings.append(pooled)
        loss = torch.stack(losses).mean()
        return {
            "loss": loss,
            "reconstruction_mse": loss,
            "embedding": torch.stack(embeddings).mean(dim=0),
        }
