"""Shared one-dimensional patch Transformer encoder."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding


class SpectralPatchTransformer(nn.Module):
    """Patchify a spectrum and encode it with a compact Transformer.

    MAE and LeJEPA share this complete module. A boolean patch mask replaces
    hidden patches with a learnable token while retaining identical positions.
    """

    def __init__(
        self,
        input_length: int,
        patch_size: int = 6,
        embedding_dim: int = 64,
        depth: int = 2,
        heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_length / patch_size)
        self.padded_length = self.n_patches * patch_size
        self.embedding_dim = embedding_dim
        self.patch_projection = nn.Linear(patch_size, embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(self.n_patches, embedding_dim).unsqueeze(0),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.ndim != 2 or spectra.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] spectra")
        if self.padded_length > self.input_length:
            spectra = F.pad(spectra, (0, self.padded_length - self.input_length), mode="replicate")
        return spectra.reshape(spectra.shape[0], self.n_patches, self.patch_size)

    def forward(
        self,
        spectra: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patchify(spectra)
        tokens = self.patch_projection(patches)
        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2]:
                raise ValueError("patch_mask must have shape [batch, n_patches]")
            tokens = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        encoded = self.norm(self.transformer(tokens + self.position_encoding.to(tokens.dtype)))
        if patch_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            visible = (~patch_mask).unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
        return encoded, pooled


class SpectralUNet(nn.Module):
    """Compact one-dimensional U-Net backbone for spectral signals.

    ``forward`` exposes the bottleneck as patch-like tokens, keeping the same
    interface as :class:`SpectralPatchTransformer`.  ``reconstruct`` uses the
    U-Net decoder and is used by MAE; LeJEPA only consumes the encoder output.
    """

    def __init__(
        self,
        input_length: int,
        patch_size: int = 6,
        embedding_dim: int = 64,
        depth: int = 2,
        heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del depth, heads, ffn_dim  # retained in the factory for fair CLI comparisons
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_length / patch_size)
        self.padded_length = self.n_patches * patch_size
        self.embedding_dim = embedding_dim
        base = max(8, embedding_dim // 2)
        self.enc1 = nn.Sequential(nn.Conv1d(1, base, 3, padding=1), nn.GELU(), nn.Dropout(dropout))
        self.enc2 = nn.Sequential(nn.Conv1d(base, embedding_dim, 4, stride=2, padding=1), nn.GELU())
        self.bottleneck = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), nn.GELU(),
        )
        self.up = nn.ConvTranspose1d(embedding_dim, base, 4, stride=2, padding=1)
        self.dec = nn.Sequential(nn.Conv1d(base * 2, base, 3, padding=1), nn.GELU())
        self.output = nn.Conv1d(base, 1, 1)
        self.mask_value = nn.Parameter(torch.zeros(1))

    def patchify(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.ndim != 2 or spectra.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] spectra")
        if self.padded_length > self.input_length:
            spectra = F.pad(spectra, (0, self.padded_length - self.input_length), mode="replicate")
        return spectra.reshape(spectra.shape[0], self.n_patches, self.patch_size)

    def _masked_input(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None) -> torch.Tensor:
        if patch_mask is None:
            return spectra
        if patch_mask.shape != (spectra.shape[0], self.n_patches):
            raise ValueError("patch_mask must have shape [batch, n_patches]")
        patches = self.patchify(spectra).clone()
        patches = torch.where(patch_mask.unsqueeze(-1), self.mask_value, patches)
        return patches.reshape(spectra.shape[0], self.padded_length)[:, : self.input_length]

    def _encode(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None):
        x = self._masked_input(spectra, patch_mask).unsqueeze(1)
        skip = self.enc1(x)
        latent = self.bottleneck(self.enc2(skip))
        return skip, latent

    def forward(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None):
        skip, latent = self._encode(spectra, patch_mask)
        tokens = F.interpolate(latent, size=self.n_patches, mode="linear", align_corners=False).transpose(1, 2)
        if patch_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            visible = (~patch_mask).unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
        return tokens, pooled

    def reconstruct(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        skip, latent = self._encode(spectra, patch_mask)
        up = self.up(latent)
        up = F.interpolate(up, size=skip.shape[-1], mode="linear", align_corners=False)
        decoded = self.dec(torch.cat([up, skip], dim=1))
        reconstructed = self.output(decoded).squeeze(1)
        reconstructed = F.interpolate(reconstructed.unsqueeze(1), size=self.padded_length,
                                      mode="linear", align_corners=False).squeeze(1)
        return reconstructed.reshape(spectra.shape[0], self.n_patches, self.patch_size)
