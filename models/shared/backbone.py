"""Shared one-dimensional patch Transformer encoder."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    """生成固定的正弦余弦位置编码。"""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding


class SpectralPatchTransformer(nn.Module):
    """One-dimensional Vision Transformer for spectra.

    A strided one-dimensional convolution creates non-overlapping spectral
    patches.  A learnable CLS token and positional embedding are then processed
    by a pre-normalized Transformer encoder. MAE and LeJEPA share this module.
    ``patch_mask`` replaces projected tokens with a learned mask token before
    positional embeddings and attention. E1C uses this interface; other
    experiments can also mask spectral points before calling the encoder.
    """

    def __init__(
        self,
        input_length: int,
        patch_size: int = 6,
        embedding_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        input_channels: int = 1,
    ) -> None:
        """建立补丁投影、位置嵌入和 Transformer 编码层。"""
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_length / patch_size)
        self.padded_length = self.n_patches * patch_size
        self.embedding_dim = embedding_dim
        if input_channels not in (1, 2):
            raise ValueError("input_channels must be 1 or 2")
        self.input_channels = input_channels
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        self.patch_projection = nn.Conv1d(
            input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embedding_dim)
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
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def patchify(self, spectra: torch.Tensor) -> torch.Tensor:
        """复制填充末端后，将一维光谱切成等长补丁。"""
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
        """返回补丁特征与 CLS 池化表示。"""
        if spectra.ndim == 2:
            if self.input_channels == 2:
                spectra = torch.stack([spectra, torch.zeros_like(spectra)], dim=1)
            else:
                spectra = spectra.unsqueeze(1)
        if spectra.ndim != 3 or spectra.shape[1] != self.input_channels or spectra.shape[2] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_channels}, {self.input_length}] input")
        if self.padded_length > self.input_length:
            spectra = F.pad(spectra, (0, self.padded_length - self.input_length), mode="replicate")
        tokens = self.patch_projection(spectra).transpose(1, 2)
        # Whole-token replacement preserves positions and the sequence length.
        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2]:
                raise ValueError("patch_mask must have shape [batch, n_patches]")
            tokens = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        cls = self.cls_token.expand(spectra.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.norm(self.transformer(sequence + self.position_embedding.to(sequence.dtype)))
        return encoded[:, 1:], encoded[:, 0]


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
        input_channels: int = 1,
    ) -> None:
        """建立共享的卷积编码器和用于重建的解码器。"""
        super().__init__()
        del depth, heads, ffn_dim  # retained in the factory for fair CLI comparisons
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_length / patch_size)
        self.padded_length = self.n_patches * patch_size
        self.embedding_dim = embedding_dim
        if input_channels not in (1, 2):
            raise ValueError("input_channels must be 1 or 2")
        self.input_channels = input_channels
        base = max(8, embedding_dim // 2)
        self.enc1 = nn.Sequential(nn.Conv1d(input_channels, base, 3, padding=1), nn.GELU(), nn.Dropout(dropout))
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
        """复制填充末端后，将一维光谱切成等长补丁。"""
        if spectra.ndim != 2 or spectra.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] spectra")
        if self.padded_length > self.input_length:
            spectra = F.pad(spectra, (0, self.padded_length - self.input_length), mode="replicate")
        return spectra.reshape(spectra.shape[0], self.n_patches, self.patch_size)

    def _masked_input(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None) -> torch.Tensor:
        """整理输入通道，并在提供补丁掩码时替换对应片段。"""
        if spectra.ndim == 2:
            if self.input_channels == 2:
                spectra = torch.stack([spectra, torch.zeros_like(spectra)], dim=1)
            else:
                spectra = spectra.unsqueeze(1)
        if spectra.ndim != 3 or spectra.shape[1] != self.input_channels or spectra.shape[2] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_channels}, {self.input_length}] input")
        if patch_mask is None:
            return spectra
        if patch_mask.shape != (spectra.shape[0], self.n_patches):
            raise ValueError("patch_mask must have shape [batch, n_patches]")
        if self.padded_length > self.input_length:
            spectra = F.pad(spectra, (0, self.padded_length - self.input_length), mode="replicate")
        patches = spectra.reshape(spectra.shape[0], self.input_channels, self.n_patches, self.patch_size).clone()
        patches = torch.where(patch_mask[:, None, :, None], self.mask_value, patches)
        return patches.reshape(spectra.shape[0], self.input_channels, self.padded_length)[:, :, : self.input_length]

    def _encode(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None):
        """计算浅层跳连特征和瓶颈特征。"""
        x = self._masked_input(spectra, patch_mask)
        skip = self.enc1(x)
        latent = self.bottleneck(self.enc2(skip))
        return skip, latent

    def forward(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None):
        """将瓶颈特征映射为补丁序列及池化表示。"""
        skip, latent = self._encode(spectra, patch_mask)
        tokens = F.interpolate(latent, size=self.n_patches, mode="linear", align_corners=False).transpose(1, 2)
        if patch_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            visible = (~patch_mask).unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
        return tokens, pooled

    def reconstruct(self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """经跳连解码并返回按补丁排列的重建光谱。"""
        skip, latent = self._encode(spectra, patch_mask)
        up = self.up(latent)
        up = F.interpolate(up, size=skip.shape[-1], mode="linear", align_corners=False)
        decoded = self.dec(torch.cat([up, skip], dim=1))
        reconstructed = self.output(decoded).squeeze(1)
        reconstructed = F.interpolate(reconstructed.unsqueeze(1), size=self.padded_length,
                                      mode="linear", align_corners=False).squeeze(1)
        return reconstructed.reshape(spectra.shape[0], self.n_patches, self.patch_size)
