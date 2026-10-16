"""Small-data, single-channel Laya adaptation for E1D."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from x_transformers import Encoder

from .sigreg import SIGReg


def spectral_block_mask(
    batch_size: int,
    n_patches: int,
    ratio: float = 0.6,
    min_block: int = 5,
    max_block: int = 10,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample non-wrapping contiguous blocks with an exact total patch count.

    The last block or blocks on very short sequences may be shorter than
    ``min_block``. Such patches are identified by the returned boolean mask.
    """
    if batch_size < 1 or n_patches < 2:
        raise ValueError("masking needs a batch and at least two patches")
    if not 0 < ratio < 1 or min_block < 1 or max_block < min_block:
        raise ValueError("invalid mask ratio or block length")
    target = max(1, min(n_patches - 1, round(n_patches * ratio)))
    masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
    for row in range(batch_size):
        block_count = max(1, math.ceil(target / max_block))
        if target >= min_block and target + block_count - 1 > n_patches:
            raise ValueError("requested mask cannot fit as separated contiguous blocks")
        lengths: list[int] = []
        remaining = target
        for block in range(block_count):
            blocks_left = block_count - block - 1
            low = max(1 if remaining < min_block else min_block, remaining - max_block * blocks_left)
            high = min(max_block, remaining - (1 if remaining < min_block else min_block) * blocks_left)
            width = low if low == high else int(torch.randint(low, high + 1, (1,), generator=generator))
            lengths.append(width)
            remaining -= width
        gap_total = n_patches - target
        internal_gaps = [1] * (block_count - 1)
        extra = gap_total - sum(internal_gaps)
        gaps = [0] + internal_gaps + [0]
        for index in torch.randperm(block_count + 1, generator=generator).tolist():
            take = int(torch.randint(extra + 1, (1,), generator=generator)) if extra else 0
            gaps[index] += take
            extra -= take
        gaps[-1] += extra
        start = gaps[0]
        for width, gap in zip(lengths, gaps[1:]):
            masks[row, start : start + width] = True
            start += width + gap
    return masks.to(device=device)


class SpectralLayaEncoder(nn.Module):
    """Patch projection and x-transformers encoder shared by full/context views."""

    def __init__(
        self,
        input_length: int,
        patch_size: int = 6,
        embedding_dim: int = 96,
        depth: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if input_length < 2 or patch_size < 1 or embedding_dim % heads:
            raise ValueError("invalid encoder dimensions")
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_length / patch_size)
        if self.n_patches < 2:
            raise ValueError("patch_size leaves fewer than two patches")
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.patch_projection = nn.Conv1d(1, embedding_dim, patch_size, stride=patch_size)
        self.transformer = Encoder(dim=embedding_dim, depth=depth, heads=heads, rotary_pos_emb=True)
        self.norm = nn.LayerNorm(embedding_dim)
        valid = torch.full((self.n_patches,), patch_size, dtype=torch.float32)
        valid[-1] = input_length - (self.n_patches - 1) * patch_size
        self.register_buffer("valid_fraction", valid / patch_size)

    def forward(
        self, spectra: torch.Tensor, patch_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spectra.ndim != 2 or spectra.shape[1] != self.input_length:
            raise ValueError(f"expected spectra [batch, {self.input_length}]")
        if patch_mask is not None and (patch_mask.shape != (len(spectra), self.n_patches) or patch_mask.dtype != torch.bool):
            raise ValueError("patch_mask must be bool [batch, patches]")
        padded = F.pad(spectra[:, None, :], (0, self.n_patches * self.patch_size - self.input_length), mode="replicate")
        tokens = self.patch_projection(padded).transpose(1, 2)
        if patch_mask is not None:
            # Patch projection precedes masking, but no masked value can reach
            # the context transformer through token values or attention keys.
            tokens = tokens.masked_fill(patch_mask.unsqueeze(-1), 0.0)
        encoded = self.norm(self.transformer(tokens, mask=None if patch_mask is None else ~patch_mask))
        weights = self.valid_fraction.to(dtype=encoded.dtype)
        if patch_mask is not None:
            weights = weights.unsqueeze(0) * (~patch_mask).to(encoded.dtype)
        summary = (encoded * weights[..., None]).sum(dim=1) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return encoded, summary


class SpectralLaya(nn.Module):
    """Masked latent prediction plus SIGReg with user-specified convex weighting."""

    def __init__(
        self,
        encoder: SpectralLayaEncoder,
        projection_dim: int = 48,
        projector_hidden_dim: int = 192,
        predictor_depth: int = 2,
        predictor_heads: int = 4,
        sigreg_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 <= sigreg_weight <= 1 or projection_dim % predictor_heads:
            raise ValueError("invalid SIGReg weight or predictor dimensions")
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.embedding_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projection_dim),
        )
        self.predictor = Encoder(dim=projection_dim, depth=predictor_depth, heads=predictor_heads, rotary_pos_emb=True)
        self.predictor_norm = nn.LayerNorm(projection_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, projection_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.sigreg = SIGReg()
        self.sigreg_weight = sigreg_weight
        self.projection_dim = projection_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.predictor_depth = predictor_depth
        self.predictor_heads = predictor_heads

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        shape = tokens.shape
        return self.projector(tokens.reshape(-1, shape[-1])).reshape(*shape[:-1], self.projection_dim)

    def forward(self, spectra: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if mask.shape != (len(spectra), self.encoder.n_patches) or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [batch, patches]")
        if not bool(mask.any()) or bool(mask.all(dim=1).any()):
            raise ValueError("each spectrum needs masked and visible patches")
        z, z_cls = self.encoder(spectra)
        targets = self.project(z.detach())
        p_cls = self.project(z_cls)
        z_ctx, _ = self.encoder(spectra, mask)
        p_ctx = self.project(z_ctx)
        predicted_input = torch.where(mask[..., None], self.mask_token.expand_as(p_ctx), p_ctx)
        predicted = self.predictor_norm(self.predictor(predicted_input))
        # Weight the short padded final patch by its number of real points.
        weights = mask.to(predicted.dtype) * self.encoder.valid_fraction
        per_patch_mse = (predicted - targets).square().mean(dim=-1)
        mse = (per_patch_mse * weights).sum() / weights.sum().clamp_min(1.0)
        sigreg = self.sigreg(p_cls.float().unsqueeze(0))
        loss = (1 - self.sigreg_weight) * mse + self.sigreg_weight * sigreg
        return {"loss": loss, "invariance_mse": mse, "sigreg": sigreg, "embedding": z_cls, "targets": targets, "prediction": predicted}
