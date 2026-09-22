"""Configurable point-level masking for self-supervised spectral training."""

from __future__ import annotations

import torch
from torch import nn


def random_point_mask(
    batch_size: int,
    n_points: int,
    mask_ratio: float,
    block_length: int = 1,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate exact-rate point masks using single points or contiguous blocks."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie in (0, 1)")
    if n_points < 1 or block_length < 1:
        raise ValueError("n_points and block_length must be positive")
    target = min(n_points, round(n_points * mask_ratio))
    mask = torch.zeros(batch_size, n_points, dtype=torch.bool, device=device)
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            start = int(torch.randint(n_points, (1,), generator=generator).item())
            remaining = target - int(mask[row].sum())
            width = min(block_length, remaining)
            indices = (torch.arange(width, device=device) + start) % n_points
            mask[row, indices] = True
    return mask


def point_mask_to_patch_mask(point_mask: torch.Tensor, patch_size: int, n_patches: int) -> torch.Tensor:
    """Mark a patch when at least one of its NIR points is masked."""
    if point_mask.ndim != 2:
        raise ValueError("point_mask must have shape [batch, points]")
    padded = torch.nn.functional.pad(
        point_mask, (0, n_patches * patch_size - point_mask.shape[1])
    )
    return padded.reshape(point_mask.shape[0], n_patches, patch_size).any(dim=-1)


class SpectralMasker(nn.Module):
    """Apply point-level random masking and set masked NIR values to zero."""

    def __init__(self, mask_ratio: float = 0.4, mode: str = "point", block_length: int = 1):
        super().__init__()
        if mode not in {"point", "block"}:
            raise ValueError("mode must be 'point' or 'block'")
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.block_length = 1 if mode == "point" else block_length

    def forward(self, spectra: torch.Tensor, *, generator: torch.Generator | None = None):
        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [batch, points]")
        point_mask = random_point_mask(
            spectra.shape[0], spectra.shape[1], self.mask_ratio, self.block_length,
            generator=generator, device=spectra.device,
        )
        return spectra.masked_fill(point_mask, 0.0), point_mask


def random_block_mask(
    batch_size: int,
    n_patches: int,
    mask_ratio: float,
    block_patches: int = 4,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie in (0, 1)")
    target = max(1, min(n_patches - 1, round(n_patches * mask_ratio)))
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            start = int(torch.randint(n_patches, (1,), generator=generator).item())
            width = min(block_patches, target - int(mask[row].sum()))
            indices = (torch.arange(width, device=device) + start) % n_patches
            mask[row, indices] = True
    return mask
