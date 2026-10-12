"""Configurable point-level masking for self-supervised spectral training."""

from __future__ import annotations

import torch
from torch import nn


def random_patch_mask(
    batch_size: int,
    n_patches: int,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Select whole patches uniformly without replacement for each spectrum.

    Mask round(n_patches * mask_ratio) tokens, with at least one masked and
    one visible patch. No separate block length is used.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie in (0, 1)")
    if n_patches < 2 or batch_size < 1:
        raise ValueError("patch masking requires at least two patches and one sample")
    n_masked = max(1, min(n_patches - 1, round(n_patches * mask_ratio)))
    scores = torch.rand(batch_size, n_patches, generator=generator, device=device)
    selected = scores.topk(n_masked, dim=1).indices
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    return mask.scatter_(1, selected, True)


def random_point_mask(
    batch_size: int,
    n_points: int,
    mask_ratio: float,
    block_length: int = 1,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate exact-rate point masks from randomly placed contiguous blocks.

    ``mask_ratio`` controls the total number of masked spectral points and
    ``block_length`` controls the nominal length of each contiguous block.
    The final block can be shorter when the requested point count is not an
    exact multiple of ``block_length``.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie in (0, 1)")
    if n_points < 1 or block_length < 1:
        raise ValueError("n_points and block_length must be positive")
    target = min(n_points, round(n_points * mask_ratio))
    mask = torch.zeros(batch_size, n_points, dtype=torch.bool, device=device)
    # 持续放置连续块，直至每条光谱达到指定掩码点数。
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            start = int(torch.randint(n_points, (1,), generator=generator).item())
            remaining = target - int(mask[row].sum())
            width = min(block_length, remaining)
            indices = (torch.arange(width, device=device) + start) % n_points
            mask[row, indices] = True
    return mask


def apply_point_mask(spectra: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
    """Zero masked spectral points without converting the mask to patches."""
    if spectra.ndim != 2 or point_mask.shape != spectra.shape:
        raise ValueError("spectra and point_mask must both have shape [batch, points]")
    return spectra.masked_fill(point_mask, 0.0)


def add_mask_indicator(spectra: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
    """Return value and explicit binary mask channels as ``[batch, 2, points]``."""
    if spectra.ndim != 2 or point_mask.shape != spectra.shape:
        raise ValueError("spectra and point_mask must both have shape [batch, points]")
    masked = apply_point_mask(spectra, point_mask)
    return torch.stack([masked, point_mask.to(dtype=spectra.dtype)], dim=1)


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
        """配置独立点掩码或连续块掩码。"""
        super().__init__()
        if mode not in {"point", "block"}:
            raise ValueError("mode must be 'point' or 'block'")
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.block_length = 1 if mode == "point" else block_length

    def forward(self, spectra: torch.Tensor, *, generator: torch.Generator | None = None):
        """返回零填充后的光谱及对应的布尔点掩码。"""
        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [batch, points]")
        point_mask = random_point_mask(
            spectra.shape[0], spectra.shape[1], self.mask_ratio, self.block_length,
            generator=generator, device=spectra.device,
        )
        return apply_point_mask(spectra, point_mask), point_mask


def random_block_mask(
    batch_size: int,
    n_patches: int,
    mask_ratio: float,
    block_patches: int = 4,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """在补丁轴随机放置连续掩码块，保持目标掩码数量。"""
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


def random_spectral_block_mask(
    batch_size: int,
    n_points: int,
    patch_size: int,
    n_patches: int,
    mask_ratio: float,
    block_length: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Legacy helper returning a patch mask from a point-level block mask.

    E1-a no longer calls this function: its training path uses
    :func:`random_point_mask` directly and keeps masking independent of
    patchification. This wrapper remains for older callers and tests.
    """
    point_mask = random_point_mask(
        batch_size, n_points, mask_ratio, block_length,
        generator=generator, device=device,
    )
    return point_mask_to_patch_mask(point_mask, patch_size, n_patches)
