"""Shared neural and regression components used by multiple methods."""

from .regressors import RidgeRegressor

try:
    from .backbone import SpectralPatchTransformer, SpectralUNet
    from .masking import SpectralMasker, point_mask_to_patch_mask, random_block_mask, random_point_mask
    __all__ = [
        "SpectralPatchTransformer", "SpectralUNet", "SpectralMasker",
        "random_point_mask", "point_mask_to_patch_mask", "random_block_mask", "RidgeRegressor",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    __all__ = ["RidgeRegressor"]
