"""Shared neural and regression components used by multiple methods."""

from .regressors import RidgeRegressor

try:
    from .backbone import SpectralPatchTransformer, SpectralUNet
    from .masking import add_mask_indicator, apply_point_mask, SpectralMasker, point_mask_to_patch_mask, random_block_mask, random_point_mask
    from .airpls import AirPLSConfig, airpls_baseline, airpls_correct, airpls_correct_batch
    __all__ = [
        "SpectralPatchTransformer", "SpectralUNet", "SpectralMasker",
        "apply_point_mask", "add_mask_indicator", "random_point_mask", "point_mask_to_patch_mask", "random_block_mask",
        "AirPLSConfig", "airpls_baseline", "airpls_correct", "airpls_correct_batch", "RidgeRegressor",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    __all__ = ["RidgeRegressor"]
