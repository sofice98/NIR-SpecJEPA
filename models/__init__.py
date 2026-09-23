"""Matched MAE and LeJEPA models for one-dimensional spectra.

Classical baselines can be imported without installing PyTorch; neural symbols
remain available when the optional dependency is installed.
"""

try:
    from .shared.backbone import SpectralPatchTransformer, SpectralUNet
    from .mae.model import SpectralMAE
    from .lejepa.model import SpectralLeJEPA
    __all__ = ["SpectralPatchTransformer", "SpectralUNet", "SpectralMAE", "SpectralLeJEPA"]
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    __all__ = []
