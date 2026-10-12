"""Construction and parameter accounting for E1 prediction paths."""
from __future__ import annotations
import numpy as np
from preprocessing.spectral import FoldSpectralStandardizer
def parameter_count(model: object) -> int:
    """统计 PyTorch 模型中可访问的参数总数。"""
    return int(sum(p.numel() for p in model.parameters())) if hasattr(model, "parameters") else 0


def fitted_array_parameter_count(*values: object) -> int:
    """Count numeric values retained by fitted preprocessing at prediction time."""
    return int(sum(value.size for value in values if isinstance(value, np.ndarray)))


def preprocessing_parameter_count(
    method: str,
    standardizer: FoldSpectralStandardizer,
    input_length: int,
) -> int:
    """Count fitted preprocessing state used to transform an unseen spectrum."""
    total = fitted_array_parameter_count(standardizer.mean_, standardizer.scale_)
    if method.upper() == "MSC":
        # preprocess_pair fits the MSC reference as the training-fold mean spectrum.
        total += int(input_length)
    return total


def ssl_prediction_parameter_count(
    encoder: object,
    experiment_name: str,
    pretrained_model_params: int,
) -> int:
    """Count only SSL parameters traversed by the experiment's prediction path."""
    from models.shared.backbone import SpectralPatchTransformer, SpectralUNet

    if experiment_name == "mae_preprocess":
        # Reconstruction uses the MAE decoder (or the U-Net reconstruction path),
        # but point masking is already materialized in the input at inference.
        masking_only = int(encoder.mask_token.numel()) if isinstance(encoder, SpectralPatchTransformer) else 0
        if isinstance(encoder, SpectralUNet):
            masking_only = int(encoder.mask_value.numel())
        return int(pretrained_model_params - masking_only)
    if isinstance(encoder, SpectralUNet):
        return sum(parameter_count(module) for module in (encoder.enc1, encoder.enc2, encoder.bottleneck))
    if isinstance(encoder, SpectralPatchTransformer):
        return int(parameter_count(encoder) - encoder.mask_token.numel())
    return parameter_count(encoder)


def build_ssl_encoder(
    backbone: str,
    input_length: int,
    patch_size: int,
    embedding_dim: int,
    input_channels: int = 2,
) -> object:
    """Build the selected shared spectral encoder with the repository defaults."""
    from models.shared.backbone import SpectralPatchTransformer, SpectralUNet

    kwargs = {
        "input_length": input_length,
        "patch_size": patch_size,
        "embedding_dim": embedding_dim,
        "dropout": 0.0,
        "input_channels": input_channels,
    }
    if backbone == "unet":
        return SpectralUNet(**kwargs, depth=2, heads=2, ffn_dim=128)
    if backbone == "patch_transformer":
        return SpectralPatchTransformer(**kwargs, depth=4, heads=4, ffn_dim=embedding_dim * 2)
    raise ValueError(f"Unsupported SSL backbone: {backbone}")


def sklearn_parameter_count(model: object) -> int:
    """Count learned numeric state, including all Random Forest tree arrays."""
    if model is None:
        return 0
    if not hasattr(model, "__dict__"):
        return 0
    total = 0
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += value.size
        elif isinstance(value, list):
            total += sum(sklearn_parameter_count(item) for item in value)
        elif hasattr(value, "tree_"):
            tree = value.tree_
            total += sum(getattr(tree, name).size for name in ("children_left", "children_right", "feature", "threshold", "value"))
    return int(total)
