from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.shared.backbone import SpectralPatchTransformer
from models.baselines.classical import pca_ridge, pls, random_forest, raw_ridge, rbf_svr
from models.lejepa.model import SpectralLeJEPA
from models.mae.model import SpectralMAE
from models.shared.masking import SpectralMasker, point_mask_to_patch_mask, random_block_mask
from models.shared.regressors import EncoderMLPRegressor, RidgeRegressor
from preprocessing.spectral import preprocess_pair


def test_matched_models_forward_backward() -> None:
    torch.manual_seed(7)
    spectra = torch.randn(8, 517)
    base = SpectralPatchTransformer(input_length=517)
    initial = {key: value.clone() for key, value in base.state_dict().items()}

    mae_encoder = SpectralPatchTransformer(input_length=517)
    mae_encoder.load_state_dict(initial)
    jepa_encoder = SpectralPatchTransformer(input_length=517)
    jepa_encoder.load_state_dict(initial)
    assert all(torch.equal(a, b) for a, b in zip(mae_encoder.parameters(), jepa_encoder.parameters()))

    generator = torch.Generator().manual_seed(11)
    mask1 = random_block_mask(8, base.n_patches, 0.4, generator=generator)
    mask2 = random_block_mask(8, base.n_patches, 0.4, generator=generator)
    mae = SpectralMAE(mae_encoder)
    jepa = SpectralLeJEPA(jepa_encoder)
    mae_out = mae(spectra, mask1)
    jepa_out = jepa(spectra, torch.stack([mask1, mask2]))
    assert mae_out["embedding"].shape == (8, 64)
    assert jepa_out["embedding"].shape == (8, 64)
    assert torch.isfinite(mae_out["loss"])
    assert torch.isfinite(jepa_out["loss"])
    mae_out["loss"].backward()
    jepa_out["loss"].backward()


def test_ridge_regressor() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 8))
    y = x[:, :3] @ rng.normal(size=(3, 3))
    model = RidgeRegressor(alpha=1e-6).fit(x, y)
    assert model.predict(x).shape == (30, 3)
    assert np.mean((model.predict(x) - y) ** 2) < 1e-8


def test_encoder_mlp_regressor_updates_encoder_and_head() -> None:
    torch.manual_seed(17)
    encoder = SpectralPatchTransformer(input_length=24, patch_size=4, embedding_dim=16,
                                       depth=1, heads=2, ffn_dim=32)
    model = EncoderMLPRegressor(encoder, output_dim=3, hidden_dim=8, dropout=0.0)
    prediction = model(torch.randn(6, 24))
    assert prediction.shape == (6, 3)
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.head.parameters())


def test_spectral_masker_zeroes_points_and_supports_blocks() -> None:
    spectra = torch.ones(2, 20)
    masker = SpectralMasker(mask_ratio=0.25, mode="point")
    masked, point_mask = masker(spectra, generator=torch.Generator().manual_seed(1))
    assert int(point_mask.sum(dim=1)[0]) == 5
    assert torch.all(masked[point_mask] == 0)
    patch_mask = point_mask_to_patch_mask(point_mask, patch_size=4, n_patches=5)
    assert patch_mask.shape == (2, 5)
    block_masker = SpectralMasker(mask_ratio=0.4, mode="block", block_length=3)
    _, block_mask = block_masker(spectra, generator=torch.Generator().manual_seed(2))
    assert int(block_mask.sum(dim=1)[0]) == 8


def test_classical_baselines() -> None:
    rng = np.random.default_rng(13)
    train_x = rng.normal(size=(24, 12))
    test_x = rng.normal(size=(5, 12))
    train_y = train_x[:, :3] @ rng.normal(size=(3, 3)) + rng.normal(scale=0.05, size=(24, 3))
    folds = [
        (np.r_[0:8, 12:24], np.arange(8, 12)),
        (np.r_[0:12, 16:24], np.arange(12, 16)),
    ]
    results = [
        raw_ridge(train_x, train_y, test_x, folds, (0.01, 1.0)),
        pca_ridge(train_x, train_y, test_x, folds, (2, 4), (0.01,), seed=13),
        pls(train_x, train_y, test_x, folds, (2, 3), max_iter=100, tolerance=1e-6),
        rbf_svr(train_x, train_y, test_x, folds, (1.0,), (0.1,), ("scale",)),
        random_forest(
            train_x, train_y, test_x, folds, (8,), (None,), (1,), (1.0,), seed=13, jobs=1
        ),
    ]
    for result in results:
        assert result.prediction.shape == (5, 3)
        assert np.isfinite(result.prediction).all()
        assert result.selected_params


def test_selectable_spectral_preprocessing() -> None:
    x = np.linspace(0.0, 1.0, 21)
    train = np.vstack([x**2, 1.5 * x**2 + 0.2])
    test = np.vstack([0.8 * x**2 + 0.1])
    for method in ("R", "FD", "SD", "SNV", "MSC", "SG"):
        train_out, test_out = preprocess_pair(train, test, method=method)
        assert train_out.shape == train.shape
        assert test_out.shape == test.shape
        assert np.isfinite(train_out).all()
        assert np.isfinite(test_out).all()
