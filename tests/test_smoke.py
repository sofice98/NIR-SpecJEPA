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
from models.shared.masking import SpectralMasker, point_mask_to_patch_mask, random_block_mask, random_spectral_block_mask
from models.shared.regressors import EncoderMLPRegressor, RidgeRegressor
from preprocessing.spectral import preprocess_pair


def test_matched_models_forward_backward() -> None:
    """检查 MAE 与 LeJEPA 的输出形状、损失及反向传播。"""
    torch.manual_seed(7)
    spectra = torch.randn(8, 517)
    base = SpectralPatchTransformer(input_length=517, embedding_dim=64, heads=4)
    initial = {key: value.clone() for key, value in base.state_dict().items()}

    mae_encoder = SpectralPatchTransformer(input_length=517, embedding_dim=64, heads=4)
    mae_encoder.load_state_dict(initial)
    jepa_encoder = SpectralPatchTransformer(input_length=517, embedding_dim=64, heads=4)
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
    assert torch.isfinite(jepa_out["invariance_mse"])
    assert torch.isfinite(jepa_out["sigreg"])
    expected_total = 0.98 * jepa_out["invariance_mse"] + 0.02 * jepa_out["sigreg"]
    assert torch.allclose(jepa_out["loss"], expected_total)
    mae_out["loss"].backward()
    jepa_out["loss"].backward()


def test_selectable_ssl_backbones() -> None:
    """检查两种共享骨干的表示形状和模型规模。"""
    from models.shared.experiment import build_ssl_encoder
    from models.mae.model import SpectralMAE
    from models.lejepa.model import SpectralLeJEPA

    for backbone in ("unet", "patch_transformer"):
        encoder = build_ssl_encoder(backbone, input_length=41, patch_size=6, embedding_dim=16)
        tokens, pooled = encoder(torch.randn(3, 41), None)
        assert tokens.shape == (3, 7, 16)
        assert pooled.shape == (3, 16)
        assert sum(parameter.numel() for parameter in encoder.parameters()) > 0

    vit = build_ssl_encoder("patch_transformer", input_length=517, patch_size=6, embedding_dim=128)
    assert isinstance(vit.patch_projection, torch.nn.Conv1d)
    assert vit.cls_token.shape == (1, 1, 128)
    assert vit.position_embedding.shape == (1, vit.n_patches + 1, 128)
    assert sum(parameter.numel() for parameter in SpectralMAE(vit).parameters()) < 1_000_000
    vit = build_ssl_encoder("patch_transformer", input_length=517, patch_size=6, embedding_dim=128)
    assert sum(parameter.numel() for parameter in SpectralLeJEPA(vit).parameters()) < 1_000_000


def test_patch_transformer_mae_reconstructs() -> None:
    """检查 Transformer 骨干可重建补丁序列。"""
    encoder = SpectralPatchTransformer(
        input_length=41, patch_size=6, embedding_dim=16, depth=1, heads=2, ffn_dim=32
    )
    model = SpectralMAE(encoder)
    spectra = torch.randn(3, 41)
    mask = random_block_mask(3, encoder.n_patches, 0.4)
    reconstructed = model.reconstruct(spectra, mask)
    assert reconstructed.shape == (3, encoder.n_patches, encoder.patch_size)


def test_ridge_regressor() -> None:
    """检查岭回归可拟合多目标线性数据。"""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 8))
    y = x[:, :3] @ rng.normal(size=(3, 3))
    model = RidgeRegressor(alpha=1e-6).fit(x, y)
    assert model.predict(x).shape == (30, 3)
    assert np.mean((model.predict(x) - y) ** 2) < 1e-8


def test_encoder_mlp_regressor_updates_encoder_and_head() -> None:
    """检查端到端回归损失同时更新编码器和预测头。"""
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
    """检查点掩码与连续块掩码的数量和零填充结果。"""
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
    """检查经典回归基线的预测形状及有限值。"""
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


def test_fixed_representation_mlp_head() -> None:
    """检查固定表示 MLP 的预测和参数配置。"""
    from models.shared.regressors import fit_fixed_representation_mlp

    rng = np.random.default_rng(23)
    train_x = rng.normal(size=(24, 8))
    test_x = rng.normal(size=(5, 8))
    train_y = train_x[:, 0] * 2.0 - train_x[:, 1] + 0.1 * rng.normal(size=24)
    prediction, parameter_count, config = fit_fixed_representation_mlp(
        train_x, train_y, test_x, seed=23, batch_size="full",
    )
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert parameter_count == (8 * 64 + 64 + 64 + 1)
    assert config["hidden_dim"] == 64


def test_prediction_pipeline_parameter_components() -> None:
    """检查各预测路径的预处理、表示和回归参数计数。"""
    from models.shared.experiment import (
        build_ssl_encoder,
        parameter_count,
        preprocessing_parameter_count,
        ssl_prediction_parameter_count,
    )
    from preprocessing.spectral import FoldSpectralStandardizer

    train = np.arange(30, dtype=float).reshape(5, 6)
    standardizer = FoldSpectralStandardizer().fit(train)
    assert preprocessing_parameter_count("AIRPLS", standardizer, 6) == 12
    assert preprocessing_parameter_count("MSC", standardizer, 6) == 18

    encoder = build_ssl_encoder("patch_transformer", 24, 4, 16)
    mae = SpectralMAE(encoder)
    mae_preprocess_params = ssl_prediction_parameter_count(
        encoder, "mae_preprocess", parameter_count(mae)
    )
    mae_feature_params = ssl_prediction_parameter_count(
        encoder, "mae_feature", parameter_count(mae)
    )
    assert mae_preprocess_params > mae_feature_params
    assert mae_feature_params == parameter_count(encoder) - encoder.mask_token.numel()

    unet = build_ssl_encoder("unet", 24, 4, 16)
    unet_full_params = parameter_count(SpectralMAE(unet))
    unet_feature_params = ssl_prediction_parameter_count(unet, "mae_feature", unet_full_params)
    assert ssl_prediction_parameter_count(unet, "mae_preprocess", unet_full_params) > unet_feature_params

    regressor_params = 16 * 64 + 64 + 64 + 1
    assert 2 * 24 + mae_feature_params + regressor_params > mae_feature_params


def test_batch_size_full_parser(monkeypatch) -> None:
    """检查命令行可解析全批量选项。"""
    from scripts import e1a as run_e1a

    monkeypatch.setattr("sys.argv", ["run_e1a.py", "--batch-size", "full"])
    args = run_e1a.parse_args()
    assert args.batch_size == "full"


def test_spectral_mask_block_length_is_in_points() -> None:
    """检查连续掩码块长度按光谱点数解释。"""
    mask = random_spectral_block_mask(
        batch_size=2, n_points=24, patch_size=6, n_patches=4,
        mask_ratio=0.25, block_length=3,
        generator=torch.Generator().manual_seed(9),
    )
    assert mask.shape == (2, 4)


def test_run_dataset_rejects_single_sample_before_split(tmp_path) -> None:
    """检查单样本数据在交叉验证前被拒绝。"""
    from argparse import Namespace
    from data.dataset_adapters import CanonicalDataset
    from scripts.e1a import run_dataset

    dataset = CanonicalDataset(
        "tiny", np.array(["s0"]), np.arange(1, 5), np.ones((1, 4)),
        __import__("pandas").DataFrame({"target": [1.0]}), np.array(["g0"]), {},
    )
    args = Namespace(max_samples=None, folds=5)
    try:
        run_dataset(dataset, "tiny", args, tmp_path, __import__("logging").getLogger("test"))
    except ValueError as exc:
        assert "at least 2 samples" in str(exc)
    else:
        raise AssertionError("single-sample dataset should be rejected")


def test_selectable_spectral_preprocessing() -> None:
    """检查各光谱预处理方法的输出形状与有限性。"""
    x = np.linspace(0.0, 1.0, 21)
    train = np.vstack([x**2, 1.5 * x**2 + 0.2])
    test = np.vstack([0.8 * x**2 + 0.1])
    for method in ("R", "FD", "SD", "SNV", "MSC", "SG", "AIRPLS"):
        train_out, test_out = preprocess_pair(train, test, method=method)
        assert train_out.shape == train.shape
        assert test_out.shape == test.shape
        assert np.isfinite(train_out).all()
        assert np.isfinite(test_out).all()
