"""Regression checks for the mnist.py training protocol adaptation."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.lejepa.model import SpectralLeJEPA
from models.lejepa.sigreg import SIGReg
from models.shared.scheduling import build_step_scheduler
from models.shared.experiment import build_ssl_encoder
from scripts import e1b as run_e1b
from scripts import e1f as run_e1f
from scripts.e1b import joint_lejepa_representation, plot_loss_curves
from scripts.e1f import ssl_representation, supervised_embeddings


def test_sigreg_matches_reference_per_view_statistic():
    """检查 SIGReg 与参考实现的逐视图统计量一致。"""
    projected = torch.randn(2, 7, 5, requires_grad=True)
    regularizer = SIGReg()
    torch.manual_seed(81)
    actual = regularizer(projected)
    torch.manual_seed(81)
    directions = torch.randn(5, 256)
    directions /= directions.norm(p=2, dim=0)
    x_t = (projected @ directions).unsqueeze(-1) * regularizer.t
    error = (x_t.cos().mean(-3) - regularizer.phi).square() + x_t.sin().mean(-3).square()
    expected = ((error @ regularizer.weights) * projected.size(-2)).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(projected.grad).all()


def test_view_mean_loss_and_probe_gradient_isolation():
    """检查视图均值损失及在线探针的梯度隔离。"""
    encoder = build_ssl_encoder("unet", 24, 4, 16)
    model = SpectralLeJEPA(encoder)
    captured = []
    def capture_embeddings(_module, _inputs, result):
        """保留编码器输出的梯度以检查两个视图。"""
        result[1].retain_grad()
        captured.append(result[1])
    hook = encoder.register_forward_hook(capture_embeddings)
    output = model(torch.randn(5, 24), torch.rand(5, 24) > 0.5)
    hook.remove()
    projected = output["projections"]
    # Two views' distance to their mean is one quarter of pairwise MSE.
    expected = (projected[0] - projected[1]).square().mean() / 4
    torch.testing.assert_close(output["invariance_mse"], expected)
    probe = torch.nn.Linear(16, 2)
    probe(output["view_embeddings"].detach()).square().mean().backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None for p in probe.parameters())
    output["loss"].backward()
    assert encoder.enc1[0].weight.grad.abs().sum() > 0
    assert captured[0].grad[:5].abs().sum() > 0
    assert captured[0].grad[5:].abs().sum() > 0


def test_scheduler_matches_reference_steps_and_short_runs():
    """检查学习率调度与参考实现及短训练行为一致。"""
    def optimizer():
        """建立两组不同学习率的测试优化器。"""
        return torch.optim.AdamW([
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.01},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.001},
        ])

    ours, reference = optimizer(), optimizer()
    scheduler = build_step_scheduler(ours, total_steps=12, warmup_steps=3, eta_min=1e-6)
    warmup = torch.optim.lr_scheduler.LinearLR(reference, start_factor=0.01, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=9, eta_min=1e-6)
    expected = torch.optim.lr_scheduler.SequentialLR(reference, [warmup, cosine], milestones=[3])
    for _ in range(13):
        assert [g["lr"] for g in ours.param_groups] == pytest.approx([g["lr"] for g in reference.param_groups])
        ours.step(); reference.step()
        scheduler.step(); expected.step()
    short = optimizer()
    schedule = build_step_scheduler(short, total_steps=1, warmup_steps=1)
    assert short.param_groups[0]["lr"] == pytest.approx(0.0001)
    short.step(); schedule.step()
    assert short.param_groups[0]["lr"] == pytest.approx(0.01)
    assert build_step_scheduler(short, total_steps=0) is None


def test_training_labels_do_not_change_lejepa_encoder(monkeypatch):
    """检查改变标签仅影响探针，不影响 LeJEPA 编码器。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(19)
    x = rng.normal(size=(7, 24)).astype(np.float32)
    y = rng.normal(size=(7, 1)).astype(np.float32)
    params = dict(
        backbone="patch_transformer", train_x=x, test_x=x[:2], epochs=2,
        batch_size=3, mask_ratio=0.4,
        patch_size=4, embedding_dim=16, seed=15, learning_rate=1e-3,
        head_learning_rate=1e-3,
    )
    first = joint_lejepa_representation(train_y=y, **params)
    second = joint_lejepa_representation(train_y=y[::-1].copy(), **params)
    for key, value in first[1].items():
        if key.startswith("encoder.") or key.startswith("projector."):
            torch.testing.assert_close(value, second[1][key], rtol=0, atol=0)
    assert any(not torch.equal(first[1][k], second[1][k]) for k in first[1] if k.startswith("probe."))
    assert first[2].shape == (7, 16)
    assert first[1]["probe.1.weight"].shape[0] == 1
    assert first[1]["encoder.patch_projection.weight"].shape[1] == 1
    assert "encoder.mask_token" in first[1]
    assert np.isfinite(first[2]).all()
    assert len(first[4]) == 2
    assert {row["stage"] for row in first[4]} == {"joint_train"}
    assert first[4][-1]["learning_rate_end"] == pytest.approx(1e-6)
    for row in first[4]:
        assert row["loss"] == pytest.approx(row["lejepa_loss"] + row["regression_loss"], rel=1e-5)
    with pytest.raises(ValueError, match="one target with no missing labels"):
        invalid_params = dict(params)
        invalid_params["train_y"] = np.full((7, 1), np.nan)
        joint_lejepa_representation(**invalid_params)


def test_fold_assets_are_single_stage_per_target(tmp_path, monkeypatch, caplog):
    """检查每个折与目标使用有标签样本独立训练并保存资产。"""
    from data.dataset_adapters import CanonicalDataset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["run_e1b.py", "--folds", "2", "--epochs", "2", "--batch-size", "full"])
    args = run_e1b.parse_args()
    monkeypatch.setattr(run_e1b, "DEFAULT_EMBEDDING_DIM", 16)
    rng = np.random.default_rng(42)
    dataset = CanonicalDataset(
        "tiny", np.array([f"s{i}" for i in range(16)]), np.arange(1, 25),
        rng.normal(size=(16, 24)),
        pd.DataFrame({"y": [np.nan, *rng.normal(size=15)], "z": [*rng.normal(size=15), np.nan]}),
        np.array(["g"] * 16), {},
    )
    monkeypatch.setattr(run_e1b, "plot_loss_curves", lambda *a, **k: [])
    original_train = run_e1b.joint_lejepa_representation
    seen_spectra = []
    seen_targets = []

    def capture_raw_spectra(backbone, train_x, train_y, test_x, *args, **kwargs):
        seen_spectra.extend((train_x, test_x))
        seen_targets.append(train_y.copy())
        assert train_y.shape[1] == 1
        assert np.isfinite(train_y).all()
        return original_train(backbone, train_x, train_y, test_x, *args, **kwargs)

    monkeypatch.setattr(run_e1b, "joint_lejepa_representation", capture_raw_spectra)
    def forbidden(*a, **k):
        """阻止 LeJEPA 进入额外的监督微调阶段。"""
        raise AssertionError("LeJEPA must not enter a second fine-tuning stage")
    monkeypatch.setattr(run_e1b, "supervised_embeddings", forbidden, raising=False)
    with caplog.at_level(logging.INFO):
        run_e1b.run_dataset(dataset, "tiny", args, tmp_path, logging.getLogger("test"))
    assert args.backbone == "patch_transformer"
    assert not hasattr(args, "mask_block_length")
    raw_rows = {tuple(row.astype(np.float32)) for row in dataset.spectra}
    assert seen_spectra and all(tuple(row) in raw_rows for array in seen_spectra for row in array)
    assert len(seen_targets) == 4
    runs = pd.read_csv(tmp_path / "ssl_runs.csv")
    summaries = pd.read_csv(tmp_path / "run_summary.csv")
    assert (tmp_path / "first_fold_patch_masks.png").is_file()
    mask_rows = pd.read_csv(tmp_path / "first_fold_patch_masks.csv")
    assert mask_rows.dataset_index.nunique() == 3
    assert mask_rows.groupby("dataset_index").size().tolist() == [4, 4, 4]
    assert mask_rows.groupby("dataset_index").masked.sum().tolist() == [2, 2, 2]
    assert mask_rows.point_end_exclusive.max() == 24
    fold_metrics = pd.read_csv(tmp_path / "metrics.csv")
    aggregate = pd.read_csv(tmp_path / "metrics_aggregate.csv")
    from models.lejepa.probe import predict_saved_lejepa_probe
    from splits.cross_validation import pooled_group_balanced_folds

    for fold in pooled_group_balanced_folds(dataset.groups, n_splits=2, seed=args.seed):
        assert f"[tiny | fold {fold.fold + 1}/2] train samples={len(fold.train_idx)} | test samples={len(fold.test_idx)}" in caplog.text
        if fold.fold == 0:
            assert set(mask_rows.dataset_index).issubset(set(fold.train_idx))
        for col, target in enumerate(dataset.targets.columns):
            asset = tmp_path / "representations" / f"lejepa_fold{fold.fold}_target{col}"
            state = torch.load(asset / "lejepa_with_probe.pt", map_location="cpu", weights_only=True)
            with np.load(asset / "representations.npz") as representations:
                train_indices = representations["train_indices"]
                train_prediction = predict_saved_lejepa_probe(representations["train_embedding"], state)[:, 0]
            expected_indices = fold.train_idx[np.isfinite(dataset.targets[target].to_numpy()[fold.train_idx])]
            np.testing.assert_array_equal(train_indices, expected_indices)
            expected = run_e1b.metrics(dataset.targets[target].to_numpy()[train_indices], train_prediction)
            actual = fold_metrics[(fold_metrics.fold == fold.fold) & (fold_metrics.target == target)].iloc[0]
            assert (actual.train_rmse, actual.train_r2, actual.train_rpd) == pytest.approx(expected)
            assert actual.n_train == len(expected_indices)
            expected_test = np.isfinite(dataset.targets[target].to_numpy()[fold.test_idx]).sum()
            assert actual.n_test == expected_test
            assert (
                f"[tiny | fold {fold.fold + 1}/2] {target} | "
                f"train n={len(expected_indices)} | test n={expected_test}"
            ) in caplog.text
    for row in aggregate.itertuples():
        selected = fold_metrics[fold_metrics.target == row.target]
        assert row.train_rmse == pytest.approx(selected.train_rmse.mean())
        assert row.train_r2 == pytest.approx(selected.train_r2.mean())
        assert row.train_rpd == pytest.approx(selected.train_rpd.mean())
    assert len(runs) == 4
    assert len(summaries) == 4
    assert set(summaries.regressor) == {"probe"}
    assert summaries.groupby("fold").representation_run_id.nunique().eq(2).all()
    assert len(pd.read_csv(tmp_path / "loss_curve.csv")) == 8
    for asset in (tmp_path / "representations").iterdir():
        assert (asset / "lejepa_with_probe.pt").exists()
        assert (asset / "encoder.pt").exists()
        assert not (asset / "finetuned_encoder.pt").exists()
        metadata = json.loads((asset / "metadata.json").read_text())
        assert metadata["probe_detached"] is True
        assert metadata["training_mode"] == "lejepa_online_probe"
    probe_config = json.loads(summaries.iloc[0].selected_params)
    assert probe_config["source"] == "one_stage_lejepa_probe"


def test_mae_stages_and_loss_plot_use_shared_schedule(tmp_path, monkeypatch):
    """检查 MAE 各训练阶段的调度及 LeJEPA 损失图输出。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(63)
    x = rng.normal(size=(6, 24)).astype(np.float32)
    encoder, _, _, _, reconstruction, _, history, _ = ssl_representation(
        "unet", x, x[:2], 2, 3, 0.4, 1, 4, 16, 3, 1e-3,
    )
    assert np.isfinite(reconstruction).all()
    assert history[0]["learning_rate"] == pytest.approx(1e-5)
    assert history[-1]["learning_rate_end"] == pytest.approx(1e-6)
    _, _, fine_history, _ = supervised_embeddings(
        encoder, x, rng.normal(size=(6, 1)), x[:2], 2, 3, 3,
    )
    assert fine_history[-1]["learning_rate_end"] == pytest.approx(1e-6)
    joint = joint_lejepa_representation(
        "patch_transformer", x, rng.normal(size=(6, 1)), x[:2], 2, 3, 0.4, 4, 16, 3, 1e-3, 1e-3,
    )
    plots = plot_loss_curves(
        [dict(row, experiment="lejepa", fold=0) for row in joint[4]],
        tmp_path, logging.getLogger("test"),
    )
    assert len(plots) == 1
    assert plots[0].stat().st_size > 0
