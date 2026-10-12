"""Check E1C token masking, frozen probing, and fold-level data selection."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset_adapters import CanonicalDataset
from models.lejepa.model import SpectralLeJEPA
from models.shared.backbone import SpectralPatchTransformer
from models.shared.experiment import build_ssl_encoder
from models.shared.masking import random_patch_mask
from scripts import run_e1c


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    """Keep these small functional checks independent of GPU availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def test_patch_mask_count_and_reproducibility():
    first = random_patch_mask(5, 7, 0.4, generator=torch.Generator().manual_seed(8))
    second = random_patch_mask(5, 7, 0.4, generator=torch.Generator().manual_seed(8))
    assert first.dtype == torch.bool
    assert first.shape == (5, 7)
    assert first.sum(dim=1).tolist() == [3] * 5
    assert torch.equal(first, second)
    assert random_patch_mask(4, 2, 0.01).sum(dim=1).tolist() == [1] * 4
    assert random_patch_mask(4, 2, 0.99).sum(dim=1).tolist() == [1] * 4
    with pytest.raises(ValueError, match="at least two patches"):
        random_patch_mask(1, 1, 0.4)


def test_attention_receives_mask_tokens_and_positions():
    encoder = SpectralPatchTransformer(25, patch_size=6, embedding_dim=16, depth=1)
    spectra = torch.randn(2, 25)
    mask = torch.tensor(
        [[True, False, False, True, True], [False, True, False, False, True]]
    )
    captured = []

    def capture_attention_input(_module, arguments):
        captured.append(arguments[0].detach().clone())

    hook = encoder.transformer.register_forward_pre_hook(capture_attention_input)
    tokens, embedding = encoder(spectra, mask)
    hook.remove()
    assert encoder.patchify(spectra).shape == (2, 5, 6)
    assert tokens.shape == (2, 5, 16)
    assert embedding.shape == (2, 16)
    assert captured[0].shape == (2, 6, 16)
    patch_inputs = captured[0][:, 1:] - encoder.position_embedding[:, 1:]
    expected_mask_tokens = encoder.mask_token.expand(2, 5, 16)
    torch.testing.assert_close(patch_inputs[mask], expected_mask_tokens[mask])

    # Changes inside masked patches, including the padded final patch, must
    # have no effect on the masked view's representation.
    changed = spectra.clone()
    point_mask = mask.repeat_interleave(6, dim=1)[:, :25]
    changed[point_mask] += 100.0
    encoder.eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(spectra, mask)[1], encoder(changed, mask)[1])


def test_lejepa_patch_views_shapes_and_gradients():
    encoder = build_ssl_encoder("patch_transformer", 25, 6, 16, input_channels=1)
    model = SpectralLeJEPA(encoder, mask_mode="patch")
    spectra = torch.randn(3, 25)
    mask = random_patch_mask(3, encoder.n_patches, 0.4)
    model.eval()
    output = model(spectra, mask)
    assert output["view_embeddings"].shape == (2, 3, 16)
    assert output["projections"].shape == (2, 3, 16)
    torch.testing.assert_close(output["view_embeddings"][1], encoder(spectra)[1])
    output["loss"].backward()
    assert encoder.mask_token.grad.abs().sum() > 0
    assert encoder.patch_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(output["loss"])
    with pytest.raises(ValueError, match="patch_mask must be bool"):
        model(spectra, torch.zeros(3, 25, dtype=torch.bool))


def test_ssl_gradient_clipping_is_optional_and_positive(monkeypatch):
    args = run_e1c.build_parser().parse_args(
        ["--ssl-epochs", "1", "--batch-size", "full", "--ssl-grad-clip", "0.5"]
    )
    monkeypatch.setattr(run_e1c, "DEFAULT_EMBEDDING_DIM", 16)
    calls = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, maximum):
        calls.append(maximum)
        return original_clip(parameters, maximum)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    spectra = np.random.default_rng(3).normal(size=(4, 25)).astype(np.float32)
    run_e1c.pretrain(spectra, spectra, args, seed=3)
    assert calls == [0.5]


def test_e1c_fold_uses_all_spectra_and_per_target_probes(tmp_path, monkeypatch):
    args = run_e1c.build_parser().parse_args(
        [
            "--folds",
            "2",
            "--ssl-epochs",
            "1",
            "--probe-epochs",
            "2",
            "--batch-size",
            "full",
        ]
    )
    assert args.backbone == "patch_transformer"
    assert not hasattr(args, "mask_block_length")
    monkeypatch.setattr(run_e1c, "DEFAULT_EMBEDDING_DIM", 16)
    rng = np.random.default_rng(51)
    targets = rng.normal(size=(12, 2))
    targets[::2, 0] = np.nan
    targets[1::2, 1] = np.nan
    assert not np.isfinite(targets).all(axis=1).any()
    dataset = CanonicalDataset(
        "SWRI_Diesel",
        np.array([f"s{i}" for i in range(12)]),
        np.arange(1, 26),
        rng.normal(size=(12, 25)),
        pd.DataFrame(targets, columns=["a", "b"]),
        np.full(12, "SWRI_Diesel"),
    )
    original_pretrain, original_fit = run_e1c.pretrain, run_e1c.fit_probe
    frozen_encoders = []
    expected_train_metrics = {}

    def capture_pretrain(*arguments, **kwargs):
        result = original_pretrain(*arguments, **kwargs)
        assert len(arguments[0]) == 6
        assert not result.encoder.training
        assert all(
            not parameter.requires_grad for parameter in result.encoder.parameters()
        )
        snapshot = {
            name: value.clone() for name, value in result.encoder.state_dict().items()
        }
        frozen_encoders.append((result.encoder, snapshot))
        return result

    def capture_probe(train_embedding, train_targets, *arguments, **kwargs):
        assert np.isfinite(train_targets).all()
        assert len(train_embedding) == len(train_targets)
        result = original_fit(train_embedding, train_targets, *arguments, **kwargs)
        from sklearn.metrics import mean_squared_error, r2_score

        assert result.train_prediction.shape == train_targets.shape
        assert train_targets.shape[1] == 1
        y = train_targets[:, 0]
        prediction = result.train_prediction[:, 0]
        rmse = np.sqrt(mean_squared_error(y, prediction))
        fold = len(expected_train_metrics) // 2
        target_index = len(expected_train_metrics) % 2
        expected_train_metrics[fold, target_index] = (
            rmse,
            r2_score(y, prediction),
            np.std(y, ddof=1) / rmse,
        )
        encoder, snapshot = frozen_encoders[-1]
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        return result

    monkeypatch.setattr(run_e1c, "pretrain", capture_pretrain)
    monkeypatch.setattr(run_e1c, "fit_probe", capture_probe)
    run_e1c.run_dataset(dataset, "SWRI_Diesel_drop_missing", args, tmp_path)
    summary = pd.read_csv(tmp_path / "run_summary.csv")
    assert len(frozen_encoders) == 2
    assert len(summary) == 4
    assert summary.ssl_train_samples.tolist() == [6] * 4
    assert set(summary.target) == {"a", "b"}
    metric_rows = pd.read_csv(tmp_path / "metrics.csv")
    for row in metric_rows.itertuples():
        column = ["a", "b"].index(row.target)
        expected = expected_train_metrics[row.fold, column]
        assert (row.train_rmse, row.train_r2, row.train_rpd) == pytest.approx(expected)
        fold_summary = summary[
            (summary.fold == row.fold) & (summary.target == row.target)
        ]
        assert len(fold_summary) == 1
        assert row.n_train == fold_summary.probe_train_samples.iloc[0]
        assert row.r2_gap == pytest.approx(row.train_r2 - row.r2)
        assert row.rmse_gap == pytest.approx(row.rmse - row.train_rmse)
    aggregate = pd.read_csv(tmp_path / "metrics_aggregate.csv")
    for row in aggregate.itertuples():
        selected = metric_rows[metric_rows.target == row.target]
        assert row.train_r2 == pytest.approx(selected.train_r2.mean())
    loss_curve = pd.read_csv(tmp_path / "loss_curve.csv")
    assert set(loss_curve.stage) == {
        "ssl_pretrain",
        "frozen_probe",
    }
    ssl_curve = loss_curve[loss_curve.stage == "ssl_pretrain"]
    assert ssl_curve[["invariance_mse", "sigreg"]].notna().all().all()
    np.testing.assert_allclose(
        ssl_curve.loss,
        (1 - args.sigreg_weight) * ssl_curve.invariance_mse
        + args.sigreg_weight * ssl_curve.sigreg,
        rtol=1e-5,
    )
    assert (tmp_path / "first_fold_patch_masks.png").is_file()
    mask_rows = pd.read_csv(tmp_path / "first_fold_patch_masks.csv")
    assert mask_rows.dataset_index.nunique() == 3
    assert mask_rows.groupby("dataset_index").size().tolist() == [5, 5, 5]
    assert mask_rows.groupby("dataset_index").masked.sum().tolist() == [2, 2, 2]
    assert mask_rows.point_end_exclusive.max() == 25
    for asset in (tmp_path / "representations").iterdir():
        for name in (
            "lejepa_total_loss.png",
            "lejepa_invariance_loss.png",
            "lejepa_sigreg_loss.png",
            "probe_loss.png",
        ):
            assert (asset / name).is_file()
        metadata = json.loads((asset / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["mask_mode"] == "patch"
        assert metadata["input_channels"] == 1
        assert metadata["patch_size"] == 6
        assert metadata["n_patches"] == 5
        arrays = np.load(asset / "representations.npz")
        assert arrays["train_embedding"].shape == (6, 16)
        ssl_checkpoint = torch.load(asset / "ssl_checkpoint.pt", weights_only=True)
        assert ssl_checkpoint["format_version"] == 1
        assert ssl_checkpoint["context"]["fold"] in (0, 1)
        train_indices = ssl_checkpoint["context"]["train_indices"]
        for target_index, target in enumerate(("a", "b")):
            checkpoint = torch.load(
                asset / "probes" / f"target{target_index}" / "probe.pt",
                weights_only=True,
            )
            assert checkpoint["embedding_dim"] == 16
            assert checkpoint["output_dim"] == 1
            assert checkpoint["target"] == target
            expected_indices = [
                index
                for index in train_indices
                if np.isfinite(targets[index, target_index])
            ]
            assert checkpoint["train_indices"] == expected_indices


def test_e1c_reuses_ssl_checkpoint_and_rejects_changed_data(tmp_path, monkeypatch):
    args = run_e1c.build_parser().parse_args(
        [
            "--folds",
            "2",
            "--ssl-epochs",
            "1",
            "--probe-epochs",
            "1",
            "--batch-size",
            "full",
        ]
    )
    monkeypatch.setattr(run_e1c, "DEFAULT_EMBEDDING_DIM", 16)
    rng = np.random.default_rng(12)
    dataset = CanonicalDataset(
        "synthetic",
        np.array([f"s{i}" for i in range(12)]),
        np.arange(1, 26),
        rng.normal(size=(12, 25)),
        pd.DataFrame({"a": rng.normal(size=12), "b": rng.normal(size=12)}),
        np.full(12, "synthetic"),
    )
    source = tmp_path / "source" / "synthetic"
    run_e1c.run_dataset(dataset, "synthetic", args, source)
    args.ssl_checkpoint = tmp_path / "source"

    def reject_pretrain(*_arguments, **_kwargs):
        raise AssertionError("checkpoint loading must skip pretraining")

    monkeypatch.setattr(run_e1c, "pretrain", reject_pretrain)
    reused = tmp_path / "reused" / "synthetic"
    run_e1c.run_dataset(dataset, "synthetic", args, reused)
    for fold in range(2):
        relative = Path("representations") / f"lejepa_fold{fold}"
        source_embeddings = np.load(source / relative / "representations.npz")
        reused_embeddings = np.load(reused / relative / "representations.npz")
        for split in ("train_embedding", "test_embedding"):
            np.testing.assert_array_equal(
                source_embeddings[split], reused_embeddings[split]
            )
        original = torch.load(
            source / relative / "ssl_checkpoint.pt", weights_only=True
        )
        loaded = torch.load(reused / relative / "ssl_checkpoint.pt", weights_only=True)
        for name, value in original["model_state"].items():
            torch.testing.assert_close(
                value, loaded["model_state"][name], rtol=0, atol=0
            )

    changed_spectra = dataset.spectra.copy()
    changed_spectra[0, 0] += 1
    changed = CanonicalDataset(
        dataset.name,
        dataset.sample_id,
        dataset.wavelengths_nm,
        changed_spectra,
        dataset.targets,
        dataset.groups,
    )
    with pytest.raises(ValueError, match="spectral data or train/test fold"):
        run_e1c.run_dataset(
            changed, "synthetic", args, tmp_path / "changed" / "synthetic"
        )

    args.seed += 1
    with pytest.raises(ValueError, match="spectral data or train/test fold"):
        run_e1c.run_dataset(
            dataset, "synthetic", args, tmp_path / "new_split" / "synthetic"
        )


def test_regression_evaluation_handles_missing_and_degenerate_targets():
    result = run_e1c.evaluate_regression(
        np.array([1.0, np.nan, 3.0]), np.array([1.5, 100.0, 2.5])
    )
    assert result == pytest.approx(
        {"rmse": 0.5, "r2": 0.75, "rpd": np.sqrt(2) / 0.5, "n": 2}
    )
    for targets in (np.array([np.nan]), np.array([2.0]), np.array([2.0, 2.0])):
        result = run_e1c.evaluate_regression(targets, np.ones_like(targets))
        assert np.isnan(result["r2"])


def test_mlp_train_predictions_use_eval_mode_and_original_target_scale():
    args = run_e1c.build_parser().parse_args(
        ["--probe", "mlp", "--probe-epochs", "2", "--batch-size", "full"]
    )
    rng = np.random.default_rng(91)
    embedding = rng.normal(size=(8, 4)).astype(np.float32)
    targets = 50 + 10 * rng.normal(size=(8, 2))
    result = run_e1c.fit_probe(embedding, targets, embedding.copy(), args, 42)
    np.testing.assert_allclose(result.train_prediction, result.prediction)
    state = result.state
    head = torch.nn.Sequential(
        torch.nn.Linear(4, args.probe_hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(args.probe_hidden_dim, 2),
    )
    head.load_state_dict(state["probe"])
    head.eval()
    with torch.no_grad():
        prediction = head(torch.from_numpy(embedding)).numpy()
    expected = prediction * state["target_scale"].numpy() + state["target_mean"].numpy()
    np.testing.assert_allclose(result.train_prediction, expected, rtol=1e-6)
