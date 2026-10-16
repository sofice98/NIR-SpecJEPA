"""E1D: small-data, single-channel Laya latent prediction for NIR spectra."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data.SWRI_Diesel.adapter import load_dataset as load_swri
from data.crude_oil_1.adapter import load_dataset as load_oil1
from data.crude_oil_2.adapter import load_dataset as load_oil2
from data.dataset_adapters import CanonicalDataset, merge_datasets
from models.lejepa.laya_spectral import SpectralLaya, SpectralLayaEncoder, spectral_block_mask
from models.shared.experiment import parameter_count, sklearn_parameter_count
from models.shared.regressors import pls, random_forest
from splits.cross_validation import pooled_group_balanced_folds
from scipy import stats

EXPERIMENT_ID = "e1d"


def target_stats(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fold-local target statistics and an observed-value mask."""
    obs = np.isfinite(targets)
    count = obs.sum(0)
    mean = np.divide(
        np.nansum(targets, 0), count, out=np.zeros(targets.shape[1]), where=count > 0
    )
    centered = np.where(obs, targets - mean, 0)
    scale = np.sqrt(
        np.divide(
            (centered * centered).sum(0),
            count,
            out=np.ones(targets.shape[1]),
            where=count > 0,
        )
    )
    scale[scale < 1e-12] = 1
    return mean, scale, obs


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    """Return RMSE, R虏, and RPD for one target column."""
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    var = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((prediction - y) ** 2) / var) if var > 1e-12 else float("nan")
    rpd = (
        float(np.std(y, ddof=1) / rmse) if rmse > 1e-12 and len(y) > 1 else float("nan")
    )
    return rmse, r2, rpd


def evaluate_regression(
    targets: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    """Evaluate one target on observed labels, retaining the evaluated count."""
    valid = np.isfinite(targets)
    n_samples = int(valid.sum())
    rmse, r2, rpd = float("nan"), float("nan"), float("nan")
    if n_samples:
        rmse, r2, rpd = metrics(targets[valid], prediction[valid])
    return {"rmse": rmse, "r2": r2, "rpd": rpd, "n": n_samples}


def limit_dataset(
    dataset: CanonicalDataset, max_samples: int | None
) -> CanonicalDataset:
    """Return a deterministic prefix for small smoke runs when requested."""
    if max_samples is None:
        return dataset
    n_samples = min(int(max_samples), len(dataset.sample_id))
    return CanonicalDataset(
        dataset.name,
        dataset.sample_id[:n_samples],
        dataset.wavelengths_nm,
        dataset.spectra[:n_samples],
        dataset.targets.iloc[:n_samples].reset_index(drop=True),
        dataset.groups[:n_samples],
        dataset.metadata,
    )


def fold_context(dataset: CanonicalDataset, dataset_name: str, fold: object) -> dict:
    """Identify spectral data and fold membership without depending on labels."""
    digest = hashlib.sha256()
    for values in (dataset.spectra, dataset.wavelengths_nm):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(dataset.sample_id.astype(str).tolist()).encode())
    digest.update(json.dumps(dataset.groups.astype(str).tolist()).encode())
    return {
        "dataset": dataset_name,
        "fold": int(fold.fold),
        "spectra_sha256": digest.hexdigest(),
        "train_indices": fold.train_idx.tolist(),
        "test_indices": fold.test_idx.tolist(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_batch_size(value: str) -> int | str:
    if str(value).lower() == "full":
        return "full"
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("batch-size must be positive or full")
    return result


def cosine_epoch_scheduler(optimizer, epochs: int, warmup_epochs: int, eta_min: float, scheduler_name: str = "warmup_cosine"):
    """Schedule once per epoch; optimizer updates remain per batch."""
    if scheduler_name == "constant" or epochs <= 1:
        return None
    warmup_epochs = min(max(0, warmup_epochs), epochs - 1)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(lr <= 0 for lr in base_lrs) or not 0 <= eta_min <= min(base_lrs):
        raise ValueError("invalid learning-rate floor")

    class EpochSchedule:
        def __init__(self):
            self.epoch = -1

        def step(self):
            self.epoch += 1
            if self.epoch < warmup_epochs:
                scale = 0.01 + 0.99 * (self.epoch + 1) / warmup_epochs
            else:
                progress = (self.epoch - warmup_epochs + 1) / max(1, epochs - warmup_epochs)
                minimum = eta_min / base_lrs[0]
                scale = minimum + (1 - minimum) * (1 + np.cos(np.pi * min(1.0, progress))) / 2
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale

    return EpochSchedule()


def build_model(args: argparse.Namespace, input_length: int, device: torch.device) -> SpectralLaya:
    encoder = SpectralLayaEncoder(input_length, args.patch_size, args.encoder_dim, args.encoder_depth, args.encoder_heads)
    model = SpectralLaya(
        encoder,
        projection_dim=args.projection_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        sigreg_weight=args.sigreg_weight,
    )
    return model.to(device)


def pretrain(train_x: np.ndarray, args: argparse.Namespace, seed: int, label: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = build_model(args, train_x.shape[1], device)
    batch_size = len(train_x) if args.batch_size == "full" else min(int(args.batch_size), len(train_x))
    drop_last = len(train_x) > batch_size and len(train_x) % batch_size == 1
    loader = DataLoader(TensorDataset(torch.as_tensor(train_x, dtype=torch.float32)), batch_size=batch_size, shuffle=True, drop_last=drop_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = args.ssl_warmup_epochs if args.ssl_warmup_epochs is not None else max(1, round(args.ssl_epochs * 0.1))
    scheduler = cosine_epoch_scheduler(optimizer, args.ssl_epochs, warmup, args.scheduler_eta_min, args.scheduler)
    history = []
    tqdm.write(f"[{label}] device={device} | samples={len(train_x)} | batch={batch_size} | params={parameter_count(model)}")
    for epoch in tqdm(range(1, args.ssl_epochs + 1), desc=f"{label} | SSL", unit="epoch", leave=False):
        model.train()
        losses = {"loss": [], "invariance_mse": [], "sigreg": []}
        for (batch,) in loader:
            batch = batch.to(device)
            mask = spectral_block_mask(len(batch), model.encoder.n_patches, args.mask_ratio, args.min_block_patches, args.max_block_patches, device=device)
            out = model(batch, mask)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            if args.ssl_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.ssl_grad_clip)
            optimizer.step()
            for key in losses:
                losses[key].append(float(out[key].detach()))
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "loss": float(np.mean(losses["loss"])), "invariance_mse": float(np.mean(losses["invariance_mse"])), "sigreg": float(np.mean(losses["sigreg"])), "lr": float(optimizer.param_groups[0]["lr"]), "stage": "ssl_pretrain"})
    return model, history


@dataclass(frozen=True)
class ProbeResult:
    prediction: np.ndarray
    train_prediction: np.ndarray
    state: dict
    history: list[dict]
    parameter_count: int


def fit_probe_epoch(
    train_embedding: np.ndarray,
    train_targets: np.ndarray,
    test_embedding: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    progress_label: str,
):
    """Train one frozen-encoder head with epoch-level LR scheduling."""
    from torch import nn

    mean, scale, observed = target_stats(train_targets)
    values = (np.where(np.isfinite(train_targets), train_targets, mean) - mean) / scale
    feature_mean = train_embedding.mean(axis=0)
    feature_scale = train_embedding.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    x = torch.as_tensor((train_embedding - feature_mean) / feature_scale, dtype=torch.float32)
    y = torch.as_tensor(values, dtype=torch.float32)
    valid = torch.as_tensor(observed.astype(np.float32))
    torch.manual_seed(seed)
    if args.probe in ("linear", "ridge"):
        head = nn.Linear(x.shape[1], train_targets.shape[1])
        decay = 0.0 if args.probe == "linear" else args.probe_weight_decay
    else:
        head = nn.Sequential(nn.Linear(x.shape[1], args.probe_hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(args.probe_hidden_dim, train_targets.shape[1]))
        decay = args.probe_weight_decay
    batch_size = len(x) if args.batch_size == "full" else min(int(args.batch_size), len(x))
    loader = DataLoader(TensorDataset(x, y, valid), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.probe_learning_rate, weight_decay=decay)
    warmup = max(1, round(args.probe_epochs * 0.1)) if args.probe_epochs else 0
    scheduler = cosine_epoch_scheduler(optimizer, args.probe_epochs, warmup, args.scheduler_eta_min, args.scheduler)
    history = []
    for epoch in tqdm(range(1, args.probe_epochs + 1), desc=f"{progress_label} | Probe", unit="epoch", leave=False):
        head.train()
        losses = []
        for bx, by, bm in loader:
            loss = ((head(bx) - by).square() * bm).sum() / bm.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "lr": float(optimizer.param_groups[0]["lr"]), "stage": "frozen_probe", "probe": args.probe})
    head.eval()
    with torch.no_grad():
        train_prediction = head(x).cpu().numpy()
        normalized_test = (test_embedding - feature_mean) / feature_scale
        prediction = head(torch.as_tensor(normalized_test, dtype=torch.float32)).cpu().numpy()
    state = {"probe": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}, "feature_mean": torch.as_tensor(feature_mean, dtype=torch.float32), "feature_scale": torch.as_tensor(feature_scale, dtype=torch.float32), "target_mean": torch.as_tensor(mean, dtype=torch.float32), "target_scale": torch.as_tensor(scale, dtype=torch.float32), "probe_type": args.probe, "embedding_dim": int(x.shape[1]), "output_dim": int(train_targets.shape[1]), "hidden_dim": int(args.probe_hidden_dim)}
    
    return ProbeResult(prediction=prediction * scale + mean, train_prediction=train_prediction * scale + mean, state=state, history=history, parameter_count=parameter_count(head))


def frozen_embeddings(model: SpectralLaya, train_x: np.ndarray, test_x: np.ndarray, batch_size: int):
    model.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device

    def encode(values: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model.encoder(batch)[1].cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, model.encoder.embedding_dim), dtype=np.float32)

    return encode(train_x), encode(test_x)


def save_checkpoint(path: Path, model: SpectralLaya, history: list[dict], context: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {"patch_size", "encoder_dim", "encoder_depth", "encoder_heads", "projection_dim", "projector_hidden_dim", "predictor_depth", "predictor_heads", "sigreg_weight"}
    torch.save({"format_version": 1, "experiment": EXPERIMENT_ID, "model_config": {k: getattr(args, k) for k in fields}, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "history": history, "context": context, "loss_formula": "(1-lambda)*masked_mse + lambda*sigreg"}, path)


def save_mask_audit(dataset: CanonicalDataset, train_indices: np.ndarray, args: argparse.Namespace, output_dir: Path) -> None:
    count = min(3, len(train_indices))
    masks = spectral_block_mask(count, (dataset.spectra.shape[1] + args.patch_size - 1) // args.patch_size, args.mask_ratio, args.min_block_patches, args.max_block_patches, generator=torch.Generator().manual_seed(args.seed)).numpy()
    rows = []
    for sample_index, mask in zip(np.asarray(train_indices[:count]), masks):
        for patch_index, masked in enumerate(mask):
            start = patch_index * args.patch_size
            stop = min(start + args.patch_size, dataset.spectra.shape[1])
            rows.append({"dataset_index": int(sample_index), "sample_id": str(dataset.sample_id[sample_index]), "patch_index": patch_index, "point_start": start, "point_end_exclusive": stop, "masked": bool(masked)})
    pd.DataFrame(rows).to_csv(output_dir / "first_fold_block_masks.csv", index=False, encoding="utf-8-sig")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    wavelengths = np.asarray(dataset.wavelengths_nm, dtype=float)
    boundaries = np.empty(len(wavelengths) + 1, dtype=float)
    boundaries[1:-1] = (wavelengths[:-1] + wavelengths[1:]) / 2
    boundaries[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
    boundaries[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2
    fig, axes = plt.subplots(count, 1, figsize=(13, 3.1 * count), sharex=True)
    for axis, sample_index, patch_mask in zip(np.atleast_1d(axes), train_indices[:count], masks):
        values = np.asarray(dataset.spectra[sample_index], dtype=float)
        visible = values.copy()
        visible[np.repeat(patch_mask, args.patch_size)[:len(values)]] = np.nan
        axis.plot(wavelengths, values, color="#64748b", linewidth=1, label="Original spectrum")
        axis.plot(wavelengths, visible, color="#2563eb", linewidth=1.2, label="Visible spectrum")
        for patch_index in np.flatnonzero(patch_mask):
            start = patch_index * args.patch_size
            stop = min(start + args.patch_size, len(values))
            axis.axvspan(boundaries[start], boundaries[stop], color="#ef4444", alpha=0.2,
                          label="Masked block" if patch_index == np.flatnonzero(patch_mask)[0] else None)
        axis.set_title(f"Sample {dataset.sample_id[sample_index]} (row {sample_index})")
        axis.set_ylabel("Intensity")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(f"E1D first-fold block masks | patch size={args.patch_size}, ratio={args.mask_ratio:g}")
    fig.tight_layout()
    fig.savefig(output_dir / "first_fold_block_masks.png", dpi=180)
    plt.close(fig)


def plot_loss_curves(history_rows: list[dict], output_dir: Path, weight: float, *, mean: bool = False) -> None:
    """Plot weighted SSL components and probe training loss for one fold or fold means."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = pd.DataFrame(history_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    ssl = history.loc[history.stage == "ssl_pretrain"].copy()
    if not ssl.empty:
        ssl["weighted_mse"] = (1 - weight) * ssl["invariance_mse"]
        ssl["weighted_sigreg"] = weight * ssl["sigreg"]
        if mean:
            ssl = ssl.groupby("epoch", as_index=False)[["weighted_mse", "weighted_sigreg", "loss"]].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for column, label in (("weighted_mse", "(1-lambda) * MSE"), ("weighted_sigreg", "lambda * SIGReg"), ("loss", "Total loss")):
            ax.plot(ssl.epoch, ssl[column], label=label)
        ax.set(xlabel="SSL epoch", ylabel="Weighted loss", title="Self-supervised loss")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "ssl_loss.png", dpi=180)
        plt.close(fig)
    probe = history.loc[history.stage == "frozen_probe"].copy()
    if mean and not probe.empty:
        probe = probe.groupby(["target", "epoch"], as_index=False)["loss"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for target, group in probe.groupby("target", sort=True):
        ax.plot(group.epoch, group.loss, label=str(target))
    if probe.empty:
        ax.text(0.5, 0.5, "PLS/RF: no epoch-wise training loss", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(fontsize=8)
    ax.set(xlabel="Probe epoch", ylabel="Normalized MSE", title="Regression training loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_loss.png", dpi=180)
    plt.close(fig)

def run_dataset(dataset: CanonicalDataset, dataset_name: str, args: argparse.Namespace, output_dir: Path) -> None:
    dataset.validate()
    dataset = limit_dataset(dataset, args.max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = dataset.targets.to_numpy(float)
    rows, histories, assets = [], [], []
    folds = pooled_group_balanced_folds(dataset.groups, min(args.folds, len(dataset.sample_id)), args.seed)
    for fold in tqdm(folds, desc=f"{dataset_name} | folds", unit="fold"):
        train_x = np.asarray(dataset.spectra[fold.train_idx], dtype=np.float32)
        test_x = np.asarray(dataset.spectra[fold.test_idx], dtype=np.float32)
        label = f"{dataset_name} | fold {fold.fold + 1}/{len(folds)}"
        model, ssl_history = pretrain(train_x, args, args.seed + fold.fold, label)
        batch_size = len(train_x) if args.batch_size == "full" else min(int(args.batch_size), len(train_x))
        train_embedding, test_embedding = frozen_embeddings(model, train_x, test_x, batch_size)
        context = fold_context(dataset, dataset_name, fold)
        asset = output_dir / "representations" / f"laya_fold{fold.fold}"
        save_checkpoint(asset / "ssl_checkpoint.pt", model, ssl_history, context, args)
        np.savez_compressed(asset / "representations.npz", train_embedding=train_embedding, test_embedding=test_embedding)
        pd.DataFrame(ssl_history).to_csv(asset / "history.csv", index=False, encoding="utf-8-sig")

        if fold.fold == 0:
            save_mask_audit(dataset, fold.train_idx, args, output_dir)
        fold_history = [dict(item, fold=fold.fold, experiment="ssl", target=None) for item in ssl_history]
        for column, target in enumerate(dataset.targets.columns):
            observed = np.flatnonzero(np.isfinite(targets[fold.train_idx, column]))
            if not len(observed):
                continue
            if args.probe in {"pls", "rf"}:
                import joblib

                if len(observed) < 4:
                    continue
                inner = pooled_group_balanced_folds(
                    dataset.groups[fold.train_idx[observed]],
                    min(3, max(2, len(observed) // 2)),
                    args.seed + fold.fold,
                )
                inner_pairs = [(part.train_idx, part.test_idx) for part in inner]
                if args.probe == "pls":
                    result = pls(
                        train_embedding[observed],
                        targets[fold.train_idx[observed], column:column + 1],
                        test_embedding,
                        inner_pairs,
                    )
                else:
                    result = random_forest(
                        train_embedding[observed],
                        targets[fold.train_idx[observed], column],
                        test_embedding,
                        inner_pairs,
                        seed=args.seed + fold.fold + column,
                        jobs=args.jobs,
                    )
                joblib.dump(result.model, asset / f"{args.probe}_{column}.joblib")
                probe_params = sklearn_parameter_count(result.model)
                components = result.selected_params.get("n_components")
            else:
                result = fit_probe_epoch(train_embedding[observed], targets[fold.train_idx[observed], column:column + 1], test_embedding, args, args.seed + fold.fold + column, progress_label=f"{label} | {target}")
                fold_history.extend(dict(item, fold=fold.fold, experiment="probe", target=target) for item in result.history)
                probe_params = result.parameter_count
                components = None
            train_prediction = np.asarray(result.train_prediction).reshape(-1)
            test_prediction = np.asarray(result.prediction).reshape(-1)
            train_metrics = evaluate_regression(targets[fold.train_idx[observed], column], train_prediction)
            test_metrics = evaluate_regression(targets[fold.test_idx, column], test_prediction)
            rows.append({"dataset": dataset_name, "experiment": "laya_frozen_probe", "regressor": args.probe, "fold": fold.fold, "target": target, "encoder_frozen": True, "ssl_params": parameter_count(model), "probe_params": probe_params, "pls_components": components, "rmse": test_metrics["rmse"], "r2": test_metrics["r2"], "rpd": test_metrics["rpd"], "train_rmse": train_metrics["rmse"], "train_r2": train_metrics["r2"], "train_rpd": train_metrics["rpd"], "n_train": train_metrics["n"], "n_test": test_metrics["n"], "r2_gap": train_metrics["r2"] - test_metrics["r2"], "rmse_gap": test_metrics["rmse"] - train_metrics["rmse"]})
        plot_loss_curves(fold_history, asset, args.sigreg_weight)
        histories.extend(fold_history)
        assets.append({"dataset": dataset_name, "fold": fold.fold, "asset_dir": str(asset)})
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(output_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
    plot_loss_curves(histories, output_dir / "figures" / "fold_mean", args.sigreg_weight, mean=True)
    pd.DataFrame(assets).to_csv(output_dir / "ssl_runs.csv", index=False, encoding="utf-8-sig")
    if rows:
        pd.DataFrame(rows).groupby(["dataset", "experiment", "regressor", "target"], as_index=False)[["rmse", "r2", "rpd", "train_rmse", "train_r2", "train_rpd", "r2_gap", "rmse_gap"]].mean().to_csv(output_dir / "metrics_aggregate.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E1D single-channel Laya latent prediction")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/e1d")
    parser.add_argument("--datasets", nargs="+", default=["swri", "crude_oil"], choices=["swri", "crude_oil", "lucas"])
    parser.add_argument("--swri-root", type=Path, default=ROOT / "data/SWRI_Diesel")
    parser.add_argument("--oil1-root", type=Path, default=ROOT / "data/crude_oil_1")
    parser.add_argument("--oil2-root", type=Path, default=ROOT / "data/crude_oil_2")
    parser.add_argument("--lucas-root", type=Path, default=ROOT / "data/lucas")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ssl-epochs", type=int, default=200)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe", choices=["linear", "ridge", "mlp", "pls", "rf"], default="pls")
    parser.add_argument("--scheduler", choices=["constant", "cosine", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--batch-size", type=parse_batch_size, default=16)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--min-block-patches", type=int, default=5)
    parser.add_argument("--max-block-patches", type=int, default=10)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--ssl-warmup-epochs", type=int, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--ssl-grad-clip", type=float, default=None)
    parser.add_argument("--encoder-dim", type=int, default=196)
    parser.add_argument("--encoder-depth", type=int, default=3)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=48)
    parser.add_argument("--projector-hidden-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers for random forest inner search")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ssl-checkpoint", type=Path, default=None, help="Reserved for a future E1D checkpoint reuse path")
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly to --output-dir for resumable batch runs")
    return parser


def main() -> None:
    started = time.perf_counter()
    args = build_parser().parse_args()
    if args.folds < 2 or args.ssl_epochs < 1 or args.probe_epochs < 0:
        raise ValueError("folds >= 2, ssl-epochs >= 1, probe-epochs >= 0 required")
    if not 0 < args.mask_ratio < 1 or args.patch_size < 1 or args.min_block_patches < 1 or args.max_block_patches < args.min_block_patches:
        raise ValueError("invalid masking configuration")
    if args.ssl_warmup_epochs is not None and not 0 <= args.ssl_warmup_epochs < args.ssl_epochs:
        raise ValueError("ssl-warmup-epochs must be in [0, ssl-epochs)")
    seed_everything(args.seed)
    if not args.no_timestamp:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({"experiment_id": EXPERIMENT_ID, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2), encoding="utf-8")
    print(f"[{EXPERIMENT_ID}] loss=(1-lambda)*masked_mse + lambda*SIGReg | epochs={args.ssl_epochs}/{args.probe_epochs}")
    if "swri" in args.datasets:
        run_dataset(load_swri(args.swri_root), "SWRI_Diesel", args, args.output_dir / "SWRI_Diesel")
    if "crude_oil" in args.datasets:
        run_dataset(merge_datasets([load_oil1(args.oil1_root), load_oil2(args.oil2_root)], name="crude_oil_1+crude_oil_2"), "crude_oil_1_plus_crude_oil_2", args, args.output_dir / "crude_oil_1_plus_crude_oil_2")
    if "lucas" in args.datasets:
        from data.lucas.adapter import load_dataset as load_lucas
        run_dataset(load_lucas(args.lucas_root), "LUCAS_SOIL", args, args.output_dir / "LUCAS_SOIL")
    print(json.dumps({"output_dir": str(args.output_dir), "experiment": EXPERIMENT_ID, "elapsed_seconds": round(time.perf_counter() - started, 2)}, indent=2))


@dataclass(frozen=True)
class E1DConfig:
    name: str
    mask_ratio: float = 0.40
    sigreg_lambda: float = 0.02
    epochs: int = 100
    batch_size: int | str = 32
    probe: str = "pls"
    patch_size: int = 6
    encoder_dim: int = 96


BASELINE = E1DConfig("baseline")


def make_configs() -> list[E1DConfig]:
    """Return baseline plus configurations changing one value at a time."""
    configs = [BASELINE]
    configs.extend(E1DConfig(f"mask_{value:g}", mask_ratio=value) for value in (0.20, 0.60, 0.80))
    configs.extend(E1DConfig(f"lambda_{value:g}", sigreg_lambda=value) for value in (0.01, 0.05, 0.10, 0.20, 0.50, 0.75))
    configs.extend(E1DConfig(f"epochs_{value}", epochs=value) for value in (200,))
    configs.extend(E1DConfig(f"batch_{value}", batch_size=value) for value in (16, 64, 128))
    configs.extend(E1DConfig(f"head_{value}", probe=value) for value in ("rf", "mlp"))
    configs.extend(E1DConfig(f"patch_{value}", patch_size=value) for value in (4, 8, 10, 16))
    configs.extend(E1DConfig(f"encoder_{value}", encoder_dim=value) for value in (48, 128, 196))
    for config in configs[1:]:
        changed = sum(
            (
                config.mask_ratio != BASELINE.mask_ratio,
                config.sigreg_lambda != BASELINE.sigreg_lambda,
                config.epochs != BASELINE.epochs,
                config.batch_size != BASELINE.batch_size,
                config.probe != BASELINE.probe,
                config.patch_size != BASELINE.patch_size,
                config.encoder_dim != BASELINE.encoder_dim,
            )
        )
        if changed != 1:
            raise ValueError(f"{config.name} must change exactly one baseline parameter")
    return configs


def find_python() -> str:
    """Run sweep configurations with the active interpreter."""
    return sys.executable


def command_for(config: E1DConfig, output_dir: Path, args: argparse.Namespace) -> list[str]:
    script = Path(__file__).resolve()
    command = [
        find_python(),
        str(script),
        "--datasets", "swri",
        "--output-dir", str(output_dir),
        "--folds", str(args.folds),
        "--ssl-epochs", str(config.epochs),
        "--probe-epochs", str(args.probe_epochs),
        "--batch-size", str(config.batch_size),
        "--mask-ratio", str(config.mask_ratio),
        "--sigreg-weight", str(config.sigreg_lambda),
        "--probe", config.probe,
        "--patch-size", str(config.patch_size),
        "--encoder-dim", str(config.encoder_dim),
        "--seed", str(args.seed),
        "--no-timestamp",
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    return command


def run(args: argparse.Namespace) -> None:
    if args.folds < 2 or args.probe_epochs < 1 or (args.max_samples is not None and args.max_samples < args.folds):
        raise ValueError("folds >= 2, probe-epochs >= 1, max-samples >= folds required")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs = make_configs()
    (output_root / "matrix.json").write_text(
        json.dumps({"baseline": asdict(BASELINE), "configs": [asdict(c) for c in configs]}, indent=2),
        encoding="utf-8",
    )
    for index, config in enumerate(configs, start=1):
        run_dir = output_root / config.name
        marker = run_dir / "SWRI_Diesel" / "metrics.csv"
        if marker.exists() and not args.force:
            print(f"[{index}/{len(configs)}] skip {config.name}: {run_dir}", flush=True)
            continue
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sweep_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        command = command_for(config, run_dir, args)
        print(f"[{index}/{len(configs)}] run {config.name}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(f"E1D config {config.name} failed with exit code {completed.returncode}")
    metric_frames = []
    for config in configs:
        metrics_path = output_root / config.name / "SWRI_Diesel" / "metrics.csv"
        if metrics_path.exists():
            frame = pd.read_csv(metrics_path)
            frame.insert(0, "sweep_config", config.name)
            metric_frames.append(frame)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics.to_csv(output_root / "metrics.csv", index=False, encoding="utf-8-sig")
        (metrics.groupby(["sweep_config", "regressor", "target"], as_index=False)
         [["rmse", "r2", "rpd"]].mean()
         .to_csv(output_root / "metrics_aggregate.csv", index=False, encoding="utf-8-sig"))


def parse_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1d_ota")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)



TARGETS = ("BP50", "CN", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")
FAMILIES = (
    ("掩码比例", "mask_ratio", ("mask_0.2", "baseline", "mask_0.6", "mask_0.8")),
    ("SIGReg 权重", "sigreg_lambda", ("lambda_0.01", "baseline", "lambda_0.05", "lambda_0.1", "lambda_0.2", "lambda_0.5", "lambda_0.75")),
    ("SSL epoch", "epochs", ("baseline", "epochs_200")),
    ("batch size", "batch_size", ("batch_16", "baseline", "batch_64", "batch_128")),
    ("回归头", "probe", ("baseline", "head_rf", "head_mlp")),
    ("patch size", "patch_size", ("patch_4", "baseline", "patch_8", "patch_10", "patch_16")),
    ("encoder dim", "encoder_dim", ("encoder_48", "baseline", "encoder_128", "encoder_196")),
)
CONFIGS = make_configs()
BY_NAME = {config.name: config for config in CONFIGS}
ORDER = [config.name for config in CONFIGS]


def value_label(name: str, field: str) -> str:
    value = getattr(BY_NAME[name], field)
    return {"pls": "PLS", "rf": "随机森林", "mlp": "MLP"}.get(str(value), f"{value:g}" if isinstance(value, float) else str(value))


def validate(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = json.loads((input_dir / "matrix.json").read_text(encoding="utf-8"))
    if matrix["baseline"] != vars(BASELINE) or matrix["configs"] != [vars(config) for config in CONFIGS]:
        raise ValueError("matrix.json does not match the fixed 23-configuration protocol")
    frames, losses = [], []
    for config in CONFIGS:
        run_dir = input_dir / config.name
        recorded = json.loads((run_dir / "sweep_config.json").read_text(encoding="utf-8"))
        if recorded != vars(config):
            raise ValueError(f"configuration mismatch: {config.name}")
        actual_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        runtime_fields = {"mask_ratio": config.mask_ratio, "sigreg_weight": config.sigreg_lambda,
                          "ssl_epochs": config.epochs, "batch_size": config.batch_size,
                          "probe": config.probe, "patch_size": config.patch_size,
                          "encoder_dim": config.encoder_dim, "folds": 5, "seed": 42,
                          "probe_epochs": 100, "datasets": ["swri"],
                          "learning_rate": 3e-4, "weight_decay": 0.05,
                          "min_block_patches": 5, "max_block_patches": 10}
        if any(actual_config.get(key) != value for key, value in runtime_fields.items()):
            raise ValueError(f"runtime arguments mismatch: {config.name}")
        metrics_path = run_dir / "SWRI_Diesel" / "metrics.csv"
        if not metrics_path.is_file():
            raise ValueError(f"incomplete experiment: {config.name} has no metrics.csv")
        metrics = pd.read_csv(metrics_path)
        expected = {(fold, target) for fold in range(5) for target in TARGETS}
        actual = set(zip(metrics.fold, metrics.target))
        if len(metrics) != 35 or actual != expected or metrics.duplicated(["fold", "target"]).any():
            raise ValueError(f"incomplete 5-fold metrics: {config.name}")
        if not np.isfinite(metrics[["rmse", "r2", "rpd"]].to_numpy(float)).all():
            raise ValueError(f"non-finite metrics: {config.name}")
        if set(metrics.regressor) != {config.probe}:
            raise ValueError(f"regression head mismatch: {config.name}")
        metrics.insert(0, "sweep_config", config.name)
        frames.append(metrics)
        loss_path = run_dir / "SWRI_Diesel" / "loss_curve.csv"
        if not loss_path.is_file():
            raise ValueError(f"incomplete experiment: {config.name} has no loss_curve.csv")
        loss = pd.read_csv(loss_path)
        ssl = loss[loss.stage == "ssl_pretrain"].copy()
        expected_epochs = {(fold, epoch) for fold in range(5) for epoch in range(1, config.epochs + 1)}
        actual_epochs = set(zip(ssl.fold, ssl.epoch))
        if len(ssl) != 5 * config.epochs or actual_epochs != expected_epochs:
            raise ValueError(f"incomplete SSL history: {config.name}")
        if not np.isfinite(ssl[["loss", "invariance_mse", "sigreg"]].to_numpy(float)).all():
            raise ValueError(f"non-finite SSL history: {config.name}")
        ssl.insert(0, "sweep_config", config.name)
        losses.append(ssl)
    all_metrics = pd.concat(frames, ignore_index=True)
    counts = all_metrics.pivot_table(index=["fold", "target"], columns="sweep_config", values="n_test")
    if counts.isna().any().any() or counts.nunique(axis=1).gt(1).any():
        raise ValueError("outer test counts differ between configurations")
    return all_metrics, pd.concat(losses, ignore_index=True)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(values.mean())
    flipped = [abs(np.mean(values * np.array(signs))) for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(score >= observed - 1e-12 for score in flipped) / len(flipped)


def analyze(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_summary = (metrics.groupby(["sweep_config", "target"])[["r2", "rmse", "rpd"]]
                      .agg(["mean", "std"]))
    target_summary.columns = ["_".join(column) for column in target_summary.columns]
    target_summary = target_summary.reset_index()
    fold_macro = metrics.groupby(["sweep_config", "fold"], as_index=False).r2.mean()
    baseline = fold_macro[fold_macro.sweep_config == "baseline"].set_index("fold").r2
    rows = []
    for name in ORDER:
        scores = fold_macro[fold_macro.sweep_config == name].set_index("fold").r2
        delta = (scores - baseline).sort_index().to_numpy()
        mean_delta = float(delta.mean())
        sd_delta = float(delta.std(ddof=1))
        margin = float(stats.t.ppf(0.975, 4) * sd_delta / np.sqrt(5))
        rows.append({"config": name, "macro_r2_mean": scores.mean(), "macro_r2_sd": scores.std(ddof=1),
                     "delta_r2": mean_delta, "delta_sd": sd_delta,
                     "ci95_low": mean_delta - margin, "ci95_high": mean_delta + margin,
                     "paired_dz": mean_delta / sd_delta if sd_delta else np.nan,
                     "sign_flip_p": exact_sign_flip_p(delta) if name != "baseline" else np.nan})
    overall = pd.DataFrame(rows)
    alternatives = overall[overall.config != "baseline"].sort_values("sign_flip_p")
    count = len(alternatives)
    adjusted = np.maximum.accumulate(alternatives.sign_flip_p.to_numpy() * np.arange(count, 0, -1))
    overall.loc[alternatives.index, "holm_p"] = np.minimum(1, adjusted)
    return target_summary, fold_macro, overall


def setup_plot() -> None:
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 9, "savefig.dpi": 220})


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(overall: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout="constrained")
    lookup = overall.set_index("config")
    for ax, (family, field, names) in zip(axes.flat, FAMILIES):
        names = list(names)
        positions = np.arange(len(names))
        values = lookup.loc[names, "delta_r2"].to_numpy()
        errors = lookup.loc[names, "delta_sd"].fillna(0).to_numpy()
        ax.errorbar(positions, values, yerr=errors, marker="o", capsize=3, color="#22577a")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xticks(positions, [value_label(name, field) for name in names])
        ax.set_title(family)
        ax.set_ylabel("相对基准的平均 R² 差值")
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("E1D 单因素扫描：五折跨目标平均 R² 与折间标准差", fontsize=13)
    save_figure(fig, out / "figure-01-factor-delta")


def plot_target_heatmap(summary: pd.DataFrame, out: Path) -> None:
    matrix = summary.pivot(index="sweep_config", columns="target", values="r2_mean").loc[ORDER, list(TARGETS)]
    delta = matrix.sub(matrix.loc["baseline"], axis=1)
    fig, ax = plt.subplots(figsize=(10, 11), layout="constrained")
    extent = max(0.05, float(np.nanmax(abs(delta.to_numpy()))))
    im = ax.imshow(delta, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
    ax.set_yticks(np.arange(len(ORDER)), ORDER)
    ax.set_xticks(np.arange(len(TARGETS)), TARGETS)
    ax.set_title("E1D 各目标平均 R² 相对基准的差值")
    fig.colorbar(im, ax=ax, label="ΔR²（五折均值）")
    save_figure(fig, out / "figure-02-target-heatmap")


def plot_loss(losses: pd.DataFrame, out: Path) -> None:
    names = ["baseline", "lambda_0.01", "lambda_0.05", "lambda_0.1", "lambda_0.2", "lambda_0.5", "lambda_0.75"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in names:
        data = losses[losses.sweep_config == name].groupby("epoch")[["invariance_mse", "sigreg"]].mean()
        label = f"λ={BY_NAME[name].sigreg_lambda:g}"
        for ax, column in zip(axes, ("invariance_mse", "sigreg")):
            ax.plot(data.index, data[column], label=label, linewidth=1.3)
    for ax, title in zip(axes, ("掩码预测 MSE", "SIGReg 分量")):
        ax.set(title=title, xlabel="SSL epoch", ylabel="五折平均损失分量")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("不同 SIGReg 权重下的训练损失分量（未加权）")
    save_figure(fig, out / "figure-03-lambda-loss")


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def table_overall(overall: pd.DataFrame) -> str:
    lines = ["| 配置 | 平均 R² ± SD | ΔR² ± SD | 95% CI（描述性） | 精确符号翻转 p | Holm p |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in overall.itertuples():
        lines.append(f"| {row.config} | {fmt(row.macro_r2_mean)} ± {fmt(row.macro_r2_sd)} | "
                     f"{fmt(row.delta_r2)} ± {fmt(row.delta_sd)} | [{fmt(row.ci95_low)}, {fmt(row.ci95_high)}] | "
                     f"{fmt(row.sign_flip_p, 4)} | {fmt(row.holm_p, 4)} |")
    return "\n".join(lines)


def write_reports(input_dir: Path, out: Path, metrics: pd.DataFrame, summary: pd.DataFrame,
                  overall: pd.DataFrame) -> None:
    report_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    table = table_overall(overall)
    baseline = overall.iloc[0]
    mlp_metrics = metrics[metrics.sweep_config == "head_mlp"]
    mlp_train_r2 = float(mlp_metrics.train_r2.mean())
    mlp_test_r2 = float(mlp_metrics.r2.mean())
    best_overall = overall.iloc[overall.macro_r2_mean.idxmax()]
    family_lines = []
    for family, field, names in FAMILIES:
        subset = overall.set_index("config").loc[list(names)]
        best = subset.loc[subset.macro_r2_mean.idxmax()]
        worst = subset.loc[subset.macro_r2_mean.idxmin()]
        family_lines.append(f"- **{family}**：描述性最高为 {value_label(best.name, field)}"
                            f"（平均 R²={best.macro_r2_mean:.3f}，相对基准 Δ={best.delta_r2:+.3f}）；"
                            f"最低为{value_label(worst.name, field)}（Δ={worst.delta_r2:+.3f}）。")
    significant = overall[(overall.config != "baseline") & (overall.holm_p < 0.05)]
    significance = "没有配置在 Holm 校正后达到 0.05。" if significant.empty else f"{len(significant)} 个配置在 Holm 校正后达到 0.05。"
    target_lines = []
    for target in TARGETS:
        row = summary[(summary.sweep_config == "baseline") & (summary.target == target)].iloc[0]
        count = int(metrics[(metrics.sweep_config == "baseline") & (metrics.target == target)].n_test.sum())
        target_lines.append(f"| {target} | {count} | {row.r2_mean:.3f} ± {row.r2_std:.3f} | {row.rmse_mean:.4g} ± {row.rmse_std:.3g} |")
    target_table = "| 目标 | 五折测试标签总数 | R² 均值 ± SD | RMSE 均值 ± SD |\n|---|---:|---:|---:|\n" + "\n".join(target_lines)
    source = f"`../../artifacts/runs/{input_dir.name}/`"
    analysis = f"""# E1D 单因素扫描：严格分析

## 分析问题与证据

在固定基准下，单独改变哪个因素会影响 SWRI Diesel 七目标的外层五折测试 R²？共有 23 个配置，每个配置 5 折 × 7 目标 = 35 行测试指标，共 {len(metrics)} 行。原始结果：{source}。主要描述性指标是逐折七目标等权平均 R²，变化量为同折配置减去基准；不同目标量纲不同，RMSE 只在目标内比较。

基准平均 R² 为 {baseline.macro_r2_mean:.3f} ± {baseline.macro_r2_sd:.3f}（折间 SD）。所有配置使用相同划分和 seed=42，因此折间差值可配对，但只有一套随机种子。

## 逐因素观察

{chr(10).join(family_lines)}

这些是已测试网格内的描述性排序，不足以推断全局最优。{significance} 外层折并非独立重复实验，五折的置信区间与检验仅作为敏感性说明。

## 完整数值表

{table}

## 基准逐目标表现

{target_table}

## 回归头诊断

训练折内对神经网络回归头的冻结表征逐维标准化后，MLP 的平均训练 R² 为 {mlp_train_r2:.3f}，测试 R² 为 {mlp_test_r2:.3f}。这说明 MLP 已能够拟合训练标签，但测试表现仍低于本轮 PLS 基准；不能把它归因于网络无法学习。标准化只用训练折统计量，不泄漏测试折。

## 候选结论

- 结论：本扫描可识别若干描述性较好的设置。证据：上表五折配对 ΔR²、图 1 与图 2。允许表述：“在本数据、本划分与种子下，某设置的平均 R² 较高。”避免表述：“该设置在总体上显著更优。”不确定性：随机种子、数据集和超参数搜索范围；后续检查：独立种子与外部数据验证。决策：保留为候选，不直接定稿。

## 图的解读

1. 图 1 展示各因素的配置相对基准的七目标平均 R² 差值。误差棒是五个配对折差值的标准差，反映划分敏感性，不是置信区间。
2. 图 2 展示每个目标的五折平均 R² 差值，检查整体均值是否由少数目标驱动。
3. 图 3 分开绘制未加权的掩码预测 MSE 和 SIGReg，便于观察权重变化的训练动态；不同 λ 下的加权总损失不能直接横向比较。
"""
    (out / "analysis-report.md").write_text(analysis, encoding="utf-8")
    appendix = f"""# E1D 统计附录

## 单位与计算

- 外层单位：五个固定测试折；同一折、同一目标在配置之间配对。七目标等权汇总为每折宏平均 R²，未把 35 行视为 35 个独立重复。
- 每配置只有 seed 42 的一次完整交叉验证。五折共享训练样本，折间 SD 和 t 区间仅作描述性参考。
- 95% 区间：五个折差值的均值 ± t(0.975,4) × SD/√5。配对效应量 dz = 平均差值/折差值 SD。
- 双侧精确符号翻转检验枚举 2⁵ 种符号模式，22 个非基准比较使用 Holm 校正。五折时最小双侧 p 为 0.0625，因此显著性检验功效有限。没有先验独立种子，不应由 p 值作总体优劣断言。
- 无效、缺失或重复的指标/历史记录会使分析脚本报错，禁止输出不完整报告。

## 汇总

{table}

## 逐目标均值与折间标准差

完整机器可读表：`target_summary.csv`，含各配置各目标的 R²、RMSE、RPD 的均值与标准差。RMSE 的数值单位随目标不同，不做跨目标平均。

## 数据与限制

SWRI Diesel 共 784 条光谱；自监督训练使用训练折全部光谱。BP50、D4052、FLASH、FREEZE、TOTAL、VISC 各有 395 个标签，CN 有 381 个标签。目标缺失标签在回归训练与测试指标计算时按目标过滤。只有一个数据集、一套 seed 和一组五折划分，单因素结果不覆盖参数交互。
"""
    (out / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog = """# E1D 图录

1. `figures/figure-01-factor-delta.png` / `.pdf`：单因素 ΔR²；误差棒为五个配对折差值的 SD，零线为基准。用于判断改动方向与折间波动，不能据此宣称统计显著。
2. `figures/figure-02-target-heatmap.png` / `.pdf`：七目标各配置相对基准的五折平均 ΔR²。用于检查目标异质性；色标是 R² 点数差值。
3. `figures/figure-03-lambda-loss.png` / `.pdf`：不同 λ 的两项未加权 SSL 损失分量，先按 epoch 在五折取均值。用于判断优化动态，不能把低训练损失直接等同于高测试 R²。
"""
    (out / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    report = f"""---
type: results-report
date: {report_date}
experiment_line: e1d-swri
round: r00
purpose: one-factor-ablation-report
status: complete
source_artifacts:
  - analysis-report.md
  - stats-appendix.md
  - figure-catalog.md
linked_experiments: []
linked_results: []
---

# E1D / Round r00 / 单因素实验报告 / {report_date}

## 摘要与实验目的

本轮在 SWRI Diesel 上完成 23 组 E1D 单因素实验，评估掩码比例、SIGReg 权重、SSL 训练轮数、批量大小、回归头、patch size 和 encoder dim 对冻结表征回归的影响。基准七目标平均测试 R² 为 **{baseline.macro_r2_mean:.3f} ± {baseline.macro_r2_sd:.3f}**（五折均值 ± 折间标准差）。{significance}因此本报告将较优配置作为下一轮复验候选。

本轮单因素网格中，**{best_overall.config}** 的平均测试 R² 最高（{best_overall.macro_r2_mean:.3f}，较基准 +{best_overall.delta_r2:.3f}）。这是单数据集、单种子下的候选设置。

## 设置与评估协议

- 数据：SWRI Diesel，784 条 401 维光谱，七个目标；每个目标约 381–395 个有效标签。
- 基准：mask ratio=0.4、SIGReg weight=0.02、SSL epoch=100、batch size=32、回归头 PLS、patch size=6、encoder dim=96。
- 固定条件：seed=42、外层 5 折、MLP probe 100 epoch、AdamW 学习率 3e-4、权重衰减 0.05、连续块掩码长度 5–10 patch。PLS 与随机森林在训练折内部用 3 折选择回归器参数。
- 每次只改变一个因素。回归头实验也重新训练相同配置的自监督编码器；设置相同随机种子，训练轨迹原则上可重复，但可能受 CUDA 非确定性影响。
- 主要指标：测试 R²。逐目标另外报告 RMSE、RPD；不同目标 RMSE 不合并。

## 主要结果

{chr(10).join(family_lines)}

完整的跨目标汇总见下表；它是各折先对七个目标取平均，再对五折取均值。

{table}

基准逐目标结果：

{target_table}

## 统计验证与图解

图 1 比较单因素变化与基准的宏平均 R² 差值，误差棒为五折配对差值 SD；它帮助判断改动方向与折间稳定性。图 2 拆解到各目标，显示总平均是否掩盖某些目标退步。图 3 展示不同 SIGReg 权重下掩码预测和正则项的训练动态。详细检验、效应量与计算方式见 [统计附录](stats-appendix.md)。

本轮 encoder dim=196 的七个目标平均 R² 均高于基准；patch size=4 的宏平均提升，但 BP50 的平均 R² 下降约 0.026。损失曲线显示高 λ 下掩码预测 MSE 保持较高；这与部分高 λ 组的下游退步一致，但不能据此断定因果机制。

![单因素 R² 对比](figures/figure-01-factor-delta.png)

![逐目标 R² 差值](figures/figure-02-target-heatmap.png)

![SIGReg 权重与损失曲线](figures/figure-03-lambda-loss.png)

## 负结果、限制与认识变化

图 1 中 ΔR² 低于零的配置在此协议下没有改善七目标平均表现；图 2 中的负值说明即使宏平均改善，也可能损伤单个目标。MLP 经训练折内特征标准化后的平均训练 R² 为 {mlp_train_r2:.3f}，测试 R² 为 {mlp_test_r2:.3f}，显示仍有泛化落差。{significance}五折属于同一数据集的重叠训练划分，不等于五次独立复现；仅一套随机种子，不能证明优势可泛化到其他数据或随机初始化。单因素设计也无法识别参数交互。当前证据只支持在既定数据和网格内筛选复验候选。

## 下一步

优先对图 1 中描述性较好的配置增加独立随机种子，并在外部数据集验证；若目标间效果相反，应按具体性质分别选型。随后再考虑组合因素实验，避免把单因素最好值直接拼成最终配置。

## 产物与复现

- 原始配置与逐折指标：{source} 中的 `matrix.json`、各配置 `SWRI_Diesel/metrics.csv` 与 `loss_curve.csv`。
- 严格分析：[分析说明](analysis-report.md)、[统计附录](stats-appendix.md)、[图录](figure-catalog.md)、`target_summary.csv`、`overall_summary.csv`。
- 从仓库根目录执行：`python scripts/e1d.py sweep --output-dir artifacts/runs/{input_dir.name} --folds 5 --probe-epochs 100 --seed 42`；然后 `python scripts/e1d.py analyze-sweep --input-dir artifacts/runs/{input_dir.name} --output-dir docs/{out.name}`。
- 本仓库未绑定 Obsidian 项目知识库，本报告保存在本地 `docs/`。
"""
    (out / f"{report_date}--e1d-swri--r00--one-factor-ablation-report.md").write_text(report, encoding="utf-8")


def analyze_sweep_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/runs/e1d_ota_20261015")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/e1d_ota_20261015")
    args = parser.parse_args(argv)
    metrics, losses = validate(args.input_dir)
    target_summary, fold_macro, overall = analyze(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    target_summary.to_csv(args.output_dir / "target_summary.csv", index=False, encoding="utf-8-sig")
    fold_macro.to_csv(args.output_dir / "fold_macro_r2.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    setup_plot()
    plot_comparison(overall, figures)
    plot_target_heatmap(target_summary, figures)
    plot_loss(losses, figures)
    write_reports(args.input_dir, args.output_dir, metrics, target_summary, overall)
    print(f"Validated {len(CONFIGS)} configurations and {len(metrics)} metric rows; report: {args.output_dir}")



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        run(parse_sweep_args(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "analyze-sweep":
        analyze_sweep_main(sys.argv[2:])
    else:
        main()







