"""E1D: small-data, single-channel Laya latent prediction for NIR spectra."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
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
from scripts.run_e1c import evaluate_regression, fold_context, limit_dataset, target_stats
from splits.cross_validation import pooled_group_balanced_folds

EXPERIMENT_ID = "e1d"


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
    from scripts.run_e1c import ProbeResult
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


def plot_ssl_losses(history: list[dict], output_dir: Path) -> None:
    """Save the three recorded SSL loss curves for one fold."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (key, title, color) in zip(axes, (
        ("loss", "Total loss", "#c56039"),
        ("invariance_mse", "Masked latent MSE", "#16857e"),
        ("sigreg", "SIGReg loss", "#6654bd"),
    )):
        axis.plot(epochs, [row[key] for row in history], color=color, linewidth=1.8)
        axis.set(title=title, xlabel="SSL epoch", ylabel="Loss")
        axis.grid(alpha=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ssl_loss_components.png", dpi=180)
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
        plot_ssl_losses(ssl_history, asset)
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
        histories.extend(fold_history)
        assets.append({"dataset": dataset_name, "fold": fold.fold, "asset_dir": str(asset)})
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(output_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
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


if __name__ == "__main__":
    main()
