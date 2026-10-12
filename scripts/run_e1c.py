"""E1C: patch-masked Transformer LeJEPA followed by a frozen encoder probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from data.dataset_adapters import CanonicalDataset, merge_datasets
from data.SWRI_Diesel.adapter import load_dataset as load_swri
from data.crude_oil_1.adapter import load_dataset as load_oil1
from data.crude_oil_2.adapter import load_dataset as load_oil2
from data.lucas.adapter import load_dataset as load_lucas
from splits.cross_validation import pooled_group_balanced_folds
from models.shared.experiment import build_ssl_encoder, parameter_count
from models.shared.scheduling import build_step_scheduler

try:
    import torch
except ImportError:
    torch = None
DEFAULT_EMBEDDING_DIM = 128
EXPERIMENT_ID = "e1c"
PREPROCESSING = "none"


@dataclass(frozen=True)
class SSLResult:
    """Artifacts produced by the self-supervised pretraining stage."""

    encoder: object
    model_state: dict
    train_embedding: np.ndarray
    test_embedding: np.ndarray
    history: list[dict]
    parameter_count: int
    source_checkpoint: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Predictions and artifacts produced by the frozen probe stage."""

    prediction: np.ndarray
    train_prediction: np.ndarray
    state: dict
    history: list[dict]
    parameter_count: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when it is available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def parse_batch_size(value: str) -> int | str:
    """Parse a positive mini-batch size or the special value ``full``."""
    if str(value).lower() == "full":
        return "full"
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch-size must be a positive integer or 'full'"
        ) from exc
    if value < 1:
        raise argparse.ArgumentTypeError("batch-size must be positive")
    return value


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
    """Return RMSE, R², and RPD for one target column."""
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


def frozen_representation(
    model: object,
    train_x: np.ndarray,
    test_x: np.ndarray,
    batch_size: int,
    history: list[dict],
    source_checkpoint: str | None = None,
) -> SSLResult:
    """Freeze one shared Encoder and extract unmasked embeddings once per fold."""
    model.eval()
    encoder = model.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    device = next(encoder.parameters()).device

    def encode(spectra: np.ndarray) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(spectra), batch_size):
                batch = torch.as_tensor(
                    spectra[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                embeddings.append(encoder(batch, None)[1].cpu().numpy())
        if not embeddings:
            return np.empty((0, encoder.embedding_dim), dtype=np.float32)
        return np.concatenate(embeddings)

    return SSLResult(
        encoder=encoder,
        model_state={
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        train_embedding=encode(train_x),
        test_embedding=encode(test_x),
        history=history,
        parameter_count=parameter_count(model),
        source_checkpoint=source_checkpoint,
    )


def pretrain(
    train_x: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    progress_label: str = "SSL",
) -> SSLResult:
    """Train LeJEPA and extract fixed eval-mode embeddings for both splits."""
    from models.lejepa.model import SpectralLeJEPA
    from models.shared.masking import random_patch_mask
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone != "patch_transformer":
        raise ValueError(
            "E1C patch token masking requires --backbone patch_transformer"
        )
    torch.manual_seed(seed)
    enc = build_ssl_encoder(
        args.backbone,
        train_x.shape[1],
        args.patch_size,
        DEFAULT_EMBEDDING_DIM,
        input_channels=1,
    ).to(device)
    model = SpectralLeJEPA(enc, sigreg_weight=args.sigreg_weight, mask_mode="patch").to(
        device
    )
    if enc.n_patches < 2:
        raise ValueError("E1C requires at least two patches; reduce --patch-size")
    n_masked = max(1, min(enc.n_patches - 1, round(enc.n_patches * args.mask_ratio)))
    tqdm.write(
        f"[{progress_label}] Transformer on {device} | "
        f"input=[B, {enc.input_length}] | patch_size={enc.patch_size} | "
        f"tokens={enc.n_patches} | masked={n_masked} | embedding={enc.embedding_dim}"
    )
    bs = len(train_x) if args.batch_size == "full" else int(args.batch_size)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x, dtype=torch.float32)),
        batch_size=bs,
        shuffle=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=5e-2
    )
    sch = build_step_scheduler(
        opt,
        args.scheduler,
        len(loader) * args.ssl_epochs,
        len(loader),
        args.scheduler_eta_min,
    )
    hist = []
    epoch_iterator = tqdm(
        range(1, args.ssl_epochs + 1),
        desc=f"{progress_label} | LeJEPA",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iterator:
        model.train()
        losses = []
        invariance_losses = []
        sigreg_losses = []
        for (x,) in loader:
            x = x.to(device)
            out = model(
                x,
                random_patch_mask(
                    len(x),
                    enc.n_patches,
                    args.mask_ratio,
                    device=device,
                ),
            )
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if args.ssl_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.ssl_grad_clip)
            opt.step()
            if sch:
                sch.step()
            losses.append(float(out["loss"].detach()))
            invariance_losses.append(float(out["invariance_mse"].detach()))
            sigreg_losses.append(float(out["sigreg"].detach()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        hist.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "invariance_mse": float(np.mean(invariance_losses)),
                "sigreg": float(np.mean(sigreg_losses)),
                "stage": "ssl_pretrain",
            }
        )
        epoch_iterator.set_postfix(loss=f"{epoch_loss:.5g}")
    epoch_iterator.close()
    return frozen_representation(model, train_x, test_x, bs, hist)


def fit_probe(
    train_embedding: np.ndarray,
    train_targets: np.ndarray,
    test_embedding: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    progress_label: str = "Probe",
) -> ProbeResult:
    """Train a probe only on fixed embeddings and return denormalized predictions."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    mean, scale, obs = target_stats(train_targets)
    y = (np.where(np.isfinite(train_targets), train_targets, mean) - mean) / scale
    x = torch.as_tensor(train_embedding, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    mask = torch.as_tensor(obs.astype(np.float32))
    torch.manual_seed(seed)
    if args.probe in ("linear", "ridge"):
        head = nn.Linear(x.shape[1], train_targets.shape[1])
        decay = 0 if args.probe == "linear" else args.probe_weight_decay
    else:
        head = nn.Sequential(
            nn.Linear(x.shape[1], args.probe_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.probe_hidden_dim, train_targets.shape[1]),
        )
        decay = args.probe_weight_decay
    bs = len(x) if args.batch_size == "full" else int(args.batch_size)
    loader = DataLoader(TensorDataset(x, yt, mask), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.probe_learning_rate, weight_decay=decay
    )
    sch = build_step_scheduler(
        opt,
        args.scheduler,
        len(loader) * args.probe_epochs,
        len(loader),
        args.scheduler_eta_min,
    )
    hist = []
    epoch_iterator = tqdm(
        range(1, args.probe_epochs + 1),
        desc=f"{progress_label} | Probe",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iterator:
        head.train()
        ls = []
        for bx, by, bm in loader:
            loss = ((head(bx) - by) ** 2 * bm).sum() / bm.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sch:
                sch.step()
            ls.append(float(loss.detach()))
        epoch_loss = float(np.mean(ls)) if ls else float("nan")
        hist.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "stage": "frozen_probe",
                "probe": args.probe,
            }
        )
        epoch_iterator.set_postfix(loss=f"{epoch_loss:.5g}")
    epoch_iterator.close()
    head.eval()
    with torch.no_grad():
        train_prediction = head(x).cpu().numpy()
        prediction = (
            head(torch.as_tensor(test_embedding, dtype=torch.float32)).cpu().numpy()
        )
    probe_state = {
        "probe": {
            key: value.detach().cpu().clone()
            for key, value in head.state_dict().items()
        },
        "target_mean": torch.as_tensor(mean, dtype=torch.float32),
        "target_scale": torch.as_tensor(scale, dtype=torch.float32),
        "probe_type": args.probe,
        "embedding_dim": int(x.shape[1]),
        "output_dim": int(train_targets.shape[1]),
        "hidden_dim": int(args.probe_hidden_dim),
    }
    return ProbeResult(
        prediction=prediction * scale + mean,
        train_prediction=train_prediction * scale + mean,
        state=probe_state,
        history=hist,
        parameter_count=parameter_count(head),
    )


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


def load_ssl_checkpoint(
    checkpoint_path: Path,
    train_x: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    context: dict,
) -> SSLResult:
    """Restore fold-specific SSL weights and recompute frozen embeddings."""
    from models.lejepa.model import SpectralLeJEPA

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or "model_state" not in checkpoint:
        raise ValueError(
            "Use the E1C ssl_checkpoint.pt containing model config and fold context"
        )
    if checkpoint["context"] != context:
        raise ValueError(
            f"{checkpoint_path}: spectral data or train/test fold does not match"
        )
    config = checkpoint["model_config"]
    if (
        config["input_length"] != train_x.shape[1]
        or config["patch_size"] != args.patch_size
    ):
        raise ValueError(
            f"{checkpoint_path}: input length or --patch-size does not match"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_ssl_encoder(
        config["backbone"],
        config["input_length"],
        config["patch_size"],
        config["embedding_dim"],
        input_channels=config["input_channels"],
    ).to(device)
    model = SpectralLeJEPA(
        encoder, sigreg_weight=config["sigreg_weight"], mask_mode=config["mask_mode"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    batch_size = len(train_x) if args.batch_size == "full" else int(args.batch_size)
    tqdm.write(f"[SSL] Loaded {checkpoint_path}; skipping self-supervised training")
    return frozen_representation(
        model,
        train_x,
        test_x,
        batch_size,
        checkpoint["history"],
        str(checkpoint_path.resolve()),
    )


def checkpoint_for_fold(source: Path, output_dir: Path, run_id: str) -> Path:
    """Resolve a checkpoint file, fold folder, dataset folder, or run folder."""
    if source.is_file():
        return source
    candidates = (
        source / "ssl_checkpoint.pt",
        source / "representations" / run_id / "ssl_checkpoint.pt",
        source / output_dir.name / "representations" / run_id / "ssl_checkpoint.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No ssl_checkpoint.pt for {output_dir.name}/{run_id} under {source}"
    )


def save_ssl_assets(
    asset_dir: Path,
    ssl_result: SSLResult,
    context: dict,
    args: argparse.Namespace,
) -> None:
    """Save reusable SSL weights before starting any attribute probes."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "backbone": "patch_transformer",
        "mask_mode": "patch",
        "input_channels": 1,
        "input_length": ssl_result.encoder.input_length,
        "patch_size": ssl_result.encoder.patch_size,
        "embedding_dim": ssl_result.encoder.embedding_dim,
        "sigreg_weight": args.sigreg_weight,
    }
    if ssl_result.source_checkpoint:
        original = torch.load(
            ssl_result.source_checkpoint, map_location="cpu", weights_only=True
        )
        model_config = original["model_config"]
    torch.save(
        {
            "format_version": 1,
            "model_config": model_config,
            "model_state": ssl_result.model_state,
            "context": context,
            "history": ssl_result.history,
        },
        asset_dir / "ssl_checkpoint.pt",
    )
    torch.save(ssl_result.model_state, asset_dir / "lejepa_pretrained.pt")
    torch.save(
        {
            key: value.detach().cpu().clone()
            for key, value in ssl_result.encoder.state_dict().items()
        },
        asset_dir / "encoder_frozen.pt",
    )
    np.savez_compressed(
        asset_dir / "representations.npz",
        train_embedding=ssl_result.train_embedding,
        test_embedding=ssl_result.test_embedding,
    )
    pd.DataFrame(
        ssl_result.history,
        columns=["epoch", "loss", "invariance_mse", "sigreg", "stage"],
    ).to_csv(asset_dir / "history.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "training_mode": "ssl_then_frozen_probe",
        "preprocessing": PREPROCESSING,
        "backbone": "patch_transformer",
        "mask_mode": "patch",
        "input_channels": 1,
        "input_length": ssl_result.encoder.input_length,
        "patch_size": ssl_result.encoder.patch_size,
        "n_patches": ssl_result.encoder.n_patches,
        "embedding_dim": ssl_result.encoder.embedding_dim,
        "ssl_uses_all_train_spectra": True,
        "probe_missing_target_policy": "per_target_observed",
        "encoder_frozen_before_probe": True,
        "encoder_eval_during_probe": True,
        "encoder_parameters_require_grad": False,
        "ssl_params": ssl_result.parameter_count,
        "ssl_train_samples": len(ssl_result.train_embedding),
        "source_checkpoint": ssl_result.source_checkpoint,
        "context": context,
    }
    (asset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def save_probe_assets(
    asset_dir: Path,
    target: str,
    target_index: int,
    result: ProbeResult,
    train_indices: np.ndarray,
) -> None:
    """Save one scalar regression head and its training sample membership."""
    probe_dir = asset_dir / "probes" / f"target{target_index}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(result.state, target=target, train_indices=train_indices.tolist())
    torch.save(checkpoint, probe_dir / "probe.pt")
    pd.DataFrame(result.history).to_csv(
        probe_dir / "history.csv", index=False, encoding="utf-8-sig"
    )
    (probe_dir / "metadata.json").write_text(
        json.dumps(
            {
                "target": target,
                "n_train": len(train_indices),
                "probe_params": result.parameter_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_fold_losses(history_rows: list[dict], output_dir: Path) -> None:
    """Save three LeJEPA components and the per-target Probe loss separately."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = pd.DataFrame(history_rows)
    ssl = history[history["stage"] == "ssl_pretrain"].sort_values("epoch")
    for column, title, filename in (
        ("loss", "LeJEPA total loss", "lejepa_total_loss.png"),
        ("invariance_mse", "LeJEPA invariance loss", "lejepa_invariance_loss.png"),
        ("sigreg", "LeJEPA SIGReg loss", "lejepa_sigreg_loss.png"),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        if column in ssl and ssl[column].notna().any():
            valid = ssl.dropna(subset=[column])
            axis.plot(valid["epoch"], valid[column], color="#2563eb", linewidth=2)
        else:
            axis.text(
                0.5,
                0.5,
                "Not recorded in this SSL checkpoint",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    probes = history[history["stage"] == "frozen_probe"]
    for target, group in probes.groupby("target", sort=False):
        group = group.sort_values("epoch")
        axis.plot(group["epoch"], group["loss"], label=str(target), linewidth=1.5)
    axis.set_title("Frozen-encoder Probe loss")
    if not probes.empty:
        axis.legend(fontsize=8, ncol=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized target MSE")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "probe_loss.png", dpi=180)
    plt.close(fig)


def plot_first_fold_masks(
    dataset: CanonicalDataset,
    train_indices: np.ndarray,
    patch_size: int,
    mask_ratio: float,
    seed: int,
    output_dir: Path,
) -> None:
    """Plot three reproducible spectra and the patch tokens hidden in one draw."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from models.shared.masking import random_patch_mask

    count = min(3, len(train_indices))
    chosen = np.random.default_rng(seed).choice(
        train_indices, size=count, replace=False
    )
    n_points = dataset.spectra.shape[1]
    n_patches = (n_points + patch_size - 1) // patch_size
    masks = random_patch_mask(
        count,
        n_patches,
        mask_ratio,
        generator=torch.Generator().manual_seed(seed),
    ).numpy()
    wavelengths = np.asarray(dataset.wavelengths_nm, dtype=float)
    boundaries = np.empty(n_points + 1, dtype=float)
    boundaries[1:-1] = (wavelengths[:-1] + wavelengths[1:]) / 2
    boundaries[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
    boundaries[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2

    fig, axes = plt.subplots(count, 1, figsize=(13, 3.1 * count), sharex=True)
    rows = []
    axes = np.atleast_1d(axes)
    for axis, dataset_index, patch_mask in zip(axes, chosen, masks):
        values = np.asarray(dataset.spectra[dataset_index], dtype=float)
        point_mask = np.repeat(patch_mask, patch_size)[:n_points]
        visible = values.copy()
        visible[point_mask] = np.nan
        axis.plot(
            wavelengths, values, color="#64748b", linewidth=1, label="Original spectrum"
        )
        axis.plot(
            wavelengths,
            visible,
            color="#2563eb",
            linewidth=1.2,
            label="Visible patches",
        )
        for patch_index, masked in enumerate(patch_mask):
            start = patch_index * patch_size
            stop = min((patch_index + 1) * patch_size, n_points)
            if masked:
                axis.axvspan(
                    boundaries[start],
                    boundaries[stop],
                    color="#ef4444",
                    alpha=0.2,
                    label=(
                        "Masked patch interval"
                        if patch_index == np.flatnonzero(patch_mask)[0]
                        else None
                    ),
                )
            rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(dataset.sample_id[dataset_index]),
                    "patch_index": patch_index,
                    "point_start": start,
                    "point_end_exclusive": stop,
                    "wavelength_start_nm": boundaries[start],
                    "wavelength_end_nm": boundaries[stop],
                    "masked": bool(masked),
                }
            )
        axis.set_title(
            f"Sample {dataset.sample_id[dataset_index]} (row {dataset_index})"
        )
        axis.set_ylabel("Original intensity")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=3, loc="best")
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        f"First fold: illustrative patch mask (patch size={patch_size}, ratio={mask_ratio:g})\n"
        "Shading marks tokens replaced after patch projection; raw values are unchanged.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "first_fold_patch_masks.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        output_dir / "first_fold_patch_masks.csv", index=False, encoding="utf-8-sig"
    )


def run_dataset(
    dataset: CanonicalDataset,
    dataset_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Run E1C cross-validation for one canonical dataset."""
    dataset.validate()
    dataset = limit_dataset(dataset, args.max_samples)
    if len(dataset.sample_id) < 2:
        raise ValueError("at least 2 samples are required for cross-validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = dataset.targets.to_numpy(float)
    rows = []
    histories = []
    summaries = []
    assets = []
    folds = pooled_group_balanced_folds(
        dataset.groups, min(args.folds, len(dataset.sample_id)), args.seed
    )
    fold_iterator = tqdm(
        folds,
        desc=f"{dataset_name} | folds",
        unit="fold",
        position=0,
    )
    for fold in fold_iterator:
        fold_started = time.perf_counter()
        fold_histories = []
        fold_label = f"{dataset_name} | fold {fold.fold + 1}/{len(folds)}"
        train_x = np.asarray(dataset.spectra[fold.train_idx], dtype=np.float32)
        test_x = np.asarray(dataset.spectra[fold.test_idx], dtype=np.float32)
        tqdm.write(
            f"[{fold_label}] train samples={len(train_x)} | test samples={len(test_x)}"
        )
        run_id = f"lejepa_fold{fold.fold}"
        asset = output_dir / "representations" / run_id
        context = fold_context(dataset, dataset_name, fold)
        if args.ssl_checkpoint is not None:
            checkpoint_path = checkpoint_for_fold(
                args.ssl_checkpoint, output_dir, run_id
            )
            ssl_result = load_ssl_checkpoint(
                checkpoint_path, train_x, test_x, args, context
            )
        else:
            ssl_result = pretrain(
                train_x, test_x, args, args.seed + fold.fold, progress_label=fold_label
            )
        save_ssl_assets(asset, ssl_result, context, args)
        fold_histories.extend(
            dict(row, fold=fold.fold, experiment="ssl", target=None)
            for row in ssl_result.history
        )
        if fold.fold == 0:
            plot_first_fold_masks(
                dataset,
                fold.train_idx,
                args.patch_size,
                args.mask_ratio,
                args.seed,
                output_dir,
            )
        target_iterator = tqdm(
            list(enumerate(dataset.targets.columns)),
            desc=f"{fold_label} | attributes",
            unit="target",
            leave=False,
        )
        for c, target in target_iterator:
            probe_train_indices = np.flatnonzero(
                np.isfinite(targets[fold.train_idx, c])
            )
            observed_test_count = int(np.isfinite(targets[fold.test_idx, c]).sum())
            tqdm.write(
                f"[{fold_label}] {target} | train n={len(probe_train_indices)} | "
                f"test n={observed_test_count}"
            )
            if not len(probe_train_indices):
                tqdm.write(
                    f"[{fold_label}] {target}: no observed training labels; skipping head"
                )
                continue
            probe_train_targets = targets[
                fold.train_idx[probe_train_indices], c : c + 1
            ]
            probe_result = fit_probe(
                ssl_result.train_embedding[probe_train_indices],
                probe_train_targets,
                ssl_result.test_embedding,
                args,
                args.seed + fold.fold + c,
                progress_label=f"{fold_label} | {target}",
            )
            save_probe_assets(
                asset, str(target), c, probe_result, fold.train_idx[probe_train_indices]
            )
            fold_histories.extend(
                dict(row, fold=fold.fold, experiment="probe", target=target)
                for row in probe_result.history
            )
            common = {
                "dataset": dataset_name,
                "experiment": "lejepa_frozen_probe",
                "label": "LeJEPA frozen probe",
                "regressor": args.probe,
                "fold": fold.fold,
                "target": target,
                "representation_run_id": run_id,
                "preprocessing": PREPROCESSING,
                "encoder_frozen": True,
                "encoder_eval": True,
                "ssl_params": ssl_result.parameter_count,
                "probe_params": probe_result.parameter_count,
                "model_total_params": ssl_result.parameter_count
                + probe_result.parameter_count,
                "ssl_train_samples": len(train_x),
                "probe_train_samples": len(probe_train_indices),
            }
            summaries.append(common)
            train_metrics = evaluate_regression(
                probe_train_targets[:, 0], probe_result.train_prediction[:, 0]
            )
            test_metrics = evaluate_regression(
                targets[fold.test_idx, c], probe_result.prediction[:, 0]
            )
            r2_gap = train_metrics["r2"] - test_metrics["r2"]
            rmse_gap = test_metrics["rmse"] - train_metrics["rmse"]
            tqdm.write(
                f"[{fold_label}] {target} | "
                f"train n={train_metrics['n']} RMSE={train_metrics['rmse']:.5g} "
                f"R2={train_metrics['r2']:.5g} RPD={train_metrics['rpd']:.5g} | "
                f"test n={test_metrics['n']} RMSE={test_metrics['rmse']:.5g} "
                f"R2={test_metrics['r2']:.5g} RPD={test_metrics['rpd']:.5g}"
            )
            if train_metrics["n"] or test_metrics["n"]:
                rows.append(
                    {
                        **common,
                        "rmse": test_metrics["rmse"],
                        "r2": test_metrics["r2"],
                        "rpd": test_metrics["rpd"],
                        "train_rmse": train_metrics["rmse"],
                        "train_r2": train_metrics["r2"],
                        "train_rpd": train_metrics["rpd"],
                        "n_train": train_metrics["n"],
                        "n_test": test_metrics["n"],
                        "r2_gap": r2_gap,
                        "rmse_gap": rmse_gap,
                    }
                )
        target_iterator.close()
        histories.extend(fold_histories)
        plot_fold_losses(fold_histories, asset)
        assets.append(
            {
                "dataset": dataset_name,
                "fold": fold.fold,
                "representation_run_id": run_id,
                "asset_dir": str(asset),
            }
        )
        fold_iterator.set_postfix(
            fold=fold.fold + 1,
            probe=args.probe,
            seconds=f"{time.perf_counter() - fold_started:.1f}",
        )
    fold_iterator.close()
    pd.DataFrame(rows).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(histories).to_csv(
        output_dir / "loss_curve.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summaries).to_csv(
        output_dir / "run_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(assets).to_csv(
        output_dir / "ssl_runs.csv", index=False, encoding="utf-8-sig"
    )
    if rows:
        pd.DataFrame(rows).groupby(
            ["dataset", "experiment", "regressor", "target"], as_index=False
        )[
            [
                "rmse",
                "r2",
                "rpd",
                "train_rmse",
                "train_r2",
                "train_rpd",
                "r2_gap",
                "rmse_gap",
            ]
        ].mean().to_csv(
            output_dir / "metrics_aggregate.csv", index=False
        )
    print(
        f"[{dataset_name}] completed: {len(folds)} folds, "
        f"{len(rows)} metric rows -> {output_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for E1C."""
    parser = argparse.ArgumentParser(description="Run E1C frozen LeJEPA probe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/e1c")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swri", "crude_oil"],
        choices=["swri", "crude_oil", "lucas"],
    )
    parser.add_argument("--swri-root", type=Path, default=ROOT / "data/SWRI_Diesel")
    parser.add_argument("--oil1-root", type=Path, default=ROOT / "data/crude_oil_1")
    parser.add_argument("--oil2-root", type=Path, default=ROOT / "data/crude_oil_2")
    parser.add_argument("--lucas-root", type=Path, default=ROOT / "data/lucas")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe", choices=["linear", "ridge", "mlp"], default="linear")
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.4,
        help="Fraction of whole patch tokens replaced by the learned mask token",
    )
    parser.add_argument(
        "--backbone", choices=["patch_transformer"], default="patch_transformer"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=6,
        help="Number of spectral points per non-overlapping patch",
    )
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument(
        "--ssl-grad-clip",
        type=float,
        default=None,
        help="Optional maximum gradient norm during LeJEPA pretraining",
    )
    parser.add_argument("--batch-size", type=parse_batch_size, default=16)
    parser.add_argument(
        "--scheduler",
        choices=["constant", "cosine", "warmup_cosine"],
        default="warmup_cosine",
    )
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        default=None,
        help="Load SSL weights and skip pretraining: prior E1C run/dataset directory or ssl_checkpoint.pt",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args()


def main() -> None:
    """Validate configuration, run selected datasets, and report the output path."""
    started = time.perf_counter()
    args = parse_args()
    if args.folds < 2 or args.ssl_epochs < 0 or args.probe_epochs < 0:
        raise ValueError("folds must be >= 2 and epoch counts must be non-negative")
    if (
        args.learning_rate <= 0
        or args.probe_learning_rate <= 0
        or args.probe_weight_decay < 0
    ):
        raise ValueError(
            "learning rates must be positive and probe weight decay non-negative"
        )
    if not 0 < args.mask_ratio < 1 or args.patch_size < 1:
        raise ValueError("invalid masking or patch arguments")
    if args.ssl_grad_clip is not None and args.ssl_grad_clip <= 0:
        raise ValueError("ssl-grad-clip must be positive")
    if args.scheduler_eta_min < 0 or (
        args.scheduler != "constant"
        and args.scheduler_eta_min > min(args.learning_rate, args.probe_learning_rate)
    ):
        raise ValueError("invalid scheduler eta_min")
    seed_everything(args.seed)
    args.output_dir = args.output_dir.parent / (
        args.output_dir.name + "_" + time.strftime("%Y%m%d_%H%M%S")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{EXPERIMENT_ID}] start | datasets={','.join(args.datasets)} | "
        f"folds={args.folds} | ssl_epochs={args.ssl_epochs} | "
        f"probe_epochs={args.probe_epochs} | output={args.output_dir}"
    )
    print(
        f"[{EXPERIMENT_ID}] backbone={args.backbone} | mask_mode=patch | "
        f"patch_size={args.patch_size} | mask_ratio={args.mask_ratio} | input_channels=1"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "mask_mode": "patch",
                "input_channels": 1,
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if "swri" in args.datasets:
        print("[e1c] loading dataset: SWRI_Diesel")
        swri = load_swri(args.swri_root)
        print(
            f"[e1c] SWRI_Diesel loaded: samples={len(swri.sample_id)}, "
            f"points={swri.spectra.shape[1]}"
        )
        run_dataset(
            swri,
            "SWRI_Diesel_drop_missing",
            args,
            args.output_dir / "SWRI_Diesel_drop_missing",
        )
    if "crude_oil" in args.datasets:
        print("[e1c] loading dataset: crude oil")
        crude_oil = merge_datasets(
            [load_oil1(args.oil1_root), load_oil2(args.oil2_root)],
            name="crude_oil_1+crude_oil_2",
        )
        print(
            f"[e1c] crude oil loaded: samples={len(crude_oil.sample_id)}, "
            f"points={crude_oil.spectra.shape[1]}"
        )
        run_dataset(
            crude_oil,
            "crude_oil_1+crude_oil_2",
            args,
            args.output_dir / "crude_oil_1_plus_crude_oil_2",
        )
    if "lucas" in args.datasets:
        print("[e1c] loading dataset: LUCAS_SOIL")
        lucas = load_lucas(args.lucas_root)
        print(
            f"[e1c] LUCAS_SOIL loaded: samples={len(lucas.sample_id)}, "
            f"points={lucas.spectra.shape[1]}"
        )
        run_dataset(
            lucas,
            "LUCAS_SOIL",
            args,
            args.output_dir / "LUCAS_SOIL",
        )
    result = {
        "output_dir": str(args.output_dir),
        "experiment": EXPERIMENT_ID,
        "preprocessing": PREPROCESSING,
        "probe": args.probe,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
