"""Run matched MAE vs LeJEPA frozen-representation evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

# This script only writes PNG files. Select a non-interactive backend before
# importing pyplot so no Tk resources are created in IDE/worker contexts.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.shared.backbone import SpectralPatchTransformer, SpectralUNet
from models.baselines.classical import pca_ridge, pls, random_forest, raw_ridge, rbf_svr
from models.lejepa.model import SpectralLeJEPA
from models.mae.model import SpectralMAE
from models.shared.masking import SpectralMasker, point_mask_to_patch_mask
from models.shared.regressors import RidgeRegressor
from preprocessing.spectral import FoldSpectralStandardizer, preprocess_pair


SUPPORTED_METHODS = (
    "raw_ridge",
    "pca_ridge",
    "pls",
    "rbf_svr",
    "random_forest",
    "random_encoder",
    "mae",
    "lejepa",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_encoder(input_length: int, args: argparse.Namespace):
    encoder_cls = SpectralUNet if args.encoder_type == "unet" else SpectralPatchTransformer
    return encoder_cls(
        input_length=input_length,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )


def batch_plan(n_samples: int, batch_size: int, epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    plans = []
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        plans.append([order[i : i + batch_size] for i in range(0, n_samples, batch_size)])
    return plans


def train_ssl(
    method: str,
    spectra: np.ndarray,
    initial_encoder_state: dict,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    progress_label: str,
):
    if method not in {"mae", "lejepa"}:
        raise ValueError(f"Mask-based self-supervised training is only supported for mae/lejepa, got {method}")
    encoder = make_encoder(spectra.shape[1], args).to(device)
    encoder.load_state_dict(copy.deepcopy(initial_encoder_state))
    if method == "mae":
        model = SpectralMAE(encoder, decoder_hidden_dim=args.mae_decoder_hidden_dim).to(device)
    elif method == "lejepa":
        model = SpectralLeJEPA(
            encoder,
            projection_dim=args.projection_dim,
            projector_hidden_dim=args.projector_hidden_dim,
            sigreg_weight=args.sigreg_weight,
            sigreg_knots=args.sigreg_knots,
            sigreg_projections=args.sigreg_projections,
        ).to(device)
    else:
        raise ValueError(method)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, len(batch_plan(len(spectra), args.batch_size, 1, seed)[0]))
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = min(total_updates - 1, max(0, int(total_updates * args.warmup_ratio)))
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=args.warmup_start_factor, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_updates - warmup_steps), eta_min=args.min_learning_rate)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=args.min_learning_rate)
    tensor = torch.as_tensor(spectra, dtype=torch.float32, device=device)
    masker = SpectralMasker(args.mask_ratio, args.mask_mode, args.mask_block_length)
    plans = batch_plan(len(tensor), args.batch_size, args.epochs, seed)
    history = []
    updates = sum(len(batches) for batches in plans)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    epoch_bar = tqdm(
        plans, total=args.epochs, desc=progress_label, unit="epoch", dynamic_ncols=True, ascii=True
    )
    for epoch, batches in enumerate(epoch_bar):
        model.train()
        total = 0.0
        component_totals: dict[str, float] = {}
        for step, indices in enumerate(batches):
            batch = tensor[torch.as_tensor(indices, device=device)]
            # Reconstructing the generator from (epoch, step) makes masks paired
            # across independently trained MAE and LeJEPA runs.
            generator = torch.Generator().manual_seed(seed + epoch * 100003 + step)
            masked_views, masks = [], []
            for _ in range(args.views):
                masked_batch, point_mask = masker(batch, generator=generator)
                masked_views.append(masked_batch)
                masks.append(point_mask_to_patch_mask(point_mask, encoder.patch_size, encoder.n_patches))
            output = model(torch.stack(masked_views), torch.stack(masks))
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            total += float(output["loss"].detach()) * len(batch)
            for key in ("reconstruction_mse", "invariance_mse", "sigreg"):
                if key in output:
                    component_totals[key] = component_totals.get(key, 0.0) + float(
                        output[key].detach()
                    ) * len(batch)
        record = {"epoch": epoch + 1, "loss": total / len(tensor), "learning_rate": optimizer.param_groups[0]["lr"]}
        record.update({key: value / len(tensor) for key, value in component_totals.items()})
        history.append(record)
        epoch_bar.set_postfix(loss=f"{history[-1]['loss']:.5f}")
    synchronize(device)
    seconds = time.perf_counter() - started
    timing = {
        "pretrain_seconds": seconds,
        "updates": updates,
        "samples_per_second": (len(tensor) * args.epochs) / seconds,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
    }
    return model.encoder, history, timing


@torch.inference_mode()
def extract_embeddings(
    encoder,
    spectra: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoder.eval()
    values = torch.as_tensor(spectra, dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(values), batch_size):
        _, pooled = encoder(values[start : start + batch_size], None)
        output.append(pooled.cpu().numpy())
    return np.concatenate(output, axis=0)


def plot_fold_history(history: pd.DataFrame, method: str, fold: int, path: Path, dpi: int) -> None:
    columns = [column for column in history.columns if column != "epoch"]
    figure, axes = plt.subplots(len(columns), 1, figsize=(7.0, 3.1 * len(columns)), squeeze=False)
    for axis, column in zip(axes[:, 0], columns):
        axis.plot(history["epoch"], history[column], linewidth=1.8)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(column)
        axis.set_title(f"{method.upper()} fold {fold}: {column}")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_aggregate_histories(run_dir: Path, dpi: int) -> None:
    methods = ("mae", "lejepa")
    figure, axes = plt.subplots(1, len(methods), figsize=(12, 4.2), squeeze=False)
    for axis, method in zip(axes[0], methods):
        histories = [pd.read_csv(path) for path in sorted(run_dir.glob(f"fold*_{method}_history.csv"))]
        if not histories:
            axis.set_visible(False)
            continue
        loss_matrix = np.vstack([frame["loss"].to_numpy(float) for frame in histories])
        epochs = histories[0]["epoch"].to_numpy()
        mean = loss_matrix.mean(axis=0)
        std = loss_matrix.std(axis=0)
        axis.plot(epochs, mean, linewidth=2.0, label="fold mean")
        axis.fill_between(epochs, mean - std, mean + std, alpha=0.22, label="±1 SD")
        axis.set_title(f"{method.upper()} total objective")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Pretraining loss across outer folds")
    figure.tight_layout()
    figure.savefig(run_dir / "loss_curves_aggregate.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def inner_folds(n: int, seed: int, n_splits: int = 4):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_ids = np.arange(n) % min(n_splits, n)
    assigned = np.empty(n, dtype=int)
    assigned[order] = fold_ids
    return [(np.flatnonzero(assigned != i), np.flatnonzero(assigned == i)) for i in np.unique(assigned)]


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    alphas: tuple[float, ...],
    inner_splits: int,
) -> float:
    folds = inner_folds(len(x), seed, n_splits=inner_splits)
    target_scale = y.std(axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    scores = []
    for alpha in alphas:
        errors = []
        for train, valid in folds:
            prediction = RidgeRegressor(alpha).fit(x[train], y[train]).predict(x[valid])
            errors.append(np.mean(((prediction - y[valid]) / target_scale) ** 2))
        scores.append((float(np.mean(errors)), alpha))
    return min(scores)[1]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> list[dict]:
    rows = []
    for column, target in enumerate(target_names):
        residual = y_pred[:, column] - y_true[:, column]
        denom = np.sum((y_true[:, column] - y_true[:, column].mean()) ** 2)
        rows.append(
            {
                "target": target,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "r2": float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan,
                "rpd": float(np.std(y_true[:, column], ddof=1) / np.sqrt(np.mean(residual**2)))
                if len(y_true) > 1 and np.sqrt(np.mean(residual**2)) > 0 else np.nan,
            }
        )
    return rows


def parse_optional_int(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def parse_number_or_string(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def append_evaluation_rows(
    fold: int,
    method: str,
    test_idx: np.ndarray,
    predicted: np.ndarray,
    train_idx: np.ndarray,
    train_predicted: np.ndarray,
    targets: np.ndarray,
    samples: pd.DataFrame,
    target_names: list[str],
    prediction_rows: list[dict],
    metric_rows: list[dict],
) -> None:
    for local, sample_index in enumerate(test_idx):
        for column, target in enumerate(target_names):
            prediction_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "sample_id": samples.loc[sample_index, "sample_id"],
                    "group": samples.loc[sample_index, "group"],
                    "target": target,
                    "y_true": targets[sample_index, column],
                    "y_pred": predicted[local, column],
                }
            )
    for split, indices, values in (("train", train_idx, train_predicted), ("test", test_idx, predicted)):
        for row in metrics(targets[indices], values, target_names):
            metric_rows.append({"fold": fold, "method": method, "split": split, **row})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=ROOT / "artifacts" / "canonical" / "crude_oil_private",
        help="prepare_data.py 生成的规范化数据目录",
    )
    # Evaluation and runtime.
    parser.add_argument("--protocol", choices=("pooled", "logo"), default="pooled")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-start-factor", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-folds", type=int, default=None, help="Smoke/debug only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preprocessing",
        choices=("R", "FD", "SD", "SNV", "MSC", "SG"),
        default="R",
        help="Spectral transform: raw, first/second derivative, SNV, MSC, or Savitzky-Golay smoothing",
    )
    parser.add_argument("--sg-window", type=int, default=11, help="Odd Savitzky-Golay window length")
    parser.add_argument("--sg-polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument(
        "--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS),
        help="Methods to evaluate in the stated order",
    )
    # Shared encoder.
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--encoder-type", choices=("transformer", "unet"), default="transformer",
        help="Backbone used by MAE and LeJEPA; LeJEPA always uses its encoder plus projector predictor",
    )
    # Shared mask and method-specific heads.
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--mask-mode", choices=("point", "block"), default="point")
    parser.add_argument("--mask-block-length", type=int, default=4)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--mae-decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--projector-hidden-dim", type=int, default=128)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    # Frozen-regression evaluation and plots.
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    parser.add_argument("--ridge-inner-splits", type=int, default=4)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--pls-components", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--pls-max-iter", type=int, default=500)
    parser.add_argument("--pls-tolerance", type=float, default=1e-6)
    parser.add_argument("--svr-c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--svr-epsilon-values", type=float, nargs="+", default=[0.01, 0.1, 0.2])
    parser.add_argument(
        "--svr-gamma-values", type=parse_number_or_string, nargs="+", default=["scale", "auto"]
    )
    parser.add_argument("--rf-estimators", type=int, nargs="+", default=[200, 500])
    parser.add_argument(
        "--rf-max-depths", type=parse_optional_int, nargs="+", default=[None, 8, 16]
    )
    parser.add_argument("--rf-min-samples-leaf", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--rf-max-features", type=parse_number_or_string, nargs="+", default=[1.0, "sqrt"]
    )
    parser.add_argument("--rf-jobs", type=int, default=-1)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.views < 2:
        raise ValueError("--views must be at least 2")
    if args.embedding_dim % args.heads != 0:
        raise ValueError("--embedding-dim must be divisible by --heads")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("--mask-ratio must lie in (0, 1)")
    if args.mask_block_length < 1:
        raise ValueError("--mask-block-length must be positive")
    if args.sg_window % 2 == 0 or args.sg_window <= args.sg_polyorder:
        raise ValueError("--sg-window must be odd and greater than --sg-polyorder")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    if args.ridge_inner_splits < 2:
        raise ValueError("--ridge-inner-splits must be at least 2")
    set_seed(args.seed)
    canonical = args.canonical_dir
    if not canonical.is_absolute():
        canonical = ROOT / canonical
    if not (canonical / "spectra_resampled.npy").exists():
        legacy = ROOT / "artifacts" / "canonical"
        if (legacy / "spectra_resampled.npy").exists():
            canonical = legacy
        else:
            raise FileNotFoundError(
                f"未找到规范化数据：{canonical}。请先运行 scripts/prepare_data.py。"
            )
    spectra = np.load(canonical / "spectra_resampled.npy")
    target_frame = pd.read_csv(canonical / "targets.csv")
    target_names = target_frame.columns.tolist()
    targets = target_frame.to_numpy(float)
    samples = pd.read_csv(canonical / "samples.csv")
    split_name = "folds_pooled.json" if args.protocol == "pooled" else "folds_leave_one_group_out.json"
    folds = json.loads((canonical / split_name).read_text(encoding="utf-8"))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    device = torch.device(args.device)
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.protocol}_{timestamp}_seed{args.seed}_ep{args.epochs}"
    run_dir = ROOT / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"

    def log(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        tqdm.write(line)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(
        f"start protocol={args.protocol} folds={len(folds)} epochs={args.epochs} "
        f"batch_size={args.batch_size} preprocessing={args.preprocessing} device={device} output={run_dir}"
    )
    prediction_rows, metric_rows, cost_rows = [], [], []

    fold_bar = tqdm(folds, desc="outer folds", unit="fold", dynamic_ncols=True, ascii=True)
    for fold_data in fold_bar:
        fold = int(fold_data["fold"])
        fold_bar.set_postfix(fold=fold)
        train_idx = np.asarray(fold_data["train_idx"], dtype=int)
        test_idx = np.asarray(fold_data["test_idx"], dtype=int)
        transformed_train, transformed_test = preprocess_pair(
            spectra[train_idx],
            spectra[test_idx],
            method=args.preprocessing,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
        )
        standardizer = FoldSpectralStandardizer().fit(transformed_train)
        train_x = standardizer.transform(transformed_train).astype(np.float32)
        test_x = standardizer.transform(transformed_test).astype(np.float32)
        set_seed(args.seed + fold)
        initial_encoder = make_encoder(spectra.shape[1], args).state_dict()
        log(f"fold={fold} train={len(train_idx)} test={len(test_idx)} begin")

        train_y = targets[train_idx]
        regression_folds = inner_folds(len(train_x), args.seed + fold, args.ridge_inner_splits)

        classical_runners = {
            "raw_ridge": lambda: raw_ridge(
                train_x, train_y, test_x, regression_folds, tuple(args.ridge_alphas)
            ),
            "pca_ridge": lambda: pca_ridge(
                train_x, train_y, test_x, regression_folds, tuple(args.pca_components),
                tuple(args.ridge_alphas), args.seed + fold,
            ),
            "pls": lambda: pls(
                train_x, train_y, test_x, regression_folds, tuple(args.pls_components),
                args.pls_max_iter, args.pls_tolerance,
            ),
            "rbf_svr": lambda: rbf_svr(
                train_x, train_y, test_x, regression_folds, tuple(args.svr_c_values),
                tuple(args.svr_epsilon_values), tuple(args.svr_gamma_values),
            ),
            "random_forest": lambda: random_forest(
                train_x, train_y, test_x, regression_folds, tuple(args.rf_estimators),
                tuple(args.rf_max_depths), tuple(args.rf_min_samples_leaf),
                tuple(args.rf_max_features), args.seed + fold, args.rf_jobs,
            ),
        }
        for method in args.methods:
            if method not in classical_runners:
                continue
            method_started = time.perf_counter()
            log(f"fold={fold} method={method} fitting begin")
            result = classical_runners[method]()
            total_seconds = time.perf_counter() - method_started
            cost_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "pretrain_seconds": 0.0,
                    "embedding_seconds": 0.0,
                    "regression_seconds": total_seconds,
                    "ridge_seconds": total_seconds if "ridge" in method else 0.0,
                    "total_seconds": total_seconds,
                    "updates": 0,
                    "samples_per_second": len(train_x) / total_seconds,
                    "model_parameters": np.nan,
                    "encoder_parameters": 0,
                    "peak_gpu_memory_mb": 0.0,
                    "selected_params": json.dumps(result.selected_params, ensure_ascii=False),
                }
            )
            append_evaluation_rows(
                fold, method, test_idx, result.prediction, train_idx,
                result.train_prediction, targets, samples,
                target_names,
                prediction_rows, metric_rows,
            )
            log(
                f"fold={fold} method={method} done total_seconds={total_seconds:.2f} "
                f"selected_params={result.selected_params}"
            )

        if "random_encoder" in args.methods:
            method = "random_encoder"
            method_started = time.perf_counter()
            log(f"fold={fold} method={method} embedding begin")
            encoder = make_encoder(spectra.shape[1], args).to(device)
            encoder.load_state_dict(copy.deepcopy(initial_encoder))
            synchronize(device)
            embedding_started = time.perf_counter()
            train_z = extract_embeddings(encoder, train_x, device, args.embedding_batch_size)
            test_z = extract_embeddings(encoder, test_x, device, args.embedding_batch_size)
            synchronize(device)
            embedding_seconds = time.perf_counter() - embedding_started
            regression_started = time.perf_counter()
            alpha = select_ridge_alpha(
                train_z, train_y, args.seed + fold, tuple(args.ridge_alphas),
                args.ridge_inner_splits,
            )
            predictor = RidgeRegressor(alpha).fit(train_z, train_y)
            predicted = predictor.predict(test_z)
            train_predicted = predictor.predict(train_z)
            regression_seconds = time.perf_counter() - regression_started
            total_seconds = time.perf_counter() - method_started
            encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
            cost_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "pretrain_seconds": 0.0,
                    "embedding_seconds": embedding_seconds,
                    "regression_seconds": regression_seconds,
                    "ridge_seconds": regression_seconds,
                    "total_seconds": total_seconds,
                    "updates": 0,
                    "samples_per_second": len(train_x) / total_seconds,
                    "model_parameters": encoder_parameters,
                    "encoder_parameters": encoder_parameters,
                    "peak_gpu_memory_mb": 0.0,
                    "selected_params": json.dumps({"alpha": alpha}),
                }
            )
            append_evaluation_rows(
                fold, method, test_idx, predicted, train_idx, train_predicted,
                targets, samples,
                target_names,
                prediction_rows, metric_rows,
            )
            log(
                f"fold={fold} method={method} done embedding_seconds={embedding_seconds:.2f} "
                f"ridge_seconds={regression_seconds:.2f} total_seconds={total_seconds:.2f} "
                f"ridge_alpha={alpha:g}"
            )
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for method in [name for name in args.methods if name in ("mae", "lejepa")]:
            method_started = time.perf_counter()
            log(f"fold={fold} method={method} pretraining begin")
            encoder, history, timing = train_ssl(
                method, train_x, initial_encoder, args, args.seed + fold, device,
                progress_label=f"fold {fold} {method}",
            )
            synchronize(device)
            embedding_started = time.perf_counter()
            train_z = extract_embeddings(encoder, train_x, device, args.embedding_batch_size)
            test_z = extract_embeddings(encoder, test_x, device, args.embedding_batch_size)
            synchronize(device)
            embedding_seconds = time.perf_counter() - embedding_started
            ridge_started = time.perf_counter()
            alpha = select_ridge_alpha(
                train_z,
                targets[train_idx],
                args.seed + fold,
                tuple(args.ridge_alphas),
                args.ridge_inner_splits,
            )
            predictor = RidgeRegressor(alpha).fit(train_z, targets[train_idx])
            predicted = predictor.predict(test_z)
            train_predicted = predictor.predict(train_z)
            ridge_seconds = time.perf_counter() - ridge_started
            torch.save(
                {"encoder": encoder.state_dict(), "fold": fold, "method": method},
                run_dir / f"fold{fold}_{method}_encoder.pt",
            )
            history_frame = pd.DataFrame(history)
            history_frame.to_csv(run_dir / f"fold{fold}_{method}_history.csv", index=False)
            plot_fold_history(
                history_frame,
                method,
                fold,
                run_dir / f"fold{fold}_{method}_loss.png",
                args.plot_dpi,
            )
            total_seconds = time.perf_counter() - method_started
            cost_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    **timing,
                    "embedding_seconds": embedding_seconds,
                    "regression_seconds": ridge_seconds,
                    "ridge_seconds": ridge_seconds,
                    "total_seconds": total_seconds,
                    "ridge_alpha": alpha,
                    "selected_params": json.dumps({"alpha": alpha}),
                }
            )
            log(
                f"fold={fold} method={method} done pretrain_seconds={timing['pretrain_seconds']:.2f} "
                f"embedding_seconds={embedding_seconds:.2f} ridge_seconds={ridge_seconds:.2f} "
                f"total_seconds={total_seconds:.2f} updates={timing['updates']} "
                f"peak_gpu_memory_mb={timing['peak_gpu_memory_mb']:.1f} ridge_alpha={alpha:g} "
                f"final_loss={history[-1]['loss']:.6f}"
            )
            append_evaluation_rows(
                fold, method, test_idx, predicted, train_idx, train_predicted,
                targets, samples,
                target_names,
                prediction_rows, metric_rows,
            )
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Persist all completed folds immediately so Ctrl+C never loses them.
        pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.partial.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(run_dir / "fold_metrics.partial.csv", index=False)
        pd.DataFrame(cost_rows).to_csv(run_dir / "costs.partial.csv", index=False)
        log(f"fold={fold} complete")

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(run_dir / "costs.csv", index=False)
    summary = []
    for (method, split, target), frame in pd.DataFrame(metric_rows).groupby(["method", "split", "target"]):
        summary.append({"method": method, "split": split, "target": target, "n": len(frame), **{
            key: float(frame[key].mean()) for key in ("rmse", "mae", "bias", "r2", "rpd")
        }})
    pd.DataFrame(summary).to_csv(run_dir / "summary_metrics.csv", index=False)
    plot_aggregate_histories(run_dir, args.plot_dpi)
    run_config = {
        **vars(args),
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_directory": str(run_dir.resolve()),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for partial in run_dir.glob("*.partial.csv"):
        partial.unlink()
    log("run complete")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
