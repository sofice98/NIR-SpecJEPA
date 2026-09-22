"""Compare LeJEPA downstream regression strategies on one canonical dataset."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.shared.regressors import EncoderMLPRegressor, RidgeRegressor
from preprocessing.spectral import FoldSpectralStandardizer, preprocess_pair
from scripts.run_e1a import (
    append_evaluation_rows,
    extract_embeddings,
    inner_folds,
    make_encoder,
    select_ridge_alpha,
    set_seed,
    synchronize,
    train_ssl,
)


METHODS = ("ridge", "xgboost", "lightgbm", "finetune")


def require_regressor(method: str) -> type:
    """Load an optional tree regressor only when its method is requested."""
    if method == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError(
                "E1D method 'xgboost' requires xgboost. Install project requirements or "
                "rerun without xgboost in --methods."
            ) from error
        return XGBRegressor
    if method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError(
                "E1D method 'lightgbm' requires lightgbm. Install project requirements or "
                "rerun without lightgbm in --methods."
            ) from error
        return LGBMRegressor
    raise ValueError(f"Unsupported nonlinear regressor: {method}")


def parameter_grid(method: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if method == "xgboost":
        return [
            {"n_estimators": n, "max_depth": depth, "learning_rate": rate}
            for n, depth, rate in itertools.product(
                args.xgb_estimators, args.xgb_max_depths, args.xgb_learning_rates
            )
        ]
    if method == "lightgbm":
        return [
            {"n_estimators": n, "num_leaves": leaves, "learning_rate": rate}
            for n, leaves, rate in itertools.product(
                args.lgbm_estimators, args.lgbm_num_leaves, args.lgbm_learning_rates
            )
        ]
    raise ValueError(method)


def make_tree_model(
    method: str,
    model_class: type,
    params: dict[str, Any],
    seed: int,
    jobs: int,
) -> Any:
    if method == "xgboost":
        return model_class(
            objective="reg:squarederror",
            tree_method="hist",
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=jobs,
            verbosity=0,
            **params,
        )
    return model_class(
        objective="regression",
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=jobs,
        verbose=-1,
        **params,
    )


def fit_tree_targets(
    method: str,
    model_class: type,
    params: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    jobs: int,
) -> list[Any]:
    models = []
    for target_index in range(train_y.shape[1]):
        model = make_tree_model(method, model_class, params, seed + target_index, jobs)
        model.fit(train_x, train_y[:, target_index])
        models.append(model)
    return models


def predict_tree_targets(models: list[Any], values: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(values) for model in models])


def select_tree_params(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[dict[str, Any]],
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    model_class = require_regressor(method)
    target_scale = train_y.std(axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    scored = []
    for candidate_index, params in enumerate(candidates):
        errors = []
        for split_index, (fit_idx, valid_idx) in enumerate(folds):
            models = fit_tree_targets(
                method,
                model_class,
                params,
                train_x[fit_idx],
                train_y[fit_idx],
                seed + candidate_index * 1009 + split_index * 97,
                jobs,
            )
            prediction = predict_tree_targets(models, train_x[valid_idx])
            errors.append(np.mean(((prediction - train_y[valid_idx]) / target_scale) ** 2))
        scored.append((float(np.mean(errors)), candidate_index, params))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


class TargetStandardizer:
    def fit(self, values: np.ndarray) -> "TargetStandardizer":
        values = np.asarray(values, dtype=np.float32)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale_ + self.mean_


def train_supervised_epoch(
    model: EncoderMLPRegressor,
    spectra: torch.Tensor,
    targets: torch.Tensor,
    optimizer: AdamW,
    batch_size: int,
    generator: torch.Generator,
    gradient_clip_norm: float,
) -> float:
    model.train()
    order = torch.randperm(len(spectra), generator=generator, device="cpu")
    total = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size].to(spectra.device)
        prediction = model(spectra[indices])
        loss = nn.functional.mse_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(spectra)


@torch.inference_mode()
def predict_finetuned(
    model: EncoderMLPRegressor,
    spectra: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(spectra, dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(tensor), batch_size):
        output.append(model(tensor[start : start + batch_size]).cpu().numpy())
    return np.concatenate(output, axis=0)


def choose_finetune_epochs(
    encoder_state: dict[str, torch.Tensor],
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    fit_idx, valid_idx = inner_folds(len(train_x), seed, args.finetune_inner_splits)[0]
    scaler = TargetStandardizer().fit(train_y[fit_idx])
    encoder = make_encoder(train_x.shape[1], args).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    model = EncoderMLPRegressor(
        encoder, train_y.shape[1], args.mlp_hidden_dim, args.mlp_dropout
    ).to(device)
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.finetune_learning_rate},
            {"params": model.head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.finetune_weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.finetune_epochs), eta_min=args.finetune_min_learning_rate
    )
    fit_x = torch.as_tensor(train_x[fit_idx], dtype=torch.float32, device=device)
    fit_y = torch.as_tensor(scaler.transform(train_y[fit_idx]), dtype=torch.float32, device=device)
    best_epoch, best_loss, stale = 1, float("inf"), 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_supervised_epoch(
            model, fit_x, fit_y, optimizer, args.batch_size, generator, args.gradient_clip_norm
        )
        valid_scaled = predict_finetuned(model, train_x[valid_idx], device, args.embedding_batch_size)
        valid_loss = float(np.mean((valid_scaled - scaler.transform(train_y[valid_idx])) ** 2))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - args.finetune_min_delta:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
        else:
            stale += 1
        scheduler.step()
        if args.finetune_patience > 0 and stale >= args.finetune_patience:
            break
    return best_epoch, history


def fit_finetuned_model(
    encoder_state: dict[str, torch.Tensor],
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[EncoderMLPRegressor, TargetStandardizer, list[dict[str, float]]]:
    scaler = TargetStandardizer().fit(train_y)
    encoder = make_encoder(train_x.shape[1], args).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    model = EncoderMLPRegressor(
        encoder, train_y.shape[1], args.mlp_hidden_dim, args.mlp_dropout
    ).to(device)
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.finetune_learning_rate},
            {"params": model.head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.finetune_weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=args.finetune_min_learning_rate
    )
    tensor_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    tensor_y = torch.as_tensor(scaler.transform(train_y), dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_supervised_epoch(
            model, tensor_x, tensor_y, optimizer, args.batch_size, generator,
            args.gradient_clip_norm,
        )
        history.append({"epoch": epoch, "train_loss": loss})
        scheduler.step()
    return model, scaler, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1D: LeJEPA Ridge/tree baselines and end-to-end fine-tuning"
    )
    parser.add_argument(
        "--canonical-dir", type=Path,
        default=ROOT / "artifacts" / "canonical" / "crude_oil_private",
    )
    parser.add_argument("--protocol", choices=("pooled", "logo"), default="pooled")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None, help="Smoke/debug only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preprocessing", choices=("R", "FD", "SD", "SNV", "MSC", "SG"), default="R")
    parser.add_argument("--sg-window", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=2)

    parser.add_argument("--ssl-epochs", dest="epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ssl-learning-rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--ssl-min-learning-rate", dest="min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-start-factor", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--mask-mode", choices=("point", "block"), default="point")
    parser.add_argument("--mask-block-length", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--projector-hidden-dim", type=int, default=128)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    parser.add_argument("--mae-decoder-hidden-dim", type=int, default=64)

    parser.add_argument("--encoder-type", choices=("transformer", "unet"), default="unet")
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--embedding-batch-size", type=int, default=256)

    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    parser.add_argument("--ridge-inner-splits", type=int, default=4)
    parser.add_argument("--tree-inner-splits", type=int, default=4)
    parser.add_argument("--tree-jobs", type=int, default=-1)
    parser.add_argument("--xgb-estimators", type=int, nargs="+", default=[200, 500])
    parser.add_argument("--xgb-max-depths", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--xgb-learning-rates", type=float, nargs="+", default=[0.03, 0.1])
    parser.add_argument("--lgbm-estimators", type=int, nargs="+", default=[200, 500])
    parser.add_argument("--lgbm-num-leaves", type=int, nargs="+", default=[7, 15, 31])
    parser.add_argument("--lgbm-learning-rates", type=float, nargs="+", default=[0.03, 0.1])

    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--finetune-inner-splits", type=int, default=5)
    parser.add_argument("--finetune-patience", type=int, default=30)
    parser.add_argument("--finetune-min-delta", type=float, default=1e-5)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    if args.views < 2:
        raise ValueError("--views must be at least 2")
    if args.embedding_dim % args.heads != 0:
        raise ValueError("--embedding-dim must be divisible by --heads")
    if not 0 < args.mask_ratio < 1:
        raise ValueError("--mask-ratio must lie in (0, 1)")
    if args.ridge_inner_splits < 2 or args.tree_inner_splits < 2 or args.finetune_inner_splits < 2:
        raise ValueError("All inner split counts must be at least 2")
    if min(args.epochs, args.finetune_epochs, args.batch_size) < 1:
        raise ValueError("Epoch and batch-size arguments must be positive")
    for method in args.methods:
        if method in {"xgboost", "lightgbm"}:
            require_regressor(method)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    canonical = args.canonical_dir if args.canonical_dir.is_absolute() else ROOT / args.canonical_dir
    required = ["spectra_resampled.npy", "targets.csv", "samples.csv"]
    missing = [name for name in required if not (canonical / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing canonical dataset files in {canonical}: {missing}")
    spectra = np.load(canonical / "spectra_resampled.npy")
    target_frame = pd.read_csv(canonical / "targets.csv")
    target_names = target_frame.columns.tolist()
    targets = target_frame.to_numpy(float)
    samples = pd.read_csv(canonical / "samples.csv")
    split_file = "folds_pooled.json" if args.protocol == "pooled" else "folds_leave_one_group_out.json"
    folds = json.loads((canonical / split_file).read_text(encoding="utf-8"))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    device = torch.device(args.device)
    started_at = datetime.now().astimezone()
    run_name = f"e1d_{canonical.name}_{args.protocol}_{started_at:%Y%m%d_%H%M%S}_seed{args.seed}"
    run_dir = ROOT / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        tqdm.write(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start dataset={canonical.name} protocol={args.protocol} folds={len(folds)} methods={args.methods} device={device}")
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    cost_rows: list[dict] = []

    for fold_data in tqdm(folds, desc="E1D outer folds", unit="fold", ascii=True, dynamic_ncols=True):
        fold = int(fold_data["fold"])
        train_idx = np.asarray(fold_data["train_idx"], dtype=int)
        test_idx = np.asarray(fold_data["test_idx"], dtype=int)
        transformed_train, transformed_test = preprocess_pair(
            spectra[train_idx], spectra[test_idx], args.preprocessing,
            args.sg_window, args.sg_polyorder,
        )
        standardizer = FoldSpectralStandardizer().fit(transformed_train)
        train_x = standardizer.transform(transformed_train).astype(np.float32)
        test_x = standardizer.transform(transformed_test).astype(np.float32)
        train_y = targets[train_idx]
        set_seed(args.seed + fold)
        initial_encoder_state = make_encoder(train_x.shape[1], args).state_dict()

        log(f"fold={fold} LeJEPA pretraining begin train={len(train_idx)} test={len(test_idx)}")
        encoder, ssl_history, ssl_timing = train_ssl(
            "lejepa", train_x, initial_encoder_state, args, args.seed + fold, device,
            progress_label=f"E1D fold {fold} LeJEPA",
        )
        pretrained_state = copy.deepcopy(encoder.state_dict())
        pd.DataFrame(ssl_history).to_csv(run_dir / f"fold{fold}_lejepa_history.csv", index=False)
        torch.save(
            {"encoder": pretrained_state, "fold": fold, "dataset": canonical.name},
            run_dir / f"fold{fold}_lejepa_encoder.pt",
        )
        synchronize(device)
        embedding_started = time.perf_counter()
        train_z = extract_embeddings(encoder, train_x, device, args.embedding_batch_size)
        test_z = extract_embeddings(encoder, test_x, device, args.embedding_batch_size)
        synchronize(device)
        embedding_seconds = time.perf_counter() - embedding_started
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if "ridge" in args.methods:
            started = time.perf_counter()
            alpha = select_ridge_alpha(
                train_z, train_y, args.seed + fold, tuple(args.ridge_alphas), args.ridge_inner_splits
            )
            regressor = RidgeRegressor(alpha).fit(train_z, train_y)
            train_prediction = regressor.predict(train_z)
            test_prediction = regressor.predict(test_z)
            seconds = time.perf_counter() - started
            append_evaluation_rows(
                fold, "ridge", test_idx, test_prediction, train_idx, train_prediction,
                targets, samples, target_names, prediction_rows, metric_rows,
            )
            cost_rows.append({
                "fold": fold, "method": "ridge", "shared_ssl_seconds": ssl_timing["pretrain_seconds"],
                "embedding_seconds": embedding_seconds, "downstream_seconds": seconds,
                "selected_params": json.dumps({"alpha": alpha}),
            })
            log(f"fold={fold} method=ridge alpha={alpha:g} seconds={seconds:.2f}")

        regression_folds = inner_folds(len(train_z), args.seed + fold, args.tree_inner_splits)
        for method in (name for name in args.methods if name in {"xgboost", "lightgbm"}):
            started = time.perf_counter()
            candidates = parameter_grid(method, args)
            selected = select_tree_params(
                method, train_z, train_y, regression_folds, candidates,
                args.seed + fold, args.tree_jobs,
            )
            model_class = require_regressor(method)
            models = fit_tree_targets(
                method, model_class, selected, train_z, train_y, args.seed + fold, args.tree_jobs
            )
            train_prediction = predict_tree_targets(models, train_z)
            test_prediction = predict_tree_targets(models, test_z)
            seconds = time.perf_counter() - started
            append_evaluation_rows(
                fold, method, test_idx, test_prediction, train_idx, train_prediction,
                targets, samples, target_names, prediction_rows, metric_rows,
            )
            cost_rows.append({
                "fold": fold, "method": method, "shared_ssl_seconds": ssl_timing["pretrain_seconds"],
                "embedding_seconds": embedding_seconds, "downstream_seconds": seconds,
                "selected_params": json.dumps(selected),
            })
            log(f"fold={fold} method={method} params={selected} seconds={seconds:.2f}")

        if "finetune" in args.methods:
            started = time.perf_counter()
            selected_epochs, selection_history = choose_finetune_epochs(
                pretrained_state, train_x, train_y, args, args.seed + fold, device
            )
            model, target_scaler, final_history = fit_finetuned_model(
                pretrained_state, train_x, train_y, selected_epochs, args,
                args.seed + fold + 100_000, device,
            )
            train_prediction = target_scaler.inverse_transform(
                predict_finetuned(model, train_x, device, args.embedding_batch_size)
            )
            test_prediction = target_scaler.inverse_transform(
                predict_finetuned(model, test_x, device, args.embedding_batch_size)
            )
            seconds = time.perf_counter() - started
            append_evaluation_rows(
                fold, "finetune", test_idx, test_prediction, train_idx, train_prediction,
                targets, samples, target_names, prediction_rows, metric_rows,
            )
            pd.DataFrame(selection_history).to_csv(
                run_dir / f"fold{fold}_finetune_selection_history.csv", index=False
            )
            pd.DataFrame(final_history).to_csv(
                run_dir / f"fold{fold}_finetune_history.csv", index=False
            )
            torch.save(
                {"model": model.state_dict(), "fold": fold, "selected_epochs": selected_epochs},
                run_dir / f"fold{fold}_finetuned_model.pt",
            )
            cost_rows.append({
                "fold": fold, "method": "finetune", "shared_ssl_seconds": ssl_timing["pretrain_seconds"],
                "embedding_seconds": 0.0, "downstream_seconds": seconds,
                "selected_params": json.dumps({
                    "epochs": selected_epochs,
                    "encoder_lr": args.finetune_learning_rate,
                    "head_lr": args.head_learning_rate,
                    "hidden_dim": args.mlp_hidden_dim,
                }),
            })
            log(f"fold={fold} method=finetune selected_epochs={selected_epochs} seconds={seconds:.2f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.partial.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(run_dir / "fold_metrics.partial.csv", index=False)
        pd.DataFrame(cost_rows).to_csv(run_dir / "costs.partial.csv", index=False)

    metrics_frame = pd.DataFrame(metric_rows)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    metrics_frame.to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(run_dir / "costs.csv", index=False)
    summary = (
        metrics_frame.groupby(["method", "split", "target"], as_index=False)
        .agg(n=("fold", "size"), rmse=("rmse", "mean"), r2=("r2", "mean"), rpd=("rpd", "mean"))
    )
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    test_summary = summary[summary["split"] == "test"].copy()
    ridge_reference = test_summary[test_summary["method"] == "ridge"][
        ["target", "rmse", "r2", "rpd"]
    ].rename(columns={"rmse": "ridge_rmse", "r2": "ridge_r2", "rpd": "ridge_rpd"})
    comparison = test_summary.merge(ridge_reference, on="target", how="left")
    comparison["rmse_improvement_vs_ridge"] = comparison["ridge_rmse"] - comparison["rmse"]
    comparison["r2_improvement_vs_ridge"] = comparison["r2"] - comparison["ridge_r2"]
    comparison["rpd_improvement_vs_ridge"] = comparison["rpd"] - comparison["ridge_rpd"]
    comparison.to_csv(run_dir / "comparison_vs_ridge.csv", index=False)
    run_config = {
        **vars(args), "canonical_dir": str(canonical.resolve()), "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_directory": str(run_dir.resolve()),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    for path in run_dir.glob("*.partial.csv"):
        path.unlink()
    log("run complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
