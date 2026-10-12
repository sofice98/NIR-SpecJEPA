"""Run E1B LeJEPA with a detached online regression probe.

The encoder learns from the self-supervised LeJEPA objective while a small
probe reads detached embeddings. The probe supplies a useful supervised
signal for prediction without allowing target labels to change the encoder.
"""
from __future__ import annotations
import argparse, json, logging, random, time
from datetime import datetime
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
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

class TqdmLoggingHandler(logging.Handler):
    """通过 tqdm 输出日志，避免打断进度条。"""

    def emit(self, record: logging.LogRecord) -> None:
        """将格式化后的日志写到进度条上方。"""
        tqdm.write(self.format(record))

def _target_statistics(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按目标统计已观测标签的均值、尺度和有效位置。"""
    observed = np.isfinite(targets)
    count = observed.sum(axis=0)
    mean = np.divide(np.nansum(targets, axis=0), count, out=np.zeros(targets.shape[1]), where=count > 0)
    centered = np.where(observed, targets - mean, 0.0)
    scale = np.sqrt(np.divide((centered * centered).sum(axis=0), count, out=np.ones(targets.shape[1]), where=count > 0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return mean, scale, observed

def configure_logging(output_dir: Path, level: str = "INFO", experiment_id: str = "e1") -> logging.Logger:
    """Create a fresh console/file logger for one E1 invocation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(experiment_id)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def timestamped_run_dir(base_dir: Path) -> Path:
    """Return a unique ``<name>_YYYYMMDD_HHMMSS`` run directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.parent / f"{base_dir.name}_{stamp}"


def plot_loss_curves(history_rows: list[dict], out_dir: Path, logger: logging.Logger) -> list[Path]:
    """Write separate loss plots for each model and training stage."""
    history = pd.DataFrame(history_rows)
    if history.empty or "loss" not in history.columns:
        logger.info("No loss history available; skipping loss plots in %s", out_dir)
        return []
    history = history[np.isfinite(pd.to_numeric(history["loss"], errors="coerce"))].copy()
    if history.empty:
        logger.info("Loss history contains no finite values; skipping loss plots in %s", out_dir)
        return []
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is unavailable; cannot generate loss PNGs in %s", out_dir)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    loss_columns = ["loss", "regression_loss", "invariance_mse", "sigreg"]
    for experiment, subset in history.groupby("experiment", sort=True):
        fig, axes = plt.subplots(
            len(loss_columns), 1, figsize=(12, 5 * len(loss_columns)), squeeze=False, sharex=True
        )
        for ax, loss_column in zip(axes[:, 0], loss_columns):
            values = pd.to_numeric(subset.get(loss_column), errors="coerce")
            component_subset = subset[np.isfinite(values)].copy()
            for fold, group in component_subset.groupby("fold", sort=True):
                group = group.sort_values("epoch")
                ax.plot(
                    group["epoch"], group[loss_column], alpha=0.45, linewidth=1.0,
                    label=f"fold {fold}",
                )
            mean_history = component_subset.groupby("epoch", as_index=False)[loss_column].mean().sort_values("epoch")
            ax.plot(
                mean_history["epoch"], mean_history[loss_column],
                color="black", linewidth=2.5, label="mean",
            )
            title = {
                "loss": "total loss", "regression_loss": "regression loss",
                "invariance_mse": "invariance loss", "sigreg": "SIGReg loss",
            }[loss_column]
            ax.set_title(f"{experiment} joint training: {title}")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=2, loc="best")
        axes[-1, 0].set_xlabel("Epoch")
        fig.tight_layout()
        path = out_dir / f"loss_curves_{experiment.lower()}_joint_train.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(path)
    logger.info("Generated loss plots: %s", ", ".join(str(item.name) for item in generated))
    return generated


def parse_batch_size(value: str) -> int | str:
    """解析正整数批量大小或表示全批量的 ``full``。"""
    if str(value).lower() == "full":
        return "full"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch-size must be a positive integer or 'full'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch-size must be positive")
    return parsed


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和可用的 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """计算单个目标的 RMSE、R² 和 RPD。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    variance = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / variance) if variance > 1e-12 else float("nan")
    rpd = float(np.std(y_true, ddof=1) / rmse) if rmse > 1e-12 and len(y_true) > 1 else float("nan")
    return rmse, r2, rpd


def save_representation_assets(
    asset_dir: Path,
    pretrained_model_state: dict[str, torch.Tensor],
    pretrained_state: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
    history: list[dict],
    metadata: dict[str, object],
) -> None:
    """Persist one reusable fold-level SSL representation run."""
    import torch

    asset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pretrained_model_state, asset_dir / "lejepa_with_probe.pt")
    torch.save(pretrained_state, asset_dir / "encoder.pt")
    np.savez_compressed(asset_dir / "representations.npz", **arrays)
    pd.DataFrame(history).to_csv(asset_dir / "history.csv", index=False, encoding="utf-8-sig")
    (asset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def detached_state_dict(module: object) -> dict[str, torch.Tensor]:
    """Copy a module state to CPU for a stable fold asset."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def joint_lejepa_representation(
    backbone: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int | str,
    mask_ratio: float,
    patch_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    head_learning_rate: float,
    scheduler_name: str = "warmup_cosine",
    scheduler_eta_min: float = 1e-6,
    sigreg_weight: float = 0.02,
    logger: logging.Logger | None = None,
) -> tuple[object, dict[str, torch.Tensor], np.ndarray, np.ndarray, list[dict], int, int]:
    """Train LeJEPA and its detached online probe for one outer fold.

    A patch-masked view and a clean view are optimized with the LeJEPA invariance/SIGReg losses.
    The probe sees ``view_embeddings.detach()`` from both views, so its
    regression loss updates only the probe; encoder representations remain
    governed by the self-supervised objective.
    """
    from models.lejepa.model import SpectralLeJEPA
    from models.shared.masking import random_patch_mask
    from torch.utils.data import DataLoader, TensorDataset
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if backbone != "patch_transformer":
        raise ValueError("Patch masking requires --backbone patch_transformer")
    encoder = build_ssl_encoder(backbone, len(train_x[0]), patch_size, embedding_dim, input_channels=1).to(device)
    model = SpectralLeJEPA(encoder, sigreg_weight=sigreg_weight, mask_mode="patch").to(device)
    if encoder.n_patches < 2:
        raise ValueError("Patch masking requires at least two patches; reduce --patch-size")
    # 用训练折的观测标签归一化目标；缺失值只参与填充，不计入探针损失。
    target_mean, target_scale, observed = _target_statistics(train_y)
    filled_y = np.where(np.isfinite(train_y), train_y, target_mean)
    mask_y = observed.astype(np.float32)
    normalized_y = (filled_y - target_mean) / target_scale
    head = nn.Sequential(nn.LayerNorm(encoder.embedding_dim), nn.Linear(encoder.embedding_dim, train_y.shape[1])).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": learning_rate, "weight_decay": 5e-2},
        {"params": head.parameters(), "lr": head_learning_rate, "weight_decay": 1e-7},
    ])
    effective_batch_size = len(train_x) if batch_size == "full" else int(batch_size)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(normalized_y, dtype=torch.float32),
            torch.as_tensor(mask_y, dtype=torch.float32),
        ), batch_size=effective_batch_size, shuffle=True, drop_last=False,
    )
    scheduler = build_step_scheduler(
        optimizer, scheduler_name, total_steps=len(loader) * max(0, epochs),
        warmup_steps=len(loader), eta_min=scheduler_eta_min,
    )
    history: list[dict] = []
    # 两个掩码视图共同训练 LeJEPA；探针读取分离的表示以阻断标签梯度。
    started = time.perf_counter()
    epoch_bar = tqdm(range(1, max(0, epochs) + 1), desc="LeJEPA + online probe", unit="epoch", leave=False)
    for epoch in epoch_bar:
        model.train(); head.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        probe_learning_rate = float(optimizer.param_groups[1]["lr"])
        losses: dict[str, list[float]] = {"loss": [], "lejepa_loss": [], "regression_loss": [], "invariance_mse": [], "sigreg": []}
        for batch_x, batch_y, batch_mask in loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            patch_mask = random_patch_mask(
                len(batch_x), encoder.n_patches, mask_ratio, device=device,
            )
            # One random patch mask creates the two views consumed by LeJEPA.
            # The same mask is sampled only for this batch and never reused at
            # evaluation time, where the unmasked encoder representation is used.
            output = model(batch_x, patch_mask)
            # Both views train the probe, but labels never train the encoder.
            prediction = head(output["view_embeddings"].detach())
            residual = (prediction - batch_y.unsqueeze(0)).square() * batch_mask.unsqueeze(0)
            # Missing targets contribute zero through batch_mask; the denominator
            # counts observed target/view pairs rather than padded values.
            regression_loss = residual.sum() / (batch_mask.sum() * prediction.shape[0]).clamp_min(1.0)
            loss = output["loss"] + regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for name, value in (("loss", loss), ("regression_loss", regression_loss),
                                ("lejepa_loss", output["loss"]),
                                ("invariance_mse", output["invariance_mse"]), ("sigreg", output["sigreg"])):
                losses[name].append(float(value.detach().cpu()))
        values = {name: float(np.mean(items)) for name, items in losses.items()}
        history.append({
            "epoch": epoch, **values, "learning_rate": epoch_learning_rate,
            "probe_learning_rate": probe_learning_rate,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "probe_learning_rate_end": float(optimizer.param_groups[1]["lr"]),
            "elapsed_sec": time.perf_counter() - started, "stage": "joint_train", "backbone": backbone,
        })
        epoch_bar.set_postfix(loss=f"{values['loss']:.5g}", lr=f"{epoch_learning_rate:.2g}")
        if logger is not None and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            logger.info(
                "Joint LeJEPA epoch %d/%d | total=%.6g regression=%.6g invariance=%.6g sigreg=%.6g | lr=%.6g | elapsed=%.1fs",
                epoch, epochs, values["loss"], values["regression_loss"], values["invariance_mse"],
                values["sigreg"], epoch_learning_rate, time.perf_counter() - started,
            )
    epoch_bar.close()
    model.eval(); head.eval()
    def encode(values: np.ndarray) -> np.ndarray:
        """分批提取未掩码光谱的池化表示。"""
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                batch = torch.as_tensor(values[start:start + effective_batch_size], dtype=torch.float32, device=device)
                outputs.append(encoder(batch, None)[1].cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, embedding_dim))
    state = detached_state_dict(model)
    # 保存探针及标签归一化参数，使测试预测可直接由持久化状态复现。
    state.update({f"probe.{key}": value for key, value in detached_state_dict(head).items()})
    state["target_mean"] = torch.as_tensor(target_mean)
    state["target_scale"] = torch.as_tensor(target_scale)
    return encoder, state, encode(train_x), encode(test_x), history, parameter_count(model), parameter_count(head)


def run_dataset(dataset: CanonicalDataset, dataset_name: str, args: argparse.Namespace,
                out_dir: Path, logger: logging.Logger) -> None:
    """逐折训练 LeJEPA 与在线探针，并保存表示、预测指标和训练记录。"""
    from models.lejepa.probe import predict_saved_lejepa_probe

    dataset.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset.sample_id))
        dataset = CanonicalDataset(dataset.name, dataset.sample_id[:n], dataset.wavelengths_nm,
                                   dataset.spectra[:n], dataset.targets.iloc[:n].reset_index(drop=True),
                                   dataset.groups[:n], dataset.metadata)
    if len(dataset.sample_id) < 2:
        raise ValueError("at least 2 samples are required for cross-validation")
    folds = pooled_group_balanced_folds(dataset.groups, n_splits=min(args.folds, len(dataset.sample_id)), seed=args.seed)
    targets = dataset.targets.to_numpy(float)
    rows: list[dict] = []
    histories: list[dict] = []
    summaries: list[dict] = []
    assets: list[dict] = []
    for fold in folds:
        started = time.perf_counter()
        train_idx, test_idx = fold.train_idx, fold.test_idx
        train_x = np.asarray(dataset.spectra[train_idx], dtype=np.float32)
        test_x = np.asarray(dataset.spectra[test_idx], dtype=np.float32)
        encoder, state, train_embedding, test_embedding, history, model_params, probe_params = joint_lejepa_representation(
            args.backbone, train_x, targets[train_idx], test_x, args.epochs, args.batch_size,
            args.mask_ratio, args.patch_size, DEFAULT_EMBEDDING_DIM,
            args.seed + fold.fold, args.learning_rate, args.probe_learning_rate,
            args.scheduler, args.scheduler_eta_min, args.sigreg_weight, logger,
        )
        # Evaluate the persisted probe state, matching the representation that is
        # written to disk for later reproducibility checks.
        prediction = predict_saved_lejepa_probe(test_embedding, state)
        fit_seconds = time.perf_counter() - started
        run_id = f"lejepa_fold{fold.fold}"
        asset_dir = out_dir / "representations" / run_id
        fold_history = [dict(row, experiment="lejepa", fold=fold.fold) for row in history]
        # 每折单独保存完整模型、编码器和表示，供后续分析复用。
        save_representation_assets(
            asset_dir, state, detached_state_dict(encoder),
            {"train_embedding": train_embedding, "test_embedding": test_embedding}, fold_history,
            {"dataset": dataset_name, "fold": fold.fold, "kind": "lejepa",
             "training_mode": "lejepa_online_probe", "probe_detached": True,
             "model_params": model_params, "probe_params": probe_params},
        )
        histories.extend(dict(row, dataset=dataset_name, representation_run_id=run_id) for row in fold_history)
        assets.append({"representation_run_id": run_id, "dataset": dataset_name,
                       "fold": fold.fold, "asset_dir": str(asset_dir)})
        selected_params = json.dumps({"source": "one_stage_lejepa_probe", "detached": True})
        common = {"dataset": dataset_name, "experiment": "lejepa", "label": "LeJEPA",
                  "regressor": "probe", "fold": fold.fold, "representation_run_id": run_id,
                  "train_time_sec": fit_seconds, "model_params": model_params,
                  "probe_params": probe_params, "model_total_params": model_params + probe_params,
                  "selected_params": selected_params}
        # 对缺失标签逐目标筛选后计算测试指标。
        for col, target in enumerate(dataset.targets.columns):
            valid = np.isfinite(targets[test_idx, col]) & np.isfinite(prediction[:, col])
            if np.any(valid):
                rmse, r2, rpd = metrics(targets[test_idx, col][valid], prediction[valid, col])
                rows.append({**common, "target": target, "rmse": rmse, "r2": r2,
                             "rpd": rpd, "n_test": int(valid.sum())})
        summaries.append(common)
    # 输出逐折结果、跨折均值、训练曲线和表示文件索引。
    # These four files provide complementary views of a run: fold metrics,
    # epoch losses, per-fold assets, and the aggregate run metadata.
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(out_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assets).to_csv(out_dir / "ssl_runs.csv", index=False, encoding="utf-8-sig")
    if rows:
        aggregate = pd.DataFrame(rows).groupby(
            ["dataset", "experiment", "label", "regressor", "target"], as_index=False
        )[["rmse", "r2", "rpd", "train_time_sec", "model_total_params"]].mean()
        aggregate.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")
    plot_loss_curves(histories, out_dir, logger)
    logger.info("Wrote %d metric rows, %d history rows and %d summaries to %s",
                len(rows), len(histories), len(summaries), out_dir)


def parse_args() -> argparse.Namespace:
    """解析数据、LeJEPA、探针和训练调度参数。"""
    parser = argparse.ArgumentParser(description="Run E1B LeJEPA and online probe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1b")
    parser.add_argument("--swri-root", type=Path, default=ROOT / "data" / "SWRI_Diesel")
    parser.add_argument("--oil1-root", type=Path, default=ROOT / "data" / "crude_oil_1")
    parser.add_argument("--oil2-root", type=Path, default=ROOT / "data" / "crude_oil_2")
    parser.add_argument("--lucas-root", type=Path, default=ROOT / "data" / "lucas")
    parser.add_argument(
        "--datasets", nargs="+", default=["swri", "crude_oil"],
        choices=["swri", "crude_oil", "lucas"], help="Datasets to run",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--backbone", choices=["patch_transformer"], default="patch_transformer")
    parser.add_argument("--patch-size", "--patch-length", dest="patch_size", type=int, default=6)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--batch-size", type=parse_batch_size, default=16)
    parser.add_argument("--scheduler", choices=["constant", "cosine", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small smoke-test limit per dataset")
    args = parser.parse_args()
    return args


def main() -> None:
    """校验配置、载入数据并运行 E1B。"""
    args = parse_args()
    if args.epochs < 0 or args.learning_rate <= 0 or args.patch_size < 1 or not 0 < args.mask_ratio < 1:
        raise ValueError("Invalid SSL epochs, learning rate, patch size, or mask ratio")
    if args.probe_learning_rate <= 0 or not 0 <= args.sigreg_weight <= 1:
        raise ValueError("Invalid probe learning rate or SIGReg weight")
    if args.scheduler_eta_min < 0:
        raise ValueError("scheduler-eta-min must be non-negative")
    rates = [args.learning_rate, args.probe_learning_rate]
    if args.scheduler != "constant" and args.scheduler_eta_min > min(rates):
        raise ValueError("scheduler-eta-min must not exceed the initial learning rates")
    seed_everything(args.seed)
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir = timestamped_run_dir(base_output_dir)
    logger = configure_logging(args.output_dir, args.log_level, "e1b")
    run_started = time.perf_counter()
    logger.info("E1B started | folds=%d datasets=%s seed=%d", args.folds, ",".join(args.datasets), args.seed)
    logger.info("Learning-rate scheduler=%s eta_min=%g", args.scheduler, args.scheduler_eta_min)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Loading datasets: %s", ",".join(args.datasets))
    swri = None
    oils = None
    lucas = None
    if "swri" in args.datasets:
        swri = load_swri(args.swri_root)
        logger.info("Loaded SWRI_Diesel: samples=%d, wavelengths=%d", len(swri.sample_id), swri.spectra.shape[1])
    if "crude_oil" in args.datasets:
        oils = merge_datasets([load_oil1(args.oil1_root), load_oil2(args.oil2_root)], name="crude_oil_1+crude_oil_2")
        logger.info("Loaded crude oil mixture: samples=%d, wavelengths=%d", len(oils.sample_id), oils.spectra.shape[1])
    if "lucas" in args.datasets:
        lucas = load_lucas(args.lucas_root)
        logger.info("Loaded LUCAS_SOIL: samples=%d, wavelengths=%d", len(lucas.sample_id), lucas.spectra.shape[1])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["experiment_id"] = "e1b"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if swri is not None:
        run_dataset(swri, "SWRI_Diesel_drop_missing", args, args.output_dir / "SWRI_Diesel_drop_missing", logger)
    if oils is not None:
        run_dataset(oils, "crude_oil_1+crude_oil_2", args, args.output_dir / "crude_oil_1_plus_crude_oil_2", logger)
    if lucas is not None:
        run_dataset(lucas, "LUCAS_SOIL", args, args.output_dir / "LUCAS_SOIL", logger)
    elapsed = time.perf_counter() - run_started
    logger.info("E1B completed successfully in %.1f seconds", elapsed)
    selected_datasets = []
    if "swri" in args.datasets:
        selected_datasets.append("SWRI_Diesel_drop_missing")
    if "crude_oil" in args.datasets:
        selected_datasets.append("crude_oil_1+crude_oil_2")
    if "lucas" in args.datasets:
        selected_datasets.append("LUCAS_SOIL")
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": selected_datasets,
                      "experiment": "lejepa", "regressor": "probe"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
