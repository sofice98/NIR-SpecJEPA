"""Run E1C masked autoencoder experiments.

E1C pretrains a spectral MAE inside each outer fold. Downstream evaluation can
use reconstructed spectra (``mae_preprocess``) or encoder features
(``mae_feature``), with the latter optionally fine-tuned using observed labels.
"""
from __future__ import annotations
import argparse, json, logging, random, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from data.dataset_adapters import CanonicalDataset, merge_datasets
from data.SWRI_Diesel.adapter import load_dataset as load_swri
from data.crude_oil_1.adapter import load_dataset as load_oil1
from data.crude_oil_2.adapter import load_dataset as load_oil2
from data.lucas.adapter import load_dataset as load_lucas
from preprocessing.spectral import FoldSpectralStandardizer, preprocess_pair
from splits.cross_validation import pooled_group_balanced_folds
from models.shared.experiment import build_ssl_encoder, parameter_count, ssl_prediction_parameter_count
from models.shared.scheduling import build_step_scheduler
try:
    import torch
except ImportError:
    torch = None

@dataclass(frozen=True)
class Experiment:
    """描述一种 MAE 下游使用方式。"""
    name: str
    label: str
EXPERIMENTS = (Experiment("mae_preprocess", "MAE预处理"), Experiment("mae_feature", "MAE特征提取"))

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
    history["stage"] = history["stage"].fillna("ssl_pretrain")
    # Pretraining is shared by the selected MAE downstream paths within each fold.
    history["model"] = history["experiment"].astype(str)

    # Keep pretraining and supervised fine-tuning on separate axes.
    stage_labels = {
        "ssl_pretrain": "self-supervised pretraining",
        "supervised_finetune": "supervised fine-tuning",
    }
    for (model, stage), subset in history.groupby(["model", "stage"], sort=True):
        loss_columns = ["loss"]
        fig, axes = plt.subplots(
            len(loss_columns), 1, figsize=(12, 5 * len(loss_columns)), squeeze=False, sharex=True
        )
        for ax, loss_column in zip(axes[:, 0], loss_columns):
            values = pd.to_numeric(subset.get(loss_column), errors="coerce")
            component_subset = subset[np.isfinite(values)].copy()
            group_columns = ["experiment", "fold"]
            if "regressor" in component_subset.columns:
                group_columns.insert(1, "regressor")
            for group_key, group in component_subset.groupby(group_columns, sort=True):
                group = group.sort_values("epoch")
                label = "/".join(str(item) for item in (
                    group_key if isinstance(group_key, tuple) else (group_key,)
                ))
                ax.plot(
                    group["epoch"], group[loss_column], alpha=0.45, linewidth=1.0,
                    label=label,
                )
            mean_history = component_subset.groupby("epoch", as_index=False)[loss_column].mean().sort_values("epoch")
            ax.plot(
                mean_history["epoch"], mean_history[loss_column],
                color="black", linewidth=2.5, label="mean",
            )
            title = {
                "loss": "total loss",
            }[loss_column]
            ax.set_title(f"{model} {stage_labels.get(stage, stage)}: {title}")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=2, loc="best")
        axes[-1, 0].set_xlabel("Epoch")
        fig.tight_layout()
        path = out_dir / f"loss_curves_{model.lower()}_{stage}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(path)
    logger.info("Generated loss plots: %s", ", ".join(str(item.name) for item in generated))
    return generated


def save_mask_example(
    spectra: torch.Tensor,
    masks: torch.Tensor,
    path: Path,
) -> None:
    """Save the exact point-level zero-masked input used by an SSL job."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if masks.ndim != 2:
        raise ValueError("masks must have shape [batch, points]")
    sample_count = min(3, spectra.shape[0])
    spectra_np = spectra[:sample_count].detach().cpu().numpy()
    masks_np = masks[:sample_count].detach().cpu().numpy()
    fig, axes = plt.subplots(sample_count, 1, figsize=(13, 3.2 * sample_count), squeeze=False)
    for sample_index in range(sample_count):
        ax = axes[sample_index, 0]
        original = spectra_np[sample_index]
        masked = original.copy()
        masked[masks_np[sample_index]] = 0.0
        ax.plot(original, color="0.65", linewidth=1.0, label="original")
        ax.plot(masked, color="tab:blue", linewidth=1.0, label="zero-masked input")
        ax.fill_between(
            np.arange(len(original)), original.min(), original.max(),
            where=masks_np[sample_index], color="tab:red", alpha=0.16,
            label="masked points",
        )
        ax.set_title(f"sample {sample_index}")
        ax.set_xlabel("spectral point")
        ax.set_ylabel("standardized intensity")
        ax.grid(True, alpha=0.2)
        if sample_index == 0:
            ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def inner_folds(indices: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """在外层训练集内生成按组划分的调参折，返回局部索引。"""
    local_groups = np.asarray(groups)[indices]
    folds = pooled_group_balanced_folds(local_groups, n_splits=max(2, min(n_splits, len(indices))), seed=seed)
    return [(f.train_idx, f.test_idx) for f in folds]


def fit_downstream(
    regressor: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    jobs: int,
    batch_size: int | str = 64,
    scheduler_name: str = "warmup_cosine",
    scheduler_eta_min: float = 1e-6,
) -> tuple[np.ndarray, int, dict]:
    """Fit one regressor per target, which naturally supports SWRI missing labels."""
    from models.shared.experiment import sklearn_parameter_count
    from models.shared.regressors import fit_fixed_representation_mlp, pls, random_forest
    prediction = np.full((len(test_x), train_y.shape[1]), np.nan, dtype=float)
    selected: dict[str, object] = {}
    total_params = 0
    for column in range(train_y.shape[1]):
        # 逐目标过滤缺失标签，并将调参折映射到该目标的有效训练样本。
        valid_train = np.isfinite(train_y[:, column])
        # PLS and the shared candidate scorer use a 2-D target, whereas
        # RandomForest expects a 1-D target for one output.
        y = train_y[valid_train, column, None]
        x = train_x[valid_train]
        if len(x) < 3:
            continue
        valid_folds = []
        valid_positions = np.flatnonzero(valid_train)
        for fold_train, fold_valid in folds:
            a = np.intersect1d(fold_train, valid_positions, assume_unique=False)
            b = np.intersect1d(fold_valid, valid_positions, assume_unique=False)
            if len(a) >= 2 and len(b) >= 1:
                valid_folds.append((np.searchsorted(valid_positions, a), np.searchsorted(valid_positions, b)))
        if not valid_folds:
            continue
        if regressor == "pls":
            result = pls(x, y, test_x, valid_folds)
            result_prediction = np.asarray(result.prediction, dtype=float)
            selected[str(column)] = result.selected_params
            total_params += sklearn_parameter_count(result.model)
        elif regressor == "rf":
            result = random_forest(x, y[:, 0], test_x, valid_folds, seed=seed, jobs=jobs)
            result_prediction = np.asarray(result.prediction, dtype=float)
            selected[str(column)] = result.selected_params
            total_params += sklearn_parameter_count(result.model)
        else:
            result_prediction, mlp_params, mlp_config = fit_fixed_representation_mlp(
                x, y[:, 0], test_x, seed + column, batch_size,
                scheduler_name=scheduler_name, scheduler_eta_min=scheduler_eta_min,
            )
            selected[str(column)] = mlp_config
            total_params += mlp_params
        prediction[:, column] = result_prediction[:, 0] if result_prediction.ndim == 2 else result_prediction
    return prediction, total_params, selected


def save_representation_assets(
    asset_dir: Path,
    pretrained_model_state: dict[str, torch.Tensor],
    pretrained_state: dict[str, torch.Tensor],
    finetuned_state: dict[str, torch.Tensor] | None,
    arrays: dict[str, np.ndarray],
    history: list[dict],
    metadata: dict[str, object],
) -> None:
    """Persist one reusable fold-level SSL representation run."""
    import torch

    asset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pretrained_model_state, asset_dir / "pretrained_ssl_model.pt")
    torch.save(pretrained_state, asset_dir / "pretrained_encoder.pt")
    if finetuned_state is not None:
        torch.save(finetuned_state, asset_dir / "finetuned_encoder.pt")
    np.savez_compressed(asset_dir / "representations.npz", **arrays)
    pd.DataFrame(history).to_csv(asset_dir / "history.csv", index=False, encoding="utf-8-sig")
    (asset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def detached_state_dict(module: object) -> dict[str, torch.Tensor]:
    """Copy a module state to CPU so later fine-tuning cannot mutate it."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def ssl_representation(
    backbone: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int | str,
    mask_ratio: float,
    mask_block_length: int,
    patch_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    scheduler_name: str = "warmup_cosine",
    scheduler_eta_min: float = 1e-6,
    mask_example_path: Path | None = None,
    logger: logging.Logger | None = None,
    save_pretrained_embeddings: bool = True,
    reconstruct_spectra: bool = True,
) -> tuple[object, dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    """预训练 MAE，并按需提取编码表示及掩码重建光谱。"""
    from models.mae.model import SpectralMAE
    from models.shared.masking import apply_point_mask, random_point_mask
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    encoder = build_ssl_encoder(backbone, len(train_x[0]), patch_size, embedding_dim).to(device)
    model = SpectralMAE(encoder)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    effective_batch_size = len(train_x) if batch_size == "full" else int(batch_size)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x, dtype=torch.float32)),
        batch_size=effective_batch_size, shuffle=True,
    )
    scheduler = build_step_scheduler(
        optimizer, scheduler_name, total_steps=len(loader) * max(0, epochs),
        warmup_steps=len(loader), eta_min=scheduler_eta_min,
    )
    history: list[dict] = []
    # 每批随机掩码并优化重建损失，首批另存掩码输入示例。
    started = time.perf_counter()
    mask_example_saved = False
    epoch_bar = tqdm(range(1, max(0, epochs) + 1), desc="SSL epochs", unit="epoch", leave=False)
    for epoch in epoch_bar:
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        component_losses: dict[str, list[float]] = {"loss": [], "reconstruction_mse": []}
        for (batch,) in loader:
            batch = batch.to(device)
            mask = random_point_mask(
                len(batch), encoder.input_length, mask_ratio, mask_block_length, device=device,
            )
            # The MAE receives a randomly masked spectrum and learns to recover
            # the original signal. The first mask is saved as a visual audit.
            output = model(batch, mask)
            if not mask_example_saved and mask_example_path is not None:
                save_mask_example(batch, mask, mask_example_path)
                mask_example_saved = True
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for name in component_losses:
                if name in output:
                    component_losses[name].append(float(output[name].detach().cpu()))
        epoch_values = {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in component_losses.items()
        }
        epoch_loss = epoch_values["loss"]
        history.append({
            "epoch": epoch,
            **epoch_values,
            "learning_rate": epoch_learning_rate,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.perf_counter() - started,
            "stage": "ssl_pretrain",
            "backbone": backbone,
        })
        epoch_bar.set_postfix(loss=f"{epoch_loss:.5g}", lr=f"{epoch_learning_rate:.2g}")
        if logger is not None and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            logger.info("SSL epoch %d/%d | loss=%.6g | lr=%.6g | elapsed=%.1fs", epoch, epochs, epoch_loss, epoch_learning_rate, time.perf_counter() - started)
    epoch_bar.close()
    model.eval()
    def encode(values: np.ndarray) -> np.ndarray:
        """分批提取未掩码光谱的池化表示。"""
        out = []
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                batch = torch.as_tensor(values[start:start + effective_batch_size], dtype=torch.float32, device=device)
                _, pooled = encoder(batch, None)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, embedding_dim))
    train_embedding = encode(train_x) if save_pretrained_embeddings else np.empty((0, embedding_dim))
    test_embedding = encode(test_x) if save_pretrained_embeddings else np.empty((0, embedding_dim))
    # Reconstruction is deliberately separate from feature extraction: it
    # returns spectra in the original point order for the preprocessing path.
    def reconstruct(values: np.ndarray) -> np.ndarray:
        """重新采样点掩码，分批生成与原光谱等长的重建结果。"""
        out = []
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                batch = torch.as_tensor(values[start:start + effective_batch_size], dtype=torch.float32, device=device)
                point_mask = random_point_mask(
                    len(batch), encoder.input_length, mask_ratio, mask_block_length,
                    device=device,
                )
                if getattr(encoder, "input_channels", 1) == 2:
                    batch = torch.stack([
                        apply_point_mask(batch, point_mask),
                        point_mask.to(dtype=batch.dtype),
                    ], dim=1)
                else:
                    batch = apply_point_mask(batch, point_mask)
                patches = model.reconstruct(batch, None)
                out.append(patches.reshape(len(batch), -1)[:, : values.shape[1]].cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty_like(values)
    train_reconstruction = reconstruct(train_x) if reconstruct_spectra else np.empty((0, train_x.shape[1]))
    test_reconstruction = reconstruct(test_x) if reconstruct_spectra else np.empty((0, test_x.shape[1]))
    return (
        encoder, detached_state_dict(model), train_embedding, test_embedding,
        train_reconstruction, test_reconstruction, history, parameter_count(model),
    )


def supervised_embeddings(
    encoder: object,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int | str,
    seed: int,
    logger: logging.Logger | None = None,
    learning_rate: float = 3e-4,
    scheduler_name: str = "warmup_cosine",
    scheduler_eta_min: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Fine-tune the SSL encoder with a temporary supervised regression head.

    Missing targets are filled only to keep batches rectangular; ``mask_y``
    removes those filled entries from the supervised loss. The returned
    embeddings therefore use a single, explicitly trained encoder state.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = next(encoder.parameters()).device
    torch.manual_seed(seed)
    target_mean, target_scale, observed = _target_statistics(train_y)
    # 缺失标签先用训练折均值填充，再用掩码排除其监督损失。
    filled_y = np.where(np.isfinite(train_y), train_y, target_mean)
    mask_y = observed.astype(np.float32)
    normalized_y = (filled_y - target_mean) / target_scale
    head = nn.Linear(encoder.embedding_dim, train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=learning_rate)
    effective_batch_size = len(train_x) if batch_size == "full" else int(batch_size)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x, dtype=torch.float32), torch.as_tensor(normalized_y, dtype=torch.float32), torch.as_tensor(mask_y, dtype=torch.float32)),
        batch_size=effective_batch_size, shuffle=True,
    )
    scheduler = build_step_scheduler(
        optimizer, scheduler_name, total_steps=max(1, len(loader) * max(1, epochs)),
        warmup_steps=len(loader), eta_min=scheduler_eta_min,
    )
    history = []
    started = time.perf_counter()
    epoch_bar = tqdm(range(1, max(1, epochs) + 1), desc="Fine-tune epochs", unit="epoch", leave=False)
    for epoch in epoch_bar:
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        encoder.train(); head.train(); losses = []
        for batch_x, batch_y, batch_mask in loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            _, pooled = encoder(batch_x, None)
            residual = (head(pooled) - batch_y).square() * batch_mask
            loss = residual.sum() / batch_mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": epoch_loss, "learning_rate": epoch_learning_rate, "learning_rate_end": float(optimizer.param_groups[0]["lr"]), "elapsed_sec": time.perf_counter() - started, "stage": "supervised_finetune"})
        epoch_bar.set_postfix(loss=f"{epoch_loss:.5g}", lr=f"{epoch_learning_rate:.2g}")
        if logger is not None and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            logger.info("Fine-tune epoch %d/%d | loss=%.6g | lr=%.6g | elapsed=%.1fs", epoch, epochs, epoch_loss, epoch_learning_rate, time.perf_counter() - started)
    epoch_bar.close()
    encoder.eval()
    def encode(values: np.ndarray) -> np.ndarray:
        """用微调后的编码器分批提取池化表示。"""
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                batch = torch.as_tensor(values[start:start + effective_batch_size], dtype=torch.float32, device=device)
                outputs.append(encoder(batch, None)[1].cpu().numpy())
        return np.concatenate(outputs)
    return encode(train_x), encode(test_x), history, parameter_count(head)


def run_dataset(dataset: CanonicalDataset, dataset_name: str, args: argparse.Namespace,
                out_dir: Path, logger: logging.Logger) -> None:
    """逐折比较 MAE 重建预处理与微调表示，并保存结果和模型资产。"""
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
    labels = {experiment.name: experiment.label for experiment in EXPERIMENTS}
    for fold in folds:
        # 当前训练折拟合预处理；同一折的两个下游路径共用一次 MAE 预训练。
        fold_started = time.perf_counter()
        train_idx, test_idx = fold.train_idx, fold.test_idx
        # The fold owns its preprocessing statistics and its MAE weights. This
        # keeps both downstream paths free of test-fold leakage.
        train_x, test_x = preprocess_pair(dataset.spectra[train_idx], dataset.spectra[test_idx], method="AIRPLS")
        standardizer = FoldSpectralStandardizer().fit(train_x)
        train_x, test_x = standardizer.transform(train_x), standardizer.transform(test_x)
        train_y, test_y = targets[train_idx], targets[test_idx]
        (encoder, pretrained_state, train_embedding, test_embedding,
         train_reconstruction, test_reconstruction, pretrain_history, pretrained_params) = ssl_representation(
            args.backbone, train_x, test_x, args.epochs, args.batch_size, args.mask_ratio,
            args.mask_block_length, args.patch_size, DEFAULT_EMBEDDING_DIM, args.seed + fold.fold,
            args.learning_rate, args.scheduler, args.scheduler_eta_min,
            out_dir / f"mask_example_fold{fold.fold}.png", logger,
            save_pretrained_embeddings="mae_feature" in args.experiments,
            reconstruct_spectra="mae_preprocess" in args.experiments,
        )
        pretrain_seconds = time.perf_counter() - fold_started
        representations = {}
        # Materialize only the representation paths requested on the command
        # line; this avoids unnecessary arrays and keeps output files explicit.
        if "mae_preprocess" in args.experiments:
            representations["mae_preprocess"] = (train_reconstruction, test_reconstruction)
        history = [dict(row, experiment="mae", fold=fold.fold) for row in pretrain_history]
        finetuned_state = None
        finetune_head_params = 0
        finetune_seconds = 0.0
        pretrained_encoder_state = detached_state_dict(encoder)
        if "mae_feature" in args.experiments:
            # 仅特征路径追加监督微调，预训练编码器状态已提前独立保存。
            finetune_started = time.perf_counter()
            feature_train, feature_test, finetune_history, finetune_head_params = supervised_embeddings(
                encoder, train_x, train_y, test_x, args.finetune_epochs, args.batch_size,
                args.seed + fold.fold, logger, args.finetune_learning_rate,
                args.scheduler, args.scheduler_eta_min,
            )
            finetune_seconds = time.perf_counter() - finetune_started
            representations["mae_feature"] = (feature_train, feature_test)
            history.extend(dict(row, experiment="mae_feature", fold=fold.fold) for row in finetune_history)
            finetuned_state = detached_state_dict(encoder)
        run_id = f"mae_fold{fold.fold}"
        asset_dir = out_dir / "representations" / run_id
        arrays = {}
        if "mae_feature" in args.experiments:
            arrays.update(train_embedding_pretrained=train_embedding, test_embedding_pretrained=test_embedding)
        if "mae_preprocess" in args.experiments:
            arrays.update(train_reconstruction=train_reconstruction, test_reconstruction=test_reconstruction)
        # 分开保存预训练与微调权重，以及两条路径实际生成的表示。
        save_representation_assets(
            asset_dir, pretrained_state, pretrained_encoder_state, finetuned_state,
            arrays,
            history,
            {"dataset": dataset_name, "fold": fold.fold, "kind": "mae",
             "model_params": pretrained_params, "finetune_head_params": finetune_head_params},
        )
        histories.extend(dict(row, dataset=dataset_name, representation_run_id=run_id) for row in history)
        assets.append({"representation_run_id": run_id, "dataset": dataset_name,
                       "fold": fold.fold, "asset_dir": str(asset_dir)})
        validation_folds = inner_folds(train_idx, dataset.groups, min(3, len(train_idx)), args.seed + fold.fold)
        # 各表示配合同一组回归器；训练时间包含对应的表示学习阶段。
        for name in args.experiments:
            for reg in args.regressors:
                started = time.perf_counter()
                prediction, regressor_params, chosen = fit_downstream(
                    reg, representations[name][0], train_y, representations[name][1],
                    validation_folds, args.seed, args.jobs, args.batch_size,
                    args.scheduler, args.scheduler_eta_min,
                )
                fit_seconds = time.perf_counter() - started
                ssl_params = ssl_prediction_parameter_count(encoder, name, pretrained_params)
                total_params = ssl_params + regressor_params
                total_seconds = pretrain_seconds + (finetune_seconds if name == "mae_feature" else 0.0) + fit_seconds
                common = {"dataset": dataset_name, "experiment": name, "label": labels[name],
                          "regressor": reg, "fold": fold.fold, "representation_run_id": run_id,
                          "train_time_sec": total_seconds, "pretrain_time_sec": pretrain_seconds,
                          "finetune_time_sec": finetune_seconds if name == "mae_feature" else 0.0,
                          "regressor_time_sec": fit_seconds, "ssl_params": ssl_params,
                          "finetune_head_params": finetune_head_params if name == "mae_feature" else 0,
                          "regressor_params": regressor_params, "model_total_params": total_params,
                          "selected_params": json.dumps(chosen, ensure_ascii=False)}
                # 只在标签与预测同时有效的测试样本上逐目标评分。
                for col, target in enumerate(dataset.targets.columns):
                    valid = np.isfinite(test_y[:, col]) & np.isfinite(prediction[:, col])
                    if np.any(valid):
                        rmse, r2, rpd = metrics(test_y[valid, col], prediction[valid, col])
                        rows.append({**common, "target": target, "rmse": rmse, "r2": r2,
                                     "rpd": rpd, "n_test": int(valid.sum())})
                summaries.append(common)
    # 输出逐折和跨折指标、损失历史及可复用模型资产索引。
    # Save fold metrics, training histories, run summaries, and reusable MAE
    # assets separately so each can be consumed without rerunning training.
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
    """解析 MAE、微调、回归器和数据集参数。"""
    parser = argparse.ArgumentParser(description="Run E1C MAE experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1c")
    parser.add_argument("--swri-root", type=Path, default=ROOT / "data" / "SWRI_Diesel")
    parser.add_argument("--oil1-root", type=Path, default=ROOT / "data" / "crude_oil_1")
    parser.add_argument("--oil2-root", type=Path, default=ROOT / "data" / "crude_oil_2")
    parser.add_argument("--lucas-root", type=Path, default=ROOT / "data" / "lucas")
    parser.add_argument(
        "--datasets", nargs="+", default=["swri", "crude_oil"],
        choices=["swri", "crude_oil", "lucas"], help="Datasets to run",
    )
    parser.add_argument("--experiments", nargs="+", default=[e.name for e in EXPERIMENTS], choices=[e.name for e in EXPERIMENTS])
    parser.add_argument("--regressors", nargs="+", default=["pls", "rf", "mlp"], choices=["pls", "rf", "mlp"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--mask-block-length", type=int, default=1)
    parser.add_argument("--backbone", choices=["unet", "patch_transformer"], default="unet")
    parser.add_argument("--patch-size", "--patch-length", dest="patch_size", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=parse_batch_size, default=16)
    parser.add_argument("--scheduler", choices=["constant", "cosine", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small smoke-test limit per dataset")
    args = parser.parse_args()
    return args


def main() -> None:
    """校验配置、载入数据并运行 E1C。"""
    args = parse_args()
    if args.epochs < 0 or args.learning_rate <= 0 or args.patch_size < 1 or args.mask_block_length < 1:
        raise ValueError("Invalid SSL epochs, learning rate, patch size, or mask block length")
    if args.finetune_epochs < 1 or args.finetune_learning_rate <= 0:
        raise ValueError("Invalid MAE fine-tuning epochs or learning rate")
    if args.scheduler_eta_min < 0:
        raise ValueError("scheduler-eta-min must be non-negative")
    rates = [1e-3, args.learning_rate, args.finetune_learning_rate]
    if args.scheduler != "constant" and args.scheduler_eta_min > min(rates):
        raise ValueError("scheduler-eta-min must not exceed the initial learning rates")
    seed_everything(args.seed)
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir = timestamped_run_dir(base_output_dir)
    logger = configure_logging(args.output_dir, args.log_level, "e1c")
    run_started = time.perf_counter()
    logger.info("%s started | experiments=%s regressors=%s folds=%d datasets=%s seed=%d",
                "E1C", ",".join(args.experiments), ",".join(args.regressors),
                args.folds, ",".join(args.datasets), args.seed)
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
    config["experiment_id"] = "e1c"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if swri is not None:
        run_dataset(swri, "SWRI_Diesel_drop_missing", args, args.output_dir / "SWRI_Diesel_drop_missing", logger)
    if oils is not None:
        run_dataset(oils, "crude_oil_1+crude_oil_2", args, args.output_dir / "crude_oil_1_plus_crude_oil_2", logger)
    if lucas is not None:
        run_dataset(lucas, "LUCAS_SOIL", args, args.output_dir / "LUCAS_SOIL", logger)
    elapsed = time.perf_counter() - run_started
    logger.info("E1C completed successfully in %.1f seconds", elapsed)
    selected_datasets = []
    if "swri" in args.datasets:
        selected_datasets.append("SWRI_Diesel_drop_missing")
    if "crude_oil" in args.datasets:
        selected_datasets.append("crude_oil_1+crude_oil_2")
    if "lucas" in args.datasets:
        selected_datasets.append("LUCAS_SOIL")
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": selected_datasets, "experiments": args.experiments, "regressors": args.regressors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
