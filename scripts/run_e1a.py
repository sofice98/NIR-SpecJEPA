"""Run E1-a single-dataset baselines and representation comparisons.

The default protocol evaluates the eight rows in the E1-a table with PLS and
Random Forest downstream regressors (16 configuration/regressor runs) on:

* SWRI_Diesel, using the ``drop_missing`` policy; and
* crude_oil_1 merged with crude_oil_2 on their common wavelength grid.

Each run writes a long-form ``metrics.csv``, ``loss_curve.csv`` and
``run_summary.csv``.  The implementation deliberately keeps all fold fitting
inside this script so that preprocessing, masking and downstream models are
fit on training data only.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - requirements.txt includes tqdm
    class _NullProgress:
        def __init__(self, iterable=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, _n=1):
            return None

        def set_postfix(self, **_kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **_kwargs):
        return _NullProgress(iterable)


class TqdmLoggingHandler(logging.Handler):
    """Send console logs above active tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    """Create a fresh console/file logger for one E1-a invocation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("e1a")
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
    history["stage"] = history.get("stage", pd.Series(index=history.index, dtype=str)).fillna("ssl_pretrain")
    # Keep the two MAE experiments independent: ``mae_preprocess`` and
    # ``mae_feature`` have different downstream semantics and should not share
    # a loss plot even though both use the MAE encoder.
    history["model"] = history["experiment"].astype(str)

    # Keep model and stage separate so pretraining and supervised fine-tuning
    # are never mixed on one axis.  Each figure contains all fold/regressor
    # curves for that model-stage pair, plus a bold mean curve across them.
    stage_labels = {
        "ssl_pretrain": "self-supervised pretraining",
        "supervised_finetune": "supervised fine-tuning",
    }
    for (model, stage), subset in history.groupby(["model", "stage"], sort=True):
        fig, ax = plt.subplots(figsize=(12, 7))
        for (experiment, regressor, fold), group in subset.groupby(["experiment", "regressor", "fold"], sort=True):
            group = group.sort_values("epoch")
            ax.plot(
                group["epoch"], group["loss"], alpha=0.45, linewidth=1.0,
                label=f"{experiment}/{regressor}/f{fold}",
            )
        mean_history = subset.groupby("epoch", as_index=False)["loss"].mean().sort_values("epoch")
        ax.plot(mean_history["epoch"], mean_history["loss"], color="black", linewidth=2.5, label="mean")
        ax.set_title(f"E1-a {model} {stage_labels.get(stage, stage)} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        path = out_dir / f"loss_curves_{model.lower()}_{stage}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(path)
    logger.info("Generated loss plots: %s", ", ".join(str(item.name) for item in generated))
    return generated
try:
    import torch
    from torch import nn
except ImportError:  # allows CLI help and static validation without torch installed
    torch = None
    class nn:  # type: ignore[no-redef]
        class Module:  # minimal placeholder; SSL execution reports a clear error
            pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.SWRI_Diesel.adapter import load_dataset as load_swri
from data.dataset_adapters import CanonicalDataset, merge_datasets
from data.crude_oil_1.adapter import load_dataset as load_oil1
from data.crude_oil_2.adapter import load_dataset as load_oil2
from preprocessing.spectral import FoldSpectralStandardizer, preprocess_pair
from splits.cross_validation import pooled_group_balanced_folds


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    label: str


EXPERIMENTS = (
    Experiment("raw", "preprocess", "Raw"),
    Experiment("fd", "preprocess", "一阶导"),
    Experiment("sd", "preprocess", "二阶导"),
    Experiment("snv", "preprocess", "SNV"),
    Experiment("msc", "preprocess", "MSC"),
    Experiment("mae_preprocess", "mae_preprocess", "MAE预处理"),
    Experiment("mae_feature", "mae_feature", "MAE特征提取"),
    Experiment("lejepa", "lejepa", "LeJEPA"),
)


class ExactDenseUNet(nn.Module):
    """Requested m-2048-1024-512-256-512-1024-2048-m dense U-Net."""

    def __init__(self, input_length: int, patch_size: int = 1, **_: object) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MAE/LeJEPA experiments; install requirements.txt")
        nn.Module.__init__(self)
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = input_length
        self.padded_length = input_length
        self.embedding_dim = 256
        self.enc1 = nn.Sequential(nn.Linear(input_length, 2048), nn.GELU())
        self.enc2 = nn.Sequential(nn.Linear(2048, 1024), nn.GELU())
        self.enc3 = nn.Sequential(nn.Linear(1024, 512), nn.GELU())
        self.bottleneck = nn.Sequential(nn.Linear(512, 256), nn.GELU())
        self.dec1 = nn.Sequential(nn.Linear(256, 512), nn.GELU())
        self.dec2 = nn.Sequential(nn.Linear(512, 1024), nn.GELU())
        self.dec3 = nn.Sequential(nn.Linear(1024, 2048), nn.GELU())
        self.output = nn.Linear(2048, input_length)

    def patchify(self, spectra):
        return spectra.unsqueeze(-1)

    def _masked(self, spectra, patch_mask):
        if patch_mask is None:
            return spectra
        if patch_mask.shape != spectra.shape:
            raise ValueError("With length-1 masks, patch_mask must match spectra")
        return spectra.masked_fill(patch_mask, 0.0)

    def forward(self, spectra, patch_mask=None):
        x = self._masked(spectra, patch_mask)
        latent = self.bottleneck(self.enc3(self.enc2(self.enc1(x))))
        # patch_size=1 means every input point is one patch.  Return one
        # scalar token per point so the encoder follows the shared MAE/LeJEPA
        # contract: [batch, n_patches, embedding_dim].
        tokens = latent.unsqueeze(1).expand(-1, self.n_patches, -1)
        return tokens, latent

    def reconstruct(self, spectra, patch_mask=None):
        latent = self.forward(spectra, patch_mask)[1]
        decoded = self.output(self.dec3(self.dec2(self.dec1(latent))))
        return decoded.unsqueeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1a")
    parser.add_argument("--swri-root", type=Path, default=ROOT / "data" / "SWRI_Diesel")
    parser.add_argument("--oil1-root", type=Path, default=ROOT / "data" / "crude_oil_1")
    parser.add_argument("--oil2-root", type=Path, default=ROOT / "data" / "crude_oil_2")
    parser.add_argument("--experiments", nargs="+", default=[e.name for e in EXPERIMENTS], choices=[e.name for e in EXPERIMENTS])
    parser.add_argument("--regressors", nargs="+", default=["pls", "rf"], choices=["pls", "rf"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100, help="SSL pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="Supervised fine-tuning epochs for MAE features/LeJEPA")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="MAE/LeJEPA self-supervised learning rate")
    parser.add_argument("--finetune-learning-rate", type=float, default=3e-4, help="Supervised fine-tuning learning rate for MAE features/LeJEPA")
    parser.add_argument("--scheduler", choices=["constant", "cosine"], default="cosine", help="Learning-rate scheduler for neural-network stages")
    parser.add_argument("--scheduler-eta-min", type=float, default=0.0, help="Minimum learning rate used by cosine scheduling")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small smoke-test limit per dataset")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
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
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    variance = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / variance) if variance > 1e-12 else float("nan")
    rpd = float(np.std(y_true, ddof=1) / rmse) if rmse > 1e-12 and len(y_true) > 1 else float("nan")
    return rmse, r2, rpd


def parameter_count(model: object) -> int:
    return int(sum(p.numel() for p in model.parameters())) if hasattr(model, "parameters") else 0


def sklearn_parameter_count(model: object) -> int:
    """Count learned numeric state, including all Random Forest tree arrays."""
    if model is None:
        return 0
    if not hasattr(model, "__dict__"):
        return 0
    total = 0
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += value.size
        elif isinstance(value, list):
            total += sum(sklearn_parameter_count(item) for item in value)
        elif hasattr(value, "tree_"):
            tree = value.tree_
            total += sum(getattr(tree, name).size for name in ("children_left", "children_right", "feature", "threshold", "value"))
    return int(total)


def inner_folds(indices: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
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
) -> tuple[np.ndarray, int, dict]:
    """Fit one regressor per target, which naturally supports SWRI missing labels."""
    from models.baselines.classical import pls, random_forest
    prediction = np.full((len(test_x), train_y.shape[1]), np.nan, dtype=float)
    selected: dict[str, object] = {}
    total_params = 0
    for column in range(train_y.shape[1]):
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
            result = pls(x, y, test_x, valid_folds, (2, 4, 8, 16), 500, 1e-6)
        else:
            result = random_forest(x, y[:, 0], test_x, valid_folds, (200,), (None, 20), (1, 2), ("sqrt", 1.0), seed, jobs)
        result_prediction = np.asarray(result.prediction, dtype=float)
        prediction[:, column] = result_prediction[:, 0] if result_prediction.ndim == 2 else result_prediction
        selected[str(column)] = result.selected_params
        total_params += sklearn_parameter_count(result.model)
    return prediction, total_params, selected


def ssl_representation(
    kind: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int,
    mask_ratio: float,
    patch_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    scheduler_name: str = "cosine",
    scheduler_eta_min: float = 0.0,
    logger: logging.Logger | None = None,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    from models.lejepa.model import SpectralLeJEPA
    from models.mae.model import SpectralMAE
    from models.shared.masking import random_block_mask
    if torch is None:
        raise RuntimeError("PyTorch is required for MAE/LeJEPA experiments; install requirements.txt")
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    del embedding_dim  # fixed at the requested 256-dimensional bottleneck
    encoder = ExactDenseUNet(len(train_x[0]), patch_size=patch_size).to(device)
    model = SpectralMAE(encoder) if kind == "mae" else SpectralLeJEPA(encoder)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if scheduler_name == "cosine" and epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=scheduler_eta_min
        )
    loader = DataLoader(TensorDataset(torch.as_tensor(train_x, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    history: list[dict] = []
    started = time.perf_counter()
    epoch_bar = tqdm(range(1, max(0, epochs) + 1), desc="SSL epochs", unit="epoch", leave=False)
    for epoch in epoch_bar:
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            if kind == "mae":
                mask = random_block_mask(len(batch), encoder.n_patches, mask_ratio, block_patches=1, device=device)
                output = model(batch, mask)
            else:
                mask1 = random_block_mask(len(batch), encoder.n_patches, mask_ratio, block_patches=1, device=device)
                mask2 = random_block_mask(len(batch), encoder.n_patches, mask_ratio, block_patches=1, device=device)
                output = model(batch, torch.stack([mask1, mask2]))
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            losses.append(float(output["loss"].detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "loss": epoch_loss, "learning_rate": epoch_learning_rate, "elapsed_sec": time.perf_counter() - started})
        epoch_bar.set_postfix(loss=f"{epoch_loss:.5g}", lr=f"{epoch_learning_rate:.2g}")
        if logger is not None and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            logger.info("SSL epoch %d/%d | loss=%.6g | lr=%.6g | elapsed=%.1fs", epoch, epochs, epoch_loss, epoch_learning_rate, time.perf_counter() - started)
    epoch_bar.close()
    model.eval()
    def encode(values: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
                _, pooled = encoder(batch, None)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, embedding_dim))
    train_embedding, test_embedding = encode(train_x), encode(test_x)
    def reconstruct(values: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
                patches = encoder.reconstruct(batch, None)
                out.append(patches.reshape(len(batch), -1)[:, : values.shape[1]].cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty_like(values)
    train_reconstruction = reconstruct(train_x) if kind == "mae" else train_x
    test_reconstruction = reconstruct(test_x) if kind == "mae" else test_x
    return encoder, train_embedding, test_embedding, train_reconstruction, test_reconstruction, history, parameter_count(model)


def supervised_embeddings(
    encoder: object,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    logger: logging.Logger | None = None,
    learning_rate: float = 3e-4,
    scheduler_name: str = "cosine",
    scheduler_eta_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Fine-tune the SSL encoder with a temporary supervised regression head."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = next(encoder.parameters()).device
    torch.manual_seed(seed)
    target_mean = np.nanmean(train_y, axis=0)
    target_scale = np.nanstd(train_y, axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    filled_y = np.where(np.isfinite(train_y), train_y, target_mean)
    mask_y = np.isfinite(train_y).astype(np.float32)
    normalized_y = (filled_y - target_mean) / target_scale
    head = nn.Linear(encoder.embedding_dim, train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=learning_rate)
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=scheduler_eta_min
        )
    loader = DataLoader(TensorDataset(torch.as_tensor(train_x, dtype=torch.float32), torch.as_tensor(normalized_y, dtype=torch.float32), torch.as_tensor(mask_y, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
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
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "loss": epoch_loss, "learning_rate": epoch_learning_rate, "elapsed_sec": time.perf_counter() - started, "stage": "supervised_finetune"})
        epoch_bar.set_postfix(loss=f"{epoch_loss:.5g}", lr=f"{epoch_learning_rate:.2g}")
        if logger is not None and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            logger.info("Fine-tune epoch %d/%d | loss=%.6g | lr=%.6g | elapsed=%.1fs", epoch, epochs, epoch_loss, epoch_learning_rate, time.perf_counter() - started)
    epoch_bar.close()
    encoder.eval()
    def encode(values: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
                outputs.append(encoder(batch, None)[1].cpu().numpy())
        return np.concatenate(outputs)
    return encode(train_x), encode(test_x), history, parameter_count(head)


def run_dataset(dataset: CanonicalDataset, dataset_name: str, args: argparse.Namespace, out_dir: Path, logger: logging.Logger) -> None:
    dataset.validate()
    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset.sample_id))
        dataset = CanonicalDataset(dataset.name, dataset.sample_id[:n], dataset.wavelengths_nm, dataset.spectra[:n], dataset.targets.iloc[:n].reset_index(drop=True), dataset.groups[:n], dataset.metadata)
    folds = pooled_group_balanced_folds(dataset.groups, n_splits=min(args.folds, len(dataset.sample_id)), seed=args.seed)
    metrics_rows, history_rows, summary_rows = [], [], []
    selected_experiments = [e for e in EXPERIMENTS if e.name in args.experiments]
    total_jobs = len(selected_experiments) * len(args.regressors) * len(folds)
    job_bar = tqdm(total=total_jobs, desc=f"{dataset_name} jobs", unit="job")
    logger.info("Dataset %s | samples=%d wavelengths=%d folds=%d jobs=%d", dataset_name, len(dataset.sample_id), dataset.spectra.shape[1], len(folds), total_jobs)
    for experiment in selected_experiments:
        for regressor in args.regressors:
            for fold in folds:
                job_started = time.perf_counter()
                logger.info("Start dataset=%s experiment=%s regressor=%s fold=%d", dataset_name, experiment.name, regressor, fold.fold)
                train_idx, test_idx = fold.train_idx, fold.test_idx
                train_y = dataset.targets.to_numpy(float)[train_idx]
                test_y = dataset.targets.to_numpy(float)[test_idx]
                train_x, test_x = preprocess_pair(dataset.spectra[train_idx], dataset.spectra[test_idx], method={"raw": "R", "fd": "FD", "sd": "SD", "snv": "SNV", "msc": "MSC"}.get(experiment.name, "R"))
                standardizer = FoldSpectralStandardizer().fit(train_x)
                train_x, test_x = standardizer.transform(train_x), standardizer.transform(test_x)
                ssl_params = 0
                history = [{"epoch": 0, "loss": float("nan"), "elapsed_sec": 0.0}]
                if experiment.name in {"mae_preprocess", "mae_feature", "lejepa"}:
                    kind = "lejepa" if experiment.name == "lejepa" else "mae"
                    encoder, train_repr, test_repr, train_recon, test_recon, history, ssl_params = ssl_representation(
                        kind, train_x, test_x, args.epochs, args.batch_size, args.mask_ratio,
                        args.patch_size, args.embedding_dim, args.seed + fold.fold,
                        args.learning_rate, args.scheduler, args.scheduler_eta_min, logger,
                    )
                    if experiment.name == "mae_preprocess":
                        # Freeze MAE and feed its reconstructed spectra to the
                        # classical regressor.
                        train_x, test_x = train_recon, test_recon
                    else:
                        train_repr, test_repr, supervised_history, head_params = supervised_embeddings(
                            encoder, train_x, train_y, test_x, args.finetune_epochs,
                            args.batch_size, args.seed + fold.fold, logger,
                            args.finetune_learning_rate, args.scheduler, args.scheduler_eta_min,
                        )
                        history.extend(supervised_history)
                        ssl_params += head_params
                        train_x, test_x = train_repr, test_repr
                started = time.perf_counter()
                prediction, model_params, selected = fit_downstream(regressor, train_x, train_y, test_x, inner_folds(train_idx, dataset.groups, min(3, len(train_idx)), args.seed + fold.fold), args.seed, args.jobs)
                fit_seconds = time.perf_counter() - started
                for epoch_row in history:
                    history_rows.append({"dataset": dataset_name, "experiment": experiment.name, "regressor": regressor, "fold": fold.fold, **epoch_row})
                for target_index, target in enumerate(dataset.targets.columns):
                    valid = np.isfinite(test_y[:, target_index]) & np.isfinite(prediction[:, target_index])
                    if not np.any(valid):
                        continue
                    rmse, r2, rpd = metrics(test_y[valid, target_index], prediction[valid, target_index])
                    metrics_rows.append({"dataset": dataset_name, "experiment": experiment.name, "label": experiment.label, "regressor": regressor, "fold": fold.fold, "target": target, "rmse": rmse, "r2": r2, "rpd": rpd, "train_time_sec": fit_seconds, "model_total_params": int(ssl_params + model_params), "n_test": int(valid.sum()), "selected_params": json.dumps(selected, ensure_ascii=False)})
                summary_rows.append({"dataset": dataset_name, "experiment": experiment.name, "label": experiment.label, "regressor": regressor, "fold": fold.fold, "train_time_sec": fit_seconds, "model_total_params": int(ssl_params + model_params), "selected_params": json.dumps(selected, ensure_ascii=False)})
                job_seconds = time.perf_counter() - job_started
                logger.info("Done dataset=%s experiment=%s regressor=%s fold=%d | fit=%.1fs total=%.1fs", dataset_name, experiment.name, regressor, fold.fold, fit_seconds, job_seconds)
                job_bar.update(1)
    job_bar.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(out_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    plot_loss_curves(history_rows, out_dir, logger)
    if metrics_rows:
        aggregate = pd.DataFrame(metrics_rows).groupby(["dataset", "experiment", "label", "regressor", "target"], as_index=False)[["rmse", "r2", "rpd", "train_time_sec", "model_total_params"]].mean()
        aggregate.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")
    logger.info("Wrote %d metric rows, %d history rows and %d summaries to %s", len(metrics_rows), len(history_rows), len(summary_rows), out_dir)


def main() -> None:
    args = parse_args()
    if args.epochs < 0 or args.finetune_epochs < 1:
        raise ValueError("epochs must be non-negative and finetune-epochs must be positive")
    if args.learning_rate <= 0 or args.finetune_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.scheduler_eta_min < 0:
        raise ValueError("scheduler-eta-min must be non-negative")
    if args.scheduler == "cosine" and args.scheduler_eta_min > min(args.learning_rate, args.finetune_learning_rate):
        raise ValueError("scheduler-eta-min must not exceed the initial learning rates")
    seed_everything(args.seed)
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir = timestamped_run_dir(base_output_dir)
    logger = configure_logging(args.output_dir, args.log_level)
    run_started = time.perf_counter()
    logger.info(
        "E1-a started | experiments=%s regressors=%s folds=%d epochs=%d finetune_epochs=%d "
        "lr=%g finetune_lr=%g seed=%d",
        ",".join(args.experiments), ",".join(args.regressors), args.folds,
        args.epochs, args.finetune_epochs, args.learning_rate, args.finetune_learning_rate, args.seed,
    )
    logger.info("Learning-rate scheduler=%s eta_min=%g", args.scheduler, args.scheduler_eta_min)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Loading datasets...")
    swri = load_swri(args.swri_root)
    oils = merge_datasets([load_oil1(args.oil1_root), load_oil2(args.oil2_root)], name="crude_oil_1+crude_oil_2")
    logger.info("Loaded SWRI_Diesel: samples=%d, wavelengths=%d", len(swri.sample_id), swri.spectra.shape[1])
    logger.info("Loaded crude oil mixture: samples=%d, wavelengths=%d", len(oils.sample_id), oils.spectra.shape[1])
    config = {
        "experiments": args.experiments,
        "regressors": args.regressors,
        "folds": args.folds,
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "learning_rate": args.learning_rate,
        "finetune_learning_rate": args.finetune_learning_rate,
        "scheduler": args.scheduler,
        "scheduler_eta_min": args.scheduler_eta_min,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    run_dataset(swri, "SWRI_Diesel_drop_missing", args, args.output_dir / "SWRI_Diesel_drop_missing", logger)
    run_dataset(oils, "crude_oil_1+crude_oil_2", args, args.output_dir / "crude_oil_1_plus_crude_oil_2", logger)
    elapsed = time.perf_counter() - run_started
    logger.info("E1-a completed successfully in %.1f seconds", elapsed)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": ["SWRI_Diesel_drop_missing", "crude_oil_1+crude_oil_2"], "experiments": args.experiments, "regressors": args.regressors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
