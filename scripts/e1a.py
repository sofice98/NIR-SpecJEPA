"""Run E1A preprocessing baselines.

E1A is the fully supervised reference experiment. Each outer fold fits
spectral preprocessing on the training split only, then compares multiple
regression heads on the same transformed representation. The resulting
baseline helps separate gains from preprocessing, representation learning,
and the downstream regressor.
"""
from __future__ import annotations
import argparse, json, logging, random, time
from dataclasses import dataclass
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
from preprocessing.spectral import FoldSpectralStandardizer, preprocess_pair
from splits.cross_validation import pooled_group_balanced_folds
from models.shared.experiment import preprocessing_parameter_count

@dataclass(frozen=True)
class Experiment:
    """描述一种待比较的光谱预处理方案。"""
    name: str
    label: str
EXPERIMENTS = (
    Experiment("raw", "Raw"), Experiment("fd", "一阶导"),
    Experiment("sd", "二阶导"), Experiment("snv", "SNV"),
    Experiment("msc", "MSC"), Experiment("airpls", "airPLS"),
)

class TqdmLoggingHandler(logging.Handler):
    """通过 tqdm 输出日志，避免打断进度条。"""

    def emit(self, record: logging.LogRecord) -> None:
        """将格式化后的日志写到进度条上方。"""
        tqdm.write(self.format(record))

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
    """固定 Python 和 NumPy 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)


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
    """Fit one regressor per target while tolerating missing analyte labels.

    SWRI can have a missing value for one target but valid values for another.
    Filtering each target independently keeps those usable samples and the
    inner validation folds are remapped to the filtered row positions.
    """
    from models.shared.experiment import sklearn_parameter_count
    from models.shared.regressors import fit_fixed_representation_mlp, pls, random_forest
    prediction = np.full((len(test_x), train_y.shape[1]), np.nan, dtype=float)
    selected: dict[str, object] = {}
    total_params = 0
    for column in range(train_y.shape[1]):
        # 每个目标单独过滤缺失标签，并将外层训练索引映射到该目标的有效样本。
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


def run_dataset(dataset: CanonicalDataset, dataset_name: str, args: argparse.Namespace,
                out_dir: Path, logger: logging.Logger) -> None:
    """对一个数据集执行预处理与回归器的外层交叉验证并保存结果。"""
    # 可选截取少量样本供快速运行；之后按样本组建立外层折。
    dataset.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset.sample_id))
        dataset = CanonicalDataset(dataset.name, dataset.sample_id[:n], dataset.wavelengths_nm,
                                   dataset.spectra[:n], dataset.targets.iloc[:n].reset_index(drop=True),
                                   dataset.groups[:n], dataset.metadata)
    if len(dataset.sample_id) < 2:
        raise ValueError("at least 2 samples are required for cross-validation")
    # Group-aware folds keep related spectra together, avoiding optimistic scores
    # caused by placing near-duplicate spectra in both train and test sets.
    folds = pooled_group_balanced_folds(dataset.groups, n_splits=min(args.folds, len(dataset.sample_id)), seed=args.seed)
    targets = dataset.targets.to_numpy(float)
    rows: list[dict] = []
    summaries: list[dict] = []
    methods = {"raw": "R", "fd": "FD", "sd": "SD", "snv": "SNV", "msc": "MSC", "airpls": "AIRPLS"}
    selected_experiments = [item for item in EXPERIMENTS if item.name in args.experiments]
    total_jobs = len(selected_experiments) * len(folds) * len(args.regressors)
    logger.info("Dataset %s | samples=%d wavelengths=%d folds=%d jobs=%d",
                dataset_name, len(dataset.sample_id), dataset.spectra.shape[1], len(folds), total_jobs)
    job_bar = tqdm(total=total_jobs, desc=f"{dataset_name} jobs", unit="job")
    try:
        for exp in selected_experiments:
            method = methods[exp.name]
            for fold in folds:
                # 预处理和标准化均在当前训练折拟合，随后共用同一表示比较回归器。
                train_idx, test_idx = fold.train_idx, fold.test_idx
                # Data-dependent preprocessing is applied after the split. The
                # standardizer below is fitted on train_x and reused unchanged.
                train_x, test_x = preprocess_pair(dataset.spectra[train_idx], dataset.spectra[test_idx], method=method)
                standardizer = FoldSpectralStandardizer().fit(train_x)
                train_x, test_x = standardizer.transform(train_x), standardizer.transform(test_x)
                train_y, test_y = targets[train_idx], targets[test_idx]
                validation_folds = inner_folds(train_idx, dataset.groups, min(3, len(train_idx)), args.seed + fold.fold)
                preprocessing_params = preprocessing_parameter_count(method, standardizer, dataset.spectra.shape[1])
                for reg in args.regressors:
                    job_bar.set_postfix_str(f"{exp.name}/{reg} fold={fold.fold}")
                    logger.info("Start dataset=%s experiment=%s regressor=%s fold=%d | train=%d test=%d",
                                dataset_name, exp.name, reg, fold.fold, len(train_idx), len(test_idx))
                    started = time.perf_counter()
                    prediction, regressor_params, chosen = fit_downstream(
                        reg, train_x, train_y, test_x, validation_folds, args.seed, args.jobs,
                        args.batch_size, args.scheduler, args.scheduler_eta_min,
                    )
                    fit_seconds = time.perf_counter() - started
                    selected_params = json.dumps(chosen, ensure_ascii=False)
                    common = {"dataset": dataset_name, "experiment": exp.name, "label": exp.label,
                              "regressor": reg, "fold": fold.fold, "train_time_sec": fit_seconds,
                              "preprocessing_params": preprocessing_params, "regressor_params": regressor_params,
                              "model_total_params": preprocessing_params + regressor_params,
                              "selected_params": selected_params}
                    metric_count = 0
                    # 只在标签和预测都有效的测试样本上逐目标计算指标。
                    # Score only samples with both a finite reference label and
                    # a finite prediction; this is important for sparse SWRI labels.
                    for col, target in enumerate(dataset.targets.columns):
                        valid = np.isfinite(test_y[:, col]) & np.isfinite(prediction[:, col])
                        if np.any(valid):
                            rmse, r2, rpd = metrics(test_y[valid, col], prediction[valid, col])
                            rows.append({**common, "target": target, "rmse": rmse, "r2": r2,
                                         "rpd": rpd, "n_test": int(valid.sum())})
                            metric_count += 1
                    summaries.append(common)
                    logger.info("Done dataset=%s experiment=%s regressor=%s fold=%d | fit=%.1fs targets=%d/%d params=%d",
                                dataset_name, exp.name, reg, fold.fold, fit_seconds, metric_count,
                                targets.shape[1], preprocessing_params + regressor_params)
                    job_bar.update(1)
    finally:
        job_bar.close()
    # 同时输出逐折指标、运行摘要和跨折均值，便于复现与比较。
    # Keep detailed fold metrics separate from the compact run summary so the
    # former can be used for statistical analysis and the latter for auditing.
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    if rows:
        aggregate = pd.DataFrame(rows).groupby(
            ["dataset", "experiment", "label", "regressor", "target"], as_index=False
        )[["rmse", "r2", "rpd", "train_time_sec", "model_total_params"]].mean()
        aggregate.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")
    logger.info("Wrote %d metric rows and %d summaries to %s", len(rows), len(summaries), out_dir)


def parse_args() -> argparse.Namespace:
    """解析数据集、预处理方法、回归器和训练参数。"""
    parser = argparse.ArgumentParser(description="Run E1A preprocessing baselines")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1a")
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
    """校验配置、载入数据并依次运行选定数据集。"""
    args = parse_args()
    if args.scheduler_eta_min < 0:
        raise ValueError("scheduler-eta-min must be non-negative")
    if args.scheduler != "constant" and args.scheduler_eta_min > 1e-3:
        raise ValueError("scheduler-eta-min must not exceed the initial learning rates")
    seed_everything(args.seed)
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir = timestamped_run_dir(base_output_dir)
    logger = configure_logging(args.output_dir, args.log_level, "e1a")
    run_started = time.perf_counter()
    logger.info("%s started | experiments=%s regressors=%s folds=%d datasets=%s seed=%d",
                "E1A", ",".join(args.experiments), ",".join(args.regressors),
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
    config["experiment_id"] = "e1a"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if swri is not None:
        run_dataset(swri, "SWRI_Diesel_drop_missing", args, args.output_dir / "SWRI_Diesel_drop_missing", logger)
    if oils is not None:
        run_dataset(oils, "crude_oil_1+crude_oil_2", args, args.output_dir / "crude_oil_1_plus_crude_oil_2", logger)
    if lucas is not None:
        run_dataset(lucas, "LUCAS_SOIL", args, args.output_dir / "LUCAS_SOIL", logger)
    elapsed = time.perf_counter() - run_started
    logger.info("E1A completed successfully in %.1f seconds", elapsed)
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
