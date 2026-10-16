"""E1D one-factor-at-a-time experiment driver.

The training implementation remains in :mod:`scripts.run_e1d`.  This module
only defines the fixed baseline, creates one-factor configurations, and runs
the three requested downstream heads (PLS, random forest, and MLP).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


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
    """Use the project environment when the driver is launched by system Python."""
    cuda_env = Path(r"D:\ProgramFiles\miniconda3\envs\spectra\python.exe")
    if cuda_env.is_file():
        return str(cuda_env)
    local = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(local) if local.is_file() else sys.executable


def command_for(config: E1DConfig, output_dir: Path, args: argparse.Namespace) -> list[str]:
    script = ROOT / "scripts" / "run_e1d.py"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "e1d_ota")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

