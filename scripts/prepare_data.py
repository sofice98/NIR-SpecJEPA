"""Create canonical spectral data and reproducible split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.crude_oil_private.adapter import load_dataset
from splits.cross_validation import leave_one_group_out_folds, pooled_group_balanced_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data" / "crude_oil_private",
        help="包含原始数据和 adapter 的任务数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "canonical" / "crude_oil_private",
        help="规范化数据和固定划分清单的输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = load_dataset(dataset_root)
    groups = pooled.groups.astype(str)
    np.save(output_dir / "common_wavelengths_nm.npy", pooled.wavelengths_nm)
    np.save(output_dir / "spectra_resampled.npy", pooled.spectra.astype(np.float32))
    pooled.targets.to_csv(output_dir / "targets.csv", index=False)
    pd.DataFrame({"sample_id": pooled.sample_id, "group": groups}).to_csv(
        output_dir / "samples.csv", index=False
    )

    def serialise(folds):
        return [
            {
                "fold": fold.fold,
                "train_idx": fold.train_idx.tolist(),
                "test_idx": fold.test_idx.tolist(),
                "test_group": fold.test_group,
            }
            for fold in folds
        ]

    (output_dir / "folds_pooled.json").write_text(
        json.dumps(serialise(pooled_group_balanced_folds(groups)), indent=2), encoding="utf-8"
    )
    (output_dir / "folds_leave_one_group_out.json").write_text(
        json.dumps(serialise(leave_one_group_out_folds(groups)), indent=2), encoding="utf-8"
    )
    manifest = {
        **pooled.metadata,
        "n_samples": int(len(pooled.sample_id)),
        "n_wavelengths": int(len(pooled.wavelengths_nm)),
        "wavelength_min_nm": float(pooled.wavelengths_nm.min()),
        "wavelength_max_nm": float(pooled.wavelengths_nm.max()),
        "target_columns": pooled.targets.columns.tolist(),
        "groups": {group: int(np.sum(groups == group)) for group in np.unique(groups)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
