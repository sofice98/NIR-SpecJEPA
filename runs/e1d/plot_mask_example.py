"""Plot an E1D continuous block-mask audit in the E1C figure style."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.SWRI_Diesel.adapter import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "SWRI_Diesel"
    dataset = load_dataset(Path(__file__).resolve().parents[2] / "data" / "SWRI_Diesel")
    audit = pd.read_csv(out / "first_fold_block_masks.csv")
    chosen = audit[["dataset_index", "sample_id"]].drop_duplicates().head(3)
    wavelengths = np.asarray(dataset.wavelengths_nm, dtype=float)
    n_points = len(wavelengths)
    boundaries = np.empty(n_points + 1)
    boundaries[1:-1] = (wavelengths[:-1] + wavelengths[1:]) / 2
    boundaries[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
    boundaries[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 3.1 * len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (_, sample) in zip(axes, chosen.iterrows()):
        idx = int(sample.dataset_index)
        values = np.asarray(dataset.spectra[idx], dtype=float)
        rows = audit[audit.dataset_index == idx].sort_values("patch_index")
        axis.plot(wavelengths, values, color="#64748b", linewidth=1, label="Original spectrum")
        masked = rows.masked.to_numpy(dtype=bool)
        visible = values.copy()
        visible[np.repeat(masked, int(rows.point_end_exclusive.iloc[0] - rows.point_start.iloc[0]))[:n_points]] = np.nan
        axis.plot(wavelengths, visible, color="#2563eb", linewidth=1.2, label="Visible patches")
        for row in rows.itertuples():
            if row.masked:
                start = int(row.point_start)
                stop = min(int(row.point_end_exclusive), n_points)
                axis.axvspan(boundaries[start], boundaries[stop], color="#ef4444", alpha=0.2,
                             label="Masked contiguous block" if row.patch_index == rows.loc[rows.masked, "patch_index"].iloc[0] else None)
        axis.set_title(f"Sample {sample.sample_id} (row {idx})")
        axis.set_ylabel("Original intensity")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=3, loc="best")
    axes[-1].set_xlabel("Wavelength (nm)")
    cfg = pd.read_json(run_dir / "config.json", typ="series")
    fig.suptitle(f"E1D SWRI: illustrative continuous block mask (patch size={int(cfg['patch_size'])}, ratio={float(cfg['mask_ratio']):g})\n"
                 "Red shading marks contiguous patch blocks masked after patch projection; raw spectrum is unchanged.", fontsize=11)
    fig.tight_layout()
    path = out / "first_fold_block_masks.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

