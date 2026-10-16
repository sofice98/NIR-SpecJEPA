"""Plot the saved SSL losses for a standalone E1D run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    weight = float(config["sigreg_weight"])
    history = pd.read_csv(run_dir / "SWRI_Diesel" / "loss_curve.csv")
    ssl = history.loc[history["stage"] == "ssl_pretrain"].copy()
    ssl["weighted_invariance"] = (1 - weight) * ssl["invariance_mse"]
    ssl["weighted_sigreg"] = weight * ssl["sigreg"]
    if not np.allclose(ssl["loss"], ssl["weighted_invariance"] + ssl["weighted_sigreg"], rtol=1e-5, atol=1e-4):
        raise ValueError("Saved total loss differs from the weighted component sum")
    columns = ["invariance_mse", "sigreg", "loss", "weighted_invariance", "weighted_sigreg"]
    mean = ssl.groupby("epoch", as_index=False)[columns].mean()
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    mean.to_csv(figures / "ssl_loss_mean.csv", index=False, encoding="utf-8-sig")
    names = {"invariance_mse": "Invariance MSE", "sigreg": "SIGReg loss", "loss": "Combined LeJEPA loss"}
    colors = {"invariance_mse": "#16857e", "sigreg": "#6654bd", "loss": "#c56039"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, key in zip(axes, names):
        for _, fold in ssl.groupby("fold"):
            ax.plot(fold.epoch, fold[key], color=colors[key], alpha=0.18, linewidth=1)
        ax.plot(mean.epoch, mean[key], color=colors[key], linewidth=2.5)
        ax.set(title=names[key], xlabel="SSL epoch", ylabel="Loss")
        ax.grid(alpha=0.2)
    fig.suptitle(f"E1D SWRI | lambda={weight:g}, mask={config['mask_ratio']:g} | 5 folds")
    fig.savefig(figures / "ssl_loss_components.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for key, label, color in (("weighted_invariance", "(1-lambda) x invariance MSE", "#16857e"),
                              ("weighted_sigreg", "lambda x SIGReg", "#6654bd"),
                              ("loss", "Combined LeJEPA loss", "#c56039")):
        ax.plot(mean.epoch, mean[key], label=label, color=color, linewidth=2)
    ax.set(xlabel="SSL epoch", ylabel="Weighted loss", title="E1D weighted objective")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(figures / "ssl_loss_weighted.png", dpi=180)
    plt.close(fig)
    metrics = pd.read_csv(run_dir / "SWRI_Diesel" / "metrics.csv")
    # PLS has a final fit only; plot scale-free final errors rather than an epoch curve.
    grouped = metrics.groupby("target", sort=True)[["train_r2", "r2"]].mean()
    positions = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(positions - 0.18, 1 - grouped.train_r2, width=0.36, label="Train: 1 - R虏", color="#16857e")
    ax.bar(positions + 0.18, 1 - grouped.r2, width=0.36, label="Test: 1 - R虏", color="#c56039")
    ax.set(xticks=positions, xticklabels=grouped.index, ylabel="Relative squared error (1 - R虏)",
           title="PLS final regression error | mean across 5 folds")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(figures / "pls_final_regression_error.png", dpi=180)
    plt.close(fig)
    print(figures)


if __name__ == "__main__":
    main()

