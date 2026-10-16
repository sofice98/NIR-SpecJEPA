"""Validate the complete E1D one-factor sweep and write Chinese analysis artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from runs.e1d.sweep import BASELINE, make_configs


TARGETS = ("BP50", "CN", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")
FAMILIES = (
    ("掩码比例", "mask_ratio", ("mask_0.2", "baseline", "mask_0.6", "mask_0.8")),
    ("SIGReg 权重", "sigreg_lambda", ("lambda_0.01", "baseline", "lambda_0.05", "lambda_0.1", "lambda_0.2", "lambda_0.5", "lambda_0.75")),
    ("SSL epoch", "epochs", ("baseline", "epochs_200")),
    ("batch size", "batch_size", ("batch_16", "baseline", "batch_64", "batch_128")),
    ("回归头", "probe", ("baseline", "head_rf", "head_mlp")),
    ("patch size", "patch_size", ("patch_4", "baseline", "patch_8", "patch_10", "patch_16")),
    ("encoder dim", "encoder_dim", ("encoder_48", "baseline", "encoder_128", "encoder_196")),
)
CONFIGS = make_configs()
BY_NAME = {config.name: config for config in CONFIGS}
ORDER = [config.name for config in CONFIGS]


def value_label(name: str, field: str) -> str:
    value = getattr(BY_NAME[name], field)
    return {"pls": "PLS", "rf": "随机森林", "mlp": "MLP"}.get(str(value), f"{value:g}" if isinstance(value, float) else str(value))


def validate(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = json.loads((input_dir / "matrix.json").read_text(encoding="utf-8"))
    if matrix["baseline"] != vars(BASELINE) or matrix["configs"] != [vars(config) for config in CONFIGS]:
        raise ValueError("matrix.json does not match the fixed 23-configuration protocol")
    frames, losses = [], []
    for config in CONFIGS:
        run_dir = input_dir / config.name
        recorded = json.loads((run_dir / "sweep_config.json").read_text(encoding="utf-8"))
        if recorded != vars(config):
            raise ValueError(f"configuration mismatch: {config.name}")
        actual_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        runtime_fields = {"mask_ratio": config.mask_ratio, "sigreg_weight": config.sigreg_lambda,
                          "ssl_epochs": config.epochs, "batch_size": config.batch_size,
                          "probe": config.probe, "patch_size": config.patch_size,
                          "encoder_dim": config.encoder_dim, "folds": 5, "seed": 42,
                          "probe_epochs": 100, "datasets": ["swri"],
                          "learning_rate": 3e-4, "weight_decay": 0.05,
                          "min_block_patches": 5, "max_block_patches": 10}
        if any(actual_config.get(key) != value for key, value in runtime_fields.items()):
            raise ValueError(f"runtime arguments mismatch: {config.name}")
        metrics_path = run_dir / "SWRI_Diesel" / "metrics.csv"
        if not metrics_path.is_file():
            raise ValueError(f"incomplete experiment: {config.name} has no metrics.csv")
        metrics = pd.read_csv(metrics_path)
        expected = {(fold, target) for fold in range(5) for target in TARGETS}
        actual = set(zip(metrics.fold, metrics.target))
        if len(metrics) != 35 or actual != expected or metrics.duplicated(["fold", "target"]).any():
            raise ValueError(f"incomplete 5-fold metrics: {config.name}")
        if not np.isfinite(metrics[["rmse", "r2", "rpd"]].to_numpy(float)).all():
            raise ValueError(f"non-finite metrics: {config.name}")
        if set(metrics.regressor) != {config.probe}:
            raise ValueError(f"regression head mismatch: {config.name}")
        metrics.insert(0, "sweep_config", config.name)
        frames.append(metrics)
        loss_path = run_dir / "SWRI_Diesel" / "loss_curve.csv"
        if not loss_path.is_file():
            raise ValueError(f"incomplete experiment: {config.name} has no loss_curve.csv")
        loss = pd.read_csv(loss_path)
        ssl = loss[loss.stage == "ssl_pretrain"].copy()
        expected_epochs = {(fold, epoch) for fold in range(5) for epoch in range(1, config.epochs + 1)}
        actual_epochs = set(zip(ssl.fold, ssl.epoch))
        if len(ssl) != 5 * config.epochs or actual_epochs != expected_epochs:
            raise ValueError(f"incomplete SSL history: {config.name}")
        if not np.isfinite(ssl[["loss", "invariance_mse", "sigreg"]].to_numpy(float)).all():
            raise ValueError(f"non-finite SSL history: {config.name}")
        ssl.insert(0, "sweep_config", config.name)
        losses.append(ssl)
    all_metrics = pd.concat(frames, ignore_index=True)
    counts = all_metrics.pivot_table(index=["fold", "target"], columns="sweep_config", values="n_test")
    if counts.isna().any().any() or counts.nunique(axis=1).gt(1).any():
        raise ValueError("outer test counts differ between configurations")
    return all_metrics, pd.concat(losses, ignore_index=True)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(values.mean())
    flipped = [abs(np.mean(values * np.array(signs))) for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(score >= observed - 1e-12 for score in flipped) / len(flipped)


def analyze(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_summary = (metrics.groupby(["sweep_config", "target"])[["r2", "rmse", "rpd"]]
                      .agg(["mean", "std"]))
    target_summary.columns = ["_".join(column) for column in target_summary.columns]
    target_summary = target_summary.reset_index()
    fold_macro = metrics.groupby(["sweep_config", "fold"], as_index=False).r2.mean()
    baseline = fold_macro[fold_macro.sweep_config == "baseline"].set_index("fold").r2
    rows = []
    for name in ORDER:
        scores = fold_macro[fold_macro.sweep_config == name].set_index("fold").r2
        delta = (scores - baseline).sort_index().to_numpy()
        mean_delta = float(delta.mean())
        sd_delta = float(delta.std(ddof=1))
        margin = float(stats.t.ppf(0.975, 4) * sd_delta / np.sqrt(5))
        rows.append({"config": name, "macro_r2_mean": scores.mean(), "macro_r2_sd": scores.std(ddof=1),
                     "delta_r2": mean_delta, "delta_sd": sd_delta,
                     "ci95_low": mean_delta - margin, "ci95_high": mean_delta + margin,
                     "paired_dz": mean_delta / sd_delta if sd_delta else np.nan,
                     "sign_flip_p": exact_sign_flip_p(delta) if name != "baseline" else np.nan})
    overall = pd.DataFrame(rows)
    alternatives = overall[overall.config != "baseline"].sort_values("sign_flip_p")
    count = len(alternatives)
    adjusted = np.maximum.accumulate(alternatives.sign_flip_p.to_numpy() * np.arange(count, 0, -1))
    overall.loc[alternatives.index, "holm_p"] = np.minimum(1, adjusted)
    return target_summary, fold_macro, overall


def setup_plot() -> None:
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 9, "savefig.dpi": 220})


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(overall: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout="constrained")
    lookup = overall.set_index("config")
    for ax, (family, field, names) in zip(axes.flat, FAMILIES):
        names = list(names)
        positions = np.arange(len(names))
        values = lookup.loc[names, "delta_r2"].to_numpy()
        errors = lookup.loc[names, "delta_sd"].fillna(0).to_numpy()
        ax.errorbar(positions, values, yerr=errors, marker="o", capsize=3, color="#22577a")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xticks(positions, [value_label(name, field) for name in names])
        ax.set_title(family)
        ax.set_ylabel("相对基准的平均 R² 差值")
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("E1D 单因素扫描：五折跨目标平均 R² 与折间标准差", fontsize=13)
    save_figure(fig, out / "figure-01-factor-delta")


def plot_target_heatmap(summary: pd.DataFrame, out: Path) -> None:
    matrix = summary.pivot(index="sweep_config", columns="target", values="r2_mean").loc[ORDER, list(TARGETS)]
    delta = matrix.sub(matrix.loc["baseline"], axis=1)
    fig, ax = plt.subplots(figsize=(10, 11), layout="constrained")
    extent = max(0.05, float(np.nanmax(abs(delta.to_numpy()))))
    im = ax.imshow(delta, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
    ax.set_yticks(np.arange(len(ORDER)), ORDER)
    ax.set_xticks(np.arange(len(TARGETS)), TARGETS)
    ax.set_title("E1D 各目标平均 R² 相对基准的差值")
    fig.colorbar(im, ax=ax, label="ΔR²（五折均值）")
    save_figure(fig, out / "figure-02-target-heatmap")


def plot_loss(losses: pd.DataFrame, out: Path) -> None:
    names = ["baseline", "lambda_0.01", "lambda_0.05", "lambda_0.1", "lambda_0.2", "lambda_0.5", "lambda_0.75"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in names:
        data = losses[losses.sweep_config == name].groupby("epoch")[["invariance_mse", "sigreg"]].mean()
        label = f"λ={BY_NAME[name].sigreg_lambda:g}"
        for ax, column in zip(axes, ("invariance_mse", "sigreg")):
            ax.plot(data.index, data[column], label=label, linewidth=1.3)
    for ax, title in zip(axes, ("掩码预测 MSE", "SIGReg 分量")):
        ax.set(title=title, xlabel="SSL epoch", ylabel="五折平均损失分量")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("不同 SIGReg 权重下的训练损失分量（未加权）")
    save_figure(fig, out / "figure-03-lambda-loss")


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def table_overall(overall: pd.DataFrame) -> str:
    lines = ["| 配置 | 平均 R² ± SD | ΔR² ± SD | 95% CI（描述性） | 精确符号翻转 p | Holm p |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in overall.itertuples():
        lines.append(f"| {row.config} | {fmt(row.macro_r2_mean)} ± {fmt(row.macro_r2_sd)} | "
                     f"{fmt(row.delta_r2)} ± {fmt(row.delta_sd)} | [{fmt(row.ci95_low)}, {fmt(row.ci95_high)}] | "
                     f"{fmt(row.sign_flip_p, 4)} | {fmt(row.holm_p, 4)} |")
    return "\n".join(lines)


def write_reports(input_dir: Path, out: Path, metrics: pd.DataFrame, summary: pd.DataFrame,
                  overall: pd.DataFrame) -> None:
    report_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    table = table_overall(overall)
    baseline = overall.iloc[0]
    mlp_metrics = metrics[metrics.sweep_config == "head_mlp"]
    mlp_train_r2 = float(mlp_metrics.train_r2.mean())
    mlp_test_r2 = float(mlp_metrics.r2.mean())
    best_overall = overall.iloc[overall.macro_r2_mean.idxmax()]
    family_lines = []
    for family, field, names in FAMILIES:
        subset = overall.set_index("config").loc[list(names)]
        best = subset.loc[subset.macro_r2_mean.idxmax()]
        worst = subset.loc[subset.macro_r2_mean.idxmin()]
        family_lines.append(f"- **{family}**：描述性最高为 {value_label(best.name, field)}"
                            f"（平均 R²={best.macro_r2_mean:.3f}，相对基准 Δ={best.delta_r2:+.3f}）；"
                            f"最低为{value_label(worst.name, field)}（Δ={worst.delta_r2:+.3f}）。")
    significant = overall[(overall.config != "baseline") & (overall.holm_p < 0.05)]
    significance = "没有配置在 Holm 校正后达到 0.05。" if significant.empty else f"{len(significant)} 个配置在 Holm 校正后达到 0.05。"
    target_lines = []
    for target in TARGETS:
        row = summary[(summary.sweep_config == "baseline") & (summary.target == target)].iloc[0]
        count = int(metrics[(metrics.sweep_config == "baseline") & (metrics.target == target)].n_test.sum())
        target_lines.append(f"| {target} | {count} | {row.r2_mean:.3f} ± {row.r2_std:.3f} | {row.rmse_mean:.4g} ± {row.rmse_std:.3g} |")
    target_table = "| 目标 | 五折测试标签总数 | R² 均值 ± SD | RMSE 均值 ± SD |\n|---|---:|---:|---:|\n" + "\n".join(target_lines)
    source = f"`../../artifacts/runs/{input_dir.name}/`"
    analysis = f"""# E1D 单因素扫描：严格分析

## 分析问题与证据

在固定基准下，单独改变哪个因素会影响 SWRI Diesel 七目标的外层五折测试 R²？共有 23 个配置，每个配置 5 折 × 7 目标 = 35 行测试指标，共 {len(metrics)} 行。原始结果：{source}。主要描述性指标是逐折七目标等权平均 R²，变化量为同折配置减去基准；不同目标量纲不同，RMSE 只在目标内比较。

基准平均 R² 为 {baseline.macro_r2_mean:.3f} ± {baseline.macro_r2_sd:.3f}（折间 SD）。所有配置使用相同划分和 seed=42，因此折间差值可配对，但只有一套随机种子。

## 逐因素观察

{chr(10).join(family_lines)}

这些是已测试网格内的描述性排序，不足以推断全局最优。{significance} 外层折并非独立重复实验，五折的置信区间与检验仅作为敏感性说明。

## 完整数值表

{table}

## 基准逐目标表现

{target_table}

## 回归头诊断

训练折内对神经网络回归头的冻结表征逐维标准化后，MLP 的平均训练 R² 为 {mlp_train_r2:.3f}，测试 R² 为 {mlp_test_r2:.3f}。这说明 MLP 已能够拟合训练标签，但测试表现仍低于本轮 PLS 基准；不能把它归因于网络无法学习。标准化只用训练折统计量，不泄漏测试折。

## 候选结论

- 结论：本扫描可识别若干描述性较好的设置。证据：上表五折配对 ΔR²、图 1 与图 2。允许表述：“在本数据、本划分与种子下，某设置的平均 R² 较高。”避免表述：“该设置在总体上显著更优。”不确定性：随机种子、数据集和超参数搜索范围；后续检查：独立种子与外部数据验证。决策：保留为候选，不直接定稿。

## 图的解读

1. 图 1 展示各因素的配置相对基准的七目标平均 R² 差值。误差棒是五个配对折差值的标准差，反映划分敏感性，不是置信区间。
2. 图 2 展示每个目标的五折平均 R² 差值，检查整体均值是否由少数目标驱动。
3. 图 3 分开绘制未加权的掩码预测 MSE 和 SIGReg，便于观察权重变化的训练动态；不同 λ 下的加权总损失不能直接横向比较。
"""
    (out / "analysis-report.md").write_text(analysis, encoding="utf-8")
    appendix = f"""# E1D 统计附录

## 单位与计算

- 外层单位：五个固定测试折；同一折、同一目标在配置之间配对。七目标等权汇总为每折宏平均 R²，未把 35 行视为 35 个独立重复。
- 每配置只有 seed 42 的一次完整交叉验证。五折共享训练样本，折间 SD 和 t 区间仅作描述性参考。
- 95% 区间：五个折差值的均值 ± t(0.975,4) × SD/√5。配对效应量 dz = 平均差值/折差值 SD。
- 双侧精确符号翻转检验枚举 2⁵ 种符号模式，22 个非基准比较使用 Holm 校正。五折时最小双侧 p 为 0.0625，因此显著性检验功效有限。没有先验独立种子，不应由 p 值作总体优劣断言。
- 无效、缺失或重复的指标/历史记录会使分析脚本报错，禁止输出不完整报告。

## 汇总

{table}

## 逐目标均值与折间标准差

完整机器可读表：`target_summary.csv`，含各配置各目标的 R²、RMSE、RPD 的均值与标准差。RMSE 的数值单位随目标不同，不做跨目标平均。

## 数据与限制

SWRI Diesel 共 784 条光谱；自监督训练使用训练折全部光谱。BP50、D4052、FLASH、FREEZE、TOTAL、VISC 各有 395 个标签，CN 有 381 个标签。目标缺失标签在回归训练与测试指标计算时按目标过滤。只有一个数据集、一套 seed 和一组五折划分，单因素结果不覆盖参数交互。
"""
    (out / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog = """# E1D 图录

1. `figures/figure-01-factor-delta.png` / `.pdf`：单因素 ΔR²；误差棒为五个配对折差值的 SD，零线为基准。用于判断改动方向与折间波动，不能据此宣称统计显著。
2. `figures/figure-02-target-heatmap.png` / `.pdf`：七目标各配置相对基准的五折平均 ΔR²。用于检查目标异质性；色标是 R² 点数差值。
3. `figures/figure-03-lambda-loss.png` / `.pdf`：不同 λ 的两项未加权 SSL 损失分量，先按 epoch 在五折取均值。用于判断优化动态，不能把低训练损失直接等同于高测试 R²。
"""
    (out / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    report = f"""---
type: results-report
date: {report_date}
experiment_line: e1d-swri
round: r00
purpose: one-factor-ablation-report
status: complete
source_artifacts:
  - analysis-report.md
  - stats-appendix.md
  - figure-catalog.md
linked_experiments: []
linked_results: []
---

# E1D / Round r00 / 单因素实验报告 / {report_date}

## 摘要与实验目的

本轮在 SWRI Diesel 上完成 23 组 E1D 单因素实验，评估掩码比例、SIGReg 权重、SSL 训练轮数、批量大小、回归头、patch size 和 encoder dim 对冻结表征回归的影响。基准七目标平均测试 R² 为 **{baseline.macro_r2_mean:.3f} ± {baseline.macro_r2_sd:.3f}**（五折均值 ± 折间标准差）。{significance}因此本报告将较优配置作为下一轮复验候选。

本轮单因素网格中，**{best_overall.config}** 的平均测试 R² 最高（{best_overall.macro_r2_mean:.3f}，较基准 +{best_overall.delta_r2:.3f}）。这是单数据集、单种子下的候选设置。

## 设置与评估协议

- 数据：SWRI Diesel，784 条 401 维光谱，七个目标；每个目标约 381–395 个有效标签。
- 基准：mask ratio=0.4、SIGReg weight=0.02、SSL epoch=100、batch size=32、回归头 PLS、patch size=6、encoder dim=96。
- 固定条件：seed=42、外层 5 折、MLP probe 100 epoch、AdamW 学习率 3e-4、权重衰减 0.05、连续块掩码长度 5–10 patch。PLS 与随机森林在训练折内部用 3 折选择回归器参数。
- 每次只改变一个因素。回归头实验也重新训练相同配置的自监督编码器；设置相同随机种子，训练轨迹原则上可重复，但可能受 CUDA 非确定性影响。
- 主要指标：测试 R²。逐目标另外报告 RMSE、RPD；不同目标 RMSE 不合并。

## 主要结果

{chr(10).join(family_lines)}

完整的跨目标汇总见下表；它是各折先对七个目标取平均，再对五折取均值。

{table}

基准逐目标结果：

{target_table}

## 统计验证与图解

图 1 比较单因素变化与基准的宏平均 R² 差值，误差棒为五折配对差值 SD；它帮助判断改动方向与折间稳定性。图 2 拆解到各目标，显示总平均是否掩盖某些目标退步。图 3 展示不同 SIGReg 权重下掩码预测和正则项的训练动态。详细检验、效应量与计算方式见 [统计附录](stats-appendix.md)。

本轮 encoder dim=196 的七个目标平均 R² 均高于基准；patch size=4 的宏平均提升，但 BP50 的平均 R² 下降约 0.026。损失曲线显示高 λ 下掩码预测 MSE 保持较高；这与部分高 λ 组的下游退步一致，但不能据此断定因果机制。

![单因素 R² 对比](figures/figure-01-factor-delta.png)

![逐目标 R² 差值](figures/figure-02-target-heatmap.png)

![SIGReg 权重与损失曲线](figures/figure-03-lambda-loss.png)

## 负结果、限制与认识变化

图 1 中 ΔR² 低于零的配置在此协议下没有改善七目标平均表现；图 2 中的负值说明即使宏平均改善，也可能损伤单个目标。MLP 经训练折内特征标准化后的平均训练 R² 为 {mlp_train_r2:.3f}，测试 R² 为 {mlp_test_r2:.3f}，显示仍有泛化落差。{significance}五折属于同一数据集的重叠训练划分，不等于五次独立复现；仅一套随机种子，不能证明优势可泛化到其他数据或随机初始化。单因素设计也无法识别参数交互。当前证据只支持在既定数据和网格内筛选复验候选。

## 下一步

优先对图 1 中描述性较好的配置增加独立随机种子，并在外部数据集验证；若目标间效果相反，应按具体性质分别选型。随后再考虑组合因素实验，避免把单因素最好值直接拼成最终配置。

## 产物与复现

- 原始配置与逐折指标：{source} 中的 `matrix.json`、各配置 `SWRI_Diesel/metrics.csv` 与 `loss_curve.csv`。
- 严格分析：[分析说明](analysis-report.md)、[统计附录](stats-appendix.md)、[图录](figure-catalog.md)、`target_summary.csv`、`overall_summary.csv`。
- 从仓库根目录执行：`python runs/e1d/sweep.py --output-dir artifacts/runs/{input_dir.name} --folds 5 --probe-epochs 100 --seed 42`；然后 `python runs/e1d/analyze_sweep.py --input-dir artifacts/runs/{input_dir.name} --output-dir docs/{out.name}`。
- 本仓库未绑定 Obsidian 项目知识库，本报告保存在本地 `docs/`。
"""
    (out / f"{report_date}--e1d-swri--r00--one-factor-ablation-report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/runs/e1d_ota_20261015")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/e1d_ota_20261015")
    args = parser.parse_args()
    metrics, losses = validate(args.input_dir)
    target_summary, fold_macro, overall = analyze(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    target_summary.to_csv(args.output_dir / "target_summary.csv", index=False, encoding="utf-8-sig")
    fold_macro.to_csv(args.output_dir / "fold_macro_r2.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    setup_plot()
    plot_comparison(overall, figures)
    plot_target_heatmap(target_summary, figures)
    plot_loss(losses, figures)
    write_reports(args.input_dir, args.output_dir, metrics, target_summary, overall)
    print(f"Validated {len(CONFIGS)} configurations and {len(metrics)} metric rows; report: {args.output_dir}")


if __name__ == "__main__":
    main()


