# 石油红外光谱实验代码

本项目用于光谱表征学习、传统回归基线、消融实验和跨域评估。代码按数据集、共享处理、模型方法、实验配置和实验产物分层，便于切换数据、模型、组件与超参数。

## 目录结构

```text
.
├── data/
│   ├── crude_oil_private/       # 原始 CSV、数据集 adapter 和说明
│   └── dataset_adapters.py      # CanonicalDataset 与通用校验
├── preprocessing/               # 重采样、标准化和光谱变换
├── splits/                      # pooled CV、leave-one-group-out
├── models/
│   ├── shared/                  # backbone、mask、回归器
│   ├── baselines/               # 传统方法
│   ├── mae/                     # MAE
│   └── lejepa/                  # LeJEPA 与 SIGReg
├── configs/                     # datasets/、experiments/、components/
├── scripts/                     # prepare_data.py、run_e1a.py
├── artifacts/
│   ├── canonical/<dataset>/     # 规范化数据和固定划分
│   └── runs/<run-name>/         # 日志、权重、预测、指标和图
├── tests/
```

## 所有运行参数及可取值

以下参数对应 `scripts/run_e1a.py` 的命令行接口。除特别注明外，数值参数由
`argparse` 解析；“程序强制范围”表示脚本会主动检查并在不满足时直接报错。

### 数据、评估、预处理与运行

| 参数 | 可取值/范围 | 默认值 |
|---|---|---:|
| `--canonical-dir` | 任意存在的规范化数据目录；需包含 `spectra_resampled.npy`、`targets.csv`、`samples.csv` 和划分文件 | `artifacts/canonical/crude_oil_private` |
| `--protocol` | `pooled`、`logo` | `pooled` |
| `--methods` | `raw_ridge`、`pca_ridge`、`pls`、`rbf_svr`、`random_forest`、`random_encoder`、`mae`、`lejepa` 的一个或多个；不可重复 | 全部方法 |
| `--preprocessing` | `R`、`FD`、`SD`、`SNV`、`MSC`、`SG` | `R` |
| `--sg-window` | 正奇数，且大于 `--sg-polyorder` | `11` |
| `--sg-polyorder` | 非负整数，且小于 `--sg-window` | `2` |
| `--epochs` | 正整数 | `200` |
| `--batch-size` | 正整数 | `16` |
| `--device` | PyTorch 设备字符串，如 `cpu`、`cuda`、`cuda:0` | 自动选择 |
| `--seed` | 任意整数 | `20260920` |
| `--max-folds` | `None` 或正整数 | `None` |

`R`、`FD`、`SD`、`SNV`、`MSC`、`SG` 分别表示原始光谱、一阶导数、二阶导数、标准正态变量变换、多元散射校正和 Savitzky–Golay 平滑。

### 优化器与学习率

| 参数 | 可取值/范围 | 默认值 |
|---|---|---:|
| `--learning-rate` | 正浮点数 | `1e-4` |
| `--min-learning-rate` | 非负浮点数 | `1e-6` |
| `--warmup-ratio` | `[0, 1)` | `0.05` |
| `--warmup-start-factor` | `(0, 1]` | `0.01` |
| `--weight-decay` | 非负浮点数 | `1e-4` |
| `--adam-beta1`、`--adam-beta2` | `[0, 1)` | `0.9`、`0.999` |
| `--adam-eps` | 正浮点数 | `1e-8` |
| `--gradient-clip-norm` | 非负浮点数；`0` 表示关闭 | `0.0` |

### 编码器结构

| 参数 | 可取值/范围 | 默认值 |
|---|---|---:|
| `--encoder-type` | `transformer`、`unet` | `transformer` |
| `--patch-size` | 正整数；应不大于光谱长度 | `6` |
| `--embedding-dim` | 正整数，且必须被 `--heads` 整除 | `64` |
| `--depth` | 正整数 | `2` |
| `--heads` | 正整数，且必须整除 `--embedding-dim` | `2` |
| `--ffn-dim` | 正整数 | `128` |
| `--dropout` | `[0, 1)` | `0.0` |

### 掩码与自监督方法

| 参数 | 可取值/范围 | 默认值 |
|---|---|---:|
| `--mask-ratio` | **程序强制：`(0, 1)`** | `0.40` |
| `--mask-mode` | `point`、`block` | `point` |
| `--mask-block-length` | **程序强制：正整数** | `4` |
| `--views` | **程序强制：至少为 `2`** | `2` |
| `--mae-decoder-hidden-dim` | 正整数 | `64` |
| `--projection-dim` | 正整数 | `16` |
| `--projector-hidden-dim` | 正整数 | `128` |
| `--sigreg-weight` | 非负浮点数 | `0.02` |
| `--sigreg-knots` | 正整数 | `17` |
| `--sigreg-projections` | 正整数 | `256` |

### 传统回归基线与输出

| 参数 | 可取值/范围 | 默认值 |
|---|---|---:|
| `--ridge-alphas` | 一个或多个正浮点数 | `1e-4 1e-3 1e-2 1e-1 1 10 100` |
| `--ridge-inner-splits` | **程序强制：至少为 `2`** | `4` |
| `--pca-components` | 一个或多个正整数；不应超过训练样本数或特征数 | `4 8 16 32` |
| `--pls-components` | 一个或多个正整数；不应超过算法允许的最大潜变量数 | `2 4 8 12` |
| `--pls-max-iter` | 正整数 | `500` |
| `--pls-tolerance` | 正浮点数 | `1e-6` |
| `--svr-c-values` | 一个或多个正浮点数 | `0.1 1 10 100` |
| `--svr-epsilon-values` | 一个或多个非负浮点数 | `0.01 0.1 0.2` |
| `--svr-gamma-values` | 一个或多个正浮点数，或 `scale`、`auto` | `scale auto` |
| `--rf-estimators` | 一个或多个正整数 | `200 500` |
| `--rf-max-depths` | 一个或多个正整数，或 `None` | `None 8 16` |
| `--rf-min-samples-leaf` | 一个或多个正整数 | `1 2 4` |
| `--rf-max-features` | 正浮点数，或 `sqrt` | `1.0 sqrt` |
| `--rf-jobs` | 任意整数；`-1` 表示使用全部 CPU 线程 | `-1` |
| `--embedding-batch-size` | 正整数 | `256` |
| `--plot-dpi` | 正整数 | `180` |

### `prepare_data.py` 参数

| 参数 | 可取值/范围 | 默认值 |
|---|---|---|
| `--dataset-root` | 任意存在且符合数据集 adapter 约定的目录 | `data/crude_oil_private` |
| `--output-dir` | 可创建或已存在的输出目录 | `artifacts/canonical/crude_oil_private` |

`prepare_data.py` 会生成公共波长网格、规范化光谱、目标值、样本清单以及 `pooled`/`logo` 划分文件。

示例：

```shell
python scripts/run_e1a.py ^
  --preprocessing SNV ^
  --encoder-type unet ^
  --mask-mode block ^
  --mask-block-length 4 ^
  --methods pls rbf_svr ^
  --epochs 200 ^
  --batch-size 16
```

## 环境安装

推荐使用 Conda 环境：


## 数据集约定

`data/` 下每个文件夹是一个完整的数据集任务单元，包含原始文件、`adapter.py` 和说明文档。`data/dataset_adapters.py` 只包含与具体数据集无关的规范化容器、校验和合并逻辑；文件名、列名、分组、指标、原始光谱轴及单位转换全部放在对应数据集目录的 `adapter.py` 中。规范化后的光谱轴统一为波长 `nm`。

当前 `oil1` 和 `oil2` 的密度都已存储为 `g/cm3`，代码不再执行密度单位换算。

新增数据集时：

1. 复制 `data/crude_oil_private/` 的目录结构。
2. 在数据集自己的 `adapter.py` 中实现 `load_dataset(root) -> CanonicalDataset`，将原始光谱转换为递增的 `wavelengths_nm`，并填充该数据集自己的 targets 和 groups。
3. 在 `configs/datasets/` 添加数据集 YAML。
4. 为该数据集生成独立的 `artifacts/canonical/<dataset>/`。

## 准备数据

```shell
python scripts/prepare_data.py
```

默认输出到 `artifacts/canonical/`，包含样本、波数点、目标值、样品清单、pooled 五折和 leave-one-group-out 划分。

也可指定输入与输出：

```shell
python scripts/prepare_data.py ^
  --dataset-root data/crude_oil_private ^
  --output-dir artifacts/canonical/crude_oil_private
```


## E1-a实验

### 光谱预处理

通过 `--preprocessing` 选择每个外层 fold 内使用的光谱变换：

```shell
python scripts/run_e1a.py --preprocessing R    # 原始光谱（默认）
python scripts/run_e1a.py --preprocessing FD   # 一阶导数
python scripts/run_e1a.py --preprocessing SD   # 二阶导数
python scripts/run_e1a.py --preprocessing SNV  # 标准正态变量变换
python scripts/run_e1a.py --preprocessing MSC  # 多元散射校正
python scripts/run_e1a.py --preprocessing SG   # Savitzky–Golay 平滑
```

`FD`、`SD` 和 `SG` 使用 Savitzky–Golay 参数 `--sg-window`（默认 11）与
`--sg-polyorder`（默认 2）。每个 fold 先完成所选变换，再用训练 fold 拟合中心化/缩放；
MSC 的参考谱也只由训练 fold 估计，并应用于对应测试 fold。

### pooled 五折主实验（原油数据集，U-Net，长度为 1 的点掩码）

```shell
python scripts/run_e1a.py ^
  --canonical-dir artifacts/canonical/crude_oil_private ^
  --protocol pooled ^
  --methods raw_ridge pca_ridge pls rbf_svr random_forest random_encoder mae lejepa ^
  --encoder-type unet ^
  --mask-mode block ^
  --mask-block-length 1 ^
  --mask-ratio 0.40 ^
  --epochs 200 ^
  --batch-size 16 ^
  --learning-rate 1e-4 ^
  --seed 42 ^
  --device cuda
```

上面的命令比较全部方法在原油规范化数据集上的效果。掩码参数只对 `mae` 和
`lejepa` 两种自监督预训练方法生效；传统回归方法和 `random_encoder` 不执行
掩码训练。`--mask-block-length 1`
表示每个掩码块长度为 1 个光谱点；因此这里使用 `--mask-mode block` 是为了
显式表达“长度为 1 的连续块掩码”。对于 `raw_ridge`、`pca_ridge`、`pls`、
`rbf_svr`、`random_forest` 和 `random_encoder`，程序不会进入自监督掩码训练
路径；这些方法只使用未掩码的输入完成回归或随机表征提取。

### leave-one-group-out 跨域实验（同一 U-Net/长度 1 掩码设置）

```shell
python scripts/run_e1a.py ^
  --canonical-dir artifacts/canonical/crude_oil_private ^
  --protocol logo ^
  --methods raw_ridge pca_ridge pls rbf_svr random_forest random_encoder mae lejepa ^
  --encoder-type unet ^
  --mask-mode block ^
  --mask-block-length 1 ^
  --mask-ratio 0.40 ^
  --epochs 200 ^
  --batch-size 16 ^
  --learning-rate 1e-4 ^
  --seed 42 ^
  --device cuda
```

### 只比较表征学习方法

```shell
python scripts/run_e1a.py ^
  --protocol pooled ^
  --methods random_encoder mae lejepa ^
  --epochs 200 ^
  --device cuda
```

### 只运行传统基线

```shell
python scripts/run_e1a.py ^
  --protocol pooled ^
  --methods raw_ridge pca_ridge pls rbf_svr random_forest ^
  --device cpu
```

切换数据集时使用：

```shell
python scripts/run_e1a.py --canonical-dir artifacts/canonical/another_dataset
```

查看全部参数：

```shell
python scripts/run_e1a.py --help
```

## 参数与消融实验

- 数据与协议：`--canonical-dir`、`--protocol`、`--seed`
- 方法：`--methods`
- encoder：`--patch-size`、`--embedding-dim`、`--depth`、`--heads`、`--ffn-dim`
- mask：`--mask-ratio`、`--mask-mode {point,block}`、`--mask-block-length`、`--views`；训练时先将随机掩码的 NIR 点置零，再进行 MAE/LeJEPA 自监督目标计算。
- LeJEPA：`--projection-dim`、`--projector-hidden-dim`、`--sigreg-weight`、`--sigreg-knots`、`--sigreg-projections`
- MAE：`--mae-decoder-hidden-dim`
- 编解码器：`--encoder-type transformer`（默认）或 `--encoder-type unet`。U-Net 配置下，MAE 使用 U-Net 的 encoder + decoder；LeJEPA 只使用 U-Net encoder，predictor 仍为 projector。
- 优化：`--epochs`、`--batch-size`、`--learning-rate`、`--weight-decay`
- 学习率策略：默认使用 AdamW + warmup + cosine annealing；可通过 `--warmup-ratio`、`--warmup-start-factor`、`--min-learning-rate` 调整。
- 基线搜索：Ridge、PCA、PLS、SVR 和 Random Forest 参数

消融实验建议一次只改变一个因素，并将方案记录到 `experiments/ablations/`。跨域方案放在 `experiments/cross_domain/`，传统基线方案放在 `experiments/baselines/`。每次运行的实际参数都会保存到 `run_config.json`。

## 实验输出

每次运行创建独立目录：

```text
artifacts/runs/{protocol}_{YYYYMMDD_HHMMSS}_seed{seed}_ep{epochs}/
├── run.log
├── run_config.json
├── fold*_mae_encoder.pt
├── fold*_lejepa_encoder.pt
├── fold*_*_history.csv
├── fold*_*_loss.png
├── loss_curves_aggregate.png
├── predictions.csv
├── fold_metrics.csv
├── summary_metrics.csv
└── costs.csv
```
