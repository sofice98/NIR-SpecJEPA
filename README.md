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
├── scripts/                     # 训练、验证、绘图与实验入口
│   ├── run_e1a.py               # 预处理基线 + 回归头
│   ├── run_e1b.py               # LeJEPA + 在线探针
│   └── run_e1c.py               # LeJEPA 预训练 + 冻结 Encoder Probe
├── artifacts/
│   ├── canonical/<dataset>/     # 规范化数据和固定划分
│   └── runs/<run-name>/         # 日志、权重、预测、指标和图
├── tests/
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

| 参数 | 可取值/范围 | 默认值 |
|---|---|---|
| `--dataset-root` | 任意存在且符合数据集 adapter 约定的目录 | `data/crude_oil_private` |
| `--output-dir` | 可创建或已存在的输出目录 | `artifacts/canonical/crude_oil_private` |

`prepare_data.py` 会生成公共波长网格、规范化光谱、目标值、样本清单以及 `pooled`/`logo` 划分文件。

## 实验

以下命令在项目根目录执行。三个实验均支持 `--datasets swri crude_oil lucas`（可选一个或多个数据集），默认采用 5 折交叉验证。

### E1A：光谱预处理与回归基线

比较原始光谱（`raw`）、一阶导（`fd`）、二阶导（`sd`）、SNV（`snv`）、MSC（`msc`）和 airPLS（`airpls`）与 PLS、随机森林（`rf`）、MLP 回归器的组合。预处理及后续标准化在每个训练折内拟合。

```shell
python scripts/run_e1a.py --datasets swri --experiments raw fd sd snv msc airpls --regressors pls rf mlp
```

### E1B：LeJEPA 与在线 Probe

直接使用原始光谱，以单通道 Patch Transformer 和随机完整 patch token 掩码训练 LeJEPA。每个目标独立训练一套 LeJEPA 与单目标在线 Probe，训练前排除该目标缺失标签的样本。Probe 在同一训练阶段读取两种视图中与编码器梯度断开的表示，标签损失不更新编码器。

```shell
python scripts/run_e1b.py --datasets swri --epochs 100 --batch-size 16 --patch-size 6 --mask-ratio 0.4
```

### E1C：LeJEPA 预训练与冻结 Probe

同样使用原始光谱、单通道 Patch Transformer 和完整 patch token 掩码。先完成 LeJEPA 自监督训练，再冻结编码器，针对每个目标分别训练 Probe；`--probe` 可选 `linear`、`ridge` 或 `mlp`。支持用 `--ssl-checkpoint` 复用已训练的自监督权重。

```shell
python scripts/run_e1c.py --datasets swri --ssl-epochs 100 --probe-epochs 100 --probe linear --patch-size 6 --mask-ratio 0.4
```

### E1D：固定基线单变量扫描

E1D 主入口是 `scripts/run_e1d.py`。批量实验驱动位于 `runs/e1d/sweep.py`，固定基线为 mask ratio=0.4、SIGReg weight=0.02、SSL epochs=100、batch size=32、PLS 回归头、patch size=6、encoder dim=96。扫描依次只改变一个因素：掩码比例（0.2、0.4、0.6、0.8）、SIGReg weight（0.01、0.02、0.05、0.10、0.20、0.50、0.75）、SSL epoch（100、200）、batch size（16、32、64、128）、回归头（PLS、随机森林、MLP）、patch size（4、6、8、10、16）或 encoder dim（48、96、128、196）。基准值只运行一次，共 23 个配置。每个配置独立写入 `artifacts/runs/e1d_ota/<config>/`，根目录生成 `metrics.csv` 与 `metrics_aggregate.csv`。

```shell
python runs/e1d/sweep.py --folds 5 --probe-epochs 100
```

完整扫描结束后运行 `python runs/e1d/analyze_sweep.py --input-dir artifacts/runs/e1d_ota`，生成按目标及因素比较的图表和中文实验报告。本次 2026-10-16 完成的 SWRI 扫描报告位于 `docs/e1d_ota_20261015/2026-10-16--e1d-swri--r00--one-factor-ablation-report.md`，原始逐折结果位于 `artifacts/runs/e1d_ota_20261015/`。

历史的多因素筛选脚本和绘图脚本保存在 `docs/e1d_swri_sweep_20261015/`，不作为当前 E1D 单变量协议的一部分。

E1B 与 E1C 当前的默认 SIGReg 权重均为 `0.02`。两者在控制台输出每个数据集、每折的训练与测试样本数，以及各目标的有效标签数和指标。
E1B 的 `metrics.csv` 同时记录每折训练集的 `train_rmse`、`train_r2`、`train_rpd`、`n_train` 和测试集指标；`metrics_aggregate.csv` 汇总跨折均值。
E1B 和 E1C 的首折输出 `first_fold_patch_masks.png` 及 `first_fold_patch_masks.csv`，展示最多 3 条训练光谱及对应的 patch 掩码区间。


## 实验输出

每次运行创建独立目录：

```text
artifacts/runs/{e1a,e1b,e1c}_{YYYYMMDD_HHMMSS}/
├── run.log
├── config.json
└── <dataset>/
    ├── metrics.csv
    ├── metrics_aggregate.csv
    ├── run_summary.csv
    ├── loss_curve.csv          # E1B、E1C
    ├── ssl_runs.csv            # E1B、E1C
    └── representations/        # E1B、E1C 的逐折权重和表示
```
