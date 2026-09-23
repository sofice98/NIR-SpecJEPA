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
