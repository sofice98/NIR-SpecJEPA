# LUCAS Soil 数据格式与适配器

`LUCAS.SOIL_corr.csv` 是按样本排列的宽表：

- 共 19,036 行样本、4,288 列；
- `sample.ID` 是样本标识，`country` 用作交叉验证分组；
- `spc.400` 到 `spc.2499.5` 是 4,200 个反射率通道，单位为 nm，间隔 0.5 nm；
- 其余列是采样时间、地理位置、土地覆盖和土壤属性；
- 适配器将 `coarse`、`clay`、`silt`、`sand`、pH、`OC`、`CaCO3`、`N`、`P`、`K`、`CEC` 作为数值 targets。缺失实验室测量保留为 `NaN`。

使用：

```python
from data.lucas.adapter import load_dataset

dataset = load_dataset("data/lucas")
print(dataset.spectra.shape)       # (19036, 4200)
print(dataset.wavelengths_nm[:3])  # [400.0, 400.5, 401.0]
print(dataset.targets.columns.tolist())
```

`load_dataset(..., nrows=10)` 可用于快速冒烟测试。
