# 模型目录

模型按方法和职责拆分。新增方法时建立独立子目录，不再把实现文件堆在 `models/` 根层。

```text
models/
├── shared/                 # 多种方法共享的组件
│   ├── backbone.py         # 一维光谱 Patch Transformer
│   ├── masking.py          # block mask 生成
│   └── regressors.py       # 冻结表征后的 Ridge
├── baselines/
│   └── classical.py        # Raw/PCA/PLS/SVR/Random Forest 基线
├── mae/
│   └── model.py            # masked reconstruction 方法
├── lejepa/
│   ├── model.py            # 多视图潜空间对齐
│   └── sigreg.py           # SIGReg 正则项
└── __init__.py             # 对外导出常用模型
```

模型层只接收已经统一好的张量，不读取 CSV，也不决定数据单位、波数网格或数据划分。
