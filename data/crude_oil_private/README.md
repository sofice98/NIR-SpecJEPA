# crude_oil_private 数据集

本目录是一个完整的数据集任务单元，包含原始 CSV、数据集专属 adapter 和说明：

- `原油*_光谱.csv`：光谱矩阵，第一行是波数（cm⁻¹），第一列是样品 ID。
- `原油*_属性.csv`：密度、硫含量和残炭等目标属性。
- `adapter.py`：文件名、分组、目标列和单位均属于本数据集；读取后将波数（cm⁻¹）转换为波长（nm），再交给通用合并逻辑插值到公共波长网格。

新增数据集时复制本目录，替换 CSV 和 `adapter.py`，并在
`configs/datasets/` 添加元数据。通用重采样、标准化和划分逻辑分别位于
`preprocessing/` 与 `splits/`。
