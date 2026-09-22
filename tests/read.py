from pathlib import Path

import matplotlib

# This script writes a figure and should not create Tk resources.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


data_file = (
    Path(__file__).resolve().parent.parent
    / "两套原油数据集-20260910"
    / "ECUST-原油-第二套"
    / "原油光谱数据.xlsx"
)

try:
    # .xlsx 是二进制 Excel 文件，不能使用 read_csv。
    df = pd.read_excel(data_file, index_col=0, engine="openpyxl")
except ImportError as exc:
    raise SystemExit(
        "读取 .xlsx 文件需要 openpyxl，请先运行：python -m pip install openpyxl"
    ) from exc

# 提取波数（列名转为数值）。
wavenumbers = pd.to_numeric(df.columns)

# 转置：行=波数，列=样本，方便绘图。
spectra = df.T

plt.figure(figsize=(12, 5))
plt.plot(wavenumbers, spectra.values, linewidth=0.8)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Spectral Intensity (a.u.)")
plt.title("Crude Oil Infrared Spectra")
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("crude_oil_spectra.png", dpi=150, bbox_inches="tight")
plt.close()
