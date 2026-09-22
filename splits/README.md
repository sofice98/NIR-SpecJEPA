# Splits layer

- `pooled_group_balanced_folds`：合并数据后的 5 折，保持 oil1/oil2 在每折中近似平衡；
- `leave_one_group_out_folds`：留一组域外评估，单独报告跨域表现。

划分清单会保存到 `artifacts/canonical/*.json`，训练和评估脚本必须读取清单，不得重新随机划分。
