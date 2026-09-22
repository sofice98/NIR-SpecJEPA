# E1D: LeJEPA downstream regression comparison

`scripts/run_e1d.py` compares four LeJEPA downstream strategies on one specified canonical dataset:

1. `ridge`: freeze the encoder, extract pooled features, and fit Ridge regression.
2. `xgboost`: freeze the encoder and fit one XGBoost regressor per NIR target.
3. `lightgbm`: freeze the encoder and fit one LightGBM regressor per NIR target.
4. `finetune`: unfreeze the encoder, attach a two-linear-layer MLP, and globally fine-tune with a small encoder learning rate.

Each outer fold runs LeJEPA self-supervised pretraining once. All downstream methods start from the same pretrained encoder state. Ridge, XGBoost, and LightGBM select hyperparameters by inner cross-validation. Fine-tuning selects the epoch count on an inner validation split, then restarts from the pretrained encoder and retrains on the full outer-training fold. The outer test fold never participates in selection.

## Example

```powershell
python scripts/run_e1d.py `
  --canonical-dir artifacts/canonical/crude_oil_private `
  --protocol pooled `
  --methods ridge xgboost lightgbm finetune `
  --encoder-type unet `
  --ssl-epochs 200 `
  --finetune-epochs 200 `
  --finetune-learning-rate 1e-5 `
  --head-learning-rate 1e-4 `
  --device cuda
```

For a fast smoke run:

```powershell
python scripts/run_e1d.py --methods ridge finetune --ssl-epochs 1 --finetune-epochs 2 --max-folds 1 --device cpu
```

XGBoost and LightGBM are declared in `requirements.txt`. If either optional backend is not installed, omit it from `--methods` or install the project requirements.

## Outputs

Runs are written to `artifacts/runs/e1d_<dataset>_<protocol>_<timestamp>_seed<seed>/`.

- `summary_metrics.csv`: mean train/test RMSE, R2, and RPD by method and target.
- `comparison_vs_ridge.csv`: test-set RMSE/R2/RPD changes relative to the frozen Ridge baseline.
- `fold_metrics.csv`: fold-level metrics.
- `predictions.csv`: outer-test predictions for every sample and target.
- `costs.csv`: shared pretraining, embedding, and downstream fitting time.
- `fold*_lejepa_encoder.pt`: pretrained encoder checkpoints.
- `fold*_finetuned_model.pt`: end-to-end fine-tuned checkpoints.
- `fold*_*_history.csv`: self-supervised, epoch-selection, and final fine-tuning histories.
