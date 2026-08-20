# Fraud Detection — IEEE-CIS Vesta Transaction Dataset

LightGBM binary classifier predicting e-commerce transaction fraud probability. Built as a production ML system with config-driven training, versioned datasets, temporal splits, and FastAPI serving.

## Current candidate model

lgbm_v8 on dataset_v4 (Dn-normalized D-features; raw D1–D15 still present) — test AUC-PR 0.5909

## Quick start

### Prerequisites

- Python 3.11
- Dependencies: `pip install -r requirements.txt`
- Dataset: download from Kaggle (`kaggle competitions download -c ieee-fraud-detection`), extract CSVs to `data/raw/`

### Reproduce from scratch

```bash
# 1. Merge raw CSVs into base parquet
python scripts/merge_train_data.py

# 2. Build the feature-engineered dataset (~5–15 minutes)
python scripts/build_dataset.py --config configs/dataset_v4.json

# 3. Train the model (~28 minutes, 11097 iterations)
python scripts/train.py --config configs/lgbm_v8.json

# 4. Compare all versions
python evals/compare_models.py

# 5. Run tests
python -m pytest tests/ -v
```

### Expected outputs

After training completes, these files should exist:
- `models/lgbm_v8_*.txt` — trained model binary
- `models/lgbm_v8_*_metrics.json` — evaluation results (test AUC-PR ≈ 0.5909)
- `models/lgbm_v8_*_manifest.json` — feature contract for serving
- `models/lgbm_v8_*_fitted_transforms.pkl` — fitted preprocessing state (if applicable)

### Approximate runtimes

| Step | Time |
|------|------|
| Dataset build | ~5–15 min |
| Model training | ~28 min (1709s recorded) |
| Bootstrap significance | ~5–15 min |

## Dataset

~590K Vesta e-commerce transactions, ~3.5% fraud rate. Temporal 70/15/15 split sorted by TransactionDT. Features: 479 columns after Dn normalization plus `requires_fit` encodings (parquet column_count 472; +10 fit-time features).

## Key design decisions

- **Dn normalization**: `normalize_d_columns` uses `floor(TransactionDT/86400 - D_col)` (inherited from dataset_v3). Ablation on branch `diag/d-dn-ablation` favored Dn-only, but raw D1–D15 were **not** dropped in the locked dataset_v4 / feature_list_v4 (D+Dn still present). See `docs/model_lock_report.md`.
- **`requires_fit` leakage prevention**: Frequency encodings and UID aggregations fit on training split only (§2g).
- **Tree ceiling resolution**: v7 (Optuna-tuned) hit the 5000-round ceiling; v8 raised to 15000 and converged naturally at 11097 rounds.

## Experiment history

See `evals/compare_models.py` output and `docs/model_lock_report.md` for the full version comparison, D/Dn ablation results, and temporal stability analysis.

## Serving

```bash
# Start the FastAPI endpoint
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

Returns `{"fraud_probability": float, "model_version": int, "dataset_version": int}`.
