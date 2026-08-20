# Fraud Detection — IEEE-CIS Vesta Transaction Dataset

LightGBM binary classifier predicting e-commerce transaction fraud probability. Built as a production ML system with config-driven training, versioned datasets, temporal splits, and FastAPI serving.

## Current candidate model

lgbm_v9 on dataset_v5 (Dn-only — raw D1–D15 dropped) — test AUC-PR 0.5952

## Quick start

### Prerequisites

- Python 3.11
- Dependencies: `pip install -r requirements.txt`
- Dataset: download from Kaggle (`kaggle competitions download -c ieee-fraud-detection`), extract CSVs to `data/raw/`

### Reproduce from scratch

```bash
# 1. Merge raw CSVs into base parquet
python scripts/merge_train_data.py

# 2. Build feature-engineered datasets (~5–15 minutes each)
# Build dataset_v4 (Dn normalization, raw D columns retained)
python scripts/build_dataset.py --config configs/dataset_v4.json

# Build dataset_v5 (drop raw D1-D15, Dn-only)
python scripts/build_dataset.py --config configs/dataset_v5.json

# 3. Train the locked candidate (~17 minutes, 9779 iterations)
python scripts/train.py --config configs/lgbm_v9.json

# 4. Compare all versions
python evals/compare_models.py

# 5. Run tests
python -m pytest tests/ -v
```

### Expected outputs

After training completes, these files should exist:
- `models/lgbm_v9_*.txt` — trained model binary
- `models/lgbm_v9_*_metrics.json` — evaluation results (test AUC-PR ≈ 0.5952)
- `models/lgbm_v9_*_manifest.json` — feature contract for serving
- `models/lgbm_v9_*_fitted_transforms.pkl` — fitted preprocessing state (if applicable)

### Approximate runtimes

| Step | Time |
|------|------|
| Dataset build | ~5–15 min |
| Model training | ~17 min (1040s recorded) |
| Bootstrap significance | ~5–15 min |

## Dataset

~590K Vesta e-commerce transactions, ~3.5% fraud rate. Temporal 70/15/15 split sorted by TransactionDT. Features: 464 columns after Dn-only drop (dataset_v5; 479 − 15 raw D columns) plus `requires_fit` encodings applied at train time.

## Key design decisions

- **Dn normalization**: `normalize_d_columns` uses `floor(TransactionDT/86400 - D_col)` (inherited from dataset_v3). Ablation on branch `diag/d-dn-ablation` favored Dn-only.
- **Raw D1–D15 dropped in dataset_v5**: Locked candidate trains on Dn-only. Raw D columns enable LightGBM to recover calendar day (greedy-attractor); Dn-only won all ablation windows.
- **`requires_fit` leakage prevention**: Frequency encodings and UID aggregations fit on training split only (§2g).
- **Tree ceiling resolution**: v7 (Optuna-tuned) hit the 5000-round ceiling; v8 raised to 15000 and converged at 11097. v9 reuses those hyperparameters on dataset_v5 and converged at 9779.

## Experiment history

See `evals/compare_models.py` output and `docs/model_lock_report.md` for the full version comparison, D/Dn ablation results, and temporal stability analysis.

## Serving

```bash
# Start the FastAPI endpoint
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

Returns `{"fraud_probability": float, "model_version": int, "dataset_version": int}`.
