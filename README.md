# Fraud Detection — IEEE-CIS Vesta Transaction Dataset

LightGBM binary classifier predicting e-commerce transaction fraud probability.
Config-driven training, versioned datasets, temporal splits, FastAPI serving.

## Current serving model

lgbm_v9 on dataset_v5 (Dn-only, 464 features) — test AUC-PR 0.5952

## Environment setup

- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

## Data placement

Download from Kaggle and extract to `data/raw/`:
```
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

Accept competition terms at https://www.kaggle.com/competitions/ieee-fraud-detection first.

## Reproduce the model

```bash
# 1. Merge raw CSVs (~2 min)
python scripts/merge_train_data.py

# 2. Build dataset_v5 (Dn-only — drops raw D1–D15; ~10 min)
# Historical lineage: dataset_v4 retains raw D + Dn; v5 is the locked build.
python scripts/build_dataset.py --config configs/dataset_v5.json

# 3. Train the serving model (~17 min, 9779 iterations)
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
| Merge raw CSVs | ~2 min |
| Dataset build (v5) | ~5–15 min |
| Model training | ~17 min (1039.7s recorded; early-stops at 9779) |
| Bootstrap significance | ~5–15 min |

## Dataset

~590K Vesta e-commerce transactions, ~3.5% fraud rate. Temporal 70/15/15 split sorted by TransactionDT. Features: 464 columns after Dn-only drop (dataset_v5; 479 − 15 raw D columns) plus `requires_fit` encodings applied at train time.

## Key design decisions

- **Dn normalization**: `normalize_d_columns` uses `floor(TransactionDT/86400 - D_col)` (inherited from dataset_v3). Ablation on branch `diag/d-dn-ablation` favored Dn-only (mean AUC-PR 0.6740).
- **Raw D1–D15 dropped in dataset_v5**: Serving model trains on Dn-only. Raw D columns enable LightGBM to recover calendar day (greedy-attractor); Dn-only won all ablation windows.
- **`requires_fit` leakage prevention**: Frequency encodings and UID aggregations fit on training split only (§2g).
- **Tree ceiling resolution**: v7 (Optuna-tuned) hit the 5000-round ceiling; v8 raised to 15000 and converged at 11097. v9 reuses those hyperparameters on dataset_v5 and converges at 9779.

## Serving

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

Returns `{"fraud_probability": float, "model_version": 9, "dataset_version": 5}`.

## Further documentation

- `docs/model_lock_report.md` — D/Dn comparison, temporal stability, version selection rationale
- `ENGINEERING_STANDARDS.md` — Versioning rules, config schemas, evaluation requirements
