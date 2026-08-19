# Fraud Detection — Production ML System

End-to-end fraud detection pipeline on the IEEE-CIS Vesta transaction dataset (~590K transactions, ~3.5% fraud rate). LightGBM classifier with config-driven training, versioned datasets, bootstrap significance gating, and FastAPI serving.

## Quick start

### Prerequisites
- Python 3.11
- pip packages: `pip install -r requirements.txt`

### Data placement
Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) and place the CSVs in `data/raw/`:
```
data/raw/
├── train_transaction.csv
├── train_identity.csv
```

Then merge:
```bash
python scripts/merge_train_data.py
```

### Build the dataset
```bash
python scripts/build_dataset.py --config configs/dataset_v4.json
```

### Train the model
```bash
python scripts/train.py --config configs/lgbm_v8.json
```
Approximate runtime: 1709s.

### Evaluate
```bash
python evals/compare_models.py
python evals/bootstrap_significance.py --baseline 7 --candidate 8
```

### Serve
```bash
cd services/api && uvicorn app:app --reload
```
The API reads `models/registry.json` to load whichever model has `is_serving: true`.

## Model versions

| version | dataset_version | description | test_auc_pr | best_iteration | n_features | serving |
|---------|-----------------|-------------|-------------|----------------|------------|---------|
| 1 | 1 | Raw baseline | 0.5511 | 953 | 431 | |
| 2 | 2 | Feature engineering migration | 0.5698 | 1321 | 461 | |
| 3 | 2 | L1 regularization | 0.5721 | 1993 | 461 | |
| 4 | 2 | Clean re-baseline (three-way split) | 0.5011 | 1947 | 461 | |
| 5 | 3 | Dn normalization | 0.6412* | 2000 | 476 | |
| 6 | 4 | Identity frequency encodings | 0.6527* | 1995 | 479 | |
| 7 | 4 | Optuna-tuned hyperparameters | 0.5837 | 5000 | 479 | |
| 8 | 4 | Tree ceiling test (serving) | 0.5909 | 11097 | 479 | **yes** |

\*v5 and v6 metrics JSONs store validation AUC-PR only; table uses `auc_pr` fallback per `compare_models.py`.

v8 is the locked serving model. See [docs/model_lock_report.md](docs/model_lock_report.md) for the full rationale.

## Engineering standards

See [ENGINEERING_STANDARDS.md](ENGINEERING_STANDARDS.md) for mandatory rules governing this repo.

## Key design decisions

- **Temporal split**: 70/15/15 train/val/test ordered by TransactionDT. No random splitting.
- **AUC-PR as primary metric**: Chosen over AUC-ROC due to 3.5% positive class rate.
- **Bootstrap significance gating**: 500 paired resamples, 95% CI must exclude zero before claiming improvement (§8b).
- **`requires_fit` leakage prevention**: Frequency encodings and UID aggregations fit on training split only (§2g).
- **Dn-only columns**: Raw D1–D15 dropped after ablation showed they enabled calendar-time shortcuts.
- **Tree ceiling resolution**: v7 (Optuna-tuned) hit the 5000-round ceiling; v8 raised to 15000 and converged naturally at 11097 rounds.
