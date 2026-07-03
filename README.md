# MLE Portfolio

End-to-end machine learning project for IEEE-CIS fraud detection: data prep, feature engineering, model training, evaluation, and a serving API.

## Project layout

```
mle-portfolio/
├── configs/          # Hyperparameters, feature lists, paths (edit settings without code changes)
├── data/
│   ├── raw/          # Competition CSVs and archives (gitignored)
│   └── processed/    # Merged and featured parquet files (gitignored)
├── docs/             # Session logs and write-ups
├── evals/            # Cross-version comparisons and evaluation artifacts
├── models/           # Trained model files and per-run metrics (gitignored)
├── notebooks/        # Exploratory analysis scripts
├── scripts/          # Pipeline: merge → features → train
├── services/         # FastAPI inference service
├── tests/            # API integration and latency checks
└── requirements.txt  # Pinned dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place raw IEEE-CIS data under `data/raw/` (see Kaggle competition files).

## Pipeline

Run from the repo root:

```bash
python scripts/merge_train_data.py
python scripts/feature_engineering.py
python scripts/train_baseline_lightgbm.py
python scripts/train_engineered_lightgbm.py
python evals/compare_models.py
```

## Serve predictions

```bash
uvicorn services.api.app:app --reload --app-dir .
python tests/test_api_endpoint.py
```

## Configuration

Training hyperparameters, feature-engineering settings, and file paths live in `configs/`:

- `configs/training.json` — split fraction, decision threshold, LightGBM params
- `configs/feature_engineering.json` — null thresholds, frequency-encode columns
- `configs/paths.json` — data and model artifact locations
