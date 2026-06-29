# Session Log — 2026-06-26

Work session on the IEEE-CIS fraud detection portfolio project: merged training data, baseline model training, FastAPI inference service, endpoint testing, and bug fixes discovered during integration.

---

## Summary of Accomplishments

1. **Training pipeline** — Added `scripts/train_baseline_lightgbm.py` to train a LightGBM classifier on the merged dataset with a time-based 80/20 split, baseline comparisons, and saved artifacts under `models/`.
2. **Inference API** — Added `services/api/app.py`, a FastAPI service that loads the trained booster and exposes `/predict` and `/health`.
3. **Integration test** — Added `scripts/test_api_endpoint.py` to build a real transaction payload from parquet and POST it to the running API.
4. **End-to-end validation** — Ran the test successfully after fixing categorical feature handling in the API (see Changes Required below).
5. **Data merge script** (prior commit on this branch) — `scripts/merge_train_data.py` joins Kaggle transaction and identity CSVs into `data/train_merged.parquet`.

---

## Test Results (2026-06-26)

After starting the API and running `python scripts/test_api_endpoint.py`:

```json
{
  "fraud_probability": 0.099538,
  "prediction": 0,
  "latency_ms": 148.76
}
```

- First transaction in the merged dataset scored ~10% fraud probability.
- Binary label at threshold 0.5: not fraud (`prediction: 0`).
- Round-trip latency ~149 ms on local hardware.

---

## File Reference (Granular)

### `scripts/merge_train_data.py`

**Purpose:** One-time (or repeatable) ETL to produce the modeling dataset.

| Section | Detail |
|---|---|
| Inputs | `project-1-detect-fraud/data/train_transaction.csv`, `train_identity.csv` |
| Join | Left merge on `TransactionID` (keeps all transactions; identity fields null when missing) |
| Output | `data/train_merged.parquet` (gitignored — large, derived data) |
| Constants | `REPO_ROOT`, `SOURCE_DIR`, `OUTPUT_DIR`, `OUTPUT_PATH` resolved from script location |

**Run:** `python scripts/merge_train_data.py`

---

### `scripts/train_baseline_lightgbm.py`

**Purpose:** Offline training job for the baseline fraud classifier.

| Function | Role |
|---|---|
| `load_and_split()` | Reads parquet, sorts by `TransactionDT`, splits first 80% train / last 20% test (temporal split) |
| `prepare_features()` | Drops `TransactionID` and `isFraud`; fills categorical nulls with `__MISSING__`; builds shared category vocab per column from train+test |
| `evaluate()` | Computes AUC-PR, precision, recall at a fixed probability threshold |
| `evaluate_at_flag_rate()` | Sets threshold to flag top N% of scores (prevalence-matched operating point) |
| `majority_class_baseline()` | No-skill baseline: always predict non-fraud |
| `random_ranking_baseline()` | No-skill ranking baseline at fixed flag rate |
| `business_context()` | Embeds metric interpretation and project context into metrics JSON |
| `main()` | Fits `LGBMClassifier` with early stopping, saves model + metrics |

| Constant | Value | Meaning |
|---|---|---|
| `TARGET` | `isFraud` | Binary label |
| `TRAIN_FRACTION` | 0.8 | Time-based split ratio |
| `DECISION_THRESHOLD` | 0.5 | Default cutoff (mostly for reporting; fraud scores rarely exceed 0.5) |
| `RANDOM_STATE` | 42 | Reproducibility |

**Outputs (gitignored locally):**

- `models/baseline_lightgbm.txt` — LightGBM booster (432 features)
- `models/baseline_lightgbm_metrics.json` — Full eval report, baselines, lift, business context

**Key metrics from last run:**

- Test AUC-PR: **0.250** (vs fraud rate ~0.034 baseline)
- At prevalence-matched flag rate: precision **0.30**, recall **0.33**

**Run:** `python scripts/train_baseline_lightgbm.py` (requires `data/train_merged.parquet`)

---

### `services/api/app.py`

**Purpose:** HTTP inference layer for production-style serving.

| Component | Detail |
|---|---|
| Startup | Loads `lgb.Booster` from `models/baseline_lightgbm.txt`; reads 432 `FEATURE_NAMES` and `pandas_categorical` from the saved model |
| `load_categorical_columns()` | Reads parquet schema to list object/string columns in the same order as training |
| `CATEGORICAL_COLUMNS` | Computed once at import — must match training's `cat_cols` ordering |
| `build_feature_frame()` | Maps request dict → single-row DataFrame; applies training-aligned preprocessing |
| `PredictionRequest` | Pydantic model: `{ "features": { ... } }` |
| `PredictionResponse` | `fraud_probability`, `prediction` (0/1 at 0.5), `latency_ms` |

**Endpoints:**

| Method | Path | Response |
|---|---|---|
| `POST` | `/predict` | Fraud score + binary label + latency |
| `GET` | `/health` | `{ "status": "healthy", "model_features": 432 }` |

**Run:**

```bash
uvicorn services.api.app:app --reload --app-dir .
```

**Dependencies:** `fastapi`, `uvicorn`, `lightgbm`, `pandas`, `numpy`

---

### `scripts/test_api_endpoint.py`

**Purpose:** Smoke test for the live API using real feature data.

| Function | Role |
|---|---|
| `build_test_payload()` | Row 0 from `train_merged.parquet`, drops `isFraud`, NaN → `None` for JSON |
| `save_test_payload()` | Writes `test_payload.json` at repo root |
| `call_predict_endpoint()` | `POST` to `/predict` via stdlib `urllib` (no extra deps) |
| `main()` | Builds payload, saves file, calls API, prints JSON response |

| Env var | Default |
|---|---|
| `API_URL` | `http://127.0.0.1:8000/predict` |

**Run:**

```bash
# Terminal 1
uvicorn services.api.app:app --reload --app-dir .

# Terminal 2
python scripts/test_api_endpoint.py
```

---

### Local artifacts (not committed to git)

| Path | Why excluded |
|---|---|
| `data/train_merged.parquet` | Gitignored (`data/`) — large derived dataset |
| `models/baseline_lightgbm.txt` | Gitignored (`models/**/*`) — binary model weights |
| `models/baseline_lightgbm_metrics.json` | Gitignored — contains local absolute paths |
| `test_payload.json` | Generated test fixture with real transaction fields |
| `.DS_Store` | macOS metadata |

---

## Changes Required During Integration

### 1. Categorical feature mismatch (API 500)

**Symptom:** `POST /predict` returned HTTP 500:

```
ValueError: train and valid dataset categorical_feature do not match.
```

**Root cause:** The initial API built a DataFrame without applying the exact category vocabularies stored in the LightGBM booster at training time. LightGBM's `pandas_categorical` metadata requires inference columns to use the same category lists in the same order as training.

**Fix in `services/api/app.py`:**

1. Load `model.pandas_categorical` from the saved booster.
2. At startup, derive `CATEGORICAL_COLUMNS` from `train_merged.parquet` dtypes (same logic as `prepare_features()` in the training script).
3. In `build_feature_frame()`:
   - Map JSON `null` → `np.nan` → `"__MISSING__"` for categoricals.
   - Cast each categorical column with `pd.Categorical(..., categories=PANDAS_CATEGORICAL[i])`.
   - Coerce numeric columns with `pd.to_numeric(..., errors="coerce")`.

### 2. Port 8000 already in use

**Symptom:** Background `uvicorn` failed with `[Errno 48] address already in use`.

**Resolution:** Kill the stale process on port 8000 before restarting the server.

### 3. Relative model path

**Symptom:** Model path `"models/baseline_lightgbm.txt"` depends on cwd when starting uvicorn.

**Fix:** Resolve paths from `REPO_ROOT = Path(__file__).resolve().parent.parent.parent` so the API works regardless of working directory when using `--app-dir .`.

---

## Repository Layout After This Session

```
mle-portfolio/
├── docs/
│   └── session-log-2026-06-26.md    ← this file
├── scripts/
│   ├── merge_train_data.py          ← CSV merge → parquet
│   ├── train_baseline_lightgbm.py   ← train + evaluate + save model
│   └── test_api_endpoint.py         ← API smoke test
├── services/
│   └── api/
│       └── app.py                   ← FastAPI inference service
├── models/                          ← gitignored artifacts (local only)
│   ├── baseline_lightgbm.txt
│   └── baseline_lightgbm_metrics.json
└── data/                            ← gitignored
    └── train_merged.parquet
```

---

## Suggested Next Steps

- Add `requirements.txt` or `pyproject.toml` pinning `fastapi`, `uvicorn`, `lightgbm`, `pandas`, `scikit-learn`.
- Add `Dockerfile` and mount `models/` at deploy time.
- Extract shared preprocessing into `src/` so train and serve cannot drift.
- Add unit tests with a small fixture instead of full parquet row.
- Commit metrics JSON with relative paths only, or store in MLflow / object storage.
