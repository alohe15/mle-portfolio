# ENGINEERING STANDARDS — mle-portfolio

**These are mandatory rules, not suggestions.**  
Any AI model (Cursor, Claude, Copilot) generating code for this repo MUST follow every rule below. Any human contributor MUST follow every rule below. Violations break reproducibility and comparability across model versions.

---

## 1. Model Naming Convention

Every trained model artifact MUST use this exact format:

```
lgbm_v{N}_{short_description}.txt
```

- `{N}` is a monotonically increasing integer starting at 1. Never reuse a version number.
- `{short_description}` is a lowercase, underscore-separated label describing the single change from the previous version.
- Examples: `lgbm_v1_raw_baseline.txt`, `lgbm_v2_email_match.txt`, `lgbm_v3_uid_aggs.txt`, `lgbm_v4_freq_encoding.txt`.

**NEVER** name a model `baseline_lightgbm.txt`, `lightgbm_engineered.txt`, `model_final.txt`, `best_model.txt`, or any other ad-hoc name. The version number IS the identity.

Every model file MUST be accompanied by two companion files in the same `models/` directory, using the same version key:

```
models/
├── lgbm_v3_uid_aggs.txt              # the trained model
├── lgbm_v3_uid_aggs_metrics.json     # evaluation results
└── lgbm_v3_uid_aggs_manifest.json    # feature contract + metadata
```

No orphan model files. If the metrics or manifest is missing, the model is not considered trained.

---

## 2. Dataset Versioning

The dataset is an independent, versioned artifact. It sits UPSTREAM of model training. A feature engineering change creates a new dataset version. A hyperparameter change does NOT — it reuses the same dataset version with a new model version.

### 2a. Dataset Naming Convention

Every processed dataset MUST use this exact format:

```
data/processed/dataset_v{D}_{short_description}.parquet
```

- `{D}` is a monotonically increasing integer starting at 1. Never reuse a dataset version number.
- `{short_description}` is a lowercase, underscore-separated label describing what changed from the previous dataset version.
- Examples: `dataset_v1_raw_merged.parquet`, `dataset_v2_email_match.parquet`, `dataset_v3_uid_aggs.parquet`.

**NEVER** name a dataset `train_merged.parquet`, `features_final.parquet`, `data_v2_fixed.parquet`, or any other ad-hoc name.

### 2b. Dataset Config

Every dataset version MUST have a corresponding config file that fully describes how to reproduce it from raw data:

```
configs/dataset_v{D}.json
```

Required schema:

```json
{
  "dataset_version": 2,
  "description": "Added email_match binary feature from P/R emaildomain comparison",
  "parent_version": 1,
  "raw_source": "data/raw/train_merged.parquet",
  "output_path": "data/processed/dataset_v2_email_match.parquet",
  "transformations": [
    {
      "name": "email_match",
      "type": "engineered",
      "description": "Binary flag: 1 if P_emaildomain == R_emaildomain, else 0",
      "inputs": ["P_emaildomain", "R_emaildomain"],
      "output_columns": ["email_match"],
      "function": "feature_engineering.create_email_match",
      "params": {}
    }
  ],
  "dropped_columns": [],
  "drop_reasons": {},
  "inherited_transformations_from": 1,
  "final_column_count": 433
}
```

#### Schema field rules

- `transformations` is an ORDERED list. Every engineered feature group gets its own entry. Each entry MUST specify `inputs` (source columns consumed), `output_columns` (columns produced), `function` (the exact Python function that performs it), and `params` (any arguments passed to that function, empty dict if none).
- `dropped_columns` is a flat list of every column present in the parent dataset version that is NOT present in this version.
- `drop_reasons` maps each dropped column name to a one-sentence reason. Every entry in `dropped_columns` MUST have a corresponding entry in `drop_reasons`. No silent drops.
- `inherited_transformations_from` is the parent dataset version whose transformations are included in addition to the new ones. Set to `null` for dataset_v1 (the raw merge with no engineering). This creates a clear chain: dataset_v3 inherits v2's transformations which inherit v1's.
- `function` references MUST point to real, importable Python functions in the codebase. **NEVER** use pseudocode, descriptions, or function names that don't exist.

#### Immutability

**NEVER** modify an existing dataset config. If you want to change a transformation, create a new dataset version. The config is a historical record of exactly how that dataset was built.

### 2c. Dataset Manifest

Every dataset version MUST produce a manifest file alongside the parquet:

```
data/processed/dataset_v{D}_{short_description}_manifest.json
```

Required schema:

```json
{
  "dataset_version": 2,
  "created_at": "2026-07-04T14:30:00Z",
  "config_path": "configs/dataset_v2.json",
  "row_count": 590540,
  "column_count": 433,
  "columns": {
    "TransactionAmt": {"dtype": "float64", "origin": "raw", "null_rate": 0.0},
    "email_match": {"dtype": "int64", "origin": "engineered_v2", "null_rate": 0.12},
    "V1": {"dtype": "float64", "origin": "raw", "null_rate": 0.47}
  },
  "target_column": "isFraud",
  "fraud_rate": 0.035,
  "feature_list_path": "data/processed/feature_list_v{D}.json"
}
```

#### Column metadata rules

- Every column MUST have a `dtype` and `null_rate` (fraction of nulls, computed at build time).
- `origin` MUST be one of: `"raw"` (from the original competition data), `"engineered_v{D}"` (created by dataset version D's transformations), or `"inherited_v{D}"` (an engineered column inherited from a parent dataset version).
- The manifest is auto-generated by the dataset build script. **NEVER** write it by hand.

### 2d. Feature List as Dataset Output

The feature list file (`data/processed/feature_list_v{D}.json`) is an OUTPUT of the dataset build, not a standalone artifact. It is a flat JSON array of all column names in the dataset EXCLUDING the target column (`isFraud`) and any non-feature columns (`TransactionID`, `TransactionDT`), in the order they appear in the parquet.

The feature list version number MUST match the dataset version number. They are the same thing. **NEVER** create a feature list that doesn't correspond to a dataset version.

### 2e. Dataset Build Script Contract

The dataset build script (`scripts/build_dataset.py`) MUST:

1. Accept `--config path/to/dataset_config.json` as its only required argument.
2. Load the raw source data.
3. If `inherited_transformations_from` is not null, load the parent dataset config and replay all parent transformations first, in order.
4. Apply the new transformations defined in `transformations`, in order.
5. Drop all columns listed in `dropped_columns`.
6. Save exactly three files: the `.parquet`, the `_manifest.json`, and the `feature_list_v{D}.json`.
7. Print a one-line summary to stdout: `"Built dataset_v{D}: {n_rows} rows | {n_cols} features | {n_new_features} new | {n_dropped} dropped"`.

**NEVER** produce a dataset without its manifest and feature list. All three or nothing.

### 2f. Transformation Functions

Every transformation referenced in a dataset config MUST live in `scripts/feature_engineering.py` as a standalone, importable function with this signature:

```python
def create_email_match(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add email_match column: 1 if P_emaildomain == R_emaildomain, else 0.
    
    Inputs: P_emaildomain, R_emaildomain
    Outputs: email_match
    """
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)
    return df
```

#### Function rules

- Each function takes a DataFrame and a params dict. Returns the DataFrame with new columns added (or existing columns modified).
- Each function MUST have a docstring listing its `Inputs` and `Outputs` columns.
- Functions MUST be idempotent: running the same function twice on the same DataFrame produces the same result without errors.
- Functions MUST NOT drop columns. Column dropping is handled by the build script using the config's `dropped_columns` list. Separation of concerns: functions add, the config decides what to remove.
- Functions MUST NOT access the target column (`isFraud`). Any function that reads `isFraud` during transformation is a leakage vector.

### 2g. Leakage Discipline for Dataset Builds

- **NEVER** compute aggregate statistics (mean, count, frequency) across the full dataset and inject them as features. Aggregations MUST be computed on the TRAINING SPLIT ONLY and applied to both splits, or computed with a strict temporal window that prevents future leakage.
- The dataset build script produces the FULL dataset (all rows). Train/test splitting happens DOWNSTREAM in the training script. Therefore, any transformation that requires train-only computation (UID aggregations, frequency encodings, target statistics) MUST be flagged in the dataset config with `"requires_fit": true` and handled specially by the training script — NOT by the dataset build script.
- If `"requires_fit": true`, the transformation function MUST implement a `.fit(train_df)` / `.transform(df)` pattern instead of a single-call pattern. The dataset config documents this; the training script enforces it.

```json
{
  "name": "uid_transaction_frequency",
  "type": "engineered",
  "description": "Transaction count per UID in trailing window",
  "inputs": ["card1", "addr1", "P_emaildomain", "TransactionDT"],
  "output_columns": ["uid_tx_count_7d", "uid_tx_count_30d"],
  "function": "feature_engineering.UidFrequencyEncoder",
  "params": {"windows_days": [7, 30]},
  "requires_fit": true
}
```

**Every AI model generating feature engineering code MUST pause and verify**: does this transformation look at the target? Does it compute statistics across rows that should be split? If either answer is yes, it MUST use the `requires_fit` pattern.

---

## 3. Config-Driven Training

**NEVER hardcode hyperparameters, feature lists, split parameters, or class-weight settings inside training scripts.**

Every model version MUST have a corresponding config file:

```
configs/lgbm_v{N}.json
```

### Required config schema

```json
{
  "version": 3,
  "description": "Added UID-based aggregation features (card1 + addr1 + P_emaildomain)",
  "parent_version": 2,
  "dataset_version": 3,
  "dataset_config_path": "configs/dataset_v3.json",
  "split": {
    "method": "temporal",
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "sort_column": "TransactionDT"
  },
  "lgbm_params": {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "scale_pos_weight": 27.0,
    "early_stopping_rounds": 100,
    "verbose": -1
  },
  "early_stopping": {
    "metric": "average_precision",
    "patience": 100,
    "min_delta": 0.0
  }
}
```

> **Split discipline:** The three fractions must sum to 1.0. All three slices are contiguous in time: train = earliest rows, val = middle rows, test = latest rows. The validation split is used for early stopping and model-selection decisions. The test split is evaluated exactly once after all decisions are frozen. **NEVER** use the test split for early stopping, hyperparameter selection, or any model decision.

Note: the `features` block from the old standard is GONE. The feature list is now derived from the dataset version. The training script reads the dataset config's `feature_list_path` from the dataset manifest. One source of truth, no duplication.

The training script MUST accept a config path as a CLI argument:

```bash
python scripts/train.py --config configs/lgbm_v3.json
```

**NEVER** train a model by editing values inside a .py file. If you want to change a hyperparameter, create a new config file with a new version number.

---

## 4. Metrics Artifacts

Every training run MUST produce a metrics JSON file with this exact schema:

```json
{
  "version": 3,
  "config_path": "configs/lgbm_v3.json",
  "dataset_version": 3,
  "timestamp": "2026-07-04T14:30:00Z",
  "dataset": {
    "train_rows": 413378,
    "val_rows": 88581,
    "test_rows": 88581,
    "fraud_rate_train": 0.035,
    "fraud_rate_val": 0.034,
    "fraud_rate_test": 0.034
  },
  "metrics": {
    "auc_pr": 0.0,
    "auc_roc": 0.0,
    "precision_at_budget": 0.0,
    "recall_at_budget": 0.0,
    "f1_at_budget": 0.0,
    "operating_threshold": 0.0,
    "best_iteration": 0
  },
  "training": {
    "wall_time_seconds": 0.0,
    "n_features": 445,
    "n_estimators_used": 0
  }
}
```

All numeric values MUST be actual computed values, never placeholders. The `operating_threshold` is the threshold at which `precision_at_budget` and `recall_at_budget` are measured (prevalence-matched flag rate or another documented choice).

---

## 5. Model Manifest (Feature Contract)

Every trained model MUST have a manifest file that serves as the contract between training and serving:

```json
{
  "version": 3,
  "model_file": "lgbm_v3_uid_aggs.txt",
  "created_at": "2026-07-04T14:30:00Z",
  "dataset_version": 3,
  "dataset_config_path": "configs/dataset_v3.json",
  "feature_order": ["TransactionAmt", "card1", "addr1", "...ordered list of all features..."],
  "feature_dtypes": {"TransactionAmt": "float64", "card1": "int64", "...": "..."},
  "n_features": 445,
  "config_path": "configs/lgbm_v3.json",
  "metrics_path": "models/lgbm_v3_uid_aggs_metrics.json",
  "description": "Added UID-based aggregation features"
}
```

### Why `dataset_version` and `dataset_config_path` are in the manifest

The manifest is what the API reads at startup. The API needs to know not just WHICH features the model expects, but HOW to compute engineered features from raw input. The `dataset_config_path` gives the API the full transformation recipe. Without it, the API would have to hardcode feature engineering logic — and that logic would silently drift from what training actually used.

### Serving rules

- The API MUST load the manifest at startup alongside the model.
- The API MUST load the dataset config referenced by the manifest to know which transformations to apply.
- The API MUST validate that incoming feature names, after transformation, match `feature_order` exactly.
- The API MUST reorder features to match `feature_order` before calling `model.predict()`.
- If a feature in `feature_order` is missing from the request AND cannot be computed by any transformation, fill it with `np.nan`.
- If a feature in the request is NOT in `feature_order` and is not an input to any transformation, ignore it — do not pass it to the model.

**NEVER** rely on implicit feature ordering from a DataFrame. The manifest is the single source of truth.

---

## 6. Model Registry

A single file `models/registry.json` tracks every version ever trained:

```json
[
  {
    "version": 1,
    "dataset_version": 1,
    "description": "Raw baseline — all 432 merged columns, no engineered features",
    "model_path": "models/lgbm_v1_raw_baseline.txt",
    "manifest_path": "models/lgbm_v1_raw_baseline_manifest.json",
    "metrics_path": "models/lgbm_v1_raw_baseline_metrics.json",
    "config_path": "configs/lgbm_v1.json",
    "dataset_config_path": "configs/dataset_v1.json",
    "created_at": "2026-06-26T00:00:00Z",
    "is_serving": false
  },
  {
    "version": 2,
    "dataset_version": 2,
    "description": "Added email_match feature from Week 1 EDA",
    "model_path": "models/lgbm_v2_email_match.txt",
    "manifest_path": "models/lgbm_v2_email_match_manifest.json",
    "metrics_path": "models/lgbm_v2_email_match_metrics.json",
    "config_path": "configs/lgbm_v2.json",
    "dataset_config_path": "configs/dataset_v2.json",
    "created_at": "2026-07-01T00:00:00Z",
    "is_serving": true
  }
]
```

### Registry rules

- Exactly ONE entry may have `"is_serving": true` at any time. This is the model the API loads.
- The API reads `registry.json` at startup to determine which model to load. **NEVER** hardcode a model path in `app.py`.
- `evals/compare_models.py` reads `registry.json` to discover all versions. **NEVER** hardcode model paths in evaluation scripts.
- Every new training run MUST append to the registry before the run is considered complete.
- **NEVER** delete or overwrite a registry entry. Old versions are historical records.

---

## 7. One Change Per Version

**NEVER bundle multiple feature engineering ideas, hyperparameter changes, or pipeline modifications into a single version.**

Each version tests exactly one hypothesis:

- v2: "Does adding `email_match` improve AUC-PR over v1?" → new dataset version + new model version
- v3: "Does adding UID-based aggregation features improve AUC-PR over v2?" → new dataset version + new model version
- v4: "Does tuning `num_leaves` from 63 to 127 improve AUC-PR?" → SAME dataset version, new model version only

This is why dataset and model versions are independent. A hyperparameter experiment does NOT require a new dataset build.

If you want to test two ideas, that is two versions, two configs, two training runs, two metrics files.

The ONLY exception: if two features are logically inseparable (e.g., you need the UID grouping key to compute the aggregation features — those are one idea, not two).

---

## 8. Comparison and Evaluation

### 8a. Every version MUST be compared against v1 (raw baseline)

`evals/compare_models.py` MUST read `models/registry.json`, load every version's metrics, and produce a comparison table. No model version exists in isolation. The comparison table MUST include the `dataset_version` column so readers can see which metric movements came from feature changes vs. hyperparameter changes.

### 8b. Statistical significance is required before claiming a win

Before declaring any version an improvement, you MUST run bootstrap resampling on the test set (minimum 500 iterations) and report the 95% confidence interval for the AUC-PR difference between the new version and the baseline. If the intervals overlap substantially, the result is inconclusive — say so.

### 8c. SHAP attribution is required for any version that improves AUC-PR

If a new version shows a statistically significant AUC-PR improvement, you MUST run SHAP on it and verify that the new feature(s) actually appear in the top SHAP contributors. If the lift comes from somewhere else, document that — it is a different finding than what you hypothesized.

---

## 9. Training Script Contract

The main training script (`scripts/train.py`) MUST:

1. Accept `--config path/to/config.json` as its only required argument.
2. Load the model config.
3. Load the dataset version specified by `dataset_version` in the config. Locate the dataset parquet and feature list via the dataset manifest.
4. Apply the three-way temporal split (train / val / test) defined in the config. Use the validation split for early stopping. Evaluate final metrics on the test split only. The test split MUST NOT be passed to `eval_set`, used for hyperparameter selection, or referenced in any model decision.
5. For any transformation in the dataset config with `"requires_fit": true`, fit on the training split only, then transform all three splits (train, val, test).
6. Train the model using only the columns in the feature list.
7. Save exactly three files: `{model}.txt`, `{model}_metrics.json`, `{model}_manifest.json`.
8. Append a new entry to `models/registry.json`.
9. Print a one-line summary to stdout: `"Trained v{N} on dataset_v{D}: AUC-PR={X:.4f} | {n_features} features | {best_iteration} rounds"`.

**NEVER** save a model without its metrics and manifest. **NEVER** skip the registry append.

---

## 10. API Serving Contract

`services/api/app.py` MUST:

1. On startup, read `models/registry.json` and find the entry where `is_serving` is `true`.
2. Load the model file, manifest, and dataset config from the paths in that registry entry.
3. Validate the loaded model's expected feature count against the manifest's `feature_order` length.
4. Build a transformation pipeline from the dataset config's `transformations` list. Load each referenced function from `scripts/feature_engineering.py`. For `requires_fit` transformations, load any pre-fitted state saved during training.
5. On each request:
   a. Accept raw transaction features as input.
   b. Run the transformation pipeline to compute all engineered features.
   c. Select and reorder columns to match the manifest's `feature_order`.
   d. Fill any missing features with `np.nan`.
   e. Discard any columns not in `feature_order`.
   f. Call `model.predict()`.
6. Return a JSON response with at minimum: `{"fraud_probability": float, "model_version": int, "dataset_version": int}`.

### Why the API replays transformations

The API receives RAW transaction features — the same shape a real transaction would arrive in from Vesta's systems. It does NOT receive pre-engineered features. This means:

- If model v3 was trained on dataset_v3 which adds `email_match` and `uid_tx_count_7d`, the API MUST compute those two features from the raw inputs before prediction.
- If model v5 drops `V300`–`V339`, the API simply excludes them after transformation — the manifest's `feature_order` controls what reaches the model.
- If you swap `is_serving` from v3 to v5, the API picks up a different model, a different manifest, and a different transformation pipeline. No code changes needed.

**NEVER** hardcode feature engineering logic in `app.py`. The transformation pipeline is defined by the dataset config and executed generically. `app.py` is model-agnostic and dataset-agnostic.

### Pre-fitted state for `requires_fit` transformations

Transformations marked `requires_fit` (e.g., frequency encodings, UID aggregation lookups) depend on statistics computed from the training split. During training, these fitted objects (lookup tables, encoders) MUST be serialized and saved alongside the model:

```
models/lgbm_v3_uid_aggs_fitted_transforms.pkl
```

The manifest MUST include a `fitted_transforms_path` field pointing to this file. The API loads it at startup and passes the fitted objects to the transformation functions during inference.

**NEVER** recompute fit statistics at inference time. The API uses the frozen training-time values.

---

## 11. Directory Structure Enforcement

```
mle-portfolio/
├── configs/                                        # one JSON per model version + one per dataset version
│   ├── dataset_v1.json
│   ├── dataset_v2.json
│   ├── lgbm_v1.json
│   └── lgbm_v2.json
├── data/
│   ├── raw/                                        # competition CSVs (gitignored; .gitkeep tracked)
│   └── processed/                                  # versioned parquets + manifests + feature lists (gitignored)
│       ├── dataset_v1_raw_merged.parquet
│       ├── dataset_v1_raw_merged_manifest.json
│       ├── feature_list_v1.json
│       ├── dataset_v2_email_match.parquet
│       ├── dataset_v2_email_match_manifest.json
│       └── feature_list_v2.json
├── docs/
├── evals/
│   ├── compare_models.py                           # reads registry.json, compares all versions
│   ├── bootstrap_significance.py                   # CI for AUC-PR differences
│   └── figures/
├── models/
│   ├── registry.json                               # single source of truth for all versions
│   ├── lgbm_v1_raw_baseline.txt
│   ├── lgbm_v1_raw_baseline_metrics.json
│   ├── lgbm_v1_raw_baseline_manifest.json
│   ├── lgbm_v2_email_match.txt
│   ├── lgbm_v2_email_match_metrics.json
│   ├── lgbm_v2_email_match_manifest.json
│   └── lgbm_v2_email_match_fitted_transforms.pkl   # only if requires_fit transforms exist
├── notebooks/
├── scripts/
│   ├── build_dataset.py                            # config-driven dataset build entrypoint
│   ├── train.py                                    # config-driven training entrypoint
│   ├── feature_engineering.py                      # all transformation functions live here
│   └── merge_train_data.py
├── services/api/
│   └── app.py                                      # reads registry.json → loads model + manifest + dataset config
├── tests/
├── requirements.txt
└── README.md
```

**NEVER** place model files, configs, metrics, manifests, or dataset artifacts in ad-hoc locations. Everything goes where this tree says it goes.

---

## 12. Git Discipline

- **NEVER** commit model files (`.txt`, `.pkl`), data files (`.parquet`, `.csv`), or large artifacts to git. They MUST be in `.gitignore`.
- **ALWAYS** commit: configs (both model and dataset), metrics JSONs, manifests (both model and dataset), registry.json, feature lists, and all code.
- The rationale: anyone cloning the repo can see every version's config, metrics, and feature contract without downloading large binary files. They can reproduce any version by running the dataset build script followed by the training script with the committed configs.

---

## 13. The Full Pipeline — End to End

To create a new model version from scratch, the complete sequence is:

```bash
# Step 1: Build the dataset (only if features changed)
python scripts/build_dataset.py --config configs/dataset_v3.json

# Step 2: Train the model
python scripts/train.py --config configs/lgbm_v5.json

# Step 3: Compare against baseline
python evals/compare_models.py

# Step 4: If AUC-PR improved, verify with bootstrap + SHAP
python evals/bootstrap_significance.py --baseline 1 --candidate 5
```

If ONLY hyperparameters changed (same dataset version), skip Step 1. The training script will load the existing dataset_v3 parquet.

If ONLY features changed (same hyperparameters), you still need BOTH steps — new dataset config, new model config pointing to the new dataset version.

---

## Quick Reference — What Gets Created Per Dataset Version

When you build dataset version D, the following files MUST all be created:

| File | Location | Committed to git? |
|---|---|---|
| Dataset config | `configs/dataset_v{D}.json` | Yes |
| Dataset parquet | `data/processed/dataset_v{D}_{desc}.parquet` | No (gitignored) |
| Dataset manifest | `data/processed/dataset_v{D}_{desc}_manifest.json` | Yes |
| Feature list | `data/processed/feature_list_v{D}.json` | Yes |

Missing any one of these four artifacts = incomplete dataset version. Do not train on it.

## Quick Reference — What Gets Created Per Model Version

When you train model version N, the following files MUST all be created:

| File | Location | Committed to git? |
|---|---|---|
| Model config | `configs/lgbm_v{N}.json` | Yes |
| Model | `models/lgbm_v{N}_{desc}.txt` | No (gitignored) |
| Metrics | `models/lgbm_v{N}_{desc}_metrics.json` | Yes |
| Manifest | `models/lgbm_v{N}_{desc}_manifest.json` | Yes |
| Fitted transforms | `models/lgbm_v{N}_{desc}_fitted_transforms.pkl` | No (gitignored) |
| Registry entry | `models/registry.json` (appended) | Yes |

The fitted transforms file is ONLY required if the dataset config contains any `"requires_fit": true` transformations. Otherwise it is omitted.

Missing any required artifact = incomplete version. Do not proceed to the next version.
