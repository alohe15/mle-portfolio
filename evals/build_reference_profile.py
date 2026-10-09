"""Pre-compute a training-split reference profile for lgbm_v9 monitoring.

Loads the serving stack from models/registry.json (same path as the API),
scores the temporal training split (first 70% by TransactionDT), and writes
aggregated statistics only — no raw rows — to evals/reference_profile.json.

Usage:
    python evals/build_reference_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUT_PATH = EVALS_DIR / "reference_profile.json"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from monitoring import (  # noqa: E402
    ACTION_NAMES,
    HIGH_CARDINALITY_KEEP,
    SERVING_CATEGORICALS,
    SERVING_REQUEST_FIELDS,
    action_proportions,
    apply_monitoring_transforms,
    build_model_input_batch,
    category_proportions,
    dump_json,
    git_commit_hash,
    label_metrics,
    load_serving_artifacts,
    null_rates,
    numeric_summary,
    operating_threshold_at_flag_rate,
    score_and_act,
    select_psi_numeric_features,
    utc_now_iso,
)
from train import (  # noqa: E402
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    repo_path,
    temporal_split,
)


def load_training_split(artifacts: dict) -> tuple[pd.DataFrame, dict]:
    """Load dataset_v5 and return the training slice (leakage-safe split)."""
    serving = artifacts["registry_entry"]
    model_config = load_json(repo_path(serving["config_path"]))
    dataset_manifest_path = find_dataset_manifest(int(serving["dataset_version"]))
    parquet_path = parquet_path_from_manifest(dataset_manifest_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cannot load training split; missing {parquet_path}")
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    train_df, val_df, test_df = temporal_split(
        df,
        model_config["split"],
        config_path=serving["config_path"],
    )
    split_info = {
        "method": model_config["split"]["method"],
        "train_fraction": model_config["split"]["train_fraction"],
        "val_fraction": model_config["split"]["val_fraction"],
        "test_fraction": model_config["split"]["test_fraction"],
        "sort_column": model_config["split"]["sort_column"],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_dt_min": float(train_df["TransactionDT"].min()),
        "train_dt_max": float(train_df["TransactionDT"].max()),
        "test_dt_min": float(test_df["TransactionDT"].min()),
        "test_dt_max": float(test_df["TransactionDT"].max()),
        "parquet_path": str(parquet_path.relative_to(REPO_ROOT)),
    }
    print(
        f"Split: train={split_info['train_rows']} | "
        f"val={split_info['val_rows']} | test={split_info['test_rows']}"
    )
    return train_df, split_info


def _categorical_columns_for_profile(artifacts: dict) -> list[str]:
    ordered: list[str] = []
    for name in SERVING_CATEGORICALS:
        if name not in ordered:
            ordered.append(name)
    for name in artifacts["categorical_columns"]:
        if name not in ordered:
            ordered.append(name)
    return ordered


def build_profile(
    artifacts: dict,
    train_df: pd.DataFrame,
    split_info: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    feature_order: list[str] = artifacts["feature_order"]
    print("Applying frozen fitted transforms (no refit) ...")
    transformed = apply_monitoring_transforms(train_df, artifacts)
    print("Building model input (feature_order + serving categoricals) ...")
    feature_df = build_model_input_batch(transformed, artifacts)
    if "TransactionAmt" not in transformed.columns:
        raise RuntimeError("Training split missing TransactionAmt (serving required field)")
    amounts = pd.to_numeric(transformed["TransactionAmt"], errors="coerce").to_numpy()
    print(f"Scoring {len(feature_df):,} training rows with lgbm_v9 ...")
    _raw, calibrated, actions = score_and_act(feature_df, artifacts, amounts)

    dtypes = {col: str(feature_df[col].dtype) for col in feature_order}
    miss_cols = list(dict.fromkeys(list(feature_order) + ["TransactionAmt"]))
    missingness = null_rates(transformed if "TransactionAmt" in transformed.columns else feature_df, miss_cols)
    # Prefer null rates on the pre-encoding transformed frame so they match
    # production logs of raw/serving fields rather than categorical codes.
    for col in miss_cols:
        src = transformed if col in transformed.columns else feature_df
        missingness[col] = float(src[col].isna().mean()) if col in src.columns else 1.0

    cat_set = set(artifacts["categorical_columns"])
    numerics: dict[str, dict] = {}
    for col in feature_order:
        if col in cat_set:
            continue
        numerics[col] = numeric_summary(
            pd.to_numeric(transformed[col] if col in transformed.columns else feature_df[col], errors="coerce").to_numpy()
        )

    categoricals: dict[str, dict[str, float]] = {}
    for col in _categorical_columns_for_profile(artifacts):
        src = transformed[col] if col in transformed.columns else feature_df[col]
        nunique = int(src.nunique(dropna=False))
        max_levels = None if nunique <= HIGH_CARDINALITY_KEEP else HIGH_CARDINALITY_KEEP
        if col in SERVING_CATEGORICALS:
            max_levels = None
        categoricals[col] = category_proportions(src, max_levels=max_levels)

    scores_summary = numeric_summary(calibrated)
    amounts_summary = numeric_summary(amounts)
    action_props = action_proportions(actions)

    if "isFraud" not in transformed.columns:
        raise RuntimeError("Training split has no isFraud labels; cannot build Tier 2 reference")
    labels = transformed["isFraud"].to_numpy()
    flag_rate = float(np.mean(labels))
    operating_threshold = operating_threshold_at_flag_rate(calibrated, flag_rate)
    labels_block = label_metrics(
        labels,
        calibrated,
        actions,
        amounts,
        artifacts["cost_params"],
        operating_threshold=operating_threshold,
    )

    psi_features = select_psi_numeric_features(artifacts, n=20)

    profile = {
        "schema": {
            "columns": list(feature_order),
            "dtypes": dtypes,
            "expected_column_count": int(len(feature_order)),
            "source": "serving_model_feature_order",
            "serving_request_fields": list(SERVING_REQUEST_FIELDS),
            "serving_response_fields": [
                "fraud_probability",
                "recommended_action",
                "action_costs",
                "model_version",
                "dataset_version",
                "request_id",
            ],
        },
        "missingness": missingness,
        "numerics": numerics,
        "categoricals": categoricals,
        "scores": scores_summary,
        "actions": {name: float(action_props[name]) for name in ACTION_NAMES},
        "amounts": amounts_summary,
        "labels": labels_block,
        "psi_numeric_features": psi_features,
        "metadata": {
            "model_version": int(artifacts["model_version"]),
            "model_version_label": artifacts["model_version_label"],
            "dataset_version": int(artifacts["dataset_version"]),
            "n_rows": int(len(train_df)),
            "split": "train",
            "generated_at": utc_now_iso(),
            "commit_hash": git_commit_hash(),
            "split_info": split_info,
            "feature_count": int(len(feature_order)),
            "paths": artifacts.get("paths") or {},
        },
    }
    return profile, calibrated, actions


def main() -> None:
    print("Loading serving artifacts from models/registry.json ...")
    artifacts = load_serving_artifacts()
    print(
        f"Serving model={artifacts['model_version_label']} "
        f"dataset_v{artifacts['dataset_version']} "
        f"features={len(artifacts['feature_order'])}"
    )
    train_df, split_info = load_training_split(artifacts)
    profile, calibrated, actions = build_profile(artifacts, train_df, split_info)
    dump_json(OUTPUT_PATH, profile)
    labels = profile["labels"]
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"n={profile['metadata']['n_rows']} | "
        f"features={profile['schema']['expected_column_count']} | "
        f"score_mean={profile['scores']['mean']:.6f} | "
        f"fraud_rate={labels['fraud_rate']:.4f} | "
        f"PR-AUC={labels['pr_auc']:.4f}"
    )
    print("Action bands: " + " ".join(
        f"{k}={profile['actions'][k]:.4f}" for k in ACTION_NAMES
    ))
    # Silence unused locals if scoring arrays are only used for profile stats.
    _ = (calibrated, actions)


if __name__ == "__main__":
    main()
