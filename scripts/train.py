"""Config-driven model training entrypoint.

Usage:
    python scripts/train.py --config configs/lgbm_v1.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_model_config(config_path: Path) -> dict:
    return load_json(config_path)


def find_dataset_manifest(dataset_version: int) -> Path:
    matches = sorted(PROCESSED_DIR.glob(f"dataset_v{dataset_version}_*_manifest.json"))
    if not matches:
        raise FileNotFoundError(
            f"No dataset manifest found for dataset_v{dataset_version} in {PROCESSED_DIR}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple dataset manifests found for dataset_v{dataset_version}: {matches}"
        )
    return matches[0]


def parquet_path_from_manifest(manifest_path: Path) -> Path:
    name = manifest_path.name.replace("_manifest.json", ".parquet")
    return manifest_path.with_name(name)


def dataset_config_path_for_version(version: int) -> Path:
    return REPO_ROOT / "configs" / f"dataset_v{version}.json"


def collect_inherited_transformations(config: dict) -> list[tuple[int, dict]]:
    inherited: list[tuple[int, dict]] = []
    parent_version = config.get("inherited_transformations_from")
    if parent_version is None:
        return inherited

    parent_config = load_json(dataset_config_path_for_version(parent_version))
    inherited.extend(collect_inherited_transformations(parent_config))
    for transform in parent_config.get("transformations", []):
        inherited.append((parent_version, transform))
    return inherited


def collect_all_transformations(config: dict) -> list[tuple[int, dict]]:
    transforms = collect_inherited_transformations(config)
    for transform in config.get("transformations", []):
        transforms.append((config["dataset_version"], transform))
    return transforms


def resolve_transform(function_ref: str):
    module_name, _, attr_path = function_ref.partition(".")
    if not attr_path:
        raise ValueError(f"Invalid function reference: {function_ref!r}")

    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    module = importlib.import_module(module_name)
    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def artifact_slug(model_config: dict, dataset_config: dict) -> str:
    output_stem = Path(dataset_config["output_path"]).stem
    prefix = f"dataset_v{dataset_config['dataset_version']}_"
    if output_stem.startswith(prefix):
        return output_stem[len(prefix) :]
    slug = re.sub(r"[^a-z0-9]+", "_", model_config["description"].lower()).strip("_")
    return slug or f"v{model_config['version']}"


def temporal_split(
    df: pd.DataFrame,
    split_config: dict,
    *,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = split_config["method"]
    if method != "temporal":
        raise ValueError(f"Unsupported split method: {method!r}")

    if "val_fraction" not in split_config:
        config_label = config_path or "unknown"
        raise ValueError(
            f"Config {config_label} uses the deprecated two-way split. "
            "Create a new config with train_fraction, val_fraction, and test_fraction "
            "per §3 of ENGINEERING_STANDARDS.md."
        )

    train_frac = split_config["train_fraction"]
    val_frac = split_config["val_fraction"]
    assert abs(train_frac + val_frac + split_config["test_fraction"] - 1.0) < 1e-6, (
        f"Split fractions must sum to 1.0, got "
        f"{train_frac + val_frac + split_config['test_fraction']}"
    )

    sort_column = split_config["sort_column"]
    df = df.sort_values(sort_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def apply_requires_fit_transforms(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fitted: dict[str, object] = {}
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue

        name = transform["name"]
        obj = resolve_transform(transform["function"])
        params = transform.get("params", {})

        if inspect.isclass(obj):
            instance = obj(**params) if params else obj()
            instance.fit(train_df)
            train_df = instance.transform(train_df)
            val_df = instance.transform(val_df)
            test_df = instance.transform(test_df)
            fitted[name] = instance
            continue

        raise ValueError(
            f"Transformation {name!r} requires fit/transform but {transform['function']!r} "
            "is not a class."
        )

    return train_df, val_df, test_df, fitted


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    missing = [col for col in feature_list if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature list columns missing from dataset: {missing}")

    x_train = train_df[feature_list].copy()
    x_val = val_df[feature_list].copy()
    x_test = test_df[feature_list].copy()
    cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        x_train[col] = x_train[col].fillna("__MISSING__").astype(str)
        x_val[col] = x_val[col].fillna("__MISSING__").astype(str)
        x_test[col] = x_test[col].fillna("__MISSING__").astype(str)
        categories = pd.Index(
            pd.concat(
                [x_train[col], x_val[col], x_test[col]], ignore_index=True
            ).unique()
        )
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_val[col] = pd.Categorical(x_val[col], categories=categories)
        x_test[col] = pd.Categorical(x_test[col], categories=categories)

    return x_train, x_val, x_test, cat_cols


def evaluate_at_flag_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    flag_rate: float,
) -> dict[str, float]:
    n_flag = max(int(round(len(y_true) * flag_rate)), 1)
    threshold = float(np.partition(y_score, len(y_score) - n_flag)[len(y_score) - n_flag])
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def append_registry_entry(entry: dict) -> None:
    registry_path = MODELS_DIR / "registry.json"
    registry: list[dict] = []
    if registry_path.exists():
        registry = load_json(registry_path)

    if any(item["version"] == entry["version"] for item in registry):
        raise ValueError(f"Registry already contains version {entry['version']}")

    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")


def train_model(config_path: Path) -> None:
    started_at = time.perf_counter()
    model_config = load_model_config(config_path)

    version = model_config["version"]
    dataset_version = model_config["dataset_version"]
    dataset_config = load_json(repo_path(model_config["dataset_config_path"]))

    dataset_manifest_path = find_dataset_manifest(dataset_version)
    dataset_manifest = load_json(dataset_manifest_path)
    parquet_path = parquet_path_from_manifest(dataset_manifest_path)
    feature_list_path = repo_path(dataset_manifest["feature_list_path"])
    feature_list = load_json(feature_list_path)
    target_column = dataset_manifest["target_column"]

    slug = artifact_slug(model_config, dataset_config)
    model_base = f"lgbm_v{version}_{slug}"
    model_path = MODELS_DIR / f"{model_base}.txt"
    metrics_path = MODELS_DIR / f"{model_base}_metrics.json"
    model_manifest_path = MODELS_DIR / f"{model_base}_manifest.json"
    fitted_transforms_path = MODELS_DIR / f"{model_base}_fitted_transforms.pkl"

    df = pd.read_parquet(parquet_path)
    split_cfg = model_config["split"]
    train_df, val_df, test_df = temporal_split(
        df,
        split_cfg,
        config_path=repo_relative_path(config_path.resolve()),
    )
    print(
        f"Split: train={len(train_df)} ({train_df[target_column].mean():.4f} fraud) | "
        f"val={len(val_df)} ({val_df[target_column].mean():.4f} fraud) | "
        f"test={len(test_df)} ({test_df[target_column].mean():.4f} fraud)"
    )
    train_df, val_df, test_df, fitted_transforms = apply_requires_fit_transforms(
        train_df, val_df, test_df, dataset_config
    )

    x_train, x_val, x_test, cat_cols = prepare_features(
        train_df, val_df, test_df, feature_list
    )
    y_train = train_df[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()

    lgbm_params = dict(model_config["lgbm_params"])
    early_stopping_rounds = lgbm_params.pop("early_stopping_rounds")
    eval_metric = lgbm_params.pop("metric")

    classifier_kwargs = dict(lgbm_params)
    model = lgb.LGBMClassifier(**classifier_kwargs)
    model.fit(
        x_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(x_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        ],
    )

    # --- Final evaluation on held-out test set (never seen during training or early stopping) ---
    y_score = model.predict_proba(x_test)[:, 1]
    train_fraud_rate = float(y_train.mean())
    operating = evaluate_at_flag_rate(y_test, y_score, train_fraud_rate)
    best_iteration = int(model.best_iteration_ or model_config["lgbm_params"]["n_estimators"])
    wall_time = time.perf_counter() - started_at

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))

    if fitted_transforms:
        fitted_transforms_path.write_bytes(pickle.dumps(fitted_transforms))

    metrics_doc = {
        "version": version,
        "config_path": repo_relative_path(config_path.resolve()),
        "dataset_version": dataset_version,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_val": float(y_val.mean()),
            "fraud_rate_test": float(y_test.mean()),
        },
        "metrics": {
            "auc_pr": float(average_precision_score(y_test, y_score)),
            "auc_roc": float(roc_auc_score(y_test, y_score)),
            "precision_at_budget": operating["precision"],
            "recall_at_budget": operating["recall"],
            "f1_at_budget": operating["f1"],
            "operating_threshold": operating["threshold"],
            "best_iteration": best_iteration,
        },
        "training": {
            "wall_time_seconds": float(wall_time),
            "n_features": int(len(feature_list)),
            "n_estimators_used": best_iteration,
        },
    }
    metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n")

    feature_dtypes = {col: str(x_train[col].dtype) for col in feature_list}
    model_manifest = {
        "version": version,
        "model_file": model_path.name,
        "created_at": metrics_doc["timestamp"],
        "dataset_version": dataset_version,
        "dataset_config_path": model_config["dataset_config_path"],
        "feature_order": feature_list,
        "feature_dtypes": feature_dtypes,
        "n_features": len(feature_list),
        "config_path": repo_relative_path(config_path.resolve()),
        "metrics_path": repo_relative_path(metrics_path),
        "description": model_config["description"],
    }
    if fitted_transforms:
        model_manifest["fitted_transforms_path"] = repo_relative_path(fitted_transforms_path)

    model_manifest_path.write_text(json.dumps(model_manifest, indent=2) + "\n")

    append_registry_entry(
        {
            "version": version,
            "dataset_version": dataset_version,
            "description": model_config["description"],
            "model_path": repo_relative_path(model_path),
            "manifest_path": repo_relative_path(model_manifest_path),
            "metrics_path": repo_relative_path(metrics_path),
            "config_path": repo_relative_path(config_path.resolve()),
            "dataset_config_path": model_config["dataset_config_path"],
            "created_at": metrics_doc["timestamp"],
            "is_serving": False,
        }
    )

    print(
        f"Trained v{version} on dataset_v{dataset_version}: "
        f"AUC-PR={metrics_doc['metrics']['auc_pr']:.4f} | "
        f"{len(feature_list)} features | {best_iteration} rounds | "
        f"split={len(train_df)}/{len(val_df)}/{len(test_df)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a versioned LightGBM model from config.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to model config JSON (e.g. configs/lgbm_v1.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    train_model(config_path)


if __name__ == "__main__":
    main()
