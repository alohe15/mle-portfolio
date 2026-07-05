"""Config-driven dataset build entrypoint.

Usage:
    python scripts/build_dataset.py --config configs/dataset_v1.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_COLUMN = "isFraud"
NON_FEATURE_COLUMNS = frozenset({"TransactionID", "TransactionDT", TARGET_COLUMN})


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def load_dataset_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def dataset_config_path_for_version(version: int) -> Path:
    return REPO_ROOT / "configs" / f"dataset_v{version}.json"


def manifest_path_for_output(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_manifest.json"


def feature_list_path_for_version(version: int) -> Path:
    return repo_path(f"data/processed/feature_list_v{version}.json")


def repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def collect_inherited_transformations(config: dict) -> list[tuple[int, dict]]:
    """Return (source_version, transformation) pairs from the inheritance chain."""
    inherited: list[tuple[int, dict]] = []
    parent_version = config.get("inherited_transformations_from")
    if parent_version is None:
        return inherited

    parent_config = load_dataset_config(dataset_config_path_for_version(parent_version))
    inherited.extend(collect_inherited_transformations(parent_config))
    for transform in parent_config.get("transformations", []):
        inherited.append((parent_version, transform))
    return inherited


def resolve_transform(function_ref: str):
    module_name, _, attr_path = function_ref.partition(".")
    if not attr_path:
        raise ValueError(f"Invalid function reference: {function_ref!r}")

    module = importlib.import_module(module_name)
    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def apply_transformation(df: pd.DataFrame, transform: dict) -> pd.DataFrame:
    if transform.get("requires_fit"):
        raise ValueError(
            f"Transformation {transform['name']!r} requires fit/transform and must be "
            "handled by the training script, not the dataset build script."
        )

    func = resolve_transform(transform["function"])
    params = transform.get("params", {})
    return func(df, params)


def build_feature_list(columns: list[str]) -> list[str]:
    return [col for col in columns if col not in NON_FEATURE_COLUMNS]


def build_column_metadata(
    df: pd.DataFrame,
    raw_columns: set[str],
    inherited_transforms: list[tuple[int, dict]],
    current_transforms: list[dict],
    dataset_version: int,
) -> dict[str, dict]:
    origins: dict[str, str] = {col: "raw" for col in raw_columns if col in df.columns}

    for source_version, transform in inherited_transforms:
        for col in transform.get("output_columns", []):
            if col in df.columns:
                origins[col] = f"inherited_v{source_version}"

    for transform in current_transforms:
        for col in transform.get("output_columns", []):
            if col in df.columns:
                origins[col] = f"engineered_v{dataset_version}"

    metadata: dict[str, dict] = {}
    for col in df.columns:
        metadata[col] = {
            "dtype": str(df[col].dtype),
            "origin": origins.get(col, "raw"),
            "null_rate": float(df[col].isnull().mean()),
        }
    return metadata


def build_dataset(config_path: Path) -> None:
    config = load_dataset_config(config_path)
    dataset_version = config["dataset_version"]
    raw_path = repo_path(config["raw_source"])
    output_path = repo_path(config["output_path"])
    manifest_path = manifest_path_for_output(output_path)
    feature_list_path = feature_list_path_for_version(dataset_version)

    df = pd.read_parquet(raw_path)
    raw_columns = set(df.columns)

    inherited_transforms = collect_inherited_transformations(config)
    for _, transform in inherited_transforms:
        df = apply_transformation(df, transform)

    current_transforms = config.get("transformations", [])
    for transform in current_transforms:
        df = apply_transformation(df, transform)

    dropped_columns = config.get("dropped_columns", [])
    drop_reasons = config.get("drop_reasons", {})
    missing_drops = [col for col in dropped_columns if col not in df.columns]
    if missing_drops:
        raise ValueError(f"Cannot drop columns not present in dataset: {missing_drops}")

    for col in dropped_columns:
        if col not in drop_reasons:
            raise ValueError(f"Missing drop_reason for column: {col}")

    df = df.drop(columns=dropped_columns)

    column_metadata = build_column_metadata(
        df,
        raw_columns,
        inherited_transforms,
        current_transforms,
        dataset_version,
    )
    feature_list = build_feature_list(df.columns.tolist())
    new_columns = set(df.columns) - raw_columns
    n_dropped = len(dropped_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    manifest = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": repo_relative_path(config_path.resolve()),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": column_metadata,
        "target_column": TARGET_COLUMN,
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "feature_list_path": repo_relative_path(feature_list_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    feature_list_path.write_text(json.dumps(feature_list, indent=2) + "\n")

    n_new_features = len(new_columns)
    print(
        f"Built dataset_v{dataset_version}: {len(df)} rows | "
        f"{len(feature_list)} features | {n_new_features} new | {n_dropped} dropped"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a versioned dataset from config.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to dataset config JSON (e.g. configs/dataset_v1.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    build_dataset(config_path)


if __name__ == "__main__":
    main()
