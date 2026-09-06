"""SHAP analysis for a frozen model version (validation split only).

Usage:
    python evals/shap_analysis.py --version 9 --sample-size 1000 --seed 42
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    collect_all_transformations,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    prepare_features,
    repo_path,
    resolve_transform,
    temporal_split,
)


def load_registry() -> list[dict]:
    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must contain a JSON array")
    return registry


def get_registry_entry(registry: list[dict], version: int) -> dict:
    matches = [entry for entry in registry if entry["version"] == version]
    if not matches:
        raise ValueError(f"Version {version} not found in models/registry.json")
    if len(matches) > 1:
        raise ValueError(f"Multiple registry entries found for version {version}")
    return matches[0]


def load_fitted_transforms(manifest: dict) -> dict[str, object]:
    fitted_path = manifest.get("fitted_transforms_path")
    if not fitted_path:
        return {}
    path = repo_path(fitted_path)
    if not path.exists():
        raise FileNotFoundError(f"Fitted transforms file not found: {path}")
    loaded = pickle.loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in fitted transforms file: {path}")
    return loaded


def apply_requires_fit_transforms(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue

        name = transform["name"]
        params = transform.get("params", {})
        if name in fitted_transforms:
            encoder = fitted_transforms[name]
            train_df = encoder.transform(train_df, params)
            val_df = encoder.transform(val_df, params)
            test_df = encoder.transform(test_df, params)
            continue

        obj = resolve_transform(transform["function"])
        if not inspect.isclass(obj):
            raise ValueError(
                f"Transformation {name!r} requires fit/transform but is not a class"
            )
        instance = obj()
        instance.fit(train_df, params)
        train_df = instance.transform(train_df, params)
        val_df = instance.transform(val_df, params)
        test_df = instance.transform(test_df, params)

    return train_df, val_df, test_df


def load_validation_features(registry_entry: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load and prepare the temporal VALIDATION split for SHAP (not test)."""
    model_config = load_json(repo_path(registry_entry["config_path"]))
    manifest = load_json(repo_path(registry_entry["manifest_path"]))
    dataset_config = load_json(repo_path(registry_entry["dataset_config_path"]))
    feature_list = list(manifest["feature_order"])

    manifest_path = find_dataset_manifest(registry_entry["dataset_version"])
    df = pd.read_parquet(parquet_path_from_manifest(manifest_path))
    train_df, val_df, test_df = temporal_split(
        df, model_config["split"], config_path=registry_entry["config_path"]
    )

    fitted_transforms = load_fitted_transforms(manifest)
    train_df, val_df, test_df = apply_requires_fit_transforms(
        train_df, val_df, test_df, dataset_config, fitted_transforms
    )

    _, x_val, _, _ = prepare_features(train_df, val_df, test_df, feature_list)
    if list(x_val.columns) != feature_list:
        x_val = x_val[feature_list]
    return x_val, feature_list


def sample_rows(x: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(x):
        return x.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(x), size=sample_size, replace=False)
    return x.iloc[indices].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP analysis on the validation split for a frozen model version."
    )
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="evals/figures")
    args = parser.parse_args()

    registry = load_registry()
    entry = get_registry_entry(registry, args.version)
    model_config = load_json(repo_path(entry["config_path"]))
    objective = model_config.get("lgbm_params", {}).get("objective", "binary")

    x_val, feature_order = load_validation_features(entry)
    x_sample = sample_rows(x_val, args.sample_size, args.seed)

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    if "objective" not in model.params:
        model.params["objective"] = objective

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, x_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_v{args.version}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_bar_v{args.version}.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    print(
        f"Top 20 features by mean |SHAP| "
        f"(v{args.version}, validation split, n={len(x_sample)}, seed={args.seed}):"
    )
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_order[i]:<30s}  {mean_abs[i]:.6f}")

    top_features = [
        {
            "rank": rank,
            "feature": feature_order[i],
            "mean_abs_shap": float(mean_abs[i]),
        }
        for rank, i in enumerate(top_idx, 1)
    ]
    top_path = output_dir / f"shap_top20_v{args.version}.json"
    top_path.write_text(json.dumps(top_features, indent=2) + "\n")

    print(f"\nFigures saved to {output_dir}/")
    print(f"Top-20 JSON: {top_path}")


if __name__ == "__main__":
    main()
