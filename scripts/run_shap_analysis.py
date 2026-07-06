"""SHAP attribution analysis for every model version in models/registry.json.

Usage:
    python scripts/run_shap_analysis.py
    python scripts/run_shap_analysis.py --version 1
    python scripts/run_shap_analysis.py --version 1 --sample-size 5000 --top-n 20

Reads model paths, manifests, and dataset configs from the registry — never
hardcodes artifact locations. Replays the same temporal split and feature
preparation as scripts/train.py so SHAP values align with training-time inputs.

Outputs per version (committed JSON + figure):
    evals/shap/lgbm_v{N}_{slug}_shap.json
    evals/figures/lgbm_v{N}_{slug}_shap_top{N}.png

See ENGINEERING_STANDARDS.md §8c (SHAP attribution).
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
SHAP_DIR = REPO_ROOT / "evals" / "shap"
FIGURES_DIR = REPO_ROOT / "evals" / "figures"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_TOP_N = 20
DEFAULT_RANDOM_STATE = 42

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (
    artifact_slug,
    collect_all_transformations,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    prepare_features,
    repo_path,
    repo_relative_path,
    resolve_transform,
    temporal_split,
)


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
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
    test_df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue

        name = transform["name"]
        if name in fitted_transforms:
            encoder = fitted_transforms[name]
            train_df = encoder.transform(train_df)
            test_df = encoder.transform(test_df)
            continue

        obj = resolve_transform(transform["function"])
        params = transform.get("params", {})
        if not inspect.isclass(obj):
            raise ValueError(
                f"Transformation {name!r} requires fit/transform but is not a class"
            )
        instance = obj(**params) if params else obj()
        instance.fit(train_df)
        train_df = instance.transform(train_df)
        test_df = instance.transform(test_df)

    return train_df, test_df


def load_test_features(registry_entry: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load and prepare the temporal test split feature matrix for a registry entry."""
    model_config = load_json(repo_path(registry_entry["config_path"]))
    manifest = load_json(repo_path(registry_entry["manifest_path"]))
    dataset_config = load_json(repo_path(registry_entry["dataset_config_path"]))
    feature_list = manifest["feature_order"]

    dataset_version = registry_entry["dataset_version"]
    manifest_path = find_dataset_manifest(dataset_version)
    df = pd.read_parquet(parquet_path_from_manifest(manifest_path))
    train_df, test_df = temporal_split(df, model_config["split"])

    fitted_transforms = load_fitted_transforms(manifest)
    train_df, test_df = apply_requires_fit_transforms(
        train_df, test_df, dataset_config, fitted_transforms
    )

    _, x_test, _ = prepare_features(train_df, test_df, feature_list)
    if list(x_test.columns) != feature_list:
        x_test = x_test[feature_list]
    return x_test, feature_list


def sample_rows(x_test: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(x_test):
        return x_test.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(x_test), size=sample_size, replace=False)
    return x_test.iloc[indices].reset_index(drop=True)


def compute_mean_abs_shap(
    model_path: Path,
    x_sample: pd.DataFrame,
    feature_names: list[str],
    objective: str = "binary",
) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    # LightGBM text models saved via booster_.save_model() omit params that SHAP
    # expects (notably objective). Patch from the model config before explaining.
    if "objective" not in booster.params:
        booster.params["objective"] = objective

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        # Binary classifiers may return [class_0, class_1]; use fraud class.
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP width {values.shape[1]} does not match feature count {len(feature_names)}"
        )
    return np.mean(np.abs(values), axis=0)


def output_paths(registry_entry: dict, model_config: dict, top_n: int) -> tuple[Path, Path]:
    version = registry_entry["version"]
    slug = artifact_slug(model_config)
    base = f"lgbm_v{version}_{slug}"
    json_path = SHAP_DIR / f"{base}_shap.json"
    figure_path = FIGURES_DIR / f"{base}_shap_top{top_n}.png"
    return json_path, figure_path


def save_bar_plot(
    feature_names: list[str],
    mean_abs_shap: np.ndarray,
    figure_path: Path,
    title: str,
    top_n: int,
) -> None:
    ranking = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in ranking]
    top_values = mean_abs_shap[ranking]

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values[::-1], color="#2a6fbb")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_shap_for_entry(
    registry_entry: dict,
    sample_size: int,
    top_n: int,
    random_state: int,
) -> dict:
    version = registry_entry["version"]
    model_config = load_json(repo_path(registry_entry["config_path"]))
    model_path = repo_path(registry_entry["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    x_test, feature_names = load_test_features(registry_entry)
    x_sample = sample_rows(x_test, sample_size, random_state)
    objective = model_config.get("lgbm_params", {}).get("objective", "binary")
    mean_abs_shap = compute_mean_abs_shap(model_path, x_sample, feature_names, objective)

    ranking = np.argsort(mean_abs_shap)[::-1]
    top_features = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[index],
            "mean_abs_shap": float(mean_abs_shap[index]),
        }
        for rank, index in enumerate(ranking[:top_n])
    ]

    json_path, figure_path = output_paths(registry_entry, model_config, top_n)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "version": version,
        "dataset_version": registry_entry["dataset_version"],
        "description": registry_entry["description"],
        "model_path": registry_entry["model_path"],
        "manifest_path": registry_entry["manifest_path"],
        "config_path": registry_entry["config_path"],
        "timestamp": timestamp,
        "sample": {
            "split": "test",
            "n_total": int(len(x_test)),
            "n_sampled": int(len(x_sample)),
            "random_state": random_state,
        },
        "top_n_reported": top_n,
        "top_features": top_features,
        "figure_path": repo_relative_path(figure_path),
    }

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    save_bar_plot(
        feature_names,
        mean_abs_shap,
        figure_path,
        title=f"SHAP top {top_n} — v{version} ({registry_entry['description']})",
        top_n=top_n,
    )

    print(
        f"SHAP v{version}: top feature={top_features[0]['feature']} "
        f"(|SHAP|={top_features[0]['mean_abs_shap']:.4f}) | "
        f"{len(x_sample):,} rows | saved {repo_relative_path(json_path)}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP attribution for all models in models/registry.json."
    )
    parser.add_argument(
        "--version",
        type=int,
        action="append",
        dest="versions",
        help="Model version to analyze (repeatable). Default: all registry versions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Rows sampled from the test split for SHAP (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top features to report and plot (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for test-row sampling (default: {DEFAULT_RANDOM_STATE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry()

    if args.versions:
        entries = [get_registry_entry(registry, version) for version in args.versions]
    else:
        entries = sorted(registry, key=lambda item: item["version"])

    if not entries:
        raise ValueError("No registry entries to analyze")

    for entry in entries:
        run_shap_for_entry(
            entry,
            sample_size=args.sample_size,
            top_n=args.top_n,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
