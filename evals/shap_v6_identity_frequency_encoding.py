"""SHAP attribution for lgbm_v6 on the validation split.

Usage:
    python evals/shap_v6_identity_frequency_encoding.py
"""

from __future__ import annotations

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
SCRIPTS_DIR = REPO_ROOT / "scripts"
JSON_PATH = REPO_ROOT / "evals" / "shap" / "lgbm_v6_identity_frequency_encoding_shap.json"
FIGURE_PATH = REPO_ROOT / "evals" / "figures" / "shap_v6_identity_frequency_encoding.png"
SAMPLE_SIZE = 5000
TOP_N = 20
RANDOM_STATE = 42
TARGET_FEATURES = ["DeviceInfo_freq", "id_30_freq", "id_31_freq"]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    collect_all_transformations,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    prepare_features,
    repo_path,
    repo_relative_path,
    temporal_split,
)


def load_fitted_transforms(manifest: dict) -> dict[str, object]:
    fitted_path = manifest.get("fitted_transforms_path")
    if not fitted_path:
        return {}
    loaded = pickle.loads(repo_path(fitted_path).read_bytes())
    if not isinstance(loaded, dict):
        raise TypeError("Expected dict in fitted transforms file")
    return loaded


def apply_fitted_transforms(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue
        encoder = fitted_transforms[transform["name"]]
        params = transform.get("params", {})
        train_df = encoder.transform(train_df, params)
        val_df = encoder.transform(val_df, params)
        test_df = encoder.transform(test_df, params)
    return train_df, val_df, test_df


def load_val_features() -> tuple[pd.DataFrame, list[str], Path, dict]:
    registry = load_json(REPO_ROOT / "models" / "registry.json")
    entry = next(item for item in registry if item["version"] == 6)
    model_config = load_json(repo_path(entry["config_path"]))
    manifest = load_json(repo_path(entry["manifest_path"]))
    dataset_config = load_json(repo_path(entry["dataset_config_path"]))
    feature_list = list(manifest["feature_order"])

    dataset_manifest_path = find_dataset_manifest(entry["dataset_version"])
    df = pd.read_parquet(parquet_path_from_manifest(dataset_manifest_path))
    train_df, val_df, test_df = temporal_split(
        df, model_config["split"], config_path=entry["config_path"]
    )
    train_df, val_df, test_df = apply_fitted_transforms(
        train_df,
        val_df,
        test_df,
        dataset_config,
        load_fitted_transforms(manifest),
    )
    _, x_val, _, _ = prepare_features(train_df, val_df, test_df, feature_list)
    return x_val[feature_list], feature_list, repo_path(entry["model_path"]), entry


def to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if str(out[col].dtype) in {"category", "object", "string"}:
            out[col] = pd.Categorical(out[col]).codes
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    x_val, feature_names, model_path, entry = load_val_features()
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(SAMPLE_SIZE, len(x_val))
    indices = rng.choice(len(x_val), size=n_sample, replace=False)
    x_sample = x_val.iloc[indices].reset_index(drop=True)

    booster = lgb.Booster(model_file=str(model_path))
    if "objective" not in booster.params:
        booster.params["objective"] = "binary"

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = np.asarray(shap_values)

    mean_abs = np.mean(np.abs(values), axis=0)
    ranking = np.argsort(mean_abs)[::-1]
    name_to_rank = {feature_names[i]: rank for rank, i in enumerate(ranking, start=1)}

    print(f"SHAP v6 on val split: sampled {n_sample:,} of {len(x_val):,} rows")
    print(f"Top {TOP_N} features by mean |SHAP|:")
    top_features = []
    for rank, index in enumerate(ranking[:TOP_N], start=1):
        name = feature_names[index]
        marker = "  <-- identity freq" if name in TARGET_FEATURES else ""
        print(f"  {rank:2d}. {name:30s} {mean_abs[index]:.6f}{marker}")
        top_features.append(
            {
                "rank": rank,
                "feature": name,
                "mean_abs_shap": float(mean_abs[index]),
            }
        )

    print("\nIdentity frequency feature ranks:")
    for name in TARGET_FEATURES:
        rank = name_to_rank.get(name)
        if rank is None:
            print(f"  {name:18s}  NOT IN MODEL")
            continue
        in_top = " TOP20" if rank <= TOP_N else ""
        print(
            f"  {name:18s}  rank={rank:3d}  "
            f"mean|SHAP|={mean_abs[feature_names.index(name)]:.6f}{in_top}"
        )

    in_top = [name for name in TARGET_FEATURES if name_to_rank.get(name, 10**9) <= TOP_N]
    print(
        f"\nIdentity freq in top {TOP_N}: "
        f"{in_top if in_top else 'NONE — lift may not be from the hypothesized features'}"
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "version": 6,
        "dataset_version": 4,
        "description": entry["description"],
        "model_path": entry["model_path"],
        "manifest_path": entry["manifest_path"],
        "config_path": entry["config_path"],
        "timestamp": timestamp,
        "sample": {
            "split": "val",
            "n_total": int(len(x_val)),
            "n_sampled": int(n_sample),
            "random_state": RANDOM_STATE,
        },
        "top_n_reported": TOP_N,
        "top_features": top_features,
        "target_features": [
            {
                "feature": name,
                "rank": name_to_rank.get(name),
                "mean_abs_shap": (
                    float(mean_abs[feature_names.index(name)])
                    if name in feature_names
                    else None
                ),
            }
            for name in TARGET_FEATURES
        ],
        "figure_path": repo_relative_path(FIGURE_PATH),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(result, indent=2) + "\n")

    x_plot = to_numeric(x_sample)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values,
        x_plot,
        feature_names=feature_names,
        max_display=TOP_N,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {JSON_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved beeswarm to {FIGURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
