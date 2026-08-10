"""TreeSHAP + LightGBM gain importance for a registered model version.

Usage:
    python evals/shap_analysis.py --version 6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVALS_DIR = REPO_ROOT / "evals"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
SHAP_SAMPLE_SIZE = 5000
SHAP_RANDOM_STATE = 42
TOP_N = 30

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from bootstrap_significance import (  # noqa: E402
    apply_requires_fit_transforms,
    get_registry_entry,
    load_fitted_transforms,
    load_json,
    load_registry,
    load_version_splits,
)
from train import prepare_features, repo_path  # noqa: E402


def _dn_columns(feature_names: list[str]) -> list[str]:
    return [
        name
        for name in feature_names
        if name.endswith("n")
        and name[:-1].startswith("D")
        and name[:-1][1:].isdigit()
    ]


def _mean_abs_shap(shap_values: np.ndarray | list) -> np.ndarray:
    values = shap_values
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return np.abs(values).mean(axis=0)


def run_shap_analysis(version: int) -> None:
    registry = load_registry()
    entry = get_registry_entry(registry, version)
    model_config = load_json(repo_path(entry["config_path"]))
    manifest = load_json(repo_path(entry["manifest_path"]))
    dataset_config = load_json(repo_path(entry["dataset_config_path"]))
    feature_list = list(manifest["feature_order"])

    train_df, test_df, target_column = load_version_splits(
        entry, model_config["split"]
    )
    fitted_transforms = load_fitted_transforms(manifest)
    train_df, test_df = apply_requires_fit_transforms(
        train_df.copy(), test_df.copy(), dataset_config, fitted_transforms
    )
    _, x_test, _ = prepare_features(train_df, test_df, feature_list)

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    # Native .txt boosters omit objective; shap 0.48 requires it for binary models.
    if "objective" not in model.params:
        model.params["objective"] = "binary"
    gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    gain_pct = 100.0 * gain / gain.sum() if gain.sum() else gain
    gain_order = np.argsort(-gain)

    print(f"SHAP analysis for lgbm_v{version}", flush=True)
    print("=" * 72, flush=True)
    print(f"Test rows: {len(x_test):,} | features: {len(feature_list)}", flush=True)
    print(flush=True)

    print(f"LightGBM gain importance (top {TOP_N})")
    print("-" * 72)
    print(f"{'Rank':<6}{'Feature':<32}{'Gain %':>10}{'Gain':>16}")
    for rank, idx in enumerate(gain_order[:TOP_N], start=1):
        print(
            f"{rank:<6}{feature_list[idx]:<32}{gain_pct[idx]:>10.4f}{gain[idx]:>16.1f}"
        )

    rng = np.random.default_rng(SHAP_RANDOM_STATE)
    n_sample = min(SHAP_SAMPLE_SIZE, len(x_test))
    sample_idx = rng.choice(len(x_test), size=n_sample, replace=False)
    x_sample = x_test.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    mean_abs = _mean_abs_shap(shap_values)
    shap_order = np.argsort(-mean_abs)

    print()
    print(f"TreeSHAP top {TOP_N} (mean |SHAP|, n={n_sample}, seed={SHAP_RANDOM_STATE})")
    print("-" * 72)
    print(f"{'Rank':<6}{'Feature':<32}{'Mean |SHAP|':>14}")
    top_features = []
    for rank, idx in enumerate(shap_order[:TOP_N], start=1):
        print(f"{rank:<6}{feature_list[idx]:<32}{mean_abs[idx]:>14.6f}")
        top_features.append(
            {
                "rank": rank,
                "feature": feature_list[idx],
                "mean_abs_shap": float(mean_abs[idx]),
            }
        )

    print()
    print("D1n–D15n ranks (SHAP and gain)")
    print("-" * 72)
    shap_rank = {feature_list[i]: rank for rank, i in enumerate(shap_order, start=1)}
    gain_rank = {feature_list[i]: rank for rank, i in enumerate(gain_order, start=1)}
    dn_cols = _dn_columns(feature_list)
    print(f"{'Feature':<10}{'SHAP rank':>12}{'Mean |SHAP|':>14}{'Gain rank':>12}{'Gain %':>10}")
    dn_records = []
    for col in sorted(dn_cols, key=lambda c: int(c[1:-1])):
        idx = feature_list.index(col)
        print(
            f"{col:<10}{shap_rank[col]:>12}{mean_abs[idx]:>14.6f}"
            f"{gain_rank[col]:>12}{gain_pct[idx]:>10.4f}"
        )
        dn_records.append(
            {
                "feature": col,
                "shap_rank": shap_rank[col],
                "mean_abs_shap": float(mean_abs[idx]),
                "gain_rank": gain_rank[col],
                "gain_pct": float(gain_pct[idx]),
            }
        )

    v5_shap_path = REPO_ROOT / "evals" / "shap" / "lgbm_v5_lower_learning_rate_shap.json"
    if v5_shap_path.exists():
        v5_shap = load_json(v5_shap_path)
        v5_ranks = {item["feature"]: item["rank"] for item in v5_shap["top_features"]}
        print()
        print("Crowding-out vs v5 SHAP top 20")
        print("-" * 72)
        print(f"{'Feature':<24}{'v6 rank':>10}{'v5 rank':>10}{'Δ Rank':>10}")
        for item in top_features[:20]:
            old = v5_ranks.get(item["feature"])
            old_text = str(old) if old is not None else "—"
            delta = f"{old - item['rank']:+d}" if old is not None else "—"
            print(f"{item['feature']:<24}{item['rank']:>10}{old_text:>10}{delta:>10}")

    out_dir = REPO_ROOT / "evals" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lgbm_v{version}_dn_only_shap.json"
    payload = {
        "version": version,
        "dataset_version": entry["dataset_version"],
        "description": entry["description"],
        "model_path": entry["model_path"],
        "manifest_path": entry["manifest_path"],
        "config_path": entry["config_path"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": {
            "split": "test",
            "n_total": int(len(x_test)),
            "n_sampled": int(n_sample),
            "random_state": SHAP_RANDOM_STATE,
        },
        "top_n_reported": TOP_N,
        "top_features": top_features,
        "dn_features": dn_records,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    run_shap_analysis(args.version)


if __name__ == "__main__":
    main()
