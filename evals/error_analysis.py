"""Error analysis for a frozen model on the validation split.

Produces confidence-band, ProductCD, and temporal-third tables used by
docs/final_evaluation.md.

Usage:
    python evals/error_analysis.py --version 9
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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


def get_registry_entry(version: int) -> dict:
    registry = load_json(REGISTRY_PATH)
    matches = [entry for entry in registry if entry["version"] == version]
    if not matches:
        raise ValueError(f"Version {version} not found")
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


def apply_fitted(
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
            raise ValueError(f"Transformation {name!r} is not a class")
        instance = obj()
        instance.fit(train_df, params)
        train_df = instance.transform(train_df, params)
        val_df = instance.transform(val_df, params)
        test_df = instance.transform(test_df, params)
    return train_df, val_df, test_df


def load_scored_validation(version: int) -> pd.DataFrame:
    entry = get_registry_entry(version)
    model_config = load_json(repo_path(entry["config_path"]))
    manifest = load_json(repo_path(entry["manifest_path"]))
    dataset_config = load_json(repo_path(entry["dataset_config_path"]))
    feature_list = list(manifest["feature_order"])

    dataset_manifest_path = find_dataset_manifest(entry["dataset_version"])
    dataset_manifest = load_json(dataset_manifest_path)
    target = dataset_manifest["target_column"]

    df = pd.read_parquet(parquet_path_from_manifest(dataset_manifest_path))
    train_df, val_df, test_df = temporal_split(
        df, model_config["split"], config_path=entry["config_path"]
    )
    fitted = load_fitted_transforms(manifest)
    train_df, val_df, test_df = apply_fitted(
        train_df, val_df, test_df, dataset_config, fitted
    )
    _, x_val, _, _ = prepare_features(train_df, val_df, test_df, feature_list)

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    scored = val_df.copy()
    scored["prob"] = model.predict(x_val)
    scored["_target"] = scored[target]
    return scored


def confidence_bands(val: pd.DataFrame) -> list[dict]:
    bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    rows = []
    for lo, hi in bands:
        if hi >= 1.0:
            mask = (val["prob"] >= lo) & (val["prob"] <= hi)
        else:
            mask = (val["prob"] >= lo) & (val["prob"] < hi)
        subset = val[mask]
        n_txns = int(len(subset))
        pred_fraud = int((subset["prob"] >= 0.5).sum()) if hi > 0.5 or lo >= 0.5 else 0
        actual_fraud = int(subset["_target"].sum())
        # Precision within band = fraction of band that is actually fraud
        precision = (actual_fraud / n_txns) if n_txns > 0 else 0.0
        rows.append(
            {
                "band": f"{lo:.1f}–{hi:.1f}",
                "transactions": n_txns,
                "predicted_fraud_ge_0_5": pred_fraud,
                "actual_fraud": actual_fraud,
                "precision": precision,
            }
        )
    return rows


def by_product(val: pd.DataFrame) -> list[dict]:
    rows = []
    if "ProductCD" not in val.columns:
        return rows
    for pc in sorted(val["ProductCD"].dropna().astype(str).unique()):
        subset = val[val["ProductCD"].astype(str) == pc]
        n_txns = int(len(subset))
        n_fraud = int(subset["_target"].sum())
        rate = n_fraud / n_txns if n_txns else 0.0
        if 0 < n_fraud < n_txns:
            auc = float(average_precision_score(subset["_target"], subset["prob"]))
        else:
            auc = None
        rows.append(
            {
                "ProductCD": pc,
                "transactions": n_txns,
                "fraud": n_fraud,
                "fraud_rate": rate,
                "auc_pr": auc,
            }
        )
    return rows


def temporal_thirds(val: pd.DataFrame) -> list[dict]:
    ordered = val.sort_values("TransactionDT").reset_index(drop=True)
    third = len(ordered) // 3
    slices = [
        ("Early third", ordered.iloc[:third]),
        ("Middle third", ordered.iloc[third : 2 * third]),
        ("Late third", ordered.iloc[2 * third :]),
    ]
    rows = []
    for name, s in slices:
        n_txns = int(len(s))
        n_fraud = int(s["_target"].sum())
        rate = n_fraud / n_txns if n_txns else 0.0
        if 0 < n_fraud < n_txns:
            auc = float(average_precision_score(s["_target"], s["prob"]))
        else:
            auc = None
        rows.append(
            {
                "period": name,
                "transactions": n_txns,
                "fraud": n_fraud,
                "fraud_rate": rate,
                "auc_pr": auc,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument(
        "--output",
        type=str,
        default="evals/figures/error_analysis_v9.json",
    )
    args = parser.parse_args()

    val = load_scored_validation(args.version)
    result = {
        "version": args.version,
        "split": "validation",
        "n_rows": int(len(val)),
        "confidence_bands": confidence_bands(val),
        "by_product": by_product(val),
        "temporal_thirds": temporal_thirds(val),
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("Confidence bands (validation)")
    print(f"{'Band':<12} {'Txns':>8} {'Pred≥0.5':>10} {'Actual':>8} {'Prec':>8}")
    for r in result["confidence_bands"]:
        print(
            f"{r['band']:<12} {r['transactions']:>8} "
            f"{r['predicted_fraud_ge_0_5']:>10} {r['actual_fraud']:>8} "
            f"{r['precision']:>8.4f}"
        )

    print("\nBy ProductCD (validation)")
    print(f"{'ProductCD':<12} {'Txns':>8} {'Fraud':>8} {'Rate':>8} {'AUC-PR':>8}")
    for r in result["by_product"]:
        auc = f"{r['auc_pr']:.4f}" if r["auc_pr"] is not None else "N/A"
        print(
            f"{r['ProductCD']:<12} {r['transactions']:>8} {r['fraud']:>8} "
            f"{r['fraud_rate']:>8.4f} {auc:>8}"
        )

    print("\nTemporal thirds (validation)")
    print(f"{'Period':<15} {'Txns':>8} {'Fraud':>8} {'Rate':>8} {'AUC-PR':>8}")
    for r in result["temporal_thirds"]:
        auc = f"{r['auc_pr']:.4f}" if r["auc_pr"] is not None else "N/A"
        print(
            f"{r['period']:<15} {r['transactions']:>8} {r['fraud']:>8} "
            f"{r['fraud_rate']:>8.4f} {auc:>8}"
        )

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
