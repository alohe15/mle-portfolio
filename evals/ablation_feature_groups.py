"""Feature-group ablation for a frozen model (diagnostic only).

Removes one named feature group, retrains with identical hyperparameters
(n_estimators capped at 2000 for speed), evaluates on VALIDATION only.
Does NOT write model artifacts to models/ or update the registry.

Usage:
    python evals/ablation_feature_groups.py --version 9 --drop-group dn_columns
    python evals/ablation_feature_groups.py --version 9 --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
DEFAULT_N_ESTIMATORS = 2000

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    apply_requires_fit_transforms,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    prepare_features,
    repo_path,
    temporal_split,
)

FEATURE_GROUPS = {
    "dn_columns": lambda cols: [
        c
        for c in cols
        if c.endswith("n")
        and any(c.startswith(f"D{i}") for i in range(1, 16))
        and c[1:-1].isdigit()
    ],
    "v_features": lambda cols: [c for c in cols if c.startswith("V")],
    "card_features": lambda cols: [c for c in cols if c.startswith("card")],
    "email_features": lambda cols: [
        c for c in cols if "email" in c.lower() or c == "email_match"
    ],
    "addr_features": lambda cols: [
        c for c in cols if c.startswith("addr") or c.startswith("dist")
    ],
    "id_features": lambda cols: [
        c
        for c in cols
        if c.startswith("id_")
        or c.startswith("DeviceType")
        or c.startswith("DeviceInfo")
    ],
}


def get_registry_entry(version: int) -> dict:
    registry = load_json(REGISTRY_PATH)
    matches = [entry for entry in registry if entry["version"] == version]
    if not matches:
        raise ValueError(f"Version {version} not found in models/registry.json")
    return matches[0]


def resolve_drop_columns(group: str, feature_list: list[str]) -> list[str]:
    if group not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown group {group!r}. Choose from: {sorted(FEATURE_GROUPS)}"
        )
    return FEATURE_GROUPS[group](feature_list)


def train_and_eval_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
    target_column: str,
    lgbm_params: dict,
) -> tuple[float, int]:
    x_train, x_val, _, cat_cols = prepare_features(
        train_df, val_df, test_df, feature_list
    )
    y_train = train_df[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()

    params = dict(lgbm_params)
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 200))
    eval_metric = params.pop("metric", "average_precision")
    params["n_estimators"] = DEFAULT_N_ESTIMATORS
    configured_spw = params.pop("scale_pos_weight", None)
    train_fraud_rate = float(y_train.mean())
    if configured_spw is None:
        scale_pos_weight = (1.0 - train_fraud_rate) / train_fraud_rate
    else:
        scale_pos_weight = float(configured_spw)

    model = lgb.LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)
    model.fit(
        x_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(x_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    y_val_score = model.predict_proba(x_val)[:, 1]
    val_auc_pr = float(average_precision_score(y_val, y_val_score))
    best_iteration = int(model.best_iteration_ or DEFAULT_N_ESTIMATORS)
    return val_auc_pr, best_iteration


def run_ablation(version: int, groups: list[str], output_dir: Path) -> list[dict]:
    entry = get_registry_entry(version)
    model_config = load_json(repo_path(entry["config_path"]))
    manifest = load_json(repo_path(entry["manifest_path"]))
    dataset_config = load_json(repo_path(entry["dataset_config_path"]))
    metrics_doc = load_json(repo_path(entry["metrics_path"]))
    baseline_val = float(metrics_doc["metrics"]["val_auc_pr"])

    feature_list = list(manifest["feature_order"])
    dataset_manifest_path = find_dataset_manifest(entry["dataset_version"])
    dataset_manifest = load_json(dataset_manifest_path)
    target_column = dataset_manifest["target_column"]

    df = pd.read_parquet(parquet_path_from_manifest(dataset_manifest_path))
    train_df, val_df, test_df = temporal_split(
        df, model_config["split"], config_path=entry["config_path"]
    )
    train_df, val_df, test_df, _ = apply_requires_fit_transforms(
        train_df, val_df, test_df, dataset_config
    )

    results: list[dict] = [
        {
            "group": "(full model)",
            "removed": 0,
            "remaining": len(feature_list),
            "val_auc_pr": baseline_val,
            "delta": 0.0,
            "best_iteration": metrics_doc["metrics"].get("best_iteration"),
            "columns_removed": [],
            "note": "Frozen v9 validation AUC-PR (full n_estimators); not a 2000-tree retrain",
        }
    ]

    lgbm_params = dict(model_config["lgbm_params"])

    for group in groups:
        drop_cols = resolve_drop_columns(group, feature_list)
        remaining = [c for c in feature_list if c not in set(drop_cols)]
        if not remaining:
            raise ValueError(f"Dropping group {group!r} removes all features")

        print(f"\n=== Ablating {group}: removing {len(drop_cols)} cols, "
              f"{len(remaining)} remaining ===")
        val_auc_pr, best_iteration = train_and_eval_val(
            train_df,
            val_df,
            test_df,
            remaining,
            target_column,
            lgbm_params,
        )
        delta = val_auc_pr - baseline_val
        row = {
            "group": group,
            "removed": len(drop_cols),
            "remaining": len(remaining),
            "val_auc_pr": val_auc_pr,
            "delta": delta,
            "best_iteration": best_iteration,
            "columns_removed": drop_cols,
            "n_estimators_cap": DEFAULT_N_ESTIMATORS,
        }
        results.append(row)
        print(
            f"{group}: val_auc_pr={val_auc_pr:.4f}  delta={delta:+.4f}  "
            f"best_iteration={best_iteration}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ablation_results_v{version}.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    # Bar chart of deltas (exclude full-model baseline row for bars)
    ablation_rows = [r for r in results if r["group"] != "(full model)"]
    if ablation_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["group"] for r in ablation_rows]
        deltas = [r["delta"] for r in ablation_rows]
        colors = ["#c0392b" if d < 0 else "#27ae60" for d in deltas]
        ax.barh(names[::-1], deltas[::-1], color=colors[::-1])
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta val AUC-PR vs frozen v9")
        ax.set_title(
            f"Feature-group ablation — lgbm_v{version} "
            f"(diagnostic, n_estimators≤{DEFAULT_N_ESTIMATORS}, validation)"
        )
        fig.tight_layout()
        png_path = output_dir / f"ablation_results_v{version}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {png_path}")

    print(f"Wrote {json_path}")
    print_table(results, version)
    return results


def print_table(results: list[dict], version: int) -> None:
    print()
    print(
        f"Feature Group Ablation — lgbm_v{version} "
        f"(diagnostic, validation split, n_estimators≤{DEFAULT_N_ESTIMATORS})"
    )
    print(
        f"{'Group':<20} {'Removed':>8} {'Remaining':>10} "
        f"{'Val AUC-PR':>11} {'Delta vs v9':>12}"
    )
    print("─" * 64)
    for r in results:
        print(
            f"{r['group']:<20} {r['removed']:>8} {r['remaining']:>10} "
            f"{r['val_auc_pr']:>11.4f} {r['delta']:>+12.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic feature-group ablation for a frozen model version."
    )
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--drop-group",
        type=str,
        choices=sorted(FEATURE_GROUPS),
        help="Single feature group to drop.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run ablation for every defined feature group.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/figures",
        help="Directory for JSON + PNG outputs (default: evals/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        groups = list(FEATURE_GROUPS.keys())
    elif args.drop_group:
        groups = [args.drop_group]
    else:
        raise SystemExit("Specify --drop-group GROUP or --all")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    run_ablation(args.version, groups, output_dir)


if __name__ == "__main__":
    main()
