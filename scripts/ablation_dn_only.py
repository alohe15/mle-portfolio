"""Leave-one-group-out ablation for the Dn-only experiment.

Loads the trained model and test set using the same temporal split and
requires_fit transforms as train.py / bootstrap_significance.py, NaN-masks
feature groups, re-predicts, and reports AUC-PR for each ablation.

Usage:
    python scripts/ablation_dn_only.py --version 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVALS_DIR = REPO_ROOT / "evals"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from bootstrap_significance import (
    apply_requires_fit_transforms,
    get_registry_entry,
    load_fitted_transforms,
    load_json,
    load_registry,
    load_version_splits,
)
from train import prepare_features, repo_path


def _dn_columns(columns) -> list[str]:
    return [
        name
        for name in columns
        if name.endswith("n")
        and name[:-1].startswith("D")
        and name[:-1][1:].isdigit()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()

    registry = load_registry()
    entry = get_registry_entry(registry, args.version)
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
    y_test = test_df[target_column].to_numpy()

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    y_pred_full = model.predict(x_test)
    full_ap = average_precision_score(y_test, y_pred_full)

    dn_cols = [c for c in _dn_columns(x_test.columns) if c in x_test.columns]
    x_no_dn = x_test.copy()
    x_no_dn[dn_cols] = np.nan
    no_dn_ap = average_precision_score(y_test, model.predict(x_no_dn))

    raw_d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in x_test.columns]
    x_no_raw = x_test.copy()
    x_no_raw[raw_d_cols] = np.nan
    no_raw_d_ap = average_precision_score(y_test, model.predict(x_no_raw))

    print(f"\nAblation results for lgbm_v{args.version}:")
    print(f"  Full model AUC-PR:          {full_ap:.4f}")
    print(f"  Drop D1n-D15n AUC-PR:       {no_dn_ap:.4f}  (delta: {no_dn_ap - full_ap:+.4f})")
    print(f"  Drop raw D1-D15 AUC-PR:     {no_raw_d_ap:.4f}  (delta: {no_raw_d_ap - full_ap:+.4f})")


if __name__ == "__main__":
    main()
