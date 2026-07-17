"""Bootstrap significance test for metric differences between model versions.

Usage:
    python evals/bootstrap_significance.py --baseline 1 --candidate 2
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
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
SCRIPTS_DIR = REPO_ROOT / "scripts"
N_BOOTSTRAP = 500
RANDOM_STATE = 42

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (
    collect_all_transformations,
    evaluate_at_flag_rate,
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
    test_df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue

        name = transform["name"]
        params = transform.get("params", {})
        if name in fitted_transforms:
            encoder = fitted_transforms[name]
            train_df = encoder.transform(train_df, params)
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
        test_df = instance.transform(test_df, params)

    return train_df, test_df


def load_version_splits(registry_entry: dict, split_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dataset_version = registry_entry["dataset_version"]
    manifest_path = find_dataset_manifest(dataset_version)
    dataset_manifest = load_json(manifest_path)
    df = pd.read_parquet(parquet_path_from_manifest(manifest_path))
    train_df, test_df = temporal_split(df, split_config)
    return train_df, test_df, dataset_manifest["target_column"]


def align_test_frames(
    baseline_test: pd.DataFrame,
    candidate_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(baseline_test) != len(candidate_test):
        raise ValueError(
            "Baseline and candidate test sets differ in size: "
            f"{len(baseline_test)} vs {len(candidate_test)}"
        )

    if "TransactionID" not in baseline_test.columns or "TransactionID" not in candidate_test.columns:
        return baseline_test.reset_index(drop=True), candidate_test.reset_index(drop=True)

    baseline_aligned = baseline_test.sort_values("TransactionID").reset_index(drop=True)
    candidate_aligned = candidate_test.sort_values("TransactionID").reset_index(drop=True)
    if not baseline_aligned["TransactionID"].equals(candidate_aligned["TransactionID"]):
        raise ValueError("Baseline and candidate test sets have different TransactionID values")
    return baseline_aligned, candidate_aligned


def score_version(
    registry_entry: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> np.ndarray:
    manifest = load_json(repo_path(registry_entry["manifest_path"]))
    dataset_config = load_json(repo_path(registry_entry["dataset_config_path"]))
    feature_list = manifest["feature_order"]
    fitted_transforms = load_fitted_transforms(manifest)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df, test_df = apply_requires_fit_transforms(
        train_df, test_df, dataset_config, fitted_transforms
    )

    _, x_test, _ = prepare_features(train_df, test_df, feature_list)
    model = lgb.Booster(model_file=str(repo_path(registry_entry["model_path"])))
    return model.predict(x_test)


def bootstrap_auc_pr_differences(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    n_iterations: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_rows = len(y_true)
    baseline_values = np.empty(n_iterations, dtype=float)
    candidate_values = np.empty(n_iterations, dtype=float)
    differences = np.empty(n_iterations, dtype=float)

    for iteration in range(n_iterations):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        y_sample = y_true[sample_idx]
        baseline_sample = baseline_scores[sample_idx]
        candidate_sample = candidate_scores[sample_idx]

        baseline_auc = average_precision_score(y_sample, baseline_sample)
        candidate_auc = average_precision_score(y_sample, candidate_sample)
        baseline_values[iteration] = baseline_auc
        candidate_values[iteration] = candidate_auc
        differences[iteration] = candidate_auc - baseline_auc

    return baseline_values, candidate_values, differences


def bootstrap_recall_at_budget_differences(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    flag_rate: float,
    n_iterations: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap recall at a fixed prevalence-matched flag rate (training fraud rate)."""
    rng = np.random.default_rng(random_state)
    n_rows = len(y_true)
    baseline_values = np.empty(n_iterations, dtype=float)
    candidate_values = np.empty(n_iterations, dtype=float)
    differences = np.empty(n_iterations, dtype=float)

    for iteration in range(n_iterations):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        y_sample = y_true[sample_idx]
        baseline_sample = baseline_scores[sample_idx]
        candidate_sample = candidate_scores[sample_idx]

        baseline_recall = evaluate_at_flag_rate(y_sample, baseline_sample, flag_rate)["recall"]
        candidate_recall = evaluate_at_flag_rate(y_sample, candidate_sample, flag_rate)["recall"]
        baseline_values[iteration] = baseline_recall
        candidate_values[iteration] = candidate_recall
        differences[iteration] = candidate_recall - baseline_recall

    return baseline_values, candidate_values, differences


def summarize_bootstrap_differences(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    differences: np.ndarray,
) -> dict[str, float | str]:
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    return {
        "mean_baseline": float(np.mean(baseline_values)),
        "mean_candidate": float(np.mean(candidate_values)),
        "mean_difference": float(np.mean(differences)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "verdict": classify_result(float(ci_low), float(ci_high)),
    }


def classify_result(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "statistically significant improvement"
    if ci_high < 0:
        return "regression"
    return "inconclusive"


def run_significance_test(baseline_version: int, candidate_version: int) -> None:
    registry = load_registry()
    baseline_entry = get_registry_entry(registry, baseline_version)
    candidate_entry = get_registry_entry(registry, candidate_version)

    candidate_config = load_json(repo_path(candidate_entry["config_path"]))
    split_config = candidate_config["split"]

    baseline_train, baseline_test, baseline_target = load_version_splits(
        baseline_entry, split_config
    )
    candidate_train, candidate_test, candidate_target = load_version_splits(
        candidate_entry, split_config
    )
    baseline_test, candidate_test = align_test_frames(baseline_test, candidate_test)

    y_test = candidate_test[candidate_target].to_numpy()
    baseline_y = baseline_test[baseline_target].to_numpy()
    if not np.array_equal(y_test, baseline_y):
        raise ValueError("Baseline and candidate test targets do not match after alignment")

    baseline_scores = score_version(
        baseline_entry, baseline_train, baseline_test, baseline_target
    )
    candidate_scores = score_version(
        candidate_entry, candidate_train, candidate_test, candidate_target
    )

    train_fraud_rate = float(baseline_train[baseline_target].mean())

    baseline_auc_boot, candidate_auc_boot, auc_differences = bootstrap_auc_pr_differences(
        y_test,
        baseline_scores,
        candidate_scores,
        n_iterations=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )
    auc_summary = summarize_bootstrap_differences(
        baseline_auc_boot, candidate_auc_boot, auc_differences
    )

    baseline_recall_boot, candidate_recall_boot, recall_differences = (
        bootstrap_recall_at_budget_differences(
            y_test,
            baseline_scores,
            candidate_scores,
            flag_rate=train_fraud_rate,
            n_iterations=N_BOOTSTRAP,
            random_state=RANDOM_STATE,
        )
    )
    recall_summary = summarize_bootstrap_differences(
        baseline_recall_boot, candidate_recall_boot, recall_differences
    )

    print("Bootstrap Significance Test")
    print("=" * 64)
    print(f"Baseline version:   v{baseline_version} ({baseline_entry['description']})")
    print(f"Candidate version:  v{candidate_version} ({candidate_entry['description']})")
    print(f"Test rows:          {len(y_test):,}")
    print(f"Bootstrap samples:  {N_BOOTSTRAP}")
    print(f"Recall flag rate:   {train_fraud_rate:.4%} (training-set fraud rate)")
    print()
    print("AUC-PR")
    print("-" * 64)
    print(f"Mean (baseline):          {auc_summary['mean_baseline']:.4f}")
    print(f"Mean (candidate):         {auc_summary['mean_candidate']:.4f}")
    print(f"Mean difference:          {auc_summary['mean_difference']:+.4f}")
    print(
        f"95% CI for difference:    [{auc_summary['ci_low']:+.4f}, {auc_summary['ci_high']:+.4f}]"
    )
    print(f"Result:                   {auc_summary['verdict']}")
    print()
    print("Recall @ budget")
    print("-" * 64)
    print(f"Mean (baseline):          {recall_summary['mean_baseline']:.4f}")
    print(f"Mean (candidate):         {recall_summary['mean_candidate']:.4f}")
    print(f"Mean difference:          {recall_summary['mean_difference']:+.4f}")
    print(
        f"95% CI for difference:    [{recall_summary['ci_low']:+.4f}, {recall_summary['ci_high']:+.4f}]"
    )
    print(f"Result:                   {recall_summary['verdict']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap test for AUC-PR and recall@budget differences between two model versions."
    )
    parser.add_argument(
        "--baseline",
        type=int,
        required=True,
        help="Baseline model version number (e.g. 1)",
    )
    parser.add_argument(
        "--candidate",
        type=int,
        required=True,
        help="Candidate model version number (e.g. 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_significance_test(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
