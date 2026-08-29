"""Probability calibration for a frozen model version.

Fits isotonic regression with 5-fold cross-fitting on the validation split
to avoid calibrating on the same data used for evaluation. Refits a single
deployment calibrator on the full validation split for serialization.

Usage:
    python evals/calibrate.py --version 9 --output-dir evals/figures

§2g compliance: calibrator is fit on validation labels only, never test.
§2f compliance: follows fit/transform contract, serialized for inference.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

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
        raise ValueError(f"Version {version} not found in models/registry.json")
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


def load_model_and_validation(version: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load model, score validation split, return scores and labels."""
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

    fitted_transforms = load_fitted_transforms(manifest)
    train_df, val_df, test_df = apply_fitted_transforms(
        train_df, val_df, test_df, dataset_config, fitted_transforms
    )

    _, x_val, _, _ = prepare_features(train_df, val_df, test_df, feature_list)
    if list(x_val.columns) != feature_list:
        x_val = x_val[feature_list]

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    raw_scores = model.predict(x_val)
    labels = val_df[target].values
    amounts = val_df["TransactionAmt"].values

    return raw_scores, labels, amounts, entry


def plot_calibration_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Plot predicted probability vs observed fraud rate."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers: list[float] = []
    observed_rates: list[float] = []
    counts: list[int] = []

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            observed_rates.append(float(labels[mask].mean()))
            counts.append(n_in_bin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(bin_centers, observed_rates, "bo-", label="Model")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed fraud rate")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_centers, counts, width=1.0 / n_bins * 0.8, alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Score distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return bin_centers, observed_rates, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="evals/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores, labels, amounts, _entry = load_model_and_validation(args.version)

    print("Raw score stats:")
    print(f"  mean:  {raw_scores.mean():.4f}")
    print(f"  max:   {raw_scores.max():.4f}")
    print(f"  min:   {raw_scores.min():.4f}")
    print(f"  Actual fraud rate: {labels.mean():.4f}")
    print()

    print("BEFORE calibration:")
    centers, rates, counts = plot_calibration_curve(
        raw_scores,
        labels,
        f"Before calibration (v{args.version})",
        output_dir / f"calibration_before_v{args.version}.png",
    )
    for c, r, n in zip(centers, rates, counts):
        print(f"  bin {c:.2f}: predicted={c:.4f} observed={r:.4f} n={n}")

    print(f"\nCross-fitting isotonic regression ({args.n_folds} folds)...")
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    calibrated_scores = np.zeros_like(raw_scores)

    for fold, (fit_idx, transform_idx) in enumerate(kf.split(raw_scores)):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(raw_scores[fit_idx], labels[fit_idx])
        calibrated_scores[transform_idx] = iso.transform(raw_scores[transform_idx])
        print(f"  Fold {fold + 1}: fit on {len(fit_idx)}, transform {len(transform_idx)}")

    print("\nCalibrated score stats (cross-fitted):")
    print(f"  mean:  {calibrated_scores.mean():.4f}  (fraud rate: {labels.mean():.4f})")
    print(f"  max:   {calibrated_scores.max():.4f}")
    print(f"  min:   {calibrated_scores.min():.4f}")

    print("\nAFTER calibration:")
    centers, rates, counts = plot_calibration_curve(
        calibrated_scores,
        labels,
        f"After calibration — cross-fitted (v{args.version})",
        output_dir / f"calibration_after_v{args.version}.png",
    )
    for c, r, n in zip(centers, rates, counts):
        print(f"  bin {c:.2f}: predicted={c:.4f} observed={r:.4f} n={n}")

    print("\nFitting deployment calibrator on full validation split...")
    deploy_iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    deploy_iso.fit(raw_scores, labels)

    calibrator_path = REPO_ROOT / f"models/lgbm_v{args.version}_calibrator.pkl"
    calibrator_path.write_bytes(pickle.dumps(deploy_iso))
    print(f"Saved deployment calibrator to {calibrator_path.relative_to(REPO_ROOT)}")

    scores_path = output_dir / f"calibrated_scores_v{args.version}.npz"
    np.savez(
        scores_path,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        labels=labels,
        amounts=amounts,
    )
    print(f"Saved cross-fitted scores to {scores_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
