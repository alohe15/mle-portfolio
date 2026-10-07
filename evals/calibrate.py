"""Isotonic calibration of lgbm_v9 raw scores on the validation split.

Fits out-of-fold calibrated probabilities via 5-fold cross-fitting on
validation, then refits a single deployment calibrator on the full
validation split and saves it to models/lgbm_v9_calibrator.pkl.

Usage:
    python evals/calibrate.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
CALIBRATOR_PATH = REPO_ROOT / "models" / "lgbm_v9_calibrator.pkl"
MODEL_VERSION = 9
N_FOLDS = 5
RANDOM_STATE = 42

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


def get_registry_entry(version: int) -> dict:
    registry = load_json(REGISTRY_PATH)
    matches = [entry for entry in registry if entry["version"] == version]
    if not matches:
        raise ValueError(f"Version {version} not found in models/registry.json")
    if len(matches) > 1:
        raise ValueError(f"Multiple registry entries for version {version}")
    return matches[0]


def load_v9_validation() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load validation rows with raw model scores for the serving v9 model.

    Returns
    -------
    val_df : DataFrame
        Validation rows after transforms (includes TransactionAmt / DT / label).
    raw_scores : ndarray
        LightGBM raw predict() scores on validation features.
    labels : ndarray
        Binary fraud labels.
    amounts : ndarray
        TransactionAmt values.
    n_days : float
        Validation span in days from TransactionDT.
    """
    entry = get_registry_entry(MODEL_VERSION)
    if not entry.get("is_serving"):
        raise RuntimeError(f"Expected version {MODEL_VERSION} to have is_serving=true")

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
    # test_df is produced by the three-way split contract but never scored here.
    train_df, val_df, test_df, _fitted = apply_requires_fit_transforms(
        train_df, val_df, test_df, dataset_config
    )
    _, x_val, _, _ = prepare_features(train_df, val_df, test_df, feature_list)

    model = lgb.Booster(model_file=str(repo_path(entry["model_path"])))
    raw_scores = np.asarray(model.predict(x_val), dtype=float)
    labels = np.asarray(val_df[target], dtype=float)
    amounts = np.asarray(val_df["TransactionAmt"], dtype=float)

    dt = np.asarray(val_df["TransactionDT"], dtype=float)
    n_days = float((dt.max() - dt.min()) / 86400.0) if len(dt) else 0.0
    if n_days <= 0:
        n_days = 1.0

    return val_df.reset_index(drop=True), raw_scores, labels, amounts, n_days


def fit_isotonic(y_true: np.ndarray, raw_scores: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_scores, y_true)
    return calibrator


def cross_fit_calibrated_probs(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Out-of-fold isotonic probabilities on the validation split."""
    oof = np.empty_like(raw_scores, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, hold_idx in kf.split(raw_scores):
        calibrator = fit_isotonic(y_true[train_idx], raw_scores[train_idx])
        oof[hold_idx] = calibrator.predict(raw_scores[hold_idx])
    return oof


def main() -> None:
    _val_df, raw_scores, labels, _amounts, _n_days = load_v9_validation()

    oof_probs = cross_fit_calibrated_probs(labels, raw_scores)
    deployment = fit_isotonic(labels, raw_scores)
    CALIBRATOR_PATH.write_bytes(pickle.dumps(deployment))

    # Summary uses OOF calibrated scores so the gap is not optimistically biased.
    raw_mean = float(np.mean(raw_scores))
    cal_mean = float(np.mean(oof_probs))
    fraud_rate = float(np.mean(labels))
    gap = cal_mean - fraud_rate

    print("=== lgbm_v9 isotonic calibration (validation) ===")
    print(f"n_validation          : {len(labels):,}")
    print(f"raw score mean        : {raw_mean:.6f}")
    print(f"calibrated score mean : {cal_mean:.6f}")
    print(f"actual fraud rate     : {fraud_rate:.6f}")
    print(f"calibration gap       : {gap:+.6f}  (calibrated_mean - fraud_rate)")
    print(f"calibrator saved to   : {CALIBRATOR_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
