"""Train LightGBM on engineered features and compare to the baseline model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "train_featured.parquet"
TEST_PATH = REPO_ROOT / "data" / "test_featured.parquet"
BASELINE_METRICS_PATH = REPO_ROOT / "models" / "baseline_lightgbm_metrics.json"
MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = MODELS_DIR / "engineered_lightgbm_metrics.json"
MODEL_PATH = MODELS_DIR / "engineered_lightgbm.txt"

TARGET = "isFraud"
DROP_COLS = {"TransactionID", TARGET, "uid"}
TRAIN_FRACTION = 0.8
DECISION_THRESHOLD = 0.5
RANDOM_STATE = 42


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    cat_cols = train[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()

    x_train = train[feature_cols].copy()
    x_test = test[feature_cols].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].fillna("__MISSING__").astype(str)
        x_test[col] = x_test[col].fillna("__MISSING__").astype(str)
        categories = pd.Index(
            pd.concat([x_train[col], x_test[col]], ignore_index=True).unique()
        )
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_test[col] = pd.Categorical(x_test[col], categories=categories)

    return x_train, x_test, cat_cols


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fraud_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }


def evaluate_at_flag_rate(
    y_true: np.ndarray, y_score: np.ndarray, flag_rate: float
) -> dict[str, float]:
    n_flag = max(int(round(len(y_true) * flag_rate)), 1)
    threshold = float(np.partition(y_score, len(y_score) - n_flag)[len(y_score) - n_flag])
    metrics = evaluate(y_true, y_score, threshold)
    metrics["operating_point"] = "prevalence_matched_flag_rate"
    metrics["target_flag_rate"] = flag_rate
    return metrics


def load_baseline_metrics() -> dict:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found at {BASELINE_METRICS_PATH}. "
            "Run scripts/train_baseline_lightgbm.py first."
        )
    return json.loads(BASELINE_METRICS_PATH.read_text())


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    x_train, x_test, cat_cols = prepare_features(train, test)
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    baseline = load_baseline_metrics()
    baseline_metrics = baseline["lightgbm"]["at_threshold_0_5"]
    baseline_operating = baseline["lightgbm"]["at_prevalence_matched_flag_rate"]
    train_fraud_rate = float(y_train.mean())

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(x_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_score = model.predict_proba(x_test)[:, 1]
    model_metrics = evaluate(y_test, y_score, DECISION_THRESHOLD)
    model_operating = evaluate_at_flag_rate(y_test, y_score, train_fraud_rate)
    model_metrics["best_iteration"] = int(model.best_iteration_)

    results = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_paths": {"train": str(TRAIN_PATH), "test": str(TEST_PATH)},
        "feature_count": len(x_train.columns),
        "categorical_feature_count": len(cat_cols),
        "split": {
            "method": "time_based_transaction_dt",
            "train_fraction": TRAIN_FRACTION,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_fraud_rate": train_fraud_rate,
            "test_fraud_rate": float(y_test.mean()),
        },
        "engineered_lightgbm": {
            "at_threshold_0_5": model_metrics,
            "at_prevalence_matched_flag_rate": model_operating,
        },
        "baseline_lightgbm": {
            "at_threshold_0_5": baseline_metrics,
            "at_prevalence_matched_flag_rate": baseline_operating,
            "source_metrics_path": str(BASELINE_METRICS_PATH),
        },
        "lift_vs_baseline": {
            "auc_pr": model_metrics["auc_pr"] - baseline_metrics["auc_pr"],
            "precision_at_prevalence_matched": (
                model_operating["precision"] - baseline_operating["precision"]
            ),
            "recall_at_prevalence_matched": (
                model_operating["recall"] - baseline_operating["recall"]
            ),
        },
    }

    model.booster_.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(results, indent=2))

    print("=" * 72)
    print("IEEE-CIS fraud detection — engineered LightGBM vs baseline")
    print("=" * 72)
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print(f"Features: {len(x_train.columns):,} ({len(cat_cols)} categorical)")
    print(
        f"Train fraud rate: {train_fraud_rate:.4%} | "
        f"Test fraud rate: {y_test.mean():.4%}"
    )
    print()
    print(f"{'Metric':<12} {'Baseline':>12} {'Engineered':>12} {'Lift':>12}")
    print("-" * 52)
    print(
        f"{'auc_pr':<12} {baseline_metrics['auc_pr']:>12.4f} "
        f"{model_metrics['auc_pr']:>12.4f} "
        f"{model_metrics['auc_pr'] - baseline_metrics['auc_pr']:>+12.4f}"
    )
    for metric in ("precision", "recall"):
        base = baseline_operating[metric]
        model_val = model_operating[metric]
        lift = model_val - base
        print(f"{metric + '*':<12} {base:>12.4f} {model_val:>12.4f} {lift:>+12.4f}")
    print("  * at prevalence-matched flag rate")
    print()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
