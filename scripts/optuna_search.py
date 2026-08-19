"""Optuna hyperparameter search for LightGBM on a training subsample.

Usage:
    python scripts/optuna_search.py --base-config configs/lgbm_v6.json --n-trials 100 --train-subsample-fraction 0.5

The test split is never created, loaded, or referenced. Early stopping and
the Optuna objective use the full validation split only.
"""

from __future__ import annotations

import argparse
import inspect
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "evals" / "figures" / ".mplconfig"))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    collect_all_transformations,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    repo_path,
    resolve_transform,
)

STUDY_PATH = REPO_ROOT / "evals" / "optuna_study.pkl"
HISTORY_PATH = REPO_ROOT / "evals" / "figures" / "optuna_history.png"
SAMPLER_SEED = 42
SUBSAMPLE_SEED = 42
N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 100


def train_val_split_only(df: pd.DataFrame, split_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same index math as train.temporal_split, but never materializes test."""
    if split_config["method"] != "temporal":
        raise ValueError(f"Unsupported split method: {split_config['method']!r}")
    train_frac = split_config["train_fraction"]
    val_frac = split_config["val_fraction"]
    if abs(train_frac + val_frac + split_config["test_fraction"] - 1.0) >= 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    sort_column = split_config["sort_column"]
    df = df.sort_values(sort_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    return train_df, val_df


def apply_requires_fit_train_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue
        obj = resolve_transform(transform["function"])
        params = transform.get("params", {})
        if not inspect.isclass(obj):
            raise ValueError(
                f"Transformation {transform['name']!r} requires fit/transform but is not a class"
            )
        instance = obj()
        instance.fit(train_df, params)
        train_df = instance.transform(train_df, params)
        val_df = instance.transform(val_df, params)
    return train_df, val_df


def prepare_features_train_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing = [col for col in feature_list if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature list columns missing from dataset: {missing}")

    x_train = train_df[feature_list].copy()
    x_val = val_df[feature_list].copy()
    cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        x_train[col] = x_train[col].fillna("__MISSING__").astype(str)
        x_val[col] = x_val[col].fillna("__MISSING__").astype(str)
        categories = pd.Index(
            pd.concat([x_train[col], x_val[col]], ignore_index=True).unique()
        )
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_val[col] = pd.Categorical(x_val[col], categories=categories)

    return x_train, x_val, cat_cols


def expand_feature_list(feature_list: list[str], dataset_config: dict) -> list[str]:
    feature_list = list(feature_list)
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue
        for col in transform.get("output_columns", []):
            if col not in feature_list:
                feature_list.append(col)
    return feature_list


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "binary",
        "n_estimators": N_ESTIMATORS,
        "verbose": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": 1,
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 20.0, 35.0),
    }


def make_objective(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            categorical_feature=cat_cols,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=False,
                ),
            ],
        )
        y_val_score = model.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, y_val_score))

    return objective


def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    value = trial.value
    value_str = "None" if value is None else f"{value:.4f}"
    print(
        f"Trial {trial.number}: val_auc_pr={value_str} | "
        f"best={study.best_value:.4f} (trial {study.best_trial.number})",
        flush=True,
    )


def run_search(base_config_path: Path, n_trials: int, train_subsample_fraction: float) -> None:
    model_config = load_json(base_config_path)
    dataset_version = model_config["dataset_version"]
    dataset_config = load_json(repo_path(model_config["dataset_config_path"]))
    dataset_manifest_path = find_dataset_manifest(dataset_version)
    dataset_manifest = load_json(dataset_manifest_path)
    parquet_path = parquet_path_from_manifest(dataset_manifest_path)
    feature_list = expand_feature_list(
        load_json(repo_path(dataset_manifest["feature_list_path"])),
        dataset_config,
    )
    target_column = dataset_manifest["target_column"]

    df = pd.read_parquet(parquet_path)
    train_df, val_df = train_val_split_only(df, model_config["split"])
    del df

    full_train = len(train_df)
    train_sub = train_df.sample(
        frac=train_subsample_fraction, random_state=SUBSAMPLE_SEED
    ).copy()
    del train_df
    print(
        f"Optuna training subsample: {len(train_sub)} rows "
        f"({train_subsample_fraction:.0%} of {full_train} training rows). "
        f"Validation: {len(val_df)} rows (full).",
        flush=True,
    )

    train_sub, val_df = apply_requires_fit_train_val(train_sub, val_df, dataset_config)
    x_train, x_val, cat_cols = prepare_features_train_val(train_sub, val_df, feature_list)
    y_train = train_sub[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()

    sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(x_train, y_train, x_val, y_val, cat_cols),
        n_trials=n_trials,
        callbacks=[log_trial],
    )

    best = study.best_trial
    print("Optuna search complete")
    print(f"Best trial: {best.number}")
    print(f"Best val_auc_pr: {best.value:.6f}")
    print("Best params:")
    for key, value in sorted(best.params.items()):
        print(f"  {key}: {value}")
    print(f"  bagging_freq: 1 (fixed)")
    print(f"  n_estimators: {N_ESTIMATORS} (fixed)")
    print(f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS} (fixed)")

    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_bytes(pickle.dumps(study))
    print(f"Saved study to {STUDY_PATH.relative_to(REPO_ROOT)}")

    from optuna.visualization.matplotlib import plot_optimization_history

    axes = plot_optimization_history(study)
    fig = axes.figure if hasattr(axes, "figure") else axes
    fig.savefig(HISTORY_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved history plot to {HISTORY_PATH.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for LightGBM on a train subsample.")
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Path to the parent model config (e.g. configs/lgbm_v6.json)",
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--train-subsample-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = args.base_config
    if not base_config_path.is_absolute():
        base_config_path = REPO_ROOT / base_config_path
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not 0.0 < args.train_subsample_fraction <= 1.0:
        raise ValueError("--train-subsample-fraction must be in (0, 1]")
    run_search(base_config_path, args.n_trials, args.train_subsample_fraction)


if __name__ == "__main__":
    main()
