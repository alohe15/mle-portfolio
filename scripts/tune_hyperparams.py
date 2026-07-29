"""Optuna hyperparameter tuning for LightGBM.

Tunes on a temporal holdout carved from the training split only — the
official test set is never touched. Best-trial params are written to a
model config that `train.py` can consume unchanged.

By default, each trial trains on a stratified 10% subsample of tune_train
while evaluating on the full tune_val set. Hyperparameter rankings are
stable across data sizes; the winner is retrained on 100% via train.py.

Usage:
    python scripts/tune_hyperparams.py \\
        --base-config configs/lgbm_v3.json \\
        --n-trials 100 \\
        --output-version 4 \\
        --description tuned_hyperparams \\
        --subsample-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVALS_DIR = REPO_ROOT / "evals"
CONFIGS_DIR = REPO_ROOT / "configs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    apply_requires_fit_transforms,
    collect_all_transformations,
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    prepare_features,
    repo_path,
    temporal_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter search for LightGBM."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        type=Path,
        help="Template model config (inherits dataset_version, split, etc.)",
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--output-version", type=int, required=True)
    parser.add_argument("--description", type=str, default="tuned_hyperparams")
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of tune_train used per trial (stratified by isFraud). "
            "tune_val is never subsampled."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def load_tuning_data(
    base_config: dict,
    seed: int,
    subsample_fraction: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    list[str],
    list[str],
    dict,
]:
    """Load dataset, temporal-split, fit transforms once, then subsample train.

    Test set from the primary split is discarded (never used during tuning).
    requires_fit transforms are fit on the FULL tune_train before any
    subsampling so encoders see the full training distribution.
    """
    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError(
            f"--subsample-fraction must be in (0, 1], got {subsample_fraction}"
        )

    dataset_version = base_config["dataset_version"]
    dataset_config = load_json(repo_path(base_config["dataset_config_path"]))

    dataset_manifest_path = find_dataset_manifest(dataset_version)
    dataset_manifest = load_json(dataset_manifest_path)
    parquet_path = parquet_path_from_manifest(dataset_manifest_path)
    feature_list = list(load_json(repo_path(dataset_manifest["feature_list_path"])))
    target_column = dataset_manifest["target_column"]

    df = pd.read_parquet(parquet_path)
    full_train, _test_df = temporal_split(df, base_config["split"])

    # Nested temporal split within full_train: 75% tune_train / 25% tune_val.
    tune_split_config = {
        "method": "temporal",
        "train_fraction": 0.75,
        "sort_column": base_config["split"]["sort_column"],
    }
    tune_train_df, tune_val_df = temporal_split(full_train, tune_split_config)

    # Fit requires_fit on FULL tune_train (before subsample), once.
    tune_train_df, tune_val_df, _fitted = apply_requires_fit_transforms(
        tune_train_df, tune_val_df, dataset_config
    )

    # Mirror train.py: append requires_fit output columns if missing from list.
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue
        for col in transform.get("output_columns", []):
            if col not in feature_list:
                feature_list.append(col)

    x_tune_train, x_tune_val, cat_cols = prepare_features(
        tune_train_df, tune_val_df, feature_list
    )
    y_tune_train = tune_train_df[target_column]
    y_tune_val = tune_val_df[target_column]

    tune_train_rows_full = int(len(x_tune_train))

    if subsample_fraction < 1.0:
        x_tune_sub, _, y_tune_sub, _ = train_test_split(
            x_tune_train,
            y_tune_train,
            train_size=subsample_fraction,
            stratify=y_tune_train,
            random_state=seed,
        )
        print(
            f"Subsampled tune_train: {tune_train_rows_full} → {len(x_tune_sub)} rows "
            f"({subsample_fraction:.0%}), fraud rate {float(y_tune_sub.mean()):.4f}"
        )
    else:
        x_tune_sub, y_tune_sub = x_tune_train, y_tune_train

    split_stats = {
        "tune_train_rows_full": tune_train_rows_full,
        "tune_train_rows_subsampled": int(len(x_tune_sub)),
        "subsample_fraction": float(subsample_fraction),
        "tune_val_rows": int(len(tune_val_df)),
        "tune_train_fraud_rate": float(y_tune_sub.mean()),
        "tune_val_fraud_rate": float(y_tune_val.mean()),
    }
    return (
        x_tune_sub,
        x_tune_val,
        y_tune_sub,
        y_tune_val,
        cat_cols,
        feature_list,
        split_stats,
    )


def build_objective(
    x_tune_sub: pd.DataFrame,
    y_tune_sub: pd.Series,
    x_tune_val: pd.DataFrame,
    y_tune_val: pd.Series,
    cat_cols: list[str],
    seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        bagging_fraction = trial.suggest_categorical(
            "bagging_fraction", [0.7, 0.85, 1.0]
        )
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "seed": seed,
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "min_data_in_leaf": trial.suggest_categorical(
                "min_data_in_leaf", [20, 100, 500]
            ),
            "feature_fraction": trial.suggest_categorical(
                "feature_fraction", [0.5, 0.7, 0.9]
            ),
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 1 if bagging_fraction < 1.0 else 0,
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.1, 1.0]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0.0, 0.1, 1.0]),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 12]),
            # Fixed single-value categorical — expanding this is a separate §7 hypothesis.
            "scale_pos_weight": trial.suggest_categorical(
                "scale_pos_weight", [27.0]
            ),
            "learning_rate": 0.05,
            "n_estimators": 3000,
        }
        if params["bagging_fraction"] >= 1.0:
            params["bagging_freq"] = 0

        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_tune_sub,
            y_tune_sub,
            categorical_feature=cat_cols,
            eval_set=[(x_tune_val, y_tune_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                lgb.log_evaluation(period=0),
            ],
        )

        y_prob = model.predict_proba(x_tune_val)[:, 1]
        auc_pr = float(average_precision_score(y_tune_val, y_prob))
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or 0))
        trial.set_user_attr("bagging_freq", params["bagging_freq"])
        return auc_pr

    return objective


def write_model_config(
    base_config: dict,
    best_trial: optuna.trial.FrozenTrial,
    output_version: int,
    description: str,
    n_trials: int,
    output_path: Path,
) -> dict:
    bagging_fraction = best_trial.params["bagging_fraction"]
    bagging_freq = 1 if bagging_fraction < 1.0 else 0
    best_auc = float(best_trial.value)

    config = {
        "version": output_version,
        "short_description": description,
        "description": (
            f"Optuna-tuned hyperparameters ({n_trials} trials, "
            f"best trial #{best_trial.number}, tune-val AUC-PR={best_auc:.4f})"
        ),
        "parent_version": base_config["version"],
        "dataset_version": base_config["dataset_version"],
        "dataset_config_path": base_config["dataset_config_path"],
        "is_serving": False,
        "split": dict(base_config["split"]),
        "lgbm_params": {
            "objective": "binary",
            "metric": "average_precision",
            "n_estimators": 3000,
            "learning_rate": 0.05,
            "num_leaves": best_trial.params["num_leaves"],
            "min_data_in_leaf": best_trial.params["min_data_in_leaf"],
            "feature_fraction": best_trial.params["feature_fraction"],
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "reg_alpha": best_trial.params["reg_alpha"],
            "reg_lambda": best_trial.params["reg_lambda"],
            "max_depth": best_trial.params["max_depth"],
            # Placeholder — train.py recomputes from train fraud rate.
            "scale_pos_weight": best_trial.params["scale_pos_weight"],
            "early_stopping_rounds": 100,
            "verbose": -1,
        },
        "early_stopping": {
            "metric": "average_precision",
            "patience": 100,
            "min_delta": 0.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def write_study_csv(study: optuna.Study, path: Path) -> None:
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": trial.number,
            "value": trial.value,
            "best_iteration": trial.user_attrs.get("best_iteration"),
            "bagging_freq": trial.user_attrs.get("bagging_freq"),
            **trial.params,
        }
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(
    study: optuna.Study,
    split_stats: dict,
    seed: int,
    wall_time: float,
    base_config_path: Path,
    n_trials: int,
    path: Path,
) -> dict:
    best = study.best_trial
    bagging_fraction = best.params["bagging_fraction"]
    summary = {
        "study_name": study.study_name,
        "n_trials": n_trials,
        "n_trials_complete": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "best_trial_number": best.number,
        "best_tune_val_auc_pr": float(best.value),
        "best_params": {
            **best.params,
            "bagging_freq": 1 if bagging_fraction < 1.0 else 0,
            "learning_rate": 0.05,
            "n_estimators": 3000,
        },
        "best_iteration": best.user_attrs.get("best_iteration"),
        "tune_split": split_stats,
        "seed": seed,
        "wall_time_seconds": float(wall_time),
        "base_config": str(base_config_path.relative_to(REPO_ROOT)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    base_config_path = resolve_config_path(args.base_config)
    base_config = load_json(base_config_path)

    print(f"Loading data from base config: {base_config_path.relative_to(REPO_ROOT)}")
    (
        x_tune_sub,
        x_tune_val,
        y_tune_sub,
        y_tune_val,
        cat_cols,
        feature_list,
        split_stats,
    ) = load_tuning_data(base_config, args.seed, args.subsample_fraction)

    print(
        f"Tune split: train_full={split_stats['tune_train_rows_full']:,}, "
        f"train_sub={split_stats['tune_train_rows_subsampled']:,} "
        f"(fraud={split_stats['tune_train_fraud_rate']:.4f}), "
        f"val={split_stats['tune_val_rows']:,} "
        f"(fraud={split_stats['tune_val_fraud_rate']:.4f}), "
        f"n_features={len(feature_list)}"
    )

    objective = build_objective(
        x_tune_sub,
        y_tune_sub,
        x_tune_val,
        y_tune_val,
        cat_cols,
        args.seed,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name="lgbm_v4_hyperparam_tuning",
    )

    started = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    wall_time = time.perf_counter() - started

    best = study.best_trial
    best_iteration = best.user_attrs.get("best_iteration", 0)
    if best_iteration is not None and best_iteration >= 3000 - 50:
        print(
            f"WARNING: best_iteration={best_iteration} is within 50 of "
            "n_estimators=3000 ceiling — model may want more trees."
        )

    output_version = args.output_version
    config_path = CONFIGS_DIR / f"lgbm_v{output_version}.json"
    study_csv_path = EVALS_DIR / f"optuna_v{output_version}_study.csv"
    summary_path = EVALS_DIR / f"optuna_v{output_version}_summary.json"

    write_model_config(
        base_config=base_config,
        best_trial=best,
        output_version=output_version,
        description=args.description,
        n_trials=args.n_trials,
        output_path=config_path,
    )
    write_study_csv(study, study_csv_path)
    write_summary(
        study=study,
        split_stats=split_stats,
        seed=args.seed,
        wall_time=wall_time,
        base_config_path=base_config_path,
        n_trials=args.n_trials,
        path=summary_path,
    )

    mins = wall_time / 60.0
    params = best.params
    full_n = split_stats["tune_train_rows_full"]
    sub_n = split_stats["tune_train_rows_subsampled"]
    print()
    print(f"Tuning complete: {args.n_trials} trials in {mins:.1f}m")
    print(
        f"Subsample: {args.subsample_fraction:.0%} of tune_train "
        f"({full_n} → {sub_n} rows), fraud rate preserved at "
        f"{split_stats['tune_train_fraud_rate']:.4f}"
    )
    print(f"Best trial #{best.number}: AUC-PR = {float(best.value):.4f} (tune-val)")
    print(
        "Best params: "
        f"num_leaves={params['num_leaves']}, "
        f"min_data_in_leaf={params['min_data_in_leaf']}, "
        f"feature_fraction={params['feature_fraction']}, "
        f"bagging_fraction={params['bagging_fraction']}, "
        f"reg_alpha={params['reg_alpha']}, "
        f"reg_lambda={params['reg_lambda']}, "
        f"max_depth={params['max_depth']}, "
        f"best_iteration={best_iteration}"
    )
    print(f"Config written: {config_path.relative_to(REPO_ROOT)}")
    print(f"Study saved:    {study_csv_path.relative_to(REPO_ROOT)}")
    print(f"Summary saved:  {summary_path.relative_to(REPO_ROOT)}")
    print()
    print(f"Next step: python scripts/train.py --config {config_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
