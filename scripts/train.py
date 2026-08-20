"""Config-driven model training entrypoint.

Usage:
    python scripts/train.py --config configs/lgbm_v1.json

This script is the single training entrypoint for all model versions in the
mle-portfolio project. It enforces the engineering standards defined in
ENGINEERING_STANDARDS.md by:

  - Reading ALL training parameters from a JSON config (no hardcoded values)
  - Producing exactly three required artifacts per run: model .txt, metrics
    JSON, and manifest JSON (plus fitted transforms pickle if needed)
  - Appending to models/registry.json so the API and eval scripts can
    discover every version ever trained
  - Deriving filenames from the config's description field, ensuring
    consistent naming like lgbm_v1_raw_baseline.txt

The script never needs editing to train a new version. You create a new
config JSON and point this script at it. That's the entire workflow.

See also:
  - ENGINEERING_STANDARDS.md §3 (Config-Driven Training)
  - ENGINEERING_STANDARDS.md §9 (Training Script Contract)
  - configs/lgbm_v1.json (example model config)
  - configs/dataset_v1.json (example dataset config)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# REPO_ROOT anchors all relative path resolution. Every config uses paths
# relative to the repo root (e.g. "configs/dataset_v1.json"), so this is
# the single reference point that makes those paths absolute.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where trained model artifacts land: .txt, _metrics.json, _manifest.json
MODELS_DIR = REPO_ROOT / "models"

# Where dataset parquets, manifests, and feature lists live (all gitignored
# except manifests and feature lists, which are committed as metadata).
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def repo_path(relative: str) -> Path:
    """Convert a repo-relative string path (from a config) to an absolute Path."""
    return REPO_ROOT / relative


def repo_relative_path(path: Path) -> str:
    """Convert an absolute Path back to a repo-relative POSIX string for
    storage in JSON artifacts (metrics, manifests, registry). Using POSIX
    format ensures paths work cross-platform and match what's in configs."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict | list:
    """Load and parse a JSON file. Used for configs, manifests, feature lists,
    and the model registry — all the metadata that drives the pipeline."""
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_model_config(config_path: Path) -> dict:
    """Load a model config JSON (e.g. configs/lgbm_v1.json).

    The config contains everything needed to reproduce a training run:
      - version: monotonically increasing model version number
      - description: human-readable label (also used for filename slug)
      - dataset_version: which dataset to train on
      - dataset_config_path: path to the dataset config (for transforms)
      - split: temporal split parameters (method, train_fraction, sort_column)
      - lgbm_params: all LightGBM hyperparameters
      - early_stopping: patience and metric for early stopping

    See ENGINEERING_STANDARDS.md §3 for the full required schema.
    """
    return load_json(config_path)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
# The training script doesn't hardcode dataset paths. Instead, it discovers
# them by convention:
#   1. Model config says dataset_version=1
#   2. Script globs data/processed/ for dataset_v1_*_manifest.json
#   3. Manifest contains feature_list_path pointing to feature_list_v1.json
#   4. Parquet path is derived from manifest path by swapping the suffix
#
# This indirection means adding a new dataset version requires zero changes
# to this script — just a new config and a build_dataset.py run.

def find_dataset_manifest(dataset_version: int) -> Path:
    """Locate the manifest JSON for a given dataset version by globbing
    the processed data directory.

    Raises FileNotFoundError if no manifest exists (dataset hasn't been built
    yet) and ValueError if multiple manifests match (ambiguous state that
    shouldn't happen if dataset versioning rules are followed).
    """
    matches = sorted(PROCESSED_DIR.glob(f"dataset_v{dataset_version}_*_manifest.json"))
    if not matches:
        raise FileNotFoundError(
            f"No dataset manifest found for dataset_v{dataset_version} in {PROCESSED_DIR}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple dataset manifests found for dataset_v{dataset_version}: {matches}"
        )
    return matches[0]


def parquet_path_from_manifest(manifest_path: Path) -> Path:
    """Derive the parquet file path from the manifest path.

    Convention: dataset_v1_raw_merged_manifest.json -> dataset_v1_raw_merged.parquet
    The manifest and parquet always sit side-by-side with matching stems.
    """
    name = manifest_path.name.replace("_manifest.json", ".parquet")
    return manifest_path.with_name(name)


def dataset_config_path_for_version(version: int) -> Path:
    """Return the expected config path for a dataset version.

    Convention: configs/dataset_v{D}.json — one config per dataset version,
    always in the configs/ directory.
    """
    return REPO_ROOT / "configs" / f"dataset_v{version}.json"


# ---------------------------------------------------------------------------
# Transformation chain resolution
# ---------------------------------------------------------------------------
# Dataset configs form a chain via inherited_transformations_from:
#   dataset_v3 inherits from v2, which inherits from v1.
#
# When training on dataset_v3, any requires_fit transforms from ALL ancestor
# versions need to be applied. These two functions walk the inheritance chain
# and collect every transformation in order.
#
# Non-requires_fit transforms were already applied by build_dataset.py and
# baked into the parquet. Only requires_fit transforms need to run here
# because they must be fit on the training split only (leakage discipline).

def collect_inherited_transformations(config: dict) -> list[tuple[int, dict]]:
    """Recursively walk the dataset config inheritance chain and collect all
    transformations from parent versions.

    Returns a list of (version_number, transform_dict) tuples, ordered from
    oldest ancestor to most recent parent. This ordering ensures transforms
    are replayed in the same sequence they were originally defined.
    """
    inherited: list[tuple[int, dict]] = []
    parent_version = config.get("inherited_transformations_from")
    if parent_version is None:
        return inherited

    parent_config = load_json(dataset_config_path_for_version(parent_version))
    # Recurse into grandparent, great-grandparent, etc. before appending
    # the parent's own transforms — this preserves chronological order.
    inherited.extend(collect_inherited_transformations(parent_config))
    for transform in parent_config.get("transformations", []):
        inherited.append((parent_version, transform))
    return inherited


def collect_all_transformations(config: dict) -> list[tuple[int, dict]]:
    """Collect the full ordered transformation chain: all inherited transforms
    followed by the current version's transforms.

    Each entry is (version_number, transform_dict). The version_number tells
    you which dataset version introduced each transform — useful for debugging
    and for the manifest's origin tracking.
    """
    transforms = collect_inherited_transformations(config)
    for transform in config.get("transformations", []):
        transforms.append((config["dataset_version"], transform))
    return transforms


def resolve_transform(function_ref: str):
    """Resolve a dotted function reference string (e.g.
    'feature_engineering.UidFrequencyEncoder') into an actual Python object.

    The function_ref format is 'module_name.attribute_path'. The module is
    imported from scripts/ (added to sys.path if not already there), then
    the attribute is traversed via getattr.

    This is what connects the dataset config's "function" field to real code.
    When a config says "function": "feature_engineering.create_email_match",
    this function imports scripts/feature_engineering.py and returns the
    create_email_match function from it.
    """
    module_name, _, attr_path = function_ref.partition(".")
    if not attr_path:
        raise ValueError(f"Invalid function reference: {function_ref!r}")

    # Ensure scripts/ is importable so "feature_engineering" resolves to
    # scripts/feature_engineering.py
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    module = importlib.import_module(module_name)
    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# Artifact naming
# ---------------------------------------------------------------------------

def artifact_slug(model_config: dict) -> str:
    """Derive the filename slug from the model config's description.

    This is what turns a config description like "Raw baseline — all 432
    merged columns" into a slug like "raw_baseline", producing filenames
    like lgbm_v1_raw_baseline.txt.

    Resolution order:
      1. If the config has an explicit "short_description" field, use it.
      2. Otherwise, take the description text, split on em-dash/en-dash/hyphen
         to get the short prefix, then slugify (lowercase, non-alphanum -> _).

    This addresses the engineering standard that model filenames must be
    lgbm_v{N}_{short_description}.txt — derived from config, not from the
    dataset parquet filename.
    """
    if "short_description" in model_config:
        return model_config["short_description"]
    description = model_config["description"]
    # Take everything before the first dash separator as the short label.
    # "Raw baseline — all 432 merged columns" -> "Raw baseline"
    short = re.split(r"\s[—–-]\s", description, maxsplit=1)[0].strip()
    # Slugify: "Raw baseline" -> "raw_baseline"
    slug = re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")
    return slug or f"v{model_config['version']}"


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    split_config: dict,
    *,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = split_config["method"]
    if method != "temporal":
        raise ValueError(f"Unsupported split method: {method!r}")

    if "val_fraction" not in split_config:
        config_label = config_path or "unknown"
        raise ValueError(
            f"Config {config_label} uses the deprecated two-way split. "
            "Create a new config with train_fraction, val_fraction, and test_fraction "
            "per §3 of ENGINEERING_STANDARDS.md."
        )

    train_frac = split_config["train_fraction"]
    val_frac = split_config["val_fraction"]
    assert abs(train_frac + val_frac + split_config["test_fraction"] - 1.0) < 1e-6, (
        f"Split fractions must sum to 1.0, got "
        f"{train_frac + val_frac + split_config['test_fraction']}"
    )

    sort_column = split_config["sort_column"]
    df = df.sort_values(sort_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Leakage-safe transform application
# ---------------------------------------------------------------------------

def apply_requires_fit_transforms(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fitted: dict[str, object] = {}
    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue

        name = transform["name"]
        obj = resolve_transform(transform["function"])
        params = transform.get("params", {})

        # requires_fit transforms must be implemented as classes with
        # .fit(train_df) and .transform(df) methods, following the
        # sklearn-style pattern. Plain functions can't hold fitted state.
        if inspect.isclass(obj):
            instance = obj()
            instance.fit(train_df, params)
            train_df = instance.transform(train_df, params)
            val_df = instance.transform(val_df, params)
            test_df = instance.transform(test_df, params)
            fitted[name] = instance
            continue

        raise ValueError(
            f"Transformation {name!r} requires fit/transform but {transform['function']!r} "
            "is not a class."
        )

    return train_df, val_df, test_df, fitted


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def validate_feature_list(feature_list: list[str], feature_list_path: Path) -> None:
    """Fail clearly on missing/empty/duplicated feature lists before training."""
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list JSON is missing: {feature_list_path}")
    if not feature_list:
        raise ValueError(f"Feature list is empty: {feature_list_path}")
    if len(feature_list) != len(set(feature_list)):
        dupes = sorted({f for f in feature_list if feature_list.count(f) > 1})
        raise ValueError(f"Feature list contains duplicates: {dupes}")
    forbidden = {"isFraud", "TransactionID", "TransactionDT"}
    bad = [c for c in feature_list if c in forbidden]
    if bad:
        raise ValueError(f"Feature list contains non-feature columns: {bad}")


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if len(feature_list) != len(set(feature_list)):
        dupes = sorted({f for f in feature_list if feature_list.count(f) > 1})
        raise ValueError(f"Feature list contains duplicates: {dupes}")

    missing = [col for col in feature_list if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature list columns missing from dataset: {missing}")

    # Select in feature_list order — this is the contract with the model manifest.
    x_train = train_df[feature_list].copy()
    x_val = val_df[feature_list].copy()
    x_test = test_df[feature_list].copy()
    if list(x_train.columns) != feature_list:
        raise ValueError(
            "Feature column order does not match feature_list "
            f"(got {list(x_train.columns)[:5]}... expected {feature_list[:5]}...)"
        )
    cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        # Fill nulls with a sentinel string so they become a learnable category
        x_train[col] = x_train[col].fillna("__MISSING__").astype(str)
        x_val[col] = x_val[col].fillna("__MISSING__").astype(str)
        x_test[col] = x_test[col].fillna("__MISSING__").astype(str)
        # Build a shared vocabulary from BOTH splits so every category code
        # maps to the same value in train and test
        categories = pd.Index(
            pd.concat(
                [x_train[col], x_val[col], x_test[col]], ignore_index=True
            ).unique()
        )
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_val[col] = pd.Categorical(x_val[col], categories=categories)
        x_test[col] = pd.Categorical(x_test[col], categories=categories)

    return x_train, x_val, x_test, cat_cols


# ---------------------------------------------------------------------------
# Evaluation at prevalence-matched flag rate
# ---------------------------------------------------------------------------

def evaluate_at_flag_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    flag_rate: float,
) -> dict[str, float]:
    """Compute precision, recall, and F1 at a specific flag rate.

    The flag rate determines what fraction of transactions get flagged as
    fraud. Setting it equal to the training fraud rate (prevalence-matched)
    answers the question: "If we flag exactly as many transactions as are
    actually fraudulent, how accurate are those flags?"

    This is more operationally meaningful than picking an arbitrary 0.5
    threshold, which is poorly calibrated for rare events. With a ~3.5%
    fraud rate, a 0.5 threshold would flag almost nothing.

    The threshold is derived from the scores: we find the score value such
    that exactly flag_rate fraction of transactions score at or above it.
    np.partition is used for efficiency (O(n) vs O(n log n) for full sort).
    """
    n_flag = max(int(round(len(y_true) * flag_rate)), 1)
    # np.partition puts the (len - n_flag)-th smallest element in its sorted
    # position. Everything above it is in the top n_flag scores.
    threshold = float(np.partition(y_score, len(y_score) - n_flag)[len(y_score) - n_flag])
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def append_registry_entry(entry: dict) -> None:
    """Append a new entry to models/registry.json.

    The registry is the single source of truth for all trained versions.
    The API reads it at startup to find the is_serving model. The eval
    scripts read it to discover all versions for comparison.

    Key behaviors:
      - If an entry with the same version already exists, it's replaced
        (idempotent re-runs of the same config don't create duplicates).
      - If the new entry has is_serving=True, all other entries are flipped
        to is_serving=False. Only one model serves at a time.
      - Entries are never deleted — the registry is an append-only historical
        record.

    See ENGINEERING_STANDARDS.md §6 (Model Registry).
    """
    registry_path = MODELS_DIR / "registry.json"
    registry: list[dict] = []
    if registry_path.exists():
        registry = load_json(registry_path)

    # Replace existing entry for this version (idempotent re-training)
    if any(item["version"] == entry["version"] for item in registry):
        registry = [item for item in registry if item["version"] != entry["version"]]

    # Enforce single-serving invariant: only one model can be is_serving=True
    if entry.get("is_serving"):
        for item in registry:
            item["is_serving"] = False

    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")


# ===========================================================================
# Main training function
# ===========================================================================

def train_model(config_path: Path) -> None:
    """Execute a full training run from a model config.

    This is the core function that implements the Training Script Contract
    (ENGINEERING_STANDARDS.md §9). The contract requires:

      1. Load the model config
      2. Load the dataset version specified in the config
      3. Apply the temporal split
      4. Fit requires_fit transforms on train only, transform both splits
      5. Train the model using only the feature list columns
      6. Save model .txt, metrics JSON, manifest JSON
      7. Append to registry.json
      8. Print a one-line summary to stdout

    Every step below maps to one of these contract requirements.
    """
    started_at = time.perf_counter()
    model_config = load_model_config(config_path)

    # -----------------------------------------------------------------------
    # Step 1: Resolve all paths from the config chain
    # -----------------------------------------------------------------------
    # The model config points to a dataset version, which points to a dataset
    # config, which has a manifest, which has a feature list. This chain of
    # references is what makes the system config-driven: change one config
    # and everything downstream follows.
    version = model_config["version"]
    dataset_version = model_config["dataset_version"]
    dataset_config = load_json(repo_path(model_config["dataset_config_path"]))

    # Discover the dataset manifest by globbing (not hardcoded path)
    dataset_manifest_path = find_dataset_manifest(dataset_version)
    dataset_manifest = load_json(dataset_manifest_path)
    parquet_path = parquet_path_from_manifest(dataset_manifest_path)
    # Feature list is an output of build_dataset.py — it's the authoritative
    # list of columns the model should train on (excludes target, IDs, time)
    feature_list_path = repo_path(dataset_manifest["feature_list_path"])
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list JSON is missing: {feature_list_path}")
    feature_list = load_json(feature_list_path)
    validate_feature_list(feature_list, feature_list_path)
    target_column = dataset_manifest["target_column"]

    # -----------------------------------------------------------------------
    # Step 2: Determine output artifact paths
    # -----------------------------------------------------------------------
    # All artifacts share a common base name: lgbm_v{N}_{slug}
    # The slug comes from the config description, not the parquet filename.
    # This was a bug fix — earlier versions incorrectly used the parquet stem.
    slug = artifact_slug(model_config)
    model_base = f"lgbm_v{version}_{slug}"
    model_path = MODELS_DIR / f"{model_base}.txt"
    metrics_path = MODELS_DIR / f"{model_base}_metrics.json"
    model_manifest_path = MODELS_DIR / f"{model_base}_manifest.json"
    fitted_transforms_path = MODELS_DIR / f"{model_base}_fitted_transforms.pkl"

    # -----------------------------------------------------------------------
    # Step 3: Load data and split temporally
    # -----------------------------------------------------------------------
    df = pd.read_parquet(parquet_path)
    split_cfg = model_config["split"]
    train_df, val_df, test_df = temporal_split(
        df,
        split_cfg,
        config_path=repo_relative_path(config_path.resolve()),
    )
    print(
        f"Split: train={len(train_df)} ({train_df[target_column].mean():.4f} fraud) | "
        f"val={len(val_df)} ({val_df[target_column].mean():.4f} fraud) | "
        f"test={len(test_df)} ({test_df[target_column].mean():.4f} fraud)"
    )
    train_df, val_df, test_df, fitted_transforms = apply_requires_fit_transforms(
        train_df, val_df, test_df, dataset_config
    )

    for _, transform in collect_all_transformations(dataset_config):
        if not transform.get("requires_fit"):
            continue
        for col in transform.get("output_columns", []):
            if col not in feature_list:
                feature_list.append(col)

    x_train, x_val, x_test, cat_cols = prepare_features(
        train_df, val_df, test_df, feature_list
    )
    y_train = train_df[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()

    # -----------------------------------------------------------------------
    # Step 6: Configure and train LightGBM
    # -----------------------------------------------------------------------
    # Pull hyperparameters from config. early_stopping_rounds is extracted
    # separately because it's passed to the callback, not the constructor.
    lgbm_params = dict(model_config["lgbm_params"])
    early_stopping_rounds = lgbm_params.pop("early_stopping_rounds")
    eval_metric = lgbm_params.pop("metric")
    # scale_pos_weight: honor a numeric config value (e.g. Optuna-tuned).
    # If the config leaves it null, compute (# negatives) / (# positives)
    # from the training fraud rate so it stays correct if the split changes.
    configured_spw = lgbm_params.pop("scale_pos_weight", None)
    train_fraud_rate = float(y_train.mean())
    if configured_spw is None:
        scale_pos_weight = (1.0 - train_fraud_rate) / train_fraud_rate
    else:
        scale_pos_weight = float(configured_spw)

    classifier_kwargs = dict(lgbm_params)
    classifier_kwargs["scale_pos_weight"] = scale_pos_weight
    model = lgb.LGBMClassifier(**classifier_kwargs)

    # fit() with eval_set enables early stopping: training continues up to
    # n_estimators trees, but stops early if the eval metric hasn't improved
    # for early_stopping_rounds consecutive rounds.
    #
    # The eval set is the validation split, not test. Test is held out for
    # final metrics only.
    #
    # Callbacks:
    #   - early_stopping: stops training when no improvement for N rounds
    #     on the first metric. first_metric_only=True avoids confusion when
    #     LightGBM tracks multiple metrics.
    #   - log_evaluation: prints eval metrics every 50 rounds for monitoring.
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
            lgb.log_evaluation(period=50),
        ],
    )

    # --- Final evaluation ---
    # val_auc_pr is computed on the validation split that early stopping
    # monitored. test_auc_pr is the clean holdout, evaluated only when
    # evaluate_test is true (holdout checkpoints). Intermediate experiments
    # set evaluate_test=false so test_auc_pr is null and the holdout stays unused.
    evaluate_test = bool(model_config.get("evaluate_test", True))
    y_val_score = model.predict_proba(x_val)[:, 1]
    val_auc_pr = float(average_precision_score(y_val, y_val_score))
    if evaluate_test:
        y_score = model.predict_proba(x_test)[:, 1]
        test_auc_pr = float(average_precision_score(y_test, y_score))
        operating = evaluate_at_flag_rate(y_test, y_score, train_fraud_rate)
        auc_roc = float(roc_auc_score(y_test, y_score))
    else:
        y_score = y_val_score
        test_auc_pr = None
        operating = evaluate_at_flag_rate(y_val, y_val_score, train_fraud_rate)
        auc_roc = float(roc_auc_score(y_val, y_val_score))
    # best_iteration_ is the round where early stopping found the best score.
    # If early stopping never triggered (model trained to n_estimators),
    # best_iteration_ may be 0/None, so we fall back to n_estimators.
    best_iteration = int(model.best_iteration_ or model_config["lgbm_params"]["n_estimators"])
    wall_time = time.perf_counter() - started_at

    # -----------------------------------------------------------------------
    # Step 8: Save all artifacts
    # -----------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model in LightGBM's native text format. This is the format
    # the API loads via lgb.Booster(model_file=path). It's human-readable
    # and contains the full tree structure.
    model.booster_.save_model(str(model_path))

    # Save fitted transform objects (lookup tables, encoders) so the API
    # can apply the same frozen statistics at inference time. Only created
    # when the dataset has requires_fit transforms.
    if fitted_transforms:
        fitted_transforms_path.write_bytes(pickle.dumps(fitted_transforms))

    # --- Metrics JSON ---
    # Schema defined in ENGINEERING_STANDARDS.md §4. Contains everything
    # needed to evaluate this version: dataset stats, all metric values,
    # and training metadata. All values are actual computed numbers, never
    # placeholders.
    metrics_doc = {
        "version": version,
        "config_path": repo_relative_path(config_path.resolve()),
        "dataset_version": dataset_version,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_val": float(y_val.mean()),
            "fraud_rate_test": float(y_test.mean()),
        },
        "metrics": {
            # AUC-PR is the primary metric. Given ~3.5% fraud rate, AUC-ROC
            # can be misleadingly high (a random classifier gets ~0.5 AUC-ROC
            # but only ~0.035 AUC-PR). AUC-PR is more sensitive to
            # performance on the minority class.
            "auc_pr": val_auc_pr if test_auc_pr is None else test_auc_pr,
            "val_auc_pr": val_auc_pr,
            "test_auc_pr": test_auc_pr,
            "auc_roc": auc_roc,
            # Precision/recall/F1 at the prevalence-matched flag rate
            "precision_at_budget": operating["precision"],
            "recall_at_budget": operating["recall"],
            "f1_at_budget": operating["f1"],
            # The threshold used for the above precision/recall/F1
            "operating_threshold": operating["threshold"],
            # How many boosting rounds were actually used (after early stopping)
            "best_iteration": best_iteration,
        },
        "training": {
            "wall_time_seconds": float(wall_time),
            "n_features": int(len(feature_list)),
            "n_estimators_used": best_iteration,
        },
    }
    metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n")

    # --- Model Manifest (Feature Contract) ---
    # Schema defined in ENGINEERING_STANDARDS.md §5. This is the contract
    # between training and serving. The API reads this manifest to know:
    #   - Which features the model expects, in what order
    #   - What dtypes each feature should be
    #   - Where to find the dataset config (for transformation replay)
    #
    # feature_order is the critical field: the API must reorder incoming
    # features to match this exact order before calling model.predict().
    # LightGBM uses column position, not names, internally.
    feature_dtypes = {col: str(x_train[col].dtype) for col in feature_list}
    model_manifest = {
        "version": version,
        "model_file": model_path.name,
        "created_at": metrics_doc["timestamp"],
        "dataset_version": dataset_version,
        "dataset_config_path": model_config["dataset_config_path"],
        "feature_order": feature_list,
        "feature_dtypes": feature_dtypes,
        "n_features": len(feature_list),
        "config_path": repo_relative_path(config_path.resolve()),
        "metrics_path": repo_relative_path(metrics_path),
        "description": model_config["description"],
    }
    # Only include fitted_transforms_path if there are fitted transforms.
    # dataset_v1 has none, so its manifest won't have this field.
    if fitted_transforms:
        model_manifest["fitted_transforms_path"] = repo_relative_path(fitted_transforms_path)

    model_manifest_path.write_text(json.dumps(model_manifest, indent=2) + "\n")

    # --- Registry append ---
    # This is the final step that makes this version "official". The API
    # reads registry.json to find the is_serving model. Setting is_serving
    # to True here means the API will pick up this model on next restart.
    #
    # The registry also stores paths to every artifact so eval scripts and
    # the API can discover everything from one file.
    append_registry_entry(
        {
            "version": version,
            "dataset_version": dataset_version,
            "description": model_config["description"],
            "model_path": repo_relative_path(model_path),
            "manifest_path": repo_relative_path(model_manifest_path),
            "metrics_path": repo_relative_path(metrics_path),
            "config_path": repo_relative_path(config_path.resolve()),
            "dataset_config_path": model_config["dataset_config_path"],
            "created_at": metrics_doc["timestamp"],
            "is_serving": model_config.get("is_serving", False),
        }
    )

    # -----------------------------------------------------------------------
    # Step 9: Print one-line summary (per Training Script Contract §9.9)
    # -----------------------------------------------------------------------
    test_auc_pr_str = "null" if test_auc_pr is None else f"{test_auc_pr:.4f}"
    print(
        f"Trained v{version} on dataset_v{dataset_version}: "
        f"val_auc_pr={val_auc_pr:.4f} | test_auc_pr={test_auc_pr_str} | "
        f"{len(feature_list)} features | {best_iteration} rounds | "
        f"split={len(train_df)}/{len(val_df)}/{len(test_df)}"
    )


# ===========================================================================
# CLI entrypoint
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse the single required CLI argument: --config path/to/config.json."""
    parser = argparse.ArgumentParser(description="Train a versioned LightGBM model from config.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to model config JSON (e.g. configs/lgbm_v1.json)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint: resolve the config path and run training.

    If the config path is relative, it's resolved against REPO_ROOT so you
    can run the script from any working directory:
        python scripts/train.py --config configs/lgbm_v1.json
    """
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    train_model(config_path)


if __name__ == "__main__":
    main()