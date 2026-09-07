"""Startup loader for serving artifacts (registry-driven, §10 compliant).

Builds the transformation pipeline from the serving model's dataset config
by resolving each `function` reference against `scripts/feature_engineering.py`.
Requires-fit transforms receive frozen state from the fitted-transforms pickle.
"""

from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    collect_all_transformations,
    load_json,
    repo_path,
)


def resolve_feature_engineering(function_ref: str) -> Any:
    """Load a transform object from scripts/feature_engineering.py by config ref."""
    module_name, _, attr_path = function_ref.partition(".")
    if module_name != "feature_engineering" or not attr_path:
        raise ValueError(
            f"Expected feature_engineering.<name> function ref, got {function_ref!r}"
        )
    module = importlib.import_module("feature_engineering")
    obj: Any = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def get_serving_entry(registry: list[dict]) -> dict:
    serving = [e for e in registry if e.get("is_serving") is True]
    if not serving:
        raise RuntimeError("No is_serving=true entry in models/registry.json")
    if len(serving) > 1:
        raise RuntimeError("Multiple is_serving=true entries in models/registry.json")
    return serving[0]


def get_candidate_entry(registry: list[dict], version: int) -> dict | None:
    matches = [
        e
        for e in registry
        if e.get("version") == version and e.get("status") == "Candidate"
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Candidate entries for version {version}")
    return matches[0] if matches else None


def _path_from_entries(
    key: str,
    serving: dict,
    candidate: dict | None,
    *,
    fallback_manifest: dict | None = None,
    manifest_key: str | None = None,
) -> str:
    if serving.get(key):
        return str(serving[key])
    if candidate and candidate.get(key):
        return str(candidate[key])
    if fallback_manifest and manifest_key and fallback_manifest.get(manifest_key):
        return str(fallback_manifest[manifest_key])
    raise KeyError(f"Could not resolve {key} from serving/Candidate/manifest")


def build_transformation_pipeline(
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> list[dict[str, Any]]:
    """Ordered pipeline steps: config + resolved FE callable + fitted state."""
    pipeline: list[dict[str, Any]] = []
    for source_version, transform in collect_all_transformations(dataset_config):
        fn_ref = transform["function"]
        obj = resolve_feature_engineering(fn_ref)
        name = transform["name"]
        requires_fit = bool(transform.get("requires_fit"))
        fitted = None
        if requires_fit:
            if name not in fitted_transforms:
                raise RuntimeError(
                    f"Missing fitted state for requires_fit transform {name!r}"
                )
            fitted = fitted_transforms[name]
        pipeline.append(
            {
                "name": name,
                "source_dataset_version": source_version,
                "config": transform,
                "function_ref": fn_ref,
                "callable": obj,
                "requires_fit": requires_fit,
                "fitted": fitted,
                "params": transform.get("params", {}),
            }
        )
    return pipeline


def load_serving_artifacts(registry_path: Path | str | None = None) -> dict[str, Any]:
    """Load model, config-driven transforms, calibrator, feature order, cost params."""
    path = Path(registry_path) if registry_path is not None else REGISTRY_PATH
    registry = load_json(path)
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must be a JSON array")

    serving = get_serving_entry(registry)
    version = int(serving["version"])
    candidate = get_candidate_entry(registry, version)

    manifest = load_json(repo_path(serving["manifest_path"]))
    dataset_config = load_json(repo_path(serving["dataset_config_path"]))
    feature_order = list(manifest["feature_order"])

    model_path = repo_path(serving["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    model_features = list(model.feature_name())
    if len(model_features) != len(feature_order):
        raise RuntimeError(
            f"Model feature count ({len(model_features)}) != "
            f"len(feature_order) ({len(feature_order)})"
        )
    if model_features != feature_order:
        raise RuntimeError("Model feature names do not match manifest feature_order")

    fitted_rel = _path_from_entries(
        "fitted_transforms_path",
        serving,
        candidate,
        fallback_manifest=manifest,
        manifest_key="fitted_transforms_path",
    )
    fitted_path = repo_path(fitted_rel)
    if not fitted_path.exists():
        raise FileNotFoundError(f"Fitted transforms not found: {fitted_path}")
    fitted_transforms = pickle.loads(fitted_path.read_bytes())
    if not isinstance(fitted_transforms, dict):
        raise TypeError(f"Expected dict in fitted transforms file: {fitted_path}")

    pipeline = build_transformation_pipeline(dataset_config, fitted_transforms)

    calibrator_rel = _path_from_entries("calibrator_path", serving, candidate)
    calibrator_path = repo_path(calibrator_rel)
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
    calibrator = pickle.loads(calibrator_path.read_bytes())

    policy_rel = _path_from_entries("decision_policy_config", serving, candidate)
    policy_path = repo_path(policy_rel)
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy config not found: {policy_path}")
    policy_doc = load_json(policy_path)
    cost_params = {k: float(v) for k, v in policy_doc["cost_params"].items()}

    feature_dtypes = dict(manifest.get("feature_dtypes", {}))
    categorical_columns = [
        col for col in feature_order if feature_dtypes.get(col) == "category"
    ]
    pandas_categorical = model.pandas_categorical
    if len(categorical_columns) != len(pandas_categorical or []):
        raise RuntimeError(
            "Categorical feature count does not match model.pandas_categorical length"
        )

    return {
        "model": model,
        "transformation_pipeline": pipeline,
        "calibrator": calibrator,
        "feature_order": feature_order,
        "feature_dtypes": feature_dtypes,
        "categorical_columns": categorical_columns,
        "pandas_categorical": pandas_categorical,
        "cost_params": cost_params,
        "model_version": version,
        "model_version_label": f"v{version}",
        "dataset_version": int(serving["dataset_version"]),
        "dataset_config": dataset_config,
        "registry_entry": serving,
        "paths": {
            "model_path": serving["model_path"],
            "manifest_path": serving["manifest_path"],
            "dataset_config_path": serving["dataset_config_path"],
            "fitted_transforms_path": fitted_rel,
            "calibrator_path": calibrator_rel,
            "decision_policy_config": policy_rel,
        },
    }
