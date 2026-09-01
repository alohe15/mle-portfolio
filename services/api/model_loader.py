"""Model loading and validation — runs once at startup.

Loads model, manifest, fitted transforms, calibrator, decision policy,
and feature classification. Validates cross-references between all artifacts.

§10 compliance: reads registry.json → loads serving entry's artifacts.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@dataclass
class ModelBundle:
    """Everything needed to serve predictions — loaded once at startup."""

    model: lgb.Booster
    manifest: dict
    feature_order: list[str]
    feature_dtypes: dict[str, str]
    n_features: int
    config: dict
    dataset_config: dict
    fitted_transforms: dict[str, object] = field(default_factory=dict)
    calibrator: Optional[object] = None
    decision_policy: Optional[dict] = None
    feature_classification: Optional[dict] = None
    version: int = 0
    dataset_version: int = 0
    policy_version: Optional[int] = None
    pandas_categorical: list[list[str]] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)


def _repo_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _get_serving_registry_entry(registry: list[dict]) -> dict:
    serving_entries = [entry for entry in registry if entry.get("is_serving")]
    if not serving_entries:
        raise RuntimeError("No model marked is_serving=true in models/registry.json")
    if len(serving_entries) > 1:
        raise RuntimeError(
            "Multiple models marked is_serving=true in models/registry.json"
        )
    return serving_entries[0]


def _load_fitted_transforms(manifest: dict) -> dict[str, object]:
    fitted_path = manifest.get("fitted_transforms_path")
    if not fitted_path:
        return {}
    path = _repo_path(fitted_path)
    if not path.exists():
        raise FileNotFoundError(f"Fitted transforms file not found: {path}")
    loaded = pickle.loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in fitted transforms file: {path}")
    return loaded


def _load_decision_policy(registry_entry: dict) -> dict | None:
    policy_path = registry_entry.get("decision_policy_path")
    if not policy_path:
        return None
    path = _repo_path(policy_path)
    if not path.exists():
        return None
    return _load_json(path)


def _load_calibrator(decision_policy: dict | None) -> object | None:
    if decision_policy is None:
        return None
    if not decision_policy.get("use_calibrated_scores"):
        return None
    calibrator_path = decision_policy.get("calibrator_path")
    if not calibrator_path:
        return None
    path = _repo_path(calibrator_path)
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def load_bundle() -> ModelBundle:
    """Load the serving model bundle from the registry."""
    registry = _load_json(PROJECT_ROOT / "models" / "registry.json")
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must contain a JSON array")

    entry = _get_serving_registry_entry(registry)
    version = int(entry["version"])
    dataset_version = int(entry["dataset_version"])
    logger.info("Loading serving model v%s", version)

    manifest = _load_json(_repo_path(entry["manifest_path"]))
    model = lgb.Booster(model_file=str(_repo_path(entry["model_path"])))
    config = _load_json(_repo_path(entry["config_path"]))
    dataset_config_path = entry.get(
        "dataset_config_path", config.get("dataset_config_path")
    )
    dataset_config = _load_json(_repo_path(dataset_config_path))

    feature_order: list[str] = manifest["feature_order"]
    feature_dtypes: dict[str, str] = manifest["feature_dtypes"]
    n_features = int(manifest["n_features"])
    if len(feature_order) != n_features:
        raise RuntimeError(
            f"Manifest feature_order length {len(feature_order)} != n_features {n_features}"
        )

    fitted_transforms = _load_fitted_transforms(manifest)
    if fitted_transforms:
        logger.info(
            "Loaded fitted transforms from %s",
            manifest.get("fitted_transforms_path"),
        )

    decision_policy = _load_decision_policy(entry)
    calibrator = _load_calibrator(decision_policy)
    if calibrator is not None:
        logger.info("Loaded calibrator")
    elif entry.get("calibrator_path"):
        logger.warning("Calibrator path set but calibrator not loaded")

    if decision_policy is not None:
        if decision_policy["model_version"] != version:
            raise RuntimeError(
                f"Policy model_version {decision_policy['model_version']} "
                f"!= serving v{version}"
            )
        if decision_policy["dataset_version"] != dataset_version:
            raise RuntimeError("Policy dataset_version mismatch")
        logger.info(
            "Loaded decision policy v%s", decision_policy["policy_version"]
        )

    model_features = model.feature_name()
    if len(model_features) != len(feature_order):
        raise RuntimeError(
            f"Model feature count ({len(model_features)}) does not match "
            f"manifest feature_order ({len(feature_order)})"
        )
    if list(model_features) != feature_order:
        raise RuntimeError("Model feature names do not match manifest feature_order")

    pandas_categorical = model.pandas_categorical
    categorical_columns = [
        col for col in feature_order if feature_dtypes.get(col) == "category"
    ]
    if len(categorical_columns) != len(pandas_categorical):
        raise RuntimeError(
            "Categorical feature count does not match model.pandas_categorical length"
        )

    feature_classification = None
    fc_path = PROJECT_ROOT / "services" / "api" / "feature_classification.json"
    if fc_path.exists():
        feature_classification = _load_json(fc_path)
        logger.info("Loaded feature classification")

    return ModelBundle(
        model=model,
        manifest=manifest,
        feature_order=feature_order,
        feature_dtypes=feature_dtypes,
        n_features=n_features,
        config=config,
        dataset_config=dataset_config,
        fitted_transforms=fitted_transforms,
        calibrator=calibrator,
        decision_policy=decision_policy,
        feature_classification=feature_classification,
        version=version,
        dataset_version=dataset_version,
        policy_version=(
            decision_policy["policy_version"] if decision_policy else None
        ),
        pandas_categorical=pandas_categorical,
        categorical_columns=categorical_columns,
    )
