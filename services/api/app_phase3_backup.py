"""FastAPI service for IEEE-CIS fraud detection inference.

Loads the serving model from models/registry.json at startup and exposes
health and prediction endpoints.

Run locally:
    uvicorn services.api.app:app --reload --app-dir .
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DECISION_THRESHOLD = 0.5

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_dataset import collect_inherited_transformations, resolve_transform

app = FastAPI()


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def get_serving_registry_entry(registry: list[dict]) -> dict:
    serving_entries = [entry for entry in registry if entry.get("is_serving")]
    if not serving_entries:
        raise RuntimeError("No model marked is_serving=true in models/registry.json")
    if len(serving_entries) > 1:
        raise RuntimeError("Multiple models marked is_serving=true in models/registry.json")
    return serving_entries[0]


def collect_dataset_transformations(dataset_config: dict) -> list[dict]:
    transforms = [transform for _, transform in collect_inherited_transformations(dataset_config)]
    transforms.extend(dataset_config.get("transformations", []))
    return transforms


def apply_transformation(
    df: pd.DataFrame,
    transform: dict,
    fitted_transforms: dict[str, object],
) -> pd.DataFrame:
    if transform.get("requires_fit"):
        name = transform["name"]
        if name not in fitted_transforms:
            raise RuntimeError(
                f"Missing fitted transform {name!r} required by dataset config"
            )
        encoder = fitted_transforms[name]
        return encoder.transform(df, transform.get("params", {}))

    func = resolve_transform(transform["function"])
    return func(df, transform.get("params", {}))


def apply_transformation_pipeline(
    df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> pd.DataFrame:
    for transform in collect_dataset_transformations(dataset_config):
        df = apply_transformation(df, transform, fitted_transforms)
    return df


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


def load_decision_policy(registry_entry: dict) -> dict | None:
    policy_path = registry_entry.get("decision_policy_path")
    if not policy_path:
        return None
    path = repo_path(policy_path)
    if not path.exists():
        return None
    return load_json(path)


def load_calibrator(decision_policy: dict | None) -> object | None:
    if decision_policy is None:
        return None
    if not decision_policy.get("use_calibrated_scores"):
        return None
    calibrator_path = decision_policy.get("calibrator_path")
    if not calibrator_path:
        return None
    path = repo_path(calibrator_path)
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def assign_action(calibrated_score: float, decision_policy: dict) -> str | None:
    for action_def in decision_policy["actions"]:
        lower = action_def["lower"]
        upper = action_def["upper"]
        if lower <= calibrated_score < upper:
            return action_def["action"]
        if calibrated_score >= upper and upper == 1.0:
            return action_def["action"]
    return None


class ServingContext:
    def __init__(self, registry_entry: dict) -> None:
        self.registry_entry = registry_entry
        self.model_version = int(registry_entry["version"])
        self.dataset_version = int(registry_entry["dataset_version"])

        manifest_path = repo_path(registry_entry["manifest_path"])
        model_path = repo_path(registry_entry["model_path"])
        dataset_config_path = repo_path(registry_entry["dataset_config_path"])

        self.manifest = load_json(manifest_path)
        self.dataset_config = load_json(dataset_config_path)
        self.model = lgb.Booster(model_file=str(model_path))

        self.feature_order: list[str] = self.manifest["feature_order"]
        self.feature_dtypes: dict[str, str] = self.manifest["feature_dtypes"]
        self.pandas_categorical = self.model.pandas_categorical
        self.categorical_columns = [
            col
            for col in self.feature_order
            if self.feature_dtypes.get(col) == "category"
        ]
        self.fitted_transforms = load_fitted_transforms(self.manifest)

        model_features = self.model.feature_name()
        if len(model_features) != len(self.feature_order):
            raise RuntimeError(
                "Model feature count "
                f"({len(model_features)}) does not match manifest feature_order "
                f"({len(self.feature_order)})"
            )
        if list(model_features) != self.feature_order:
            raise RuntimeError("Model feature names do not match manifest feature_order")

        if len(self.categorical_columns) != len(self.pandas_categorical):
            raise RuntimeError(
                "Categorical feature count does not match model.pandas_categorical length"
            )

        self.decision_policy = load_decision_policy(registry_entry)
        if self.decision_policy is not None:
            if self.decision_policy["model_version"] != self.model_version:
                raise RuntimeError(
                    "Policy model_version "
                    f"{self.decision_policy['model_version']} != serving version "
                    f"{self.model_version}"
                )
            if self.decision_policy["dataset_version"] != self.dataset_version:
                raise RuntimeError("Policy dataset_version mismatch")
        self.calibrator = load_calibrator(self.decision_policy)


def load_serving_context() -> ServingContext:
    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must contain a JSON array")
    return ServingContext(get_serving_registry_entry(registry))


SERVING = load_serving_context()


def raw_features_to_frame(features: dict) -> pd.DataFrame:
    row = {key: (np.nan if value is None else value) for key, value in features.items()}
    return pd.DataFrame([row])


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    feature_order = SERVING.feature_order
    row: dict[str, object] = {}
    for feature in feature_order:
        if feature in df.columns:
            value = df.at[0, feature]
        else:
            value = np.nan
        row[feature] = np.nan if value is None else value

    model_input = pd.DataFrame([row], columns=feature_order)

    for feature in feature_order:
        if feature in SERVING.categorical_columns:
            cat_idx = SERVING.categorical_columns.index(feature)
            value = model_input.at[0, feature]
            if pd.isna(value):
                value = "__MISSING__"
            model_input[feature] = pd.Categorical(
                [str(value)],
                categories=SERVING.pandas_categorical[cat_idx],
            )
        else:
            model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce")

    return model_input


class PredictionRequest(BaseModel):
    features: dict


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    latency_ms: float
    model_version: int
    dataset_version: int
    raw_score: float | None = None
    action: str | None = None
    policy_version: int | None = None


@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.perf_counter()

    df = raw_features_to_frame(request.features)
    df = apply_transformation_pipeline(df, SERVING.dataset_config, SERVING.fitted_transforms)
    model_input = build_model_input(df)
    raw_score = float(SERVING.model.predict(model_input)[0])
    elapsed_ms = (time.perf_counter() - start) * 1000

    calibrated_score = raw_score
    if SERVING.calibrator is not None:
        calibrated_score = float(SERVING.calibrator.transform([raw_score])[0])

    action = None
    policy_version = None
    if SERVING.decision_policy is not None:
        policy_version = SERVING.decision_policy["policy_version"]
        action = assign_action(calibrated_score, SERVING.decision_policy)

    score_for_prediction = calibrated_score if SERVING.calibrator is not None else raw_score

    return PredictionResponse(
        fraud_probability=round(calibrated_score, 6),
        raw_score=round(raw_score, 6) if SERVING.calibrator is not None else None,
        prediction=int(score_for_prediction >= DECISION_THRESHOLD),
        action=action,
        policy_version=policy_version,
        latency_ms=round(elapsed_ms, 2),
        model_version=SERVING.model_version,
        dataset_version=SERVING.dataset_version,
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": SERVING.model_version,
        "dataset_version": SERVING.dataset_version,
        "model_features": len(SERVING.feature_order),
    }
