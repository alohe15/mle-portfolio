"""Prediction pipeline — raw input to scored output.

Guarantees training-serving parity by importing the SAME transformation
functions from scripts/feature_engineering.py that train.py uses.

§10 compliance: transformations are driven by the dataset config, not hardcoded.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from services.api.model_loader import ModelBundle
from services.api.schemas import PredictionResponse

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_dataset import collect_inherited_transformations, resolve_transform


def collect_dataset_transformations(dataset_config: dict) -> list[dict]:
    transforms = [
        transform for _, transform in collect_inherited_transformations(dataset_config)
    ]
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


def ensure_transform_input_columns(
    df: pd.DataFrame, dataset_config: dict
) -> pd.DataFrame:
    """Add missing transform input columns as NaN so engineering functions do not KeyError."""
    all_inputs: set[str] = set()
    for transform in collect_dataset_transformations(dataset_config):
        all_inputs.update(transform.get("inputs", []))
    for col in all_inputs:
        if col not in df.columns:
            df[col] = np.nan
    return df


def apply_transformation_pipeline(
    df: pd.DataFrame,
    dataset_config: dict,
    fitted_transforms: dict[str, object],
) -> pd.DataFrame:
    for transform in collect_dataset_transformations(dataset_config):
        df = apply_transformation(df, transform, fitted_transforms)
    return df


def raw_features_to_frame(features: dict) -> pd.DataFrame:
    row = {key: (np.nan if value is None else value) for key, value in features.items()}
    return pd.DataFrame([row])


def build_model_input(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    feature_order = bundle.feature_order
    row_series = df.iloc[0]
    row: dict[str, object] = {}
    for feature in feature_order:
        if feature in df.columns:
            value = row_series[feature]
        else:
            value = np.nan
        row[feature] = np.nan if value is None else value

    model_input = pd.DataFrame([row], columns=feature_order)

    for feature in feature_order:
        if feature in bundle.categorical_columns:
            cat_idx = bundle.categorical_columns.index(feature)
            value = model_input.at[0, feature]
            if pd.isna(value):
                value = "__MISSING__"
            model_input[feature] = pd.Categorical(
                [str(value)],
                categories=bundle.pandas_categorical[cat_idx],
            )
        else:
            model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce")

    return model_input


def assign_action(calibrated_score: float, decision_policy: dict) -> str:
    for action_def in decision_policy["actions"]:
        lower = action_def["lower"]
        upper = action_def["upper"]
        if lower <= calibrated_score < upper:
            return action_def["action"]
        if calibrated_score >= upper and upper == 1.0:
            return action_def["action"]
    return "approve"


def predict(raw_input: dict, bundle: ModelBundle) -> PredictionResponse:
    """Run the full prediction pipeline on a single transaction."""
    request_id = str(uuid.uuid4())
    warnings: list[str] = []

    df = raw_features_to_frame(raw_input)
    df = ensure_transform_input_columns(df, bundle.dataset_config)

    try:
        df = apply_transformation_pipeline(
            df, bundle.dataset_config, bundle.fitted_transforms
        )
    except Exception as exc:
        warnings.append(f"Transformation pipeline error: {exc}")
        logger.warning("Transformation pipeline error: %s", type(exc).__name__)

    model_input = build_model_input(df, bundle)

    missing_count = int(model_input.isnull().any(axis=0).sum())
    if missing_count > 0:
        warnings.append(
            f"{missing_count} features filled with NaN (not in request or transforms)"
        )

    raw_score = float(bundle.model.predict(model_input)[0])

    calibrated_score = raw_score
    if bundle.calibrator is not None:
        calibrated_score = float(bundle.calibrator.transform([raw_score])[0])
        calibrated_score = max(0.0, min(1.0, calibrated_score))

    recommended_action = "approve"
    policy_version = 0
    if bundle.decision_policy is not None:
        policy_version = int(bundle.decision_policy["policy_version"])
        recommended_action = assign_action(calibrated_score, bundle.decision_policy)
    else:
        warnings.append("No decision policy loaded — defaulting to approve")

    if bundle.calibrator is None:
        warnings.append("No calibrator loaded — raw score used as fraud_probability")

    return PredictionResponse(
        fraud_probability=round(calibrated_score, 6),
        raw_score=round(raw_score, 6),
        recommended_action=recommended_action,
        model_version=bundle.version,
        dataset_version=bundle.dataset_version,
        policy_version=policy_version,
        request_id=request_id,
        warnings=warnings,
    )
