"""Config-driven prediction pipeline (§10).

Replays the dataset-config transformation chain from feature_engineering.py,
selects/reorders to the model manifest feature_order, scores with the serving
booster, calibrates, and applies the per-transaction argmin decision policy.
"""

from __future__ import annotations

import inspect
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVALS_DIR = REPO_ROOT / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from cost_model import optimal_action  # noqa: E402


def _request_to_frame(request_data: dict[str, Any]) -> pd.DataFrame:
    row = {
        key: (np.nan if value is None else value)
        for key, value in request_data.items()
        if key != "metadata"
    }
    return pd.DataFrame([row])


def ensure_transform_input_columns(
    df: pd.DataFrame, pipeline: list[dict[str, Any]]
) -> pd.DataFrame:
    """Add missing transform input columns as NaN so FE functions can run."""
    needed: set[str] = set()
    for step in pipeline:
        cfg = step["config"]
        for col in cfg.get("inputs", []) or []:
            needed.add(col)
        params = step.get("params") or {}
        for col in params.get("columns", []) or []:
            needed.add(col)
    missing = [c for c in needed if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    return df


def apply_transformation_pipeline(
    df: pd.DataFrame, pipeline: list[dict[str, Any]]
) -> pd.DataFrame:
    """Apply ordered transforms from the dataset config (never .fit at serve)."""
    df = ensure_transform_input_columns(df, pipeline)
    for step in pipeline:
        params = step.get("params") or {}
        if step["requires_fit"]:
            fitted = step["fitted"]
            # Match train.py: fitted.transform(df, params)
            sig = inspect.signature(fitted.transform)
            if len(sig.parameters) >= 2:
                df = fitted.transform(df, params)
            else:
                df = fitted.transform(df)
        else:
            func = step["callable"]
            df = func(df, params)
    return df


def build_model_input(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Select/reorder to feature_order; NaN-fill missing; drop extras; encode cats."""
    feature_order: list[str] = artifacts["feature_order"]
    categorical_columns: list[str] = artifacts["categorical_columns"]
    pandas_categorical = artifacts["pandas_categorical"]

    # Normalize to a 0-based single row (transforms may preserve non-zero index).
    df = df.reset_index(drop=True)

    row: dict[str, object] = {}
    for feature in feature_order:
        if feature in df.columns:
            value = df.iloc[0][feature]
        else:
            value = np.nan
        row[feature] = np.nan if value is None else value

    model_input = pd.DataFrame([row], columns=feature_order)

    for feature in feature_order:
        if feature in categorical_columns:
            cat_idx = categorical_columns.index(feature)
            cats = list(pandas_categorical[cat_idx])
            value = model_input.at[0, feature]
            if pd.isna(value):
                coded = None
            else:
                value = str(value)
                if value == "__MISSING__" or value in cats:
                    coded = value if value in cats else None
                else:
                    coded = "__MISSING__" if "__MISSING__" in cats else None
            model_input[feature] = pd.Series(
                pd.Categorical([coded], categories=cats)
            )
        else:
            model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce")

    return model_input


def predict(
    request_data: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    request_id: str | None = None,
    return_raw_score: bool = False,
) -> dict[str, Any]:
    """Run the full serving pipeline for one transaction.

    Parameters
    ----------
    return_raw_score:
        When True, include ``raw_score`` in the result (for parity tests only —
        never exposed on the HTTP response schema).
    """
    amount = float(request_data["TransactionAmt"])
    df = _request_to_frame(request_data)
    df = apply_transformation_pipeline(df, artifacts["transformation_pipeline"])
    model_input = build_model_input(df, artifacts)

    raw_score = float(artifacts["model"].predict(model_input)[0])
    calibrated = float(artifacts["calibrator"].predict([raw_score])[0])
    action, costs = optimal_action(calibrated, amount, artifacts["cost_params"])
    cost_dict = {k: float(v) for k, v in costs.items()}

    result = {
        "fraud_probability": calibrated,
        "recommended_action": str(action),
        "action_costs": cost_dict,
        "model_version": artifacts["model_version_label"],
        "dataset_version": int(artifacts["dataset_version"]),
        "request_id": request_id or str(uuid.uuid4()),
    }
    if return_raw_score:
        result["raw_score"] = raw_score
    return result
