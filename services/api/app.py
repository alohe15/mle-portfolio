"""FastAPI service for IEEE-CIS fraud detection inference.

Loads the baseline LightGBM model trained by scripts/train_baseline_lightgbm.py
and exposes health and prediction endpoints.

Run locally:
    uvicorn services.api.app:app --reload --app-dir .
"""

from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from pathlib import Path

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT / "models" / "baseline_lightgbm.txt"
DATA_PATH = REPO_ROOT / "data" / "train_merged.parquet"

model = lgb.Booster(model_file=str(MODEL_PATH))
FEATURE_NAMES = model.feature_name()
PANDAS_CATEGORICAL = model.pandas_categorical


def load_categorical_columns() -> list[str]:
    sample = pd.read_parquet(DATA_PATH, columns=FEATURE_NAMES)
    return sample.select_dtypes(include=["object", "string"]).columns.tolist()


CATEGORICAL_COLUMNS = load_categorical_columns()


def build_feature_frame(features: dict) -> pd.DataFrame:
    row = {}
    for feat in FEATURE_NAMES:
        value = features.get(feat, np.nan)
        row[feat] = np.nan if value is None else value

    df = pd.DataFrame([row])

    for col in df.columns:
        if col in CATEGORICAL_COLUMNS:
            cat_idx = CATEGORICAL_COLUMNS.index(col)
            value = df.at[0, col]
            if pd.isna(value):
                value = "__MISSING__"
            df[col] = pd.Categorical([str(value)], categories=PANDAS_CATEGORICAL[cat_idx])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

class PredictionRequest(BaseModel):
    features: dict  # {"TransactionAmt": 100.0, "card1": 1234, ...}

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int  # 1 = fraud, 0 = legit
    latency_ms: float

@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.perf_counter()

    df = build_feature_frame(request.features)

    # Score
    prob = model.predict(df)[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        fraud_probability=round(float(prob), 6),
        prediction=int(prob >= 0.5),
        latency_ms=round(elapsed_ms, 2),
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_features": len(FEATURE_NAMES)}


