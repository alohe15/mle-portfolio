"""Fraud detection API — serves the frozen lgbm_v9 model.

Loads all artifacts once at startup from the model registry.
Returns calibrated fraud probability, recommended action, and request_id.

§10 compliance: model-agnostic, config-driven, no hardcoded features or thresholds.

Run locally:
    uvicorn services.api.app:app --reload --app-dir .
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.api.model_loader import ModelBundle, load_bundle
from services.api.predictor import predict
from services.api.schemas import (
    HealthResponse,
    PredictionResponse,
    TransactionInput,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model bundle at startup."""
    global bundle
    try:
        bundle = load_bundle()
        logger.info(
            json.dumps(
                {
                    "event": "model_loaded",
                    "model_version": bundle.version,
                    "dataset_version": bundle.dataset_version,
                    "n_features": bundle.n_features,
                    "calibrator_loaded": bundle.calibrator is not None,
                    "policy_loaded": bundle.decision_policy is not None,
                    "policy_version": bundle.policy_version,
                }
            )
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", type(exc).__name__)
        raise
    yield


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Serves lgbm_v9 fraud predictions with calibrated scores and action recommendations"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput) -> PredictionResponse:
    """Score a transaction and return fraud probability with recommended action."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    raw_input = {k: v for k, v in transaction.model_dump().items() if v is not None}

    response = predict(raw_input, bundle)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "request_id": response.request_id,
                "model_version": response.model_version,
                "policy_version": response.policy_version,
                "recommended_action": response.recommended_action,
                "fraud_probability": response.fraud_probability,
                "latency_ms": round(elapsed_ms, 1),
                "warning_count": len(response.warnings),
            }
        )
    )

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — returns model status."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="healthy",
        model_version=bundle.version,
        dataset_version=bundle.dataset_version,
        policy_version=bundle.policy_version,
        n_features=bundle.n_features,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Validation errors — privacy-safe, no raw input echoed."""
    request_id = str(uuid.uuid4())
    logger.info(
        json.dumps(
            {
                "event": "validation_error",
                "request_id": request_id,
                "error_count": len(exc.errors()),
            }
        )
    )
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "request_id": request_id},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler — privacy-safe."""
    request_id = str(uuid.uuid4())
    logger.error(
        json.dumps(
            {
                "event": "error",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            }
        )
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id},
    )
