"""FastAPI fraud inference service for the registry serving model (lgbm_v9).

Startup loads artifacts via ``model_loader.load_serving_artifacts`` (§10).
Predict replays the dataset-config transformation pipeline, calibrates, and
applies the Phase 3 decision policy.

Run:
    uvicorn services.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.api.model_loader import load_serving_artifacts
from services.api.predictor import predict as run_predict
from services.api.schemas import TransactionRequest, TransactionResponse

logger = logging.getLogger("services.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

ARTIFACTS: dict[str, Any] | None = None
LOAD_ERROR: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ARTIFACTS, LOAD_ERROR
    try:
        ARTIFACTS = load_serving_artifacts()
        LOAD_ERROR = None
        logger.info(
            "Serving artifacts loaded model_version=%s dataset_version=%s "
            "n_features=%d n_transforms=%d",
            ARTIFACTS["model_version_label"],
            ARTIFACTS["dataset_version"],
            len(ARTIFACTS["feature_order"]),
            len(ARTIFACTS["transformation_pipeline"]),
        )
    except Exception as exc:  # noqa: BLE001 — surface load failure as 503
        ARTIFACTS = None
        LOAD_ERROR = str(exc)
        logger.exception("Failed to load serving artifacts")
    yield


app = FastAPI(title="mle-portfolio fraud API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(exc.errors())}
    )


@app.get("/health")
def health() -> dict[str, Any]:
    if ARTIFACTS is None:
        return {
            "status": "unhealthy",
            "model_version": None,
            "dataset_version": None,
            "model_loaded": False,
            "error": LOAD_ERROR,
        }
    return {
        "status": "healthy",
        "model_version": ARTIFACTS["model_version_label"],
        "dataset_version": ARTIFACTS["dataset_version"],
        "model_loaded": True,
    }


@app.post("/predict", response_model=TransactionResponse)
def predict_endpoint(body: TransactionRequest) -> TransactionResponse:
    request_id = str(uuid.uuid4())
    if ARTIFACTS is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Model not loaded", "request_id": request_id},
        )

    start = time.perf_counter()
    try:
        result = run_predict(
            body.feature_dict(),
            ARTIFACTS,
            request_id=request_id,
        )
    except Exception:
        logger.exception("Prediction failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Prediction failed", "request_id": request_id},
        ) from None

    latency_ms = (time.perf_counter() - start) * 1000.0
    # Privacy-safe log: ids, versions, action, latency only — never raw features.
    logger.info(
        "predict ok request_id=%s model_version=%s dataset_version=%s "
        "recommended_action=%s latency_ms=%.2f",
        result["request_id"],
        result["model_version"],
        result["dataset_version"],
        result["recommended_action"],
        latency_ms,
    )
    return TransactionResponse(**result)
