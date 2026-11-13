"""FastAPI fraud inference service for the registry serving model (lgbm_v9).

Startup loads artifacts via ``model_loader.load_serving_artifacts`` (§10).
Predict replays the dataset-config transformation pipeline, calibrates, and
applies the Phase 3 decision policy.

Run:
    uvicorn services.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse, Response

from services.api.metrics import record_prediction, set_model_loaded
from services.api.model_loader import load_serving_artifacts
from services.api.observability import AccessLogMiddleware, disable_uvicorn_access_log
from services.api.predictor import predict as run_predict
from services.api.schemas import TransactionRequest, TransactionResponse

logger = logging.getLogger("services.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

ARTIFACTS: dict[str, Any] | None = None
LOAD_ERROR: str | None = None
ARTIFACT_LOAD_FAILED = "artifact_load_failed"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ARTIFACTS, LOAD_ERROR
    # Uvicorn installs its access logger after this module is imported.
    disable_uvicorn_access_log()
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
        set_model_loaded(
            True,
            model_version=str(ARTIFACTS["model_version_label"]),
            dataset_version=str(ARTIFACTS["dataset_version"]),
        )
    except Exception as exc:  # noqa: BLE001 — surface load failure as 503
        ARTIFACTS = None
        LOAD_ERROR = str(exc)
        logger.exception("Failed to load serving artifacts")
        set_model_loaded(False)
    yield


app = FastAPI(title="mle-portfolio fraud API", lifespan=lifespan)
app.add_middleware(AccessLogMiddleware)


def _public_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """type, loc, and msg only. input and ctx can echo the request body."""
    public = [
        {"type": err.get("type"), "loc": err.get("loc"), "msg": err.get("msg")}
        for err in exc.errors()
    ]
    return jsonable_encoder(public)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": _public_validation_errors(exc),
            "request_id": request.state.request_id,
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    if ARTIFACTS is None:
        return {
            "status": "unhealthy",
            "model_version": None,
            "dataset_version": None,
            "model_loaded": False,
            "error": ARTIFACT_LOAD_FAILED,
        }
    return {
        "status": "healthy",
        "model_version": ARTIFACTS["model_version_label"],
        "dataset_version": ARTIFACTS["dataset_version"],
        "model_loaded": True,
    }


@app.get("/livez")
def livez() -> dict[str, str]:
    """Liveness: the process is up and can accept connections. No artifact check."""
    return {"status": "alive"}


@app.get("/readyz", response_model=None)
def readyz() -> dict[str, Any] | JSONResponse:
    """Readiness: model, calibrator, fitted transforms, and cost params loaded.

    A failed startup leaves ARTIFACTS unset, which means one of those artifacts
    did not load. Kubernetes should drop the pod from traffic, not restart it.
    """
    if ARTIFACTS is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model_loaded": False},
        )
    return {
        "status": "ready",
        "model_version": ARTIFACTS["model_version_label"],
        "dataset_version": ARTIFACTS["dataset_version"],
        "model_loaded": True,
    }


@app.post("/predict", response_model=TransactionResponse)
def predict_endpoint(body: TransactionRequest, request: Request) -> TransactionResponse:
    request_id = request.state.request_id
    if ARTIFACTS is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Model not loaded", "request_id": request_id},
        )

    try:
        result = run_predict(
            body.feature_dict(),
            ARTIFACTS,
            request_id=request_id,
        )
    except Exception as exc:
        # Class name only. The message and traceback can contain raw inputs.
        logger.error(
            "Prediction failed request_id=%s exception_class=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "Prediction failed", "request_id": request_id},
        ) from None

    request.state.recommended_action = result["recommended_action"]
    record_prediction(result["recommended_action"])
    return TransactionResponse(**result)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
