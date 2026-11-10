"""Access-log and safe-error coverage. The access logger does not propagate."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from services.api.observability import ACCESS_LOGGER_NAME

GOLDEN = Path(__file__).parent / "fixtures" / "golden_request.json"
PAYLOAD_MARKER = "payload-marker-do-not-log"
CLIENT_REQUEST_ID = "client-sent-id-should-be-ignored"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        yield test_client


@pytest.fixture
def access_records():
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    handler = _Capture()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _entries(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    assert records, "expected a completion log"
    messages = [record.getMessage() for record in records]
    for message in messages:
        assert "\n" not in message
    return [json.loads(message) for message in messages]


def _assert_timestamp(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(key)
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def test_valid_predict_logs_one_line_and_matching_request_id(
    client: TestClient, access_records: list[logging.LogRecord]
):
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    assert logger.propagate is False
    payload = json.loads(GOLDEN.read_text())
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    entries = _entries(access_records)
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == {
        "event",
        "timestamp",
        "request_id",
        "route",
        "method",
        "status_class",
        "status_code",
        "duration_ms",
        "model_version",
        "error_category",
        "recommended_action",
    }
    assert entry["event"] == "request_completed"
    _assert_timestamp(entry["timestamp"])
    assert entry["route"] == "/predict"
    assert entry["method"] == "POST"
    assert entry["status_class"] == "2xx"
    assert entry["status_code"] == 200
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0
    assert entry["model_version"] == "v9"
    assert entry["error_category"] == "none"
    assert entry["recommended_action"] == body["recommended_action"]
    request_id = body["request_id"]
    assert resp.headers["x-request-id"] == request_id == entry["request_id"]
    assert uuid.UUID(request_id).version == 4


def test_missing_transaction_amt_hides_input(
    client: TestClient, access_records: list[logging.LogRecord]
):
    payload = json.loads(GOLDEN.read_text())
    del payload["TransactionAmt"]
    payload["DeviceInfo"] = PAYLOAD_MARKER
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail", "request_id"}
    assert body["request_id"] == resp.headers["x-request-id"]
    assert isinstance(body["detail"], list) and body["detail"]
    for error in body["detail"]:
        assert set(error) == {"type", "loc", "msg"}
    assert "input" not in _keys(body)
    assert "ctx" not in _keys(body)
    entries = _entries(access_records)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["route"] == "/predict"
    assert entry["status_code"] == 422
    assert entry["status_class"] == "4xx"
    assert entry["error_category"] == "validation_error"
    assert entry["recommended_action"] is None
    assert entry["request_id"] == body["request_id"]
    logged = json.dumps(entry)
    assert PAYLOAD_MARKER not in logged
    assert "gmail.com" not in logged
    assert PAYLOAD_MARKER not in resp.text


def test_prediction_failure_hides_exception_message(
    client: TestClient,
    access_records: list[logging.LogRecord],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("secret-marker-123")

    monkeypatch.setattr("services.api.app.run_predict", _boom)
    caplog.set_level(logging.INFO)
    resp = client.post("/predict", json=json.loads(GOLDEN.read_text()))
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"]["message"] == "Prediction failed"
    assert body["detail"]["request_id"] == resp.headers["x-request-id"]
    assert "secret-marker-123" not in resp.text
    entries = _entries(access_records)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["error_category"] == "prediction_error"
    assert entry["status_code"] == 500
    assert entry["status_class"] == "5xx"
    assert entry["route"] == "/predict"
    assert entry["recommended_action"] is None
    assert entry["request_id"] == body["detail"]["request_id"]
    assert "secret-marker-123" not in json.dumps(entry)
    assert "secret-marker-123" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert entry["request_id"] in caplog.text


def test_unknown_path_is_unmatched_and_omits_raw_path(
    client: TestClient, access_records: list[logging.LogRecord]
):
    resp = client.post("/predict/abc123")
    assert resp.status_code == 404
    entries = _entries(access_records)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["route"] == "unmatched"
    assert entry["error_category"] == "not_found"
    assert entry["status_code"] == 404
    assert "abc123" not in json.dumps(entry)
    assert resp.headers["x-request-id"] == entry["request_id"]


def test_client_request_id_is_ignored(
    client: TestClient, access_records: list[logging.LogRecord]
):
    resp = client.post(
        "/predict",
        json=json.loads(GOLDEN.read_text()),
        headers={"X-Request-ID": CLIENT_REQUEST_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    entries = _entries(access_records)
    assert len(entries) == 1
    entry = entries[0]
    assert resp.headers["x-request-id"] == body["request_id"] == entry["request_id"]
    assert entry["request_id"] != CLIENT_REQUEST_ID
    assert CLIENT_REQUEST_ID not in json.dumps(entry)
    assert uuid.UUID(entry["request_id"]).version == 4
