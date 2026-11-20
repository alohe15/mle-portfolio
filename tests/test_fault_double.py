"""The prediction double fails /predict and leaves readiness alone."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.api.observability import ACCESS_LOGGER_NAME

GOLDEN = Path(__file__).parent / "fixtures" / "golden_request.json"
MARKER = "drill-injected-failure-7f3a"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from tests.support.failing_api import app

    with TestClient(app) as test_client:
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
    return [json.loads(record.getMessage()) for record in records]


def test_golden_request_hides_the_injected_failure(
    client: TestClient, access_records: list[logging.LogRecord]
):
    resp = client.post("/predict", json=json.loads(GOLDEN.read_text()))
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"]["request_id"]
    assert body["detail"]["request_id"] == resp.headers["x-request-id"]
    assert MARKER not in resp.text
    entries = _entries(access_records)
    assert len(entries) == 1
    assert entries[0]["error_category"] == "prediction_error"
    assert MARKER not in json.dumps(entries)


def test_readyz_stays_ready(client: TestClient):
    assert client.get("/readyz").status_code == 200
