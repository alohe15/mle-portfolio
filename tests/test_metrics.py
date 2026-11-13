"""Prometheus series for the serving API. Counters are process-global."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from services.api.app import app
from services.api.metrics import ACTION_NAMES

GOLDEN = Path(__file__).parent / "fixtures" / "golden_request.json"
METRIC_NAMES = (
    "fraud_api_http_requests_total",
    "fraud_api_http_request_duration_seconds",
    "fraud_api_predictions_total",
    "fraud_api_model_info",
    "fraud_api_model_loaded",
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else float(value)


def _prediction_values() -> dict[str, float]:
    """Counter values by action. The family name drops the ``_total`` suffix."""
    found: dict[str, float] = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != "fraud_api_predictions_total":
                continue
            found[sample.labels["action"]] = float(sample.value)
    return found


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        yield test_client


def test_metrics_endpoint_lists_families(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    for name in METRIC_NAMES:
        assert name in resp.text


def test_golden_predict_increments_request_and_action(client: TestClient):
    payload = json.loads(GOLDEN.read_text())
    before_requests = _sample(
        "fraud_api_http_requests_total",
        {"route": "/predict", "method": "POST", "status_class": "2xx"},
    )
    before_duration = _sample(
        "fraud_api_http_request_duration_seconds_count",
        {"route": "/predict", "method": "POST", "status_class": "2xx"},
    )
    before_predictions = _prediction_values()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    action = resp.json()["recommended_action"]
    assert (
        _sample(
            "fraud_api_http_requests_total",
            {"route": "/predict", "method": "POST", "status_class": "2xx"},
        )
        - before_requests
        == 1
    )
    assert (
        _sample(
            "fraud_api_http_request_duration_seconds_count",
            {"route": "/predict", "method": "POST", "status_class": "2xx"},
        )
        - before_duration
        == 1
    )
    after_predictions = _prediction_values()
    assert after_predictions[action] - before_predictions.get(action, 0.0) == 1


def test_validation_error_does_not_count_a_prediction(client: TestClient):
    payload = json.loads(GOLDEN.read_text())
    del payload["TransactionAmt"]
    before_requests = _sample(
        "fraud_api_http_requests_total",
        {"route": "/predict", "method": "POST", "status_class": "4xx"},
    )
    before_duration_2xx = _sample(
        "fraud_api_http_request_duration_seconds_count",
        {"route": "/predict", "method": "POST", "status_class": "2xx"},
    )
    before_predictions = _prediction_values()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert (
        _sample(
            "fraud_api_http_requests_total",
            {"route": "/predict", "method": "POST", "status_class": "4xx"},
        )
        - before_requests
        == 1
    )
    assert (
        _sample(
            "fraud_api_http_request_duration_seconds_count",
            {"route": "/predict", "method": "POST", "status_class": "2xx"},
        )
        == before_duration_2xx
    )
    assert _prediction_values() == before_predictions


def test_prediction_error_does_not_count_a_prediction(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("metrics-should-not-see-this")

    monkeypatch.setattr("services.api.app.run_predict", _boom)
    before_requests = _sample(
        "fraud_api_http_requests_total",
        {"route": "/predict", "method": "POST", "status_class": "5xx"},
    )
    before_predictions = _prediction_values()
    resp = client.post("/predict", json=json.loads(GOLDEN.read_text()))
    assert resp.status_code == 500
    assert (
        _sample(
            "fraud_api_http_requests_total",
            {"route": "/predict", "method": "POST", "status_class": "5xx"},
        )
        - before_requests
        == 1
    )
    assert _prediction_values() == before_predictions


def test_model_gauges_after_load(client: TestClient):
    assert _sample("fraud_api_model_loaded") == 1
    assert (
        _sample(
            "fraud_api_model_info",
            {"model_version": "v9", "dataset_version": "5"},
        )
        == 1
    )


def test_metrics_text_omits_request_identifiers(client: TestClient):
    resp = client.post("/predict", json=json.loads(GOLDEN.read_text()))
    assert resp.status_code == 200
    request_id = resp.json()["request_id"]
    body = client.get("/metrics").text
    assert request_id not in body
    assert "SyntheticOS" not in body

    missed = client.get("/predict/abc123")
    assert missed.status_code == 404
    body = client.get("/metrics").text
    assert "abc123" not in body
    assert 'route="unmatched"' in body


def test_prediction_action_labels_stay_finite():
    observed: set[str] = set()
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name not in {
                "fraud_api_predictions_total",
                "fraud_api_predictions_created",
            }:
                continue
            observed.add(sample.labels["action"])
    assert observed <= (set(ACTION_NAMES) | {"other"})
    assert set(ACTION_NAMES) <= observed
