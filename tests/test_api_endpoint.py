"""FastAPI TestClient integration tests for the serving API."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        yield test_client


def _base_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "card1": 10000,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 264.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "dist1": 5.0,
        "dist2": None,
        "TransactionDT": 86500.0,
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "id_30": "Windows 10",
        "id_31": "chrome",
    }
    payload.update(overrides)
    return payload


def test_successful_prediction(client: TestClient):
    resp = client.post("/predict", json=_base_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "fraud_probability",
        "recommended_action",
        "action_costs",
        "model_version",
        "dataset_version",
        "request_id",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["recommended_action"] in {"approve", "step_up", "review", "decline"}
    assert body["model_version"] == "v9"
    assert body["dataset_version"] == 5
    uuid.UUID(body["request_id"])


def test_missing_transaction_amt_returns_422(client: TestClient):
    payload = _base_payload()
    del payload["TransactionAmt"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_wrong_type_transaction_amt_returns_422(client: TestClient):
    resp = client.post("/predict", json=_base_payload(TransactionAmt="abc"))
    assert resp.status_code == 422


def test_extreme_near_zero_amount(client: TestClient):
    resp = client.post("/predict", json=_base_payload(TransactionAmt=0.001))
    assert resp.status_code == 200


def test_extreme_large_amount(client: TestClient):
    resp = client.post("/predict", json=_base_payload(TransactionAmt=50000.0))
    assert resp.status_code == 200


def test_all_null_historical_features(client: TestClient):
    hist: dict[str, Any] = {f"D{i}": None for i in range(1, 16)}
    hist.update({f"C{i}": None for i in range(1, 15)})
    hist.update({f"V{i}": None for i in range(1, 50)})  # representative V block
    resp = client.post("/predict", json=_base_payload(**hist))
    assert resp.status_code == 200


def test_unseen_categorical_productcd(client: TestClient):
    resp = client.post("/predict", json=_base_payload(ProductCD="Z"))
    assert resp.status_code == 200


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == "v9"
    assert body["dataset_version"] == 5
    assert body["model_loaded"] is True


def test_negative_amount_returns_422(client: TestClient):
    resp = client.post("/predict", json=_base_payload(TransactionAmt=-1.0))
    assert resp.status_code == 422


def test_response_has_no_raw_input_fields(client: TestClient):
    body = client.post("/predict", json=_base_payload()).json()
    for bad in (
        "TransactionAmt",
        "card1",
        "P_emaildomain",
        "addr1",
        "features",
        "ProductCD",
        "R_emaildomain",
        "dist1",
    ):
        assert bad not in body
    for key in body:
        assert not key.startswith("card")
        assert "email" not in key.lower()
        assert not key.startswith("addr")


def test_request_id_unique(client: TestClient):
    a = client.post("/predict", json=_base_payload()).json()["request_id"]
    b = client.post("/predict", json=_base_payload()).json()["request_id"]
    assert a != b


def test_action_costs_four_keys(client: TestClient):
    body = client.post("/predict", json=_base_payload()).json()
    assert set(body["action_costs"].keys()) == {
        "approve",
        "step_up",
        "review",
        "decline",
    }


def test_dataset_version_present(client: TestClient):
    body = client.post("/predict", json=_base_payload()).json()
    assert "dataset_version" in body
    assert body["dataset_version"] == 5
