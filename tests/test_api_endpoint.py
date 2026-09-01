"""API endpoint tests — success, validation, edge cases, privacy.

Run: python -m pytest tests/test_api_endpoint.py -v
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestSuccessfulPrediction:
    def test_minimal_request(self, client):
        resp = client.post("/predict", json={"TransactionAmt": 75.0})
        assert resp.status_code == 200
        data = resp.json()
        assert "fraud_probability" in data
        assert "raw_score" in data
        assert "recommended_action" in data
        assert "model_version" in data
        assert "request_id" in data
        assert data["model_version"] == 9
        assert data["recommended_action"] in ("approve", "step_up", "review", "decline")

    def test_full_request(self, client):
        resp = client.post(
            "/predict",
            json={
                "TransactionAmt": 150.0,
                "TransactionDT": 86400,
                "card1": 1000,
                "card4": "visa",
                "ProductCD": "W",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "D1": 14.0,
                "C1": 1.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["raw_score"] <= 1.0

    def test_response_has_request_id(self, client):
        resp = client.post("/predict", json={"TransactionAmt": 50.0})
        data = resp.json()
        uuid.UUID(data["request_id"])

    def test_response_has_policy_version(self, client):
        resp = client.post("/predict", json={"TransactionAmt": 50.0})
        data = resp.json()
        assert "policy_version" in data


class TestValidationErrors:
    def test_missing_transaction_amt(self, client):
        resp = client.post("/predict", json={"card1": 1000})
        assert resp.status_code == 422

    def test_invalid_type_string_for_amount(self, client):
        resp = client.post("/predict", json={"TransactionAmt": "not_a_number"})
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_null_transaction_amt(self, client):
        resp = client.post("/predict", json={"TransactionAmt": None})
        assert resp.status_code == 422


class TestEdgeCases:
    def test_extreme_high_amount(self, client):
        resp = client.post("/predict", json={"TransactionAmt": 999999999.0})
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_zero_amount(self, client):
        resp = client.post("/predict", json={"TransactionAmt": 0.01})
        assert resp.status_code == 200

    def test_unseen_card_category(self, client):
        resp = client.post(
            "/predict",
            json={
                "TransactionAmt": 50.0,
                "card4": "brand_new_card_type_never_seen",
            },
        )
        assert resp.status_code == 200

    def test_all_optional_fields_null(self, client):
        resp = client.post(
            "/predict",
            json={
                "TransactionAmt": 100.0,
                "card1": None,
                "card4": None,
                "P_emaildomain": None,
            },
        )
        assert resp.status_code == 200

    def test_extra_unknown_fields_ignored(self, client):
        resp = client.post(
            "/predict",
            json={
                "TransactionAmt": 100.0,
                "completely_unknown_field": "value",
            },
        )
        assert resp.status_code == 200


class TestPrivacy:
    def test_response_does_not_echo_input(self, client):
        resp = client.post(
            "/predict",
            json={
                "TransactionAmt": 123.45,
                "P_emaildomain": "secret@test.com",
                "card1": 99999,
            },
        )
        data = resp.json()
        response_str = json.dumps(data)
        assert "123.45" not in response_str
        assert "secret@test.com" not in response_str
        assert "99999" not in response_str

    def test_error_response_does_not_leak_input(self, client):
        resp = client.post("/predict", json={"TransactionAmt": "bad"})
        assert resp.status_code == 422
        data = resp.json()
        response_str = json.dumps(data)
        assert "bad" not in response_str.lower()
        assert "request_id" in data


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == 9
        assert data["n_features"] == 464
