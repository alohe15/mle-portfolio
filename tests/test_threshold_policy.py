"""Tests for decision policy — boundaries, edge cases, cost model, calibration, config.

Run: python -m pytest tests/test_threshold_policy.py -v
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.cost_model import (
    CostParams,
    cost_approve,
    cost_decline,
    cost_review,
    cost_step_up,
    optimal_action,
)
from evals.threshold_policy import assign_actions, compute_band_table


class TestBoundaryScores:
    """Verify action assignment at exact threshold boundaries."""

    def test_score_at_t1_is_step_up(self):
        actions = assign_actions(np.array([0.40]), 0.40, 0.80, 0.95)
        assert actions[0] == "step_up"

    def test_score_at_t2_is_review(self):
        actions = assign_actions(np.array([0.80]), 0.40, 0.80, 0.95)
        assert actions[0] == "review"

    def test_score_at_t3_is_decline(self):
        actions = assign_actions(np.array([0.95]), 0.40, 0.80, 0.95)
        assert actions[0] == "decline"

    def test_score_zero_is_approve(self):
        actions = assign_actions(np.array([0.0]), 0.40, 0.80, 0.95)
        assert actions[0] == "approve"

    def test_score_one_is_decline(self):
        actions = assign_actions(np.array([1.0]), 0.40, 0.80, 0.95)
        assert actions[0] == "decline"

    def test_score_just_below_t1_is_approve(self):
        actions = assign_actions(np.array([0.3999]), 0.40, 0.80, 0.95)
        assert actions[0] == "approve"

    def test_all_four_actions_assigned(self):
        scores = np.array([0.1, 0.5, 0.85, 0.99])
        actions = assign_actions(scores, 0.40, 0.80, 0.95)
        assert list(actions) == ["approve", "step_up", "review", "decline"]


class TestEmptyBands:
    """Thresholds that produce empty bands should not crash."""

    def test_all_in_approve(self):
        scores = np.array([0.01, 0.02, 0.03])
        actions = assign_actions(scores, 0.99, 0.995, 0.999)
        assert all(a == "approve" for a in actions)

    def test_empty_middle_bands(self):
        scores = np.array([0.01, 0.99])
        labels = np.array([0, 1])
        amounts = np.array([100.0, 200.0])
        actions = assign_actions(scores, 0.40, 0.80, 0.95)
        params = CostParams()
        rows = compute_band_table(scores, labels, amounts, actions, params)
        assert len(rows) == 4


class TestExtremeSamples:
    """All-fraud and no-fraud edge cases."""

    def test_all_fraud(self):
        scores = np.array([0.1, 0.5, 0.85, 0.99])
        labels = np.ones(4)
        amounts = np.array([100.0, 200.0, 300.0, 400.0])
        actions = assign_actions(scores, 0.40, 0.80, 0.95)
        params = CostParams()
        rows = compute_band_table(scores, labels, amounts, actions, params)
        total_fraud = sum(r["fraud_count"] for r in rows)
        assert total_fraud == 4

    def test_no_fraud(self):
        scores = np.array([0.1, 0.5, 0.85, 0.99])
        labels = np.zeros(4)
        amounts = np.array([100.0, 200.0, 300.0, 400.0])
        actions = assign_actions(scores, 0.40, 0.80, 0.95)
        params = CostParams()
        rows = compute_band_table(scores, labels, amounts, actions, params)
        total_fraud = sum(r["fraud_count"] for r in rows)
        assert total_fraud == 0
        assert all(r["precision"] == 0.0 for r in rows if r["txn_count"] > 0)


class TestCostModel:
    """Verify cost functions return expected values."""

    def setup_method(self):
        self.params = CostParams(
            fraud_cost_multiplier=2.0,
            step_up_stop_rate=0.70,
            step_up_friction_rate=0.05,
            review_catch_rate=0.95,
            review_cost_per_txn=10.00,
            decline_revenue_loss=1.0,
        )

    def test_approve_cost(self):
        assert cost_approve(0.5, 100.0, self.params) == pytest.approx(100.0)

    def test_step_up_cost(self):
        assert cost_step_up(0.5, 100.0, self.params) == pytest.approx(32.5)

    def test_review_cost(self):
        assert cost_review(0.5, 100.0, self.params) == pytest.approx(15.0)

    def test_decline_cost(self):
        assert cost_decline(0.5, 100.0, self.params) == pytest.approx(50.0)

    def test_optimal_action_at_known_crossover(self):
        action, cost = optimal_action(0.5, 100.0, self.params)
        assert action == "review"
        assert cost == pytest.approx(15.0)

    def test_approve_cheapest_at_low_p(self):
        action, _ = optimal_action(0.001, 50.0, self.params)
        assert action == "approve"

    def test_decline_cheapest_at_high_p(self):
        action, _ = optimal_action(0.999, 50.0, self.params)
        assert action == "decline"

    def test_cost_approve_increases_with_p(self):
        c1 = cost_approve(0.1, 100.0, self.params)
        c2 = cost_approve(0.9, 100.0, self.params)
        assert c2 > c1

    def test_cost_decline_decreases_with_p(self):
        c1 = cost_decline(0.1, 100.0, self.params)
        c2 = cost_decline(0.9, 100.0, self.params)
        assert c2 < c1


class TestCalibration:
    """Verify calibrator properties if it exists."""

    @pytest.fixture
    def calibrator(self):
        cal_path = PROJECT_ROOT / "models" / "lgbm_v9_calibrator.pkl"
        if not cal_path.exists():
            pytest.skip("Calibrator not found — run evals/calibrate.py first")
        with cal_path.open("rb") as f:
            return pickle.load(f)

    def test_calibrated_scores_in_range(self, calibrator):
        test_scores = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        calibrated = calibrator.transform(test_scores)
        assert all(0 <= s <= 1 for s in calibrated)

    def test_calibrator_is_idempotent_ish(self, calibrator):
        scores = np.array([0.01, 0.05, 0.10])
        cal1 = calibrator.transform(scores)
        cal2 = calibrator.transform(cal1)
        assert all(0 <= s <= 1 for s in cal2)


class TestConfigSchema:
    """Verify decision_policy_v1.json structure."""

    @pytest.fixture
    def policy(self):
        path = PROJECT_ROOT / "configs" / "decision_policy_v1.json"
        if not path.exists():
            pytest.skip("decision_policy_v1.json not found")
        return json.loads(path.read_text())

    def test_required_fields(self, policy):
        required = [
            "policy_version",
            "model_version",
            "dataset_version",
            "status",
            "cost_assumptions",
            "actions",
        ]
        for field in required:
            assert field in policy, f"Missing field: {field}"

    def test_status_is_valid(self, policy):
        assert policy["status"] in ("proposed", "active", "retired")

    def test_actions_cover_zero_to_one(self, policy):
        actions = sorted(policy["actions"], key=lambda a: a["lower"])
        assert actions[0]["lower"] == 0.0
        assert actions[-1]["upper"] == 1.0

    def test_actions_no_gaps(self, policy):
        actions = sorted(policy["actions"], key=lambda a: a["lower"])
        for i in range(len(actions) - 1):
            assert actions[i]["upper"] == actions[i + 1]["lower"], (
                f"Gap between {actions[i]['action']} upper={actions[i]['upper']} "
                f"and {actions[i + 1]['action']} lower={actions[i + 1]['lower']}"
            )

    def test_policy_matches_serving_model(self, policy):
        reg = json.loads((PROJECT_ROOT / "models" / "registry.json").read_text())
        serving = [e for e in reg if e.get("is_serving")]
        assert serving, "No serving model"
        assert policy["model_version"] == serving[0]["version"]
        assert policy["dataset_version"] == serving[0]["dataset_version"]

    def test_four_actions(self, policy):
        assert len(policy["actions"]) == 4
        names = {a["action"] for a in policy["actions"]}
        assert names == {"approve", "step_up", "review", "decline"}


class TestAPIIntegration:
    """Verify API returns action field when policy is loaded."""

    @pytest.fixture
    def api_client(self):
        try:
            from fastapi.testclient import TestClient

            sys.path.insert(0, str(PROJECT_ROOT / "services" / "api"))
            from app import app

            return TestClient(app)
        except Exception as exc:
            pytest.skip(f"Cannot load API: {exc}")

    @pytest.fixture
    def sample_features(self) -> dict:
        import pandas as pd

        data_path = PROJECT_ROOT / "data" / "raw" / "train_merged.parquet"
        if not data_path.exists():
            pytest.skip("Raw training data not available for API integration test")
        df = pd.read_parquet(data_path)
        sample = df.iloc[0].drop("isFraud", errors="ignore").to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in sample.items()}

    def test_response_has_action(self, api_client, sample_features):
        response = api_client.post(
            "/predict",
            json={"features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ("approve", "step_up", "review", "decline")
        assert "policy_version" in data
        assert "raw_score" in data
        assert "fraud_probability" in data

    def test_response_has_calibrated_score(self, api_client, sample_features):
        response = api_client.post(
            "/predict",
            json={"features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["raw_score"] <= 1
