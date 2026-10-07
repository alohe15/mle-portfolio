"""Unit tests for Phase 3 decision policy (cost model, thresholds, calibrator)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
CALIBRATOR_PATH = REPO_ROOT / "models" / "lgbm_v9_calibrator.pkl"
POLICY_CONFIG = REPO_ROOT / "configs" / "decision_policy_v1.json"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from cost_model import (  # noqa: E402
    cost_approve,
    cost_decline,
    cost_review,
    cost_step_up,
    load_cost_params,
    optimal_action,
)
from per_transaction_policy import apply_argmin_policy, run_evaluation  # noqa: E402
from threshold_policy import apply_threshold_policy, apply_threshold_policy_frame  # noqa: E402


@pytest.fixture(scope="module")
def params() -> dict[str, float]:
    return load_cost_params(POLICY_CONFIG)


def test_config_is_argmin_policy():
    import json

    doc = json.loads(POLICY_CONFIG.read_text())
    assert doc["policy_type"] == "per_transaction_argmin"
    assert "thresholds" not in doc
    assert "t1" not in doc.get("cost_params", {})


def test_cost_approve_zero_when_p_zero(params):
    for a in (1.0, 50.0, 5000.0):
        assert cost_approve(0.0, a, params) == 0.0


def test_p0_always_approve(params):
    for a in (5.0, 100.0, 5000.0, 1e6):
        action, costs = optimal_action(0.0, a, params)
        assert action == "approve"
        assert costs["approve"] <= min(costs.values())


def test_p1_approve_never_cheapest(params):
    for a in (5.0, 100.0, 5000.0):
        action, costs = optimal_action(1.0, a, params)
        assert action != "approve"
        assert costs["approve"] > min(costs.values())


def test_small_amount_never_reviewed(params):
    action, costs = optimal_action(0.5, 5.0, params)
    assert action != "review"
    assert costs["review"] >= max(
        cost_approve(0.5, 5.0, params),
        # review fixed cost alone equals max fraud loss at this amount
        5.0 * params["fraud_cost_multiplier"] * 0.5,
    )


def test_large_amount_review_efficient(params):
    """$5k at p=0.5: review beats approve; decline beats approve."""
    a, p = 5000.0, 0.5
    action, costs = optimal_action(p, a, params)
    assert costs["decline"] < costs["approve"]
    assert costs["review"] < costs["approve"]
    assert action == "review"


def test_low_risk_approval(params):
    action, costs = optimal_action(0.001, 100.0, params)
    assert action == "approve"
    assert abs(costs["approve"] - 0.20) < 1e-9


def test_threshold_monotonicity():
    t1, t2, t3 = 0.2, 0.5, 0.8
    probs = np.linspace(0.0, 1.0, 101)
    actions = apply_threshold_policy(probs, t1, t2, t3)
    rank = {"approve": 0, "step_up": 1, "review": 2, "decline": 3}
    ranks = actions.map(rank).to_numpy()
    assert np.all(np.diff(ranks) >= 0)


def test_argmin_dominates_threshold_on_random_set(params):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=500)
    a = rng.uniform(1.0, 2000.0, size=500)
    argmin_df = apply_argmin_policy(p, a, params)
    # Use mid-range thresholds as a fixed baseline.
    thr_df = apply_threshold_policy_frame(p, a, params, 0.2, 0.5, 0.8)
    assert argmin_df["chosen_cost"].sum() <= thr_df["chosen_cost"].sum() + 1e-6


def test_argmin_dominates_many_threshold_triplets(params):
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, size=200)
    a = rng.uniform(5.0, 5000.0, size=200)
    argmin_total = apply_argmin_policy(p, a, params)["chosen_cost"].sum()
    for t1, t2, t3 in [(0.1, 0.3, 0.6), (0.0, 0.4, 0.9), (0.25, 0.25, 0.75)]:
        thr_total = apply_threshold_policy_frame(p, a, params, t1, t2, t3)[
            "chosen_cost"
        ].sum()
        assert argmin_total <= thr_total + 1e-6


def test_calibrator_output_range():
    if not CALIBRATOR_PATH.exists():
        pytest.skip("calibrator pkl not present — run evals/calibrate.py first")
    calibrator = pickle.loads(CALIBRATOR_PATH.read_bytes())
    assert isinstance(calibrator, IsotonicRegression)
    x = np.linspace(-1.0, 2.0, 50)
    y = np.asarray(calibrator.predict(x), dtype=float)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_synthetic_isotonic_range():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0.0, 1.0, size=200)
    y = (raw > 0.7).astype(float)
    cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal.fit(raw, y)
    preds = cal.predict(np.linspace(-0.5, 1.5, 100))
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_vectorized_costs_match_scalars(params):
    p = np.array([0.0, 0.2, 0.8, 1.0])
    a = np.array([10.0, 50.0, 100.0, 500.0])
    for fn in (cost_approve, cost_step_up, cost_review, cost_decline):
        vec = fn(p, a, params)
        for i in range(len(p)):
            assert abs(float(vec[i]) - float(fn(p[i], a[i], params))) < 1e-9


def test_apply_argmin_columns(params):
    p = np.array([0.0, 0.5, 1.0])
    a = np.array([100.0, 100.0, 100.0])
    df = apply_argmin_policy(p, a, params)
    for col in (
        "action",
        "cost_approve",
        "cost_step_up",
        "cost_review",
        "cost_decline",
        "chosen_cost",
        "cost_gap",
    ):
        assert col in df.columns
    assert (df["cost_gap"] >= -1e-9).all()


def test_run_evaluation_schema(params):
    p = np.array([0.01, 0.4, 0.9])
    a = np.array([20.0, 200.0, 2000.0])
    labels = np.array([0.0, 1.0, 1.0])
    df = apply_argmin_policy(p, a, params)
    metrics = run_evaluation(df, labels, a, params, n_days=2.0)
    assert "fraud_caught_count" in metrics
    assert "fraud_caught_dollars" in metrics
    assert "total_expected_cost" in metrics
    assert "reviews_per_day" in metrics
    assert "declined_legit" in metrics
    assert len(metrics["per_action"]) == 4


def test_threshold_boundaries():
    actions = apply_threshold_policy(
        np.array([0.19, 0.20, 0.49, 0.50, 0.79, 0.80]), 0.20, 0.50, 0.80
    )
    assert list(actions) == [
        "approve",
        "step_up",
        "step_up",
        "review",
        "review",
        "decline",
    ]


def test_decline_cost_at_p1_is_zero(params):
    assert cost_decline(1.0, 999.0, params) == 0.0


def test_review_cost_includes_fixed_fee(params):
    # At p=0 residual fraud term vanishes → exactly review_cost_per_txn.
    assert cost_review(0.0, 1000.0, params) == params["review_cost_per_txn"]
