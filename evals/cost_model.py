"""Per-transaction expected-cost model for decision policy.

Four actions — approve, step-up, review, decline — each with an expected
cost given calibrated fraud probability p, transaction amount a, and a
shared cost-parameter dict.

Usage:
    from cost_model import load_cost_params, optimal_action
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs" / "decision_policy_v1.json"

ACTION_NAMES = ("approve", "step_up", "review", "decline")


def load_cost_params(config_path: Path | str | None = None) -> dict[str, float]:
    """Load cost_params from a decision-policy JSON config."""
    path = Path(config_path) if config_path is not None else DEFAULT_POLICY_CONFIG
    doc = json.loads(path.read_text())
    if "cost_params" not in doc:
        raise KeyError(f"Missing cost_params in {path}")
    params = dict(doc["cost_params"])
    required = {
        "fraud_cost_multiplier",
        "step_up_stop_rate",
        "step_up_friction_rate",
        "review_catch_rate",
        "review_cost_per_txn",
        "review_capacity_per_day",
        "decline_revenue_loss",
    }
    missing = required - set(params)
    if missing:
        raise KeyError(f"Missing cost param keys: {sorted(missing)}")
    return {k: float(params[k]) for k in required}


def _as_arrays(
    p: Any, a: Any
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Broadcast p and a to arrays; return (p, a, was_scalar)."""
    p_arr = np.asarray(p, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    was_scalar = p_arr.ndim == 0 and a_arr.ndim == 0
    p_arr, a_arr = np.broadcast_arrays(p_arr, a_arr)
    return p_arr, a_arr, was_scalar


def _maybe_scalar(value: np.ndarray, was_scalar: bool) -> Any:
    if was_scalar:
        return float(value)
    return value


def cost_approve(p: Any, a: Any, params: dict[str, float]) -> Any:
    """Expected cost of approving: residual fraud loss."""
    p_arr, a_arr, was_scalar = _as_arrays(p, a)
    out = p_arr * a_arr * params["fraud_cost_multiplier"]
    return _maybe_scalar(out, was_scalar)


def cost_step_up(p: Any, a: Any, params: dict[str, float]) -> Any:
    """Expected cost of step-up authentication."""
    p_arr, a_arr, was_scalar = _as_arrays(p, a)
    f = params["fraud_cost_multiplier"]
    stop = params["step_up_stop_rate"]
    friction = params["step_up_friction_rate"]
    out = (
        p_arr * a_arr * f * (1.0 - stop)
        + (1.0 - p_arr) * a_arr * friction
    )
    return _maybe_scalar(out, was_scalar)


def cost_review(p: Any, a: Any, params: dict[str, float]) -> Any:
    """Expected cost of manual review."""
    p_arr, a_arr, was_scalar = _as_arrays(p, a)
    f = params["fraud_cost_multiplier"]
    catch = params["review_catch_rate"]
    review_cost = params["review_cost_per_txn"]
    out = p_arr * a_arr * f * (1.0 - catch) + review_cost
    return _maybe_scalar(out, was_scalar)


def cost_decline(p: Any, a: Any, params: dict[str, float]) -> Any:
    """Expected cost of declining: lost revenue on legitimate traffic."""
    p_arr, a_arr, was_scalar = _as_arrays(p, a)
    out = (1.0 - p_arr) * a_arr * params["decline_revenue_loss"]
    return _maybe_scalar(out, was_scalar)


_COST_FNS = {
    "approve": cost_approve,
    "step_up": cost_step_up,
    "review": cost_review,
    "decline": cost_decline,
}


def compute_all_costs(
    p: Any, a: Any, params: dict[str, float]
) -> dict[str, Any]:
    """Return expected cost for each action (scalar or array-valued)."""
    return {name: fn(p, a, params) for name, fn in _COST_FNS.items()}


def optimal_action(
    p: Any, a: Any, params: dict[str, float]
) -> tuple[Any, dict[str, Any]]:
    """Return (action_name, cost_dict) minimizing expected cost.

    For array inputs, action_name is a numpy object array of strings.
    Ties break in favor of the earlier ACTION_NAMES entry (approve first).
    """
    costs = compute_all_costs(p, a, params)
    p_arr, _, was_scalar = _as_arrays(p, a)

    stacked = np.stack([np.asarray(costs[name], dtype=float) for name in ACTION_NAMES], axis=0)
    best_idx = np.argmin(stacked, axis=0)
    names = np.asarray(ACTION_NAMES, dtype=object)
    actions = names[best_idx]

    if was_scalar:
        return str(actions), {k: float(np.asarray(v)) for k, v in costs.items()}
    return actions, costs
