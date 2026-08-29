"""Expected value cost model for fraud decision policy.

Each function computes the expected cost of taking an action on a single
transaction with calibrated fraud probability p and dollar amount a.
All functions are pure, stateless, and idempotent (§2f compliance).

Cost parameters come from configs/decision_policy_v1.json, never hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CostParams:
    """Cost model parameters — all loaded from config, none hardcoded (§3)."""

    fraud_cost_multiplier: float = 2.0
    step_up_stop_rate: float = 0.70
    step_up_friction_rate: float = 0.05
    review_catch_rate: float = 0.95
    review_cost_per_txn: float = 10.00
    review_capacity_per_day: int = 200
    decline_revenue_loss: float = 1.0

    @classmethod
    def from_config(cls, config_path: str | Path) -> CostParams:
        config = json.loads(Path(config_path).read_text())
        return cls(**config["cost_assumptions"])

    @classmethod
    def from_dict(cls, d: dict) -> CostParams:
        return cls(**d)


def cost_approve(p: float, a: float, params: CostParams) -> float:
    """Expected cost of auto-approving a transaction."""
    return p * a * params.fraud_cost_multiplier


def cost_step_up(p: float, a: float, params: CostParams) -> float:
    """Expected cost of requesting extra authentication."""
    fraud_residual = p * a * params.fraud_cost_multiplier * (1 - params.step_up_stop_rate)
    friction_loss = (1 - p) * a * params.step_up_friction_rate
    return fraud_residual + friction_loss


def cost_review(p: float, a: float, params: CostParams) -> float:
    """Expected cost of routing to manual review."""
    fraud_residual = p * a * params.fraud_cost_multiplier * (1 - params.review_catch_rate)
    return fraud_residual + params.review_cost_per_txn


def cost_decline(p: float, a: float, params: CostParams) -> float:
    """Expected cost of auto-declining a transaction."""
    return (1 - p) * a * params.decline_revenue_loss


def optimal_action(p: float, a: float, params: CostParams) -> tuple[str, float]:
    """Return the action with minimum expected cost and that cost."""
    costs = {
        "approve": cost_approve(p, a, params),
        "step_up": cost_step_up(p, a, params),
        "review": cost_review(p, a, params),
        "decline": cost_decline(p, a, params),
    }
    best = min(costs, key=costs.get)
    return best, costs[best]


def batch_optimal_actions(
    probs: np.ndarray, amounts: np.ndarray, params: CostParams
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal action for each transaction. Returns (actions, min_costs)."""
    c_approve = probs * amounts * params.fraud_cost_multiplier
    c_step_up = (
        probs * amounts * params.fraud_cost_multiplier * (1 - params.step_up_stop_rate)
        + (1 - probs) * amounts * params.step_up_friction_rate
    )
    c_review = (
        probs * amounts * params.fraud_cost_multiplier * (1 - params.review_catch_rate)
        + params.review_cost_per_txn
    )
    c_decline = (1 - probs) * amounts * params.decline_revenue_loss

    all_costs = np.stack([c_approve, c_step_up, c_review, c_decline], axis=1)
    best_idx = np.argmin(all_costs, axis=1)
    min_costs = all_costs[np.arange(len(best_idx)), best_idx]

    action_names = np.array(["approve", "step_up", "review", "decline"])
    actions = action_names[best_idx]

    return actions, min_costs
