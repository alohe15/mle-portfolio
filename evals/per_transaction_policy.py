"""Per-transaction argmin decision policy.

Chooses approve / step_up / review / decline by minimizing expected cost
for each transaction independently given calibrated probability and amount.

Usage (library):
    from per_transaction_policy import apply_argmin_policy, run_evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVALS_DIR = Path(__file__).resolve().parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from cost_model import ACTION_NAMES, compute_all_costs  # noqa: E402

CATCH_RATE_BY_ACTION = {
    "approve": 0.0,
    "step_up": "step_up_stop_rate",
    "review": "review_catch_rate",
    "decline": 1.0,
}


def apply_argmin_policy(
    calibrated_probs: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series,
    cost_params: dict[str, float],
) -> pd.DataFrame:
    """Compute per-row costs and choose the minimum-cost action."""
    p = np.asarray(calibrated_probs, dtype=float)
    a = np.asarray(amounts, dtype=float)
    if p.shape != a.shape:
        raise ValueError(f"Shape mismatch: probs {p.shape} vs amounts {a.shape}")

    costs = compute_all_costs(p, a, cost_params)
    cost_matrix = np.column_stack([np.asarray(costs[name], dtype=float) for name in ACTION_NAMES])
    best_idx = np.argmin(cost_matrix, axis=1)
    chosen_cost = cost_matrix[np.arange(len(p)), best_idx]

    # Second-best cost for cost_gap (difference between chosen and runner-up).
    partitioned = np.partition(cost_matrix, 1, axis=1)
    second_best = partitioned[:, 1]
    cost_gap = second_best - chosen_cost

    actions = np.asarray(ACTION_NAMES, dtype=object)[best_idx]
    return pd.DataFrame(
        {
            "action": actions,
            "cost_approve": cost_matrix[:, 0],
            "cost_step_up": cost_matrix[:, 1],
            "cost_review": cost_matrix[:, 2],
            "cost_decline": cost_matrix[:, 3],
            "chosen_cost": chosen_cost,
            "cost_gap": cost_gap,
        }
    )


def _catch_rates(actions: pd.Series, cost_params: dict[str, float]) -> np.ndarray:
    rates = np.zeros(len(actions), dtype=float)
    for action, spec in CATCH_RATE_BY_ACTION.items():
        mask = actions.to_numpy() == action
        if isinstance(spec, str):
            rates[mask] = float(cost_params[spec])
        else:
            rates[mask] = float(spec)
    return rates


def run_evaluation(
    policy_df: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series,
    cost_params: dict[str, float],
    *,
    n_days: float,
) -> dict:
    """Summarize policy outcomes on labeled validation transactions."""
    labels_arr = np.asarray(labels, dtype=float)
    amounts_arr = np.asarray(amounts, dtype=float)
    actions = policy_df["action"]
    n = len(policy_df)
    if n == 0:
        raise ValueError("Empty policy_df")
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")

    catch = _catch_rates(actions, cost_params)
    is_fraud = labels_arr >= 0.5
    fraud_caught_count = float(np.sum(is_fraud * catch))
    fraud_caught_dollars = float(np.sum(is_fraud * catch * amounts_arr))

    total_expected_cost = float(policy_df["chosen_cost"].sum())
    n_review = int((actions == "review").sum())
    reviews_per_day = n_review / float(n_days)
    declined_legit = int(((actions == "decline") & (~is_fraud)).sum())

    breakdown = []
    for action in ACTION_NAMES:
        mask = actions.to_numpy() == action
        count = int(mask.sum())
        fraud_count = int(np.sum(is_fraud & mask))
        breakdown.append(
            {
                "action": action,
                "count": count,
                "percentage": 100.0 * count / n,
                "fraud_count": fraud_count,
                "fraud_percentage": (100.0 * fraud_count / count) if count else 0.0,
                "precision": (fraud_count / count) if count else 0.0,
            }
        )

    return {
        "fraud_caught_count": fraud_caught_count,
        "fraud_caught_dollars": fraud_caught_dollars,
        "total_expected_cost": total_expected_cost,
        "reviews_per_day": reviews_per_day,
        "n_review": n_review,
        "declined_legit": declined_legit,
        "n_days": float(n_days),
        "n_transactions": n,
        "per_action": breakdown,
    }
