"""Fixed-threshold decision policy with capacity-aware grid search.

Thresholds on calibrated probability only (amount ignored at decision time):
    p < t1            → approve
    t1 ≤ p < t2       → step_up
    t2 ≤ p < t3       → review
    p ≥ t3            → decline

Usage (library):
    from threshold_policy import search_thresholds, apply_threshold_policy, run_evaluation
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
from per_transaction_policy import run_evaluation as _shared_run_evaluation  # noqa: E402

CAPACITY_PENALTY_WEIGHT = 1e6


def apply_threshold_policy(
    calibrated_probs: np.ndarray | pd.Series,
    t1: float,
    t2: float,
    t3: float,
) -> pd.Series:
    """Assign actions from a fixed threshold triplet."""
    if not (0.0 <= t1 <= t2 <= t3 <= 1.0):
        raise ValueError(f"Thresholds must satisfy 0 ≤ t1 ≤ t2 ≤ t3 ≤ 1, got {t1}, {t2}, {t3}")
    p = np.asarray(calibrated_probs, dtype=float)
    actions = np.full(p.shape, "approve", dtype=object)
    actions[(p >= t1) & (p < t2)] = "step_up"
    actions[(p >= t2) & (p < t3)] = "review"
    actions[p >= t3] = "decline"
    return pd.Series(actions, name="action")


def _expected_cost_for_actions(
    actions: np.ndarray,
    calibrated_probs: np.ndarray,
    amounts: np.ndarray,
    cost_params: dict[str, float],
) -> np.ndarray:
    costs = compute_all_costs(calibrated_probs, amounts, cost_params)
    out = np.empty(len(actions), dtype=float)
    for i, name in enumerate(ACTION_NAMES):
        mask = actions == name
        out[mask] = np.asarray(costs[name], dtype=float)[mask]
    return out


def policy_objective(
    calibrated_probs: np.ndarray,
    amounts: np.ndarray,
    cost_params: dict[str, float],
    t1: float,
    t2: float,
    t3: float,
    *,
    n_days: float,
    capacity_penalty_weight: float = CAPACITY_PENALTY_WEIGHT,
) -> float:
    """Total expected cost + penalty for exceeding daily review capacity."""
    actions = apply_threshold_policy(calibrated_probs, t1, t2, t3).to_numpy()
    total_cost = float(
        _expected_cost_for_actions(actions, calibrated_probs, amounts, cost_params).sum()
    )
    n_review = int(np.sum(actions == "review"))
    reviews_per_day = n_review / float(n_days)
    capacity = float(cost_params["review_capacity_per_day"])
    excess = max(0.0, reviews_per_day - capacity)
    return total_cost + capacity_penalty_weight * excess


def _candidate_triplets(step: float, lo: float = 0.0, hi: float = 1.0) -> list[tuple[float, float, float]]:
    grid = np.round(np.arange(lo, hi + 1e-12, step), 10)
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]
    triplets: list[tuple[float, float, float]] = []
    for i, t1 in enumerate(grid):
        for j in range(i, len(grid)):
            t2 = grid[j]
            for k in range(j, len(grid)):
                t3 = grid[k]
                triplets.append((float(t1), float(t2), float(t3)))
    return triplets


def _fast_objective(
    p: np.ndarray,
    cost_approve_arr: np.ndarray,
    cost_step_up_arr: np.ndarray,
    cost_review_arr: np.ndarray,
    cost_decline_arr: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    *,
    n_days: float,
    capacity: float,
    capacity_penalty_weight: float,
) -> float:
    """Capacity-penalized total cost using precomputed per-action costs."""
    approve_m = p < t1
    step_m = (p >= t1) & (p < t2)
    review_m = (p >= t2) & (p < t3)
    decline_m = p >= t3
    total = (
        float(cost_approve_arr[approve_m].sum())
        + float(cost_step_up_arr[step_m].sum())
        + float(cost_review_arr[review_m].sum())
        + float(cost_decline_arr[decline_m].sum())
    )
    n_review = int(review_m.sum())
    excess = max(0.0, n_review / float(n_days) - capacity)
    return total + capacity_penalty_weight * excess


def search_thresholds(
    calibrated_probs: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series,
    cost_params: dict[str, float],
    *,
    n_days: float,
    coarse_step: float = 0.01,
    fine_step: float = 0.005,
    fine_radius: float = 0.05,
    capacity_penalty_weight: float = CAPACITY_PENALTY_WEIGHT,
) -> dict:
    """Two-pass grid search minimizing capacity-penalized expected cost."""
    p = np.asarray(calibrated_probs, dtype=float)
    a = np.asarray(amounts, dtype=float)
    costs = compute_all_costs(p, a, cost_params)
    c_approve = np.asarray(costs["approve"], dtype=float)
    c_step = np.asarray(costs["step_up"], dtype=float)
    c_review = np.asarray(costs["review"], dtype=float)
    c_decline = np.asarray(costs["decline"], dtype=float)
    capacity = float(cost_params["review_capacity_per_day"])

    best: dict | None = None
    for t1, t2, t3 in _candidate_triplets(coarse_step):
        obj = _fast_objective(
            p, c_approve, c_step, c_review, c_decline, t1, t2, t3,
            n_days=n_days,
            capacity=capacity,
            capacity_penalty_weight=capacity_penalty_weight,
        )
        if best is None or obj < best["objective"]:
            best = {"t1": t1, "t2": t2, "t3": t3, "objective": obj, "pass": "coarse"}

    assert best is not None

    # Fine pass: triplets within ±fine_radius of each coarse threshold.
    def _local_grid(center: float) -> np.ndarray:
        lo = max(0.0, center - fine_radius)
        hi = min(1.0, center + fine_radius)
        grid = np.round(np.arange(lo, hi + 1e-12, fine_step), 10)
        return grid[(grid >= 0.0) & (grid <= 1.0)]

    g1 = _local_grid(best["t1"])
    g2 = _local_grid(best["t2"])
    g3 = _local_grid(best["t3"])
    for t1 in g1:
        for t2 in g2:
            if t2 < t1:
                continue
            for t3 in g3:
                if t3 < t2:
                    continue
                obj = _fast_objective(
                    p, c_approve, c_step, c_review, c_decline,
                    float(t1), float(t2), float(t3),
                    n_days=n_days,
                    capacity=capacity,
                    capacity_penalty_weight=capacity_penalty_weight,
                )
                if obj < best["objective"]:
                    best = {
                        "t1": float(t1),
                        "t2": float(t2),
                        "t3": float(t3),
                        "objective": obj,
                        "pass": "fine",
                    }

    policy_df = apply_threshold_policy_frame(
        p, a, cost_params, best["t1"], best["t2"], best["t3"]
    )
    best["policy_df"] = policy_df
    return best


def apply_threshold_policy_frame(
    calibrated_probs: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series,
    cost_params: dict[str, float],
    t1: float,
    t2: float,
    t3: float,
) -> pd.DataFrame:
    """Build a policy DataFrame matching the argmin evaluator schema."""
    p = np.asarray(calibrated_probs, dtype=float)
    a = np.asarray(amounts, dtype=float)
    actions = apply_threshold_policy(p, t1, t2, t3)
    costs = compute_all_costs(p, a, cost_params)
    cost_matrix = np.column_stack([np.asarray(costs[name], dtype=float) for name in ACTION_NAMES])
    action_to_idx = {name: i for i, name in enumerate(ACTION_NAMES)}
    idx = np.array([action_to_idx[x] for x in actions.to_numpy()], dtype=int)
    chosen_cost = cost_matrix[np.arange(len(p)), idx]
    partitioned = np.partition(cost_matrix, 1, axis=1)
    return pd.DataFrame(
        {
            "action": actions.to_numpy(),
            "cost_approve": cost_matrix[:, 0],
            "cost_step_up": cost_matrix[:, 1],
            "cost_review": cost_matrix[:, 2],
            "cost_decline": cost_matrix[:, 3],
            "chosen_cost": chosen_cost,
            "cost_gap": partitioned[:, 1] - chosen_cost,
        }
    )


def run_evaluation(
    policy_df: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series,
    cost_params: dict[str, float],
    *,
    n_days: float,
) -> dict:
    """Same summary schema as per_transaction_policy.run_evaluation."""
    return _shared_run_evaluation(
        policy_df, labels, amounts, cost_params, n_days=n_days
    )
