"""Grid search for optimal threshold triplet that minimizes total expected cost.

Searches over (t1, t2, t3) where:
    p < t1      → approve
    t1 ≤ p < t2 → step_up
    t2 ≤ p < t3 → review
    t3 ≤ p      → decline

Includes capacity constraint: if reviews exceed capacity, adds a penalty.
Uses validation split only. Never loads or references the test split.

Usage:
    python evals/optimize_thresholds.py --version 9
    python evals/optimize_thresholds.py --version 9 --cost-config configs/decision_policy_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.cost_model import CostParams


def compute_policy_cost(
    cal_scores: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    params: CostParams,
    val_days: float,
    capacity_penalty_weight: float = 1e6,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Compute total expected cost for a threshold triplet."""
    actions = np.where(
        cal_scores < t1,
        0,
        np.where(cal_scores < t2, 1, np.where(cal_scores < t3, 2, 3)),
    )

    p = cal_scores
    a = amounts
    costs = np.zeros_like(p)

    mask = actions == 0
    costs[mask] = p[mask] * a[mask] * params.fraud_cost_multiplier

    mask = actions == 1
    costs[mask] = (
        p[mask] * a[mask] * params.fraud_cost_multiplier * (1 - params.step_up_stop_rate)
        + (1 - p[mask]) * a[mask] * params.step_up_friction_rate
    )

    mask = actions == 2
    costs[mask] = (
        p[mask] * a[mask] * params.fraud_cost_multiplier * (1 - params.review_catch_rate)
        + params.review_cost_per_txn
    )

    mask = actions == 3
    costs[mask] = (1 - p[mask]) * a[mask] * params.decline_revenue_loss

    total_cost = float(costs.sum())

    reviews_per_day = float((actions == 2).sum() / val_days)
    if reviews_per_day > params.review_capacity_per_day:
        excess = reviews_per_day - params.review_capacity_per_day
        total_cost += excess * capacity_penalty_weight

    return total_cost, actions, costs, reviews_per_day


def grid_search(
    cal_scores: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    params: CostParams,
    val_days: float,
    coarse_step: float = 0.01,
    fine_step: float = 0.005,
    fine_radius: float = 0.05,
) -> tuple[tuple[float, float, float], float]:
    """Two-pass grid search: coarse then fine."""
    print(f"Coarse grid search (step={coarse_step})...")
    best_cost = np.inf
    best_triplet = (0.4, 0.8, 0.95)

    coarse_grid = np.arange(0.01, 1.0, coarse_step)
    evaluated = 0
    for t1 in coarse_grid:
        for t2 in coarse_grid[coarse_grid > t1]:
            for t3 in coarse_grid[coarse_grid > t2]:
                cost, _, _, _ = compute_policy_cost(
                    cal_scores, labels, amounts, t1, t2, t3, params, val_days
                )
                evaluated += 1
                if cost < best_cost:
                    best_cost = cost
                    best_triplet = (float(t1), float(t2), float(t3))

    print(f"  Evaluated {evaluated:,} triplets")
    print(
        f"  Coarse best: ({best_triplet[0]:.2f}, {best_triplet[1]:.2f}, "
        f"{best_triplet[2]:.2f})  cost=${best_cost:,.2f}"
    )

    print(f"\nFine grid search (step={fine_step}, radius={fine_radius})...")
    fine_best_cost = best_cost
    fine_best_triplet = best_triplet

    for t1 in np.arange(
        max(0.001, best_triplet[0] - fine_radius),
        min(1.0, best_triplet[0] + fine_radius),
        fine_step,
    ):
        for t2 in np.arange(
            max(t1 + fine_step, best_triplet[1] - fine_radius),
            min(1.0, best_triplet[1] + fine_radius),
            fine_step,
        ):
            for t3 in np.arange(
                max(t2 + fine_step, best_triplet[2] - fine_radius),
                min(1.0, best_triplet[2] + fine_radius),
                fine_step,
            ):
                cost, _, _, _ = compute_policy_cost(
                    cal_scores, labels, amounts, t1, t2, t3, params, val_days
                )
                if cost < fine_best_cost:
                    fine_best_cost = cost
                    fine_best_triplet = (float(t1), float(t2), float(t3))

    print(
        f"  Fine best: ({fine_best_triplet[0]:.4f}, {fine_best_triplet[1]:.4f}, "
        f"{fine_best_triplet[2]:.4f})  cost=${fine_best_cost:,.2f}"
    )

    return fine_best_triplet, fine_best_cost


def validation_days(version: int) -> float:
    config = json.loads((REPO_ROOT / f"configs/lgbm_v{version}.json").read_text())
    ds_config = json.loads((REPO_ROOT / config["dataset_config_path"]).read_text())
    df = pd.read_parquet(REPO_ROOT / ds_config["output_path"])
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * config["split"]["train_fraction"])
    val_end = train_end + int(n * config["split"]["val_fraction"])
    val = df.iloc[train_end:val_end]
    return float((val["TransactionDT"].max() - val["TransactionDT"].min()) / 86400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument("--cost-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evals/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(output_dir / f"calibrated_scores_v{args.version}.npz")
    cal_scores = data["calibrated_scores"]
    labels = data["labels"]
    amounts = data["amounts"]

    if args.cost_config:
        params = CostParams.from_config(REPO_ROOT / args.cost_config)
    else:
        params = CostParams()

    val_days = validation_days(args.version)

    optimal_triplet, optimal_cost = grid_search(
        cal_scores, labels, amounts, params, val_days
    )

    hyp_cost, _, _, hyp_reviews = compute_policy_cost(
        cal_scores, labels, amounts, 0.40, 0.80, 0.95, params, val_days
    )

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(
        f"  Hypothesis (0.40/0.80/0.95):  cost=${hyp_cost:,.2f}  "
        f"reviews/day={hyp_reviews:.0f}"
    )
    print(f"  Optimized:                    cost=${optimal_cost:,.2f}")

    print("\nBootstrap (500 resamples)...")
    np.random.seed(42)
    n_boot = 500
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = np.random.choice(len(cal_scores), len(cal_scores), replace=True)
        b_scores = cal_scores[idx]
        b_labels = labels[idx]
        b_amounts = amounts[idx]
        h_cost, _, _, _ = compute_policy_cost(
            b_scores, b_labels, b_amounts, 0.40, 0.80, 0.95, params, val_days
        )
        o_cost, _, _, _ = compute_policy_cost(
            b_scores, b_labels, b_amounts, *optimal_triplet, params, val_days
        )
        deltas.append(h_cost - o_cost)

    deltas_arr = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas_arr, [2.5, 97.5])
    print("  Cost reduction (hypothesis - optimized):")
    print(f"  Mean:    ${deltas_arr.mean():,.2f}")
    print(f"  95% CI:  [${ci_lo:,.2f}, ${ci_hi:,.2f}]")
    print(f"  Excludes zero: {'YES' if ci_lo > 0 else 'NO'}")

    results = {
        "optimal_thresholds": list(optimal_triplet),
        "optimal_cost": float(optimal_cost),
        "hypothesis_thresholds": [0.40, 0.80, 0.95],
        "hypothesis_cost": float(hyp_cost),
        "bootstrap_delta_mean": float(deltas_arr.mean()),
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "cost_assumptions": {
            "fraud_cost_multiplier": params.fraud_cost_multiplier,
            "step_up_stop_rate": params.step_up_stop_rate,
            "step_up_friction_rate": params.step_up_friction_rate,
            "review_catch_rate": params.review_catch_rate,
            "review_cost_per_txn": params.review_cost_per_txn,
            "review_capacity_per_day": params.review_capacity_per_day,
            "decline_revenue_loss": params.decline_revenue_loss,
        },
    }
    out_path = output_dir / f"optimal_thresholds_v{args.version}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
