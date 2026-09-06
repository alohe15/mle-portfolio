"""One-at-a-time sensitivity sweeps over decision-policy cost parameters.

For each tunable cost parameter, hold others at their defaults, sweep a
small grid, rerun the per-transaction argmin policy, and report effects on
total expected cost and review volume.

Usage:
    python evals/sensitivity_analysis.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
CALIBRATOR_PATH = REPO_ROOT / "models" / "lgbm_v9_calibrator.pkl"
POLICY_CONFIG = REPO_ROOT / "configs" / "decision_policy_v1.json"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from calibrate import load_v9_validation  # noqa: E402
from cost_model import load_cost_params  # noqa: E402
from per_transaction_policy import apply_argmin_policy, run_evaluation  # noqa: E402

# Six tunable parameters (review_capacity_per_day is operational, not in the
# per-transaction cost formulas, so it is excluded from this sweep).
SWEEPS: dict[str, list[float]] = {
    "fraud_cost_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "step_up_stop_rate": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "step_up_friction_rate": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
    "review_catch_rate": [0.70, 0.80, 0.90, 0.95, 0.98, 1.00],
    "review_cost_per_txn": [2.0, 5.0, 10.0, 15.0, 25.0, 40.0],
    "decline_revenue_loss": [0.25, 0.50, 0.75, 1.00, 1.50, 2.00],
}


def load_calibrator(path: Path = CALIBRATOR_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrator not found at {path}. Run: python evals/calibrate.py"
        )
    return pickle.loads(path.read_bytes())


def sweep_parameter(
    param_name: str,
    values: list[float],
    base_params: dict[str, float],
    calibrated: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    n_days: float,
) -> list[dict]:
    rows = []
    for value in values:
        params = dict(base_params)
        params[param_name] = float(value)
        policy_df = apply_argmin_policy(calibrated, amounts, params)
        metrics = run_evaluation(
            policy_df, labels, amounts, params, n_days=n_days
        )
        rows.append(
            {
                "param": param_name,
                "value": float(value),
                "total_expected_cost": metrics["total_expected_cost"],
                "n_review": metrics["n_review"],
                "reviews_per_day": metrics["reviews_per_day"],
                "declined_legit": metrics["declined_legit"],
                "fraud_caught_count": metrics["fraud_caught_count"],
            }
        )
    return rows


def leverage(rows: list[dict]) -> tuple[float, float]:
    """Range of cost and review count across the sweep."""
    costs = [r["total_expected_cost"] for r in rows]
    reviews = [r["n_review"] for r in rows]
    return float(max(costs) - min(costs)), float(max(reviews) - min(reviews))


def main() -> None:
    base_params = load_cost_params(POLICY_CONFIG)
    _val_df, raw_scores, labels, amounts, n_days = load_v9_validation()
    calibrator = load_calibrator()
    calibrated = np.asarray(calibrator.predict(raw_scores), dtype=float)

    print("=== Cost-parameter sensitivity (argmin policy, validation) ===")
    print(f"n={len(labels):,}  days={n_days:.2f}")
    print()

    all_results: dict[str, list[dict]] = {}
    leverage_rows = []
    for param_name, values in SWEEPS.items():
        rows = sweep_parameter(
            param_name, values, base_params, calibrated, labels, amounts, n_days
        )
        all_results[param_name] = rows
        cost_lev, review_lev = leverage(rows)
        leverage_rows.append(
            {
                "param": param_name,
                "cost_range": cost_lev,
                "review_count_range": review_lev,
            }
        )

        print(f"--- {param_name} (default={base_params[param_name]}) ---")
        header = (
            f"{'value':>10} {'total_cost':>14} {'n_review':>10} "
            f"{'reviews/day':>12} {'declined_legit':>14}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['value']:10.4g} {r['total_expected_cost']:14,.0f} "
                f"{r['n_review']:10d} {r['reviews_per_day']:12.1f} "
                f"{r['declined_legit']:14d}"
            )
        print()

    leverage_rows.sort(key=lambda r: r["cost_range"], reverse=True)
    print("=== Leverage ranking (by total expected cost range) ===")
    for r in leverage_rows:
        print(
            f"{r['param']:<24} cost_range={r['cost_range']:,.0f}  "
            f"review_count_range={r['review_count_range']:,.0f}"
        )
    top = leverage_rows[0]
    print()
    print(
        f"Most leverage on total expected cost: {top['param']} "
        f"(range {top['cost_range']:,.0f})"
    )
    by_review = max(leverage_rows, key=lambda r: r["review_count_range"])
    print(
        f"Most leverage on review volume: {by_review['param']} "
        f"(range {by_review['review_count_range']:,.0f})"
    )


if __name__ == "__main__":
    main()
