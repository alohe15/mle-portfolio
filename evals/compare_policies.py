"""Side-by-side comparison of argmin vs fixed-threshold policies on validation.

Loads lgbm_v9 raw scores, applies the deployment calibrator, runs both
policies on the same validation rows, and prints:

  - summary table (fraud caught, expected cost, reviews/day, declined legit)
  - disagreement analysis stratified by amount quartile
  - 500-bootstrap 95% CI on total-expected-cost difference (argmin - threshold)

Usage:
    python evals/compare_policies.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
CALIBRATOR_PATH = REPO_ROOT / "models" / "lgbm_v9_calibrator.pkl"
POLICY_CONFIG = REPO_ROOT / "configs" / "decision_policy_v1.json"
N_BOOTSTRAP = 500
RANDOM_STATE = 42

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from calibrate import CALIBRATOR_PATH as _CAL_PATH, load_v9_validation  # noqa: E402
from cost_model import load_cost_params  # noqa: E402
from per_transaction_policy import apply_argmin_policy, run_evaluation as eval_argmin  # noqa: E402
from threshold_policy import (  # noqa: E402
    apply_threshold_policy_frame,
    search_thresholds,
)


def load_calibrator(path: Path = CALIBRATOR_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrator not found at {path}. Run: python evals/calibrate.py"
        )
    return pickle.loads(path.read_bytes())


def format_summary_table(argmin_metrics: dict, threshold_metrics: dict) -> str:
    rows = [
        (
            "fraud_caught_count",
            f"{argmin_metrics['fraud_caught_count']:.1f}",
            f"{threshold_metrics['fraud_caught_count']:.1f}",
        ),
        (
            "fraud_caught_dollars",
            f"{argmin_metrics['fraud_caught_dollars']:,.0f}",
            f"{threshold_metrics['fraud_caught_dollars']:,.0f}",
        ),
        (
            "total_expected_cost",
            f"{argmin_metrics['total_expected_cost']:,.0f}",
            f"{threshold_metrics['total_expected_cost']:,.0f}",
        ),
        (
            "reviews_per_day",
            f"{argmin_metrics['reviews_per_day']:.1f}",
            f"{threshold_metrics['reviews_per_day']:.1f}",
        ),
        (
            "declined_legit",
            f"{argmin_metrics['declined_legit']:,}",
            f"{threshold_metrics['declined_legit']:,}",
        ),
    ]
    lines = [
        f"{'metric':<22} {'argmin':>16} {'threshold':>16}",
        "-" * 56,
    ]
    for name, a, t in rows:
        lines.append(f"{name:<22} {a:>16} {t:>16}")
    return "\n".join(lines)


def disagreement_by_amount_quartile(
    argmin_actions: pd.Series,
    threshold_actions: pd.Series,
    amounts: np.ndarray,
) -> pd.DataFrame:
    disagree = argmin_actions.to_numpy() != threshold_actions.to_numpy()
    quartiles = pd.qcut(amounts, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    # qcut may return a Series or Categorical depending on pandas version.
    q_series = pd.Series(np.asarray(quartiles), dtype="object")
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = (q_series == q).to_numpy()
        n = int(mask.sum())
        d = int(np.sum(disagree & mask))
        rows.append(
            {
                "amount_quartile": str(q),
                "n": n,
                "disagreements": d,
                "disagreement_rate": (d / n) if n else 0.0,
                "amount_min": float(amounts[mask].min()) if n else float("nan"),
                "amount_max": float(amounts[mask].max()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_cost_diff_ci(
    argmin_costs: np.ndarray,
    threshold_costs: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> dict:
    rng = np.random.default_rng(random_state)
    n = len(argmin_costs)
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[b] = float(argmin_costs[idx].sum() - threshold_costs[idx].sum())
    point = float(argmin_costs.sum() - threshold_costs.sum())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "point_diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    assert _CAL_PATH == CALIBRATOR_PATH
    cost_params = load_cost_params(POLICY_CONFIG)
    _val_df, raw_scores, labels, amounts, n_days = load_v9_validation()
    calibrator = load_calibrator()
    calibrated = np.asarray(calibrator.predict(raw_scores), dtype=float)

    argmin_df = apply_argmin_policy(calibrated, amounts, cost_params)
    argmin_metrics = eval_argmin(
        argmin_df, labels, amounts, cost_params, n_days=n_days
    )

    search = search_thresholds(
        calibrated, amounts, cost_params, n_days=n_days
    )
    threshold_df = apply_threshold_policy_frame(
        calibrated,
        amounts,
        cost_params,
        search["t1"],
        search["t2"],
        search["t3"],
    )
    threshold_metrics = eval_argmin(
        threshold_df, labels, amounts, cost_params, n_days=n_days
    )

    print("=== Policy comparison on validation (lgbm_v9) ===")
    print(
        f"thresholds: t1={search['t1']:.4f}  t2={search['t2']:.4f}  "
        f"t3={search['t3']:.4f}  (search pass={search['pass']})"
    )
    print(f"validation span: {n_days:.2f} days  n={len(labels):,}")
    print()
    print(format_summary_table(argmin_metrics, threshold_metrics))
    print()

    if argmin_metrics["total_expected_cost"] > threshold_metrics["total_expected_cost"] + 1e-6:
        raise RuntimeError(
            "Invariant violated: argmin total expected cost exceeds threshold "
            f"policy ({argmin_metrics['total_expected_cost']} > "
            f"{threshold_metrics['total_expected_cost']})"
        )

    print("=== Disagreement by amount quartile ===")
    disagree = disagreement_by_amount_quartile(
        argmin_df["action"], threshold_df["action"], amounts
    )
    print(disagree.to_string(index=False))
    print()

    ci = bootstrap_cost_diff_ci(
        argmin_df["chosen_cost"].to_numpy(),
        threshold_df["chosen_cost"].to_numpy(),
    )
    print("=== Bootstrap cost difference (argmin - threshold) ===")
    print(f"point estimate : {ci['point_diff']:,.0f}")
    print(f"95% CI         : [{ci['ci_low']:,.0f}, {ci['ci_high']:,.0f}]")
    print(f"n_bootstrap    : {ci['n_bootstrap']}")


if __name__ == "__main__":
    main()
