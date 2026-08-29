"""Threshold policy evaluation — the rubric-required deliverable.

Takes validation labels, scores, transaction amounts, and candidate boundaries.
Outputs: transaction count, fraud count, precision, recall, false positives,
fraud value captured, and volume per action.

Usage:
    python evals/threshold_policy.py --version 9 --thresholds 0.40 0.80 0.95
    python evals/threshold_policy.py --version 9 --thresholds 0.12 0.45 0.88
    python evals/threshold_policy.py --version 9 --thresholds 0.12 0.45 0.88 \
        --cost-config configs/decision_policy_v1.json

§3: cost parameters from config, never hardcoded.
§4: output JSON includes split, dataset_version, model_version.
§7: does NOT create a new model version.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.cost_model import CostParams


def assign_actions(scores: np.ndarray, t1: float, t2: float, t3: float) -> np.ndarray:
    """Assign action labels based on threshold boundaries (lower-inclusive)."""
    actions = np.full(len(scores), "approve", dtype="U10")
    actions[scores >= t1] = "step_up"
    actions[scores >= t2] = "review"
    actions[scores >= t3] = "decline"
    return actions


def compute_band_table(
    scores: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    actions: np.ndarray,
    params: CostParams,
) -> list[dict]:
    """Compute per-band statistics."""
    total_fraud = labels.sum()
    total_fraud_dollars = (amounts * labels).sum()
    action_order = ["approve", "step_up", "review", "decline"]
    rows: list[dict] = []

    for action_name in action_order:
        mask = actions == action_name
        n = int(mask.sum())
        n_fraud = int(labels[mask].sum())
        n_legit = n - n_fraud
        fraud_dollars = float((amounts[mask] * labels[mask]).sum())
        precision = n_fraud / n if n > 0 else 0.0

        higher_mask = np.isin(actions, action_order[action_order.index(action_name) :])
        cumulative_fraud = labels[higher_mask].sum()
        recall = cumulative_fraud / total_fraud if total_fraud > 0 else 0.0

        p = scores[mask]
        a = amounts[mask]
        if action_name == "approve":
            band_cost = float((p * a * params.fraud_cost_multiplier).sum())
        elif action_name == "step_up":
            band_cost = float(
                (
                    p * a * params.fraud_cost_multiplier * (1 - params.step_up_stop_rate)
                    + (1 - p) * a * params.step_up_friction_rate
                ).sum()
            )
        elif action_name == "review":
            band_cost = float(
                (
                    p * a * params.fraud_cost_multiplier * (1 - params.review_catch_rate)
                    + params.review_cost_per_txn
                ).sum()
            )
        else:
            band_cost = float(((1 - p) * a * params.decline_revenue_loss).sum())

        rows.append(
            {
                "action": action_name,
                "txn_count": n,
                "txn_pct": float(n / len(actions) * 100),
                "fraud_count": n_fraud,
                "fraud_pct": float(n_fraud / total_fraud * 100) if total_fraud > 0 else 0.0,
                "fraud_dollars": fraud_dollars,
                "fraud_dollars_pct": float(fraud_dollars / total_fraud_dollars * 100)
                if total_fraud_dollars > 0
                else 0.0,
                "precision": float(precision),
                "recall_cumulative": float(recall),
                "false_positives": int(n_legit) if action_name in ("review", "decline") else 0,
                "legit_friction": int(n_legit) if action_name == "step_up" else 0,
                "expected_cost": band_cost,
            }
        )

    return rows


def print_band_table(rows: list[dict], thresholds: tuple[float, float, float]) -> None:
    """Pretty-print the band table."""
    t1, t2, t3 = thresholds
    boundaries = {
        "approve": f"[0.00, {t1:.4f})",
        "step_up": f"[{t1:.4f}, {t2:.4f})",
        "review": f"[{t2:.4f}, {t3:.4f})",
        "decline": f"[{t3:.4f}, 1.00]",
    }

    header = (
        f"{'Action':<10} {'Boundaries':<20} {'Txns':>8} {'%':>6} "
        f"{'Fraud':>7} {'F%':>6} {'Fraud$':>12} {'F$%':>6} "
        f"{'Prec':>6} {'Recall≥':>7} {'FP/Fric':>8} {'Cost':>12}"
    )
    print(header)
    print("─" * len(header))

    for row in rows:
        action = row["action"]
        fp_fric = row["false_positives"] + row["legit_friction"]
        print(
            f"{action:<10} {boundaries[action]:<20} "
            f"{row['txn_count']:>8,} {row['txn_pct']:>5.1f}% "
            f"{row['fraud_count']:>7,} {row['fraud_pct']:>5.1f}% "
            f"${row['fraud_dollars']:>10,.0f} {row['fraud_dollars_pct']:>5.1f}% "
            f"{row['precision']:>5.3f} {row['recall_cumulative']:>6.3f} "
            f"{fp_fric:>8,} ${row['expected_cost']:>10,.0f}"
        )

    total_fraud_caught = sum(r["fraud_count"] for r in rows if r["action"] != "approve")
    total_fraud = sum(r["fraud_count"] for r in rows)
    total_cost = sum(r["expected_cost"] for r in rows)
    reviews = next((r["txn_count"] for r in rows if r["action"] == "review"), 0)
    declined_legit = next((r["false_positives"] for r in rows if r["action"] == "decline"), 0)

    print("─" * len(header))
    print(
        f"Fraud caught: {total_fraud_caught:,}/{total_fraud:,} "
        f"({total_fraud_caught / total_fraud * 100:.1f}%)  "
        f"Reviews: {reviews:,}  Declined legit: {declined_legit:,}  "
        f"Total cost: ${total_cost:,.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        required=True,
        help="Three thresholds: approve|step_up step_up|review review|decline",
    )
    parser.add_argument("--cost-config", type=str, default=None)
    parser.add_argument("--use-calibrated", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="evals/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t1, t2, t3 = args.thresholds
    if not (t1 < t2 < t3):
        raise ValueError(f"Thresholds must be strictly increasing: {t1} < {t2} < {t3}")

    scores_file = output_dir / f"calibrated_scores_v{args.version}.npz"
    data = np.load(scores_file)
    if args.use_calibrated:
        scores = data["calibrated_scores"]
        score_type = "calibrated"
    else:
        scores = data["raw_scores"]
        score_type = "raw"
    labels = data["labels"]
    amounts = data["amounts"]

    if args.cost_config:
        params = CostParams.from_config(REPO_ROOT / args.cost_config)
    else:
        params = CostParams()

    actions = assign_actions(scores, t1, t2, t3)
    rows = compute_band_table(scores, labels, amounts, actions, params)

    model_config = json.loads((REPO_ROOT / f"configs/lgbm_v{args.version}.json").read_text())
    dataset_version = model_config["dataset_version"]

    print(f"\nTHRESHOLD POLICY — v{args.version} ({score_type} scores)")
    print(f"Thresholds: {t1:.4f} / {t2:.4f} / {t3:.4f}")
    print(f"Split: validation | Dataset: v{dataset_version}")
    print()
    print_band_table(rows, (t1, t2, t3))

    result = {
        "model_version": args.version,
        "dataset_version": dataset_version,
        "split": "validation",
        "score_type": score_type,
        "thresholds": [t1, t2, t3],
        "cost_assumptions": asdict(params),
        "bands": rows,
    }
    out_path = output_dir / f"threshold_policy_results_v{args.version}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
