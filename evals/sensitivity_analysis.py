"""Sensitivity analysis — how thresholds move as cost assumptions change.

Varies one parameter at a time, re-runs grid search, records optimal thresholds
and outcomes. All on validation split only.

Usage:
    python evals/sensitivity_analysis.py --version 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.cost_model import CostParams
from evals.optimize_thresholds import compute_policy_cost, grid_search, validation_days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument("--output-dir", type=str, default="evals/figures")
    parser.add_argument("--coarse-step", type=float, default=0.02)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(output_dir / f"calibrated_scores_v{args.version}.npz")
    cal_scores = data["calibrated_scores"]
    labels = data["labels"]
    amounts = data["amounts"]

    val_days = validation_days(args.version)

    sweeps = {
        "fraud_cost_multiplier": [1.0, 1.5, 2.0, 3.0, 5.0],
        "step_up_stop_rate": [0.50, 0.60, 0.70, 0.80, 0.90],
        "review_cost_per_txn": [5.0, 10.0, 20.0, 50.0],
        "review_capacity_per_day": [100, 200, 500, 10000],
    }

    all_results: list[dict] = []

    for param_name, values in sweeps.items():
        print(f"\n{'=' * 60}")
        print(f"Sweeping {param_name}: {values}")
        print(f"{'=' * 60}")

        for value in values:
            defaults = CostParams().__dict__.copy()
            defaults[param_name] = value
            params = CostParams(**defaults)

            triplet, cost = grid_search(
                cal_scores,
                labels,
                amounts,
                params,
                val_days,
                coarse_step=args.coarse_step,
            )

            _, _, _, reviews_day = compute_policy_cost(
                cal_scores, labels, amounts, *triplet, params, val_days
            )

            action_arr = np.where(
                cal_scores < triplet[0],
                "approve",
                np.where(
                    cal_scores < triplet[1],
                    "step_up",
                    np.where(cal_scores < triplet[2], "review", "decline"),
                ),
            )
            fraud_caught = labels[action_arr != "approve"].sum()
            fraud_total = labels.sum()
            fraud_caught_dollars = (amounts * labels)[action_arr != "approve"].sum()
            fraud_total_dollars = (amounts * labels).sum()

            result = {
                "parameter": param_name,
                "value": value,
                "t1": triplet[0],
                "t2": triplet[1],
                "t3": triplet[2],
                "total_cost": cost,
                "fraud_caught_pct": float(fraud_caught / fraud_total * 100),
                "fraud_dollars_caught_pct": float(
                    fraud_caught_dollars / fraud_total_dollars * 100
                ),
                "reviews_per_day": reviews_day,
            }
            all_results.append(result)

            print(
                f"  {param_name}={value}: "
                f"t=({triplet[0]:.3f},{triplet[1]:.3f},{triplet[2]:.3f}) "
                f"cost=${cost:,.0f} fraud={fraud_caught / fraud_total * 100:.1f}% "
                f"reviews/day={reviews_day:.0f}"
            )

    json_path = output_dir / f"sensitivity_analysis_v{args.version}.json"
    json_path.write_text(json.dumps(all_results, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (param_name, _values) in zip(axes.flatten(), sweeps.items()):
        param_results = [r for r in all_results if r["parameter"] == param_name]
        xs = [r["value"] for r in param_results]
        ax.plot(xs, [r["t1"] for r in param_results], "o-", label="t1 (approve→step_up)")
        ax.plot(xs, [r["t2"] for r in param_results], "s-", label="t2 (step_up→review)")
        ax.plot(xs, [r["t3"] for r in param_results], "^-", label="t3 (review→decline)")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Threshold")
        ax.set_title(f"Sensitivity to {param_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / f"sensitivity_thresholds_v{args.version}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nSaved results to {json_path.relative_to(REPO_ROOT)}")
    print(f"Saved plot to {plot_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
