"""Compare baseline vs engineered model metrics from saved run artifacts.

Usage:
    python evals/compare_models.py

Writes a human-readable summary to evals/model_comparison_results.txt.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from load_config import load_config, repo_path

paths = load_config("paths")
OUTPUT_PATH = REPO_ROOT / "evals" / "model_comparison_results.txt"


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    return json.loads(path.read_text())


def format_section(title: str, body: str) -> str:
    line = "=" * 64
    return f"{line}\n{title}\n{line}\n\n{body}\n"


def main() -> None:
    baseline_path = repo_path(paths["baseline_metrics"])
    engineered_path = repo_path(paths["engineered_metrics"])

    baseline = load_metrics(baseline_path)
    engineered = load_metrics(engineered_path)

    base_metrics = baseline["lightgbm"]["at_threshold_0_5"]
    base_operating = baseline["lightgbm"]["at_prevalence_matched_flag_rate"]
    eng_metrics = engineered["engineered_lightgbm"]["at_threshold_0_5"]
    eng_operating = engineered["engineered_lightgbm"]["at_prevalence_matched_flag_rate"]
    lift = engineered["lift_vs_baseline"]

    lines = [
        "IEEE-CIS Fraud Detection — Model Comparison",
        "=" * 64,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Sources:",
        f"  Baseline metrics:   {baseline_path}",
        f"  Engineered metrics: {engineered_path}",
        "",
        format_section(
            "Side-by-side (prevalence-matched flag rate)",
            "\n".join(
                [
                    f"{'Metric':<12} {'Baseline':>12} {'Engineered':>12} {'Lift':>12}",
                    "-" * 52,
                    f"{'auc_pr':<12} {base_metrics['auc_pr']:>12.4f} {eng_metrics['auc_pr']:>12.4f} {lift['auc_pr']:>+12.4f}",
                    f"{'precision':<12} {base_operating['precision']:>12.4f} {eng_operating['precision']:>12.4f} {lift['precision_at_prevalence_matched']:>+12.4f}",
                    f"{'recall':<12} {base_operating['recall']:>12.4f} {eng_operating['recall']:>12.4f} {lift['recall_at_prevalence_matched']:>+12.4f}",
                ]
            ),
        ),
        format_section(
            "Split configuration",
            "\n".join(
                [
                    f"Method:         {baseline['split']['method']}",
                    f"Train fraction: {baseline['split']['train_fraction']}",
                    f"Train rows:     {baseline['split']['train_rows']:,}",
                    f"Test rows:      {baseline['split']['test_rows']:,}",
                    f"Train fraud:    {baseline['split']['train_fraud_rate']:.4%}",
                    f"Test fraud:     {baseline['split']['test_fraud_rate']:.4%}",
                ]
            ),
        ),
        format_section(
            "Feature counts",
            "\n".join(
                [
                    f"Baseline model:   raw merged parquet",
                    f"Engineered model: {engineered['feature_count']} features "
                    f"({engineered['categorical_feature_count']} categorical)",
                ]
            ),
        ),
    ]

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote comparison to {OUTPUT_PATH}")
    print()
    print(lines[6])  # Side-by-side header area - print key table from file
    with OUTPUT_PATH.open() as f:
        print(f.read())


if __name__ == "__main__":
    main()
