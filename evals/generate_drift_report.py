"""Generate a two-tier drift report against a frozen reference profile.

Usage:
    python evals/generate_drift_report.py evals/reference_profile.json <comparison.parquet> [--labels]

Loads the serving model, calibrator, fitted transforms, and cost params from
models/registry.json via services/api/model_loader.py (§10). Scores the
comparison sample the same way the API does: frozen transforms → predict →
calibrate → argmin action. Writes JSON under reports/monitoring/ (gitignored)
and prints a human-readable summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
REPORTS_DIR = REPO_ROOT / "reports" / "monitoring"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from monitoring import (  # noqa: E402
    NON_SERVING_COLUMNS,
    apply_monitoring_transforms,
    build_model_input_batch,
    build_report,
    dump_json,
    format_human_summary,
    label_metrics,
    load_json,
    load_monitoring_config,
    load_serving_artifacts,
    run_tier1_checks,
    check_tier2,
    score_and_act,
    utc_now_compact,
    utc_now_iso,
)


def load_comparison(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported comparison file type: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monitoring drift report.")
    parser.add_argument(
        "reference_profile",
        type=Path,
        help="Path to evals/reference_profile.json",
    )
    parser.add_argument(
        "comparison_dataset",
        type=Path,
        help="Comparison sample (parquet or CSV)",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Comparison data includes isFraud labels (enables Tier 2)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EVALS_DIR / "monitoring_config.json",
        help="Monitoring threshold config (proposed values)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit report path (default reports/monitoring/drift_report_*.json)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    reference_path = resolve_path(args.reference_profile)
    comparison_path = resolve_path(args.comparison_dataset)
    config_path = resolve_path(args.config)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference profile not found: {reference_path}")
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison dataset not found: {comparison_path}")

    reference = load_json(reference_path)
    config = load_monitoring_config(config_path)
    print(f"Loading comparison sample from {comparison_path} ...")
    source_df = load_comparison(comparison_path)
    print(f"Comparison rows: {len(source_df):,}")

    print("Loading serving artifacts from models/registry.json ...")
    artifacts = load_serving_artifacts()
    print("Applying frozen fitted transforms ...")
    transformed = apply_monitoring_transforms(source_df, artifacts)
    feature_df = build_model_input_batch(transformed, artifacts)

    if "TransactionAmt" not in transformed.columns and "TransactionAmt" not in source_df.columns:
        raise RuntimeError("Comparison sample missing serving required field TransactionAmt")
    amount_src = transformed if "TransactionAmt" in transformed.columns else source_df
    amounts = pd.to_numeric(amount_src["TransactionAmt"], errors="coerce").to_numpy()

    print(f"Scoring {len(feature_df):,} comparison rows ...")
    _raw, calibrated, actions = score_and_act(feature_df, artifacts, amounts)

    # Schema is validated against the serving model-feature contract (464
    # columns after transform), not the training DataFrame. ID / label columns
    # that appear on processed splits are ignored as extras.
    ignore_extra = set(NON_SERVING_COLUMNS) | {"TransactionDT"}
    serving_fields = set(reference.get("schema", {}).get("serving_request_fields") or [])
    ignore_extra |= serving_fields - set(reference["schema"]["columns"])

    # Feature-frame schema: the 464 model columns the API actually scores.
    checks = run_tier1_checks(
        reference=reference,
        comparison_df=feature_df,
        scores=calibrated,
        actions=actions,
        amounts=amounts,
        config=config,
        ignore_extra=ignore_extra,
        psi_features=reference.get("psi_numeric_features"),
    )

    labels_present = bool(args.labels)
    if labels_present:
        if "isFraud" not in source_df.columns and "isFraud" not in transformed.columns:
            raise RuntimeError("--labels was set but comparison data has no isFraud column")
        label_src = transformed if "isFraud" in transformed.columns else source_df
        labels = label_src["isFraud"].to_numpy()
        operating_threshold = float(reference["labels"]["operating_threshold"])
        comparison_metrics = label_metrics(
            labels,
            calibrated,
            actions,
            amounts,
            artifacts["cost_params"],
            operating_threshold=operating_threshold,
        )
        checks.extend(check_tier2(comparison_metrics, reference["labels"], config))

    report = build_report(
        reference=reference,
        checks=checks,
        comparison_n_rows=len(source_df),
        labels_present=labels_present,
        generated_at=utc_now_iso(),
    )
    report["thresholds_status"] = config.get("status", "proposed")
    report["thresholds_description"] = config.get("description", "")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output
    if output is None:
        output = REPORTS_DIR / f"drift_report_{utc_now_compact()}.json"
    elif not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output, report)

    summary = format_human_summary(report)
    print()
    print(summary)
    print()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
