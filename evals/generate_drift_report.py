"""Generate a two-tier drift report against a frozen reference profile.

Usage:
    python evals/generate_drift_report.py evals/reference_profile.json <comparison.parquet> [--labels] [--label-age-days N]

Loads the serving model, calibrator, fitted transforms, and cost params from
models/registry.json via services/api/model_loader.py (§10).

Validation order:
  1. Read the comparison file as-is.
  2. Schema, missingness, and categorical checks on that raw frame
     (before NaN-fill, dtype coercion, column drops, or categorical encoding).
  3. Apply frozen fitted transforms.
  4. Score (model-input build does NaN-fill / coercion / column selection).
  5. PSI, score distribution, and action-band checks on the scored frame.
  6. Tier 2 only when labels are present, at least label_maturity days old,
     and the reference is not a training-split profile under tier2_reference
     "out_of_sample". Otherwise every Tier 2 check is status "not_ready"
     and no Tier 2 metric is computed.

Writes JSON under reports/monitoring/ (gitignored) and prints a summary.
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
    check_tier2,
    dump_json,
    fold_missing_produced_columns,
    format_human_summary,
    label_metrics,
    load_json,
    load_monitoring_config,
    load_serving_artifacts,
    make_tier2_not_ready,
    missingness_rates_before_nan_fill,
    replace_missingness_check,
    requires_fit_output_columns,
    run_raw_input_checks,
    run_scored_tier1_checks,
    score_and_act,
    tier2_not_ready_reason,
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
        "--label-age-days",
        type=int,
        default=None,
        help=(
            "Age in days of isFraud labels in the comparison window. "
            "Tier 2 stays not_ready unless this is >= label_maturity "
            "(default 30, chargeback lag)."
        ),
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
    produced = requires_fit_output_columns(artifacts)

    # ID / label / serving-request columns that are not model features are
    # not "extra" schema violations.
    ignore_extra = set(NON_SERVING_COLUMNS) | {"TransactionDT"}
    schema_columns = set(reference["schema"]["columns"])
    serving_fields = set(reference.get("schema", {}).get("serving_request_fields") or [])
    ignore_extra |= serving_fields - schema_columns

    # 1–4. Raw file: schema, missingness, categoricals. requires_fit outputs
    # are not on the processed parquet; they are created in the next step
    # and must not be reported as missing or as 100% null.
    print(
        "Validating schema, missingness, and categoricals on the raw "
        "comparison file (before NaN-fill, coercion, or column drops) ..."
    )
    raw_checks = run_raw_input_checks(
        reference=reference,
        raw_df=source_df,
        config=config,
        ignore_extra=ignore_extra,
        ignore_missing=produced,
    )

    # 5. Frozen transforms, then record any output that is still absent
    # BEFORE model-input build NaN-fills it.
    print("Applying frozen fitted transforms ...")
    transformed = apply_monitoring_transforms(source_df, artifacts)
    raw_checks = fold_missing_produced_columns(
        raw_checks,
        transformed,
        produced,
        list(reference["schema"]["columns"]),
    )
    raw_checks = replace_missingness_check(
        raw_checks,
        missingness_rates_before_nan_fill(
            source_df,
            transformed,
            list(reference.get("missingness") or {}),
            produced,
        ),
        dict(reference.get("missingness") or {}),
        config,
    )

    scoring_error: Exception | None = None
    scored_checks: list = []
    calibrated = None
    actions = None
    amounts = None
    try:
        # NaN-fill, dtype coercion, and column selection happen inside
        # build_model_input_batch — only after the raw checks above.
        feature_df = build_model_input_batch(transformed, artifacts)
        if (
            "TransactionAmt" not in transformed.columns
            and "TransactionAmt" not in source_df.columns
        ):
            raise RuntimeError(
                "Comparison sample missing serving required field TransactionAmt"
            )
        amount_src = transformed if "TransactionAmt" in transformed.columns else source_df
        amounts = pd.to_numeric(amount_src["TransactionAmt"], errors="coerce").to_numpy()
        print(f"Scoring {len(feature_df):,} comparison rows ...")
        _raw_scores, calibrated, actions = score_and_act(feature_df, artifacts, amounts)
        scored_checks = run_scored_tier1_checks(
            reference=reference,
            comparison_df=feature_df,
            scores=calibrated,
            actions=actions,
            amounts=amounts,
            config=config,
            psi_features=reference.get("psi_numeric_features"),
        )
    except Exception as exc:  # noqa: BLE001 — report raw findings even if scoring fails
        scoring_error = exc
        print(f"Scoring failed after raw validation: {exc}")

    labels_present = bool(args.labels)
    tier2_checks: list = []
    tier2_status = "skipped"
    tier2_message = None
    if labels_present:
        if "isFraud" not in source_df.columns and "isFraud" not in transformed.columns:
            raise RuntimeError("--labels was set but comparison data has no isFraud column")
        tier2_message = tier2_not_ready_reason(config, reference, args.label_age_days)
        if tier2_message is not None:
            # Do not compute PR-AUC or dollar capture on immature or
            # training-memorized references.
            tier2_status = "not_ready"
            tier2_checks = make_tier2_not_ready(config, tier2_message)
        elif calibrated is None or actions is None or amounts is None:
            raise RuntimeError(
                "--labels was set and Tier 2 is eligible, but scoring failed"
            ) from scoring_error
        else:
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
            tier2_checks = check_tier2(comparison_metrics, reference["labels"], config)
            tier2_status = "computed"

    checks = raw_checks + scored_checks + tier2_checks
    report = build_report(
        reference=reference,
        checks=checks,
        comparison_n_rows=len(source_df),
        labels_present=labels_present,
        generated_at=utc_now_iso(),
    )
    report["thresholds_status"] = config.get("status", "proposed")
    report["thresholds_description"] = config.get("description", "")
    report["metadata"]["validation_order"] = [
        "raw_schema",
        "raw_missingness",
        "raw_categorical",
        "fitted_transforms",
        "nan_fill_coerce_select",
        "psi_score_action",
        "tier2_if_ready",
    ]
    report["metadata"]["label_age_days"] = args.label_age_days
    report["metadata"]["label_maturity"] = config.get("label_maturity", 30)
    report["metadata"]["tier2_reference"] = config.get("tier2_reference", "out_of_sample")
    report["metadata"]["tier2_status"] = tier2_status
    report["metadata"]["tier2_message"] = tier2_message
    if scoring_error is not None:
        report["metadata"]["scoring_error"] = str(scoring_error)

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
    if scoring_error is not None:
        raise scoring_error


if __name__ == "__main__":
    main()
