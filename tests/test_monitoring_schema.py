"""Unit tests for monitoring checks — synthetic frames only, no real data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from monitoring import (  # noqa: E402
    action_proportions,
    build_report,
    check_action_band_volumes,
    check_missingness,
    check_psi_shift,
    check_schema,
    check_tier2,
    dump_json,
    format_human_summary,
    load_monitoring_config,
    overall_status,
    psi_from_values,
    run_tier1_checks,
)

CONFIG = load_monitoring_config()


def _schema_ref(columns: list[str], dtypes: dict[str, str] | None = None) -> dict:
    dtypes = dtypes or {c: "float64" for c in columns}
    return {"columns": columns, "dtypes": dtypes, "expected_column_count": len(columns)}


def _mini_reference(df: pd.DataFrame, scores: np.ndarray, actions: np.ndarray) -> dict:
    columns = list(df.columns)
    missingness = {c: float(df[c].isna().mean()) for c in columns}
    numerics = {}
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            from monitoring import numeric_summary

            numerics[c] = numeric_summary(pd.to_numeric(df[c], errors="coerce").to_numpy())
    from monitoring import numeric_summary, category_proportions

    cats = {
        c: category_proportions(df[c])
        for c in columns
        if not pd.api.types.is_numeric_dtype(df[c])
    }
    return {
        "schema": _schema_ref(
            columns,
            {c: str(df[c].dtype) for c in columns},
        ),
        "missingness": missingness,
        "numerics": numerics,
        "categoricals": cats,
        "scores": numeric_summary(scores),
        "actions": action_proportions(actions),
        "amounts": numeric_summary(
            pd.to_numeric(df["TransactionAmt"], errors="coerce").to_numpy()
            if "TransactionAmt" in df.columns
            else np.ones(len(df))
        ),
        "labels": {
            "fraud_rate": 0.1,
            "pr_auc": 0.50,
            "precision": 0.40,
            "recall": 0.40,
            "false_positive_rate": 0.05,
            "operating_threshold": 0.2,
            "fraud_dollars_captured": 1000.0,
            "fraud_dollars_total": 1250.0,
            "fraud_dollar_capture_rate": 0.80,
            "fraud_dollars_by_action": {
                "approve": 100.0,
                "step_up": 200.0,
                "review": 300.0,
                "decline": 400.0,
            },
        },
        "psi_numeric_features": [c for c in columns if pd.api.types.is_numeric_dtype(df[c])][:20],
        "metadata": {
            "model_version": 9,
            "dataset_version": 5,
            "n_rows": int(len(df)),
            "split": "train",
        },
    }


def test_schema_missing_column_is_critical():
    ref = _schema_ref(["a", "b"])
    df = pd.DataFrame({"a": [1.0, 2.0]})
    checks = check_schema(df, ref, CONFIG)
    missing = next(c for c in checks if c["name"] == "schema_missing_columns")
    assert missing["status"] == "critical"
    assert missing["observed_value"] == 1
    assert missing["reference_value"] == 0
    assert missing["threshold"] == 0
    assert "b" in missing["detail"]


def test_schema_extra_column_is_warning():
    ref = _schema_ref(["a"])
    df = pd.DataFrame({"a": [1.0], "bonus": [0.0]})
    checks = check_schema(df, ref, CONFIG)
    extra = next(c for c in checks if c["name"] == "schema_extra_columns")
    assert extra["status"] == "warning"
    assert extra["observed_value"] == 1
    assert extra["threshold"] == 0
    assert "bonus" in extra["detail"]


def test_schema_dtype_mismatch_is_critical():
    ref = _schema_ref(["amt"], {"amt": "float64"})
    df = pd.DataFrame({"amt": ["1.0", "2.0"]})
    checks = check_schema(df, ref, CONFIG)
    mismatch = next(c for c in checks if c["name"] == "schema_dtype_mismatch")
    assert mismatch["status"] == "critical"
    assert mismatch["observed_value"] == 1
    assert mismatch["threshold"] == 0


def test_schema_all_columns_correct_passes():
    ref = _schema_ref(["a", "b"], {"a": "float64", "b": "float64"})
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    checks = check_schema(df, ref, CONFIG)
    assert all(c["status"] == "pass" for c in checks)
    assert overall_status(checks) == "healthy"


def test_missingness_increase_flagged():
    reference_rates = {"feat": 0.01}
    comparison_rates = {"feat": 0.10}
    checks = check_missingness(comparison_rates, reference_rates, CONFIG)
    rec = checks[0]
    assert rec["name"] == "missingness_rate_change"
    assert rec["status"] == "warning"
    assert rec["observed_value"] == pytest.approx(0.09)
    assert rec["threshold"] == 0.05


def test_missingness_stable_passes():
    reference_rates = {"feat": 0.01}
    comparison_rates = {"feat": 0.02}
    checks = check_missingness(comparison_rates, reference_rates, CONFIG)
    rec = checks[0]
    assert rec["status"] == "pass"
    assert rec["observed_value"] == pytest.approx(0.01)


def test_psi_identical_distributions_pass():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=8000)
    psi, _ = psi_from_values(values, values.copy())
    assert psi == pytest.approx(0.0, abs=1e-12)
    rec = check_psi_shift(values, values.copy(), CONFIG)
    assert rec["status"] == "pass"
    assert rec["observed_value"] == pytest.approx(0.0, abs=1e-12)
    assert rec["threshold"] == 0.2


def test_psi_shifted_distribution_flagged():
    rng = np.random.default_rng(1)
    reference = rng.normal(0.0, 1.0, size=8000)
    shifted = rng.normal(4.0, 1.0, size=8000)
    psi, _ = psi_from_values(reference, shifted)
    assert psi > 0.2
    rec = check_psi_shift(reference, shifted, CONFIG)
    assert rec["status"] == "warning"
    assert rec["observed_value"] > 0.2


def test_action_band_unchanged_passes():
    props = {
        "approve": 0.80,
        "step_up": 0.10,
        "review": 0.05,
        "decline": 0.05,
    }
    checks = check_action_band_volumes(props, props, CONFIG)
    rec = checks[0]
    assert rec["status"] == "pass"
    assert rec["threshold"] == 0.30


def test_action_band_decline_doubles_flagged():
    reference = {
        "approve": 0.80,
        "step_up": 0.10,
        "review": 0.05,
        "decline": 0.05,
    }
    comparison = {
        "approve": 0.75,
        "step_up": 0.10,
        "review": 0.05,
        "decline": 0.10,
    }
    checks = check_action_band_volumes(comparison, reference, CONFIG)
    rec = checks[0]
    assert rec["status"] == "warning"
    flagged_actions = {row["action"] for row in rec["detail"]["flagged"]}
    assert "decline" in flagged_actions


def _tiny_scored_frame(n: int = 10) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "TransactionAmt": rng.uniform(10, 200, size=n),
            "card1": rng.normal(5000, 50, size=n),
        }
    )
    scores = rng.uniform(0.01, 0.2, size=n)
    amounts = df["TransactionAmt"].to_numpy()
    actions = np.array(["approve"] * n, dtype=object)
    return df, scores, actions, amounts


def test_report_generation_tiny_synthetic(tmp_path: Path):
    df, scores, actions, amounts = _tiny_scored_frame(10)
    reference = _mini_reference(df, scores, actions)
    checks = run_tier1_checks(
        reference=reference,
        comparison_df=df,
        scores=scores,
        actions=actions,
        amounts=amounts,
        config=CONFIG,
    )
    report = build_report(
        reference=reference,
        checks=checks,
        comparison_n_rows=len(df),
        labels_present=False,
    )
    out = tmp_path / "drift_report.json"
    dump_json(out, report)
    loaded = json.loads(out.read_text())
    assert loaded["overall_status"] in {"healthy", "warning", "critical"}
    assert isinstance(loaded["checks"], list)
    assert loaded["checks"], "expected at least one check"
    for check in loaded["checks"]:
        assert check["status"] in {"pass", "warning", "critical"}
        assert "observed_value" in check
        assert "reference_value" in check
        assert "threshold" in check
    summary = format_human_summary(loaded)
    assert "Overall status" in summary
    assert "TIER 1" in summary


def test_report_overall_status_healthy_warning_critical():
    def _check(status: str) -> dict:
        return {
            "name": "x",
            "tier": "tier1_immediate",
            "status": status,
            "observed_value": 0,
            "reference_value": 0,
            "threshold": 0,
            "severity": status if status != "pass" else "warning",
        }

    assert overall_status([_check("pass"), _check("pass")]) == "healthy"
    assert overall_status([_check("pass"), _check("warning")]) == "warning"
    assert overall_status([_check("warning"), _check("critical")]) == "critical"
    assert overall_status([_check("critical")]) == "critical"

    healthy = build_report(
        reference={"metadata": {}},
        checks=[_check("pass")],
        comparison_n_rows=10,
        labels_present=False,
    )
    warning = build_report(
        reference={"metadata": {}},
        checks=[_check("warning")],
        comparison_n_rows=10,
        labels_present=False,
    )
    critical = build_report(
        reference={"metadata": {}},
        checks=[_check("critical")],
        comparison_n_rows=10,
        labels_present=False,
    )
    assert healthy["overall_status"] == "healthy"
    assert warning["overall_status"] == "warning"
    assert critical["overall_status"] == "critical"


def test_tier2_skipped_when_labels_absent():
    df, scores, actions, amounts = _tiny_scored_frame(10)
    reference = _mini_reference(df, scores, actions)
    checks = run_tier1_checks(
        reference=reference,
        comparison_df=df,
        scores=scores,
        actions=actions,
        amounts=amounts,
        config=CONFIG,
    )
    report = build_report(
        reference=reference,
        checks=checks,
        comparison_n_rows=len(df),
        labels_present=False,
    )
    assert report["metadata"]["labels_present"] is False
    assert all(c["tier"] != "tier2_delayed" for c in report["checks"])
    summary = format_human_summary(report)
    assert "skipped" in summary.lower() or "TIER 2" in summary


def test_tier2_pr_auc_drop_is_critical():
    reference_labels = {
        "fraud_rate": 0.03,
        "pr_auc": 0.60,
        "precision": 0.50,
        "recall": 0.50,
        "false_positive_rate": 0.02,
        "fraud_dollars_captured": 10_000.0,
        "fraud_dollars_total": 12_000.0,
        "fraud_dollar_capture_rate": 10_000.0 / 12_000.0,
        "fraud_dollars_by_action": {
            "approve": 0.0,
            "step_up": 0.0,
            "review": 5000.0,
            "decline": 5000.0,
        },
    }
    comparison_metrics = {
        "fraud_rate": 0.03,
        "pr_auc": 0.50,  # drop of 0.10 >= 0.05
        "precision": 0.48,
        "recall": 0.48,
        "false_positive_rate": 0.021,
        "fraud_dollars_captured": 9_500.0,
        "fraud_dollars_total": 12_000.0,
        "fraud_dollar_capture_rate": 9_500.0 / 12_000.0,
        "fraud_dollars_by_action": {
            "approve": 0.0,
            "step_up": 0.0,
            "review": 4500.0,
            "decline": 5000.0,
        },
    }
    checks = check_tier2(comparison_metrics, reference_labels, CONFIG)
    pr = next(c for c in checks if c["name"] == "pr_auc_drop")
    assert pr["status"] == "critical"
    assert pr["observed_value"] == pytest.approx(0.50)
    assert pr["reference_value"] == pytest.approx(0.60)
    assert pr["threshold"] == 0.05
    assert overall_status(checks) == "critical"
