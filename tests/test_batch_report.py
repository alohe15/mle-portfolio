"""Batch report aggregates scores without inventing labels."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prediction_batch_report import record_from_response, summarize  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "simulated_labels.json"
PAYLOADS = ROOT / "tests" / "load" / "payloads.json"


def _rows_from_payloads() -> list[dict]:
    payloads = json.loads(PAYLOADS.read_text())
    return [
        {
            "example_id": item["example_id"],
            "fraud_probability": 0.2,
            "recommended_action": "review",
        }
        for item in payloads
    ]


def test_missing_labels_are_unavailable():
    report = summarize(_rows_from_payloads(), labels=None)
    assert report["label_status"] == "unavailable"
    assert report["performance"] is None
    assert report["matched_label_count"] is None
    assert report["label_coverage"] is None
    assert report["joined_labels"] is None


def test_partial_simulated_labels_do_not_fill_zero():
    labels = json.loads(FIXTURE.read_text())
    rows = _rows_from_payloads()
    report = summarize(rows, labels=labels)
    assert report["label_status"] == "simulated"
    assert report["matched_label_count"] == 10
    assert report["label_coverage"] == 10 / 24
    assert report["performance"] == "not computed (simulated labels, n too small)"
    assert len(report["joined_labels"]) == 10
    labeled_ids = set(labels["labels"])
    for row in rows:
        example_id = row["example_id"]
        if example_id not in labeled_ids:
            assert example_id not in report["joined_labels"]
            assert report["joined_labels"].get(example_id) != 0


def test_original_score_is_not_replaced_by_action():
    original = 0.1234567890123
    response = {
        "fraud_probability": original,
        "recommended_action": "review",
        "request_id": "req-1",
    }
    record = record_from_response("syn-001", response)
    assert record["fraud_probability"] == original
    assert record["fraud_probability"] != record["recommended_action"]
    assert record["recommended_action"] == "review"

    rows = [
        {
            "example_id": "syn-001",
            "fraud_probability": original,
            "recommended_action": "review",
        }
    ]
    report = summarize(rows)
    assert report["scores"] == [original]
    assert rows[0]["fraud_probability"] == original


def test_invalid_score_is_counted_and_excluded_from_quantiles():
    rows = [
        {"example_id": "a", "fraud_probability": 0.2, "recommended_action": "approve"},
        {"example_id": "b", "fraud_probability": None, "recommended_action": "approve"},
        {
            "example_id": "c",
            "fraud_probability": float("nan"),
            "recommended_action": "decline",
        },
        {"example_id": "d", "fraud_probability": 1.5, "recommended_action": "review"},
        {"example_id": "e", "fraud_probability": -0.1, "recommended_action": "nope"},
        {"example_id": "f", "fraud_probability": 0.8, "recommended_action": "step_up"},
    ]
    report = summarize(rows)
    assert report["row_count"] == 6
    assert report["invalid_score_count"] == 4
    assert report["score_quantiles"]["min"] == 0.2
    assert report["score_quantiles"]["max"] == 0.8
    assert 1.5 not in report["score_quantiles"].values()
    assert all(
        value is None or not math.isnan(value)
        for value in report["score_quantiles"].values()
    )
    assert report["scores"][1] is None
    assert math.isnan(report["scores"][2])
    assert report["scores"][3] == 1.5
    assert report["action_counts"]["other"] == 1
    assert report["action_counts"]["approve"] == 2
    assert report["action_counts"]["decline"] == 1
    assert report["action_counts"]["review"] == 1
    assert report["action_counts"]["step_up"] == 1
