"""Batch report aggregates scores without inventing labels."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prediction_batch_report import (  # noqa: E402
    contract_from_openapi,
    http_error_summary,
    record_from_response,
    run,
    summarize,
    validate_batch,
)

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


def _payload_rows() -> list[dict]:
    return json.loads(PAYLOADS.read_text())


def _api_contract() -> dict:
    from services.api.app import app

    return contract_from_openapi(app.openapi())


def _drop_amount(example_ids: set[str]) -> list[dict]:
    rows = []
    for item in _payload_rows():
        request = dict(item["request"])
        if item["example_id"] in example_ids:
            del request["TransactionAmt"]
        rows.append({"example_id": item["example_id"], "request": request})
    return rows


def test_validate_batch_reports_every_missing_amount():
    dropped = {"syn-003", "syn-011", "syn-019"}
    errors = validate_batch(_drop_amount(dropped), _api_contract())
    assert [(error["example_id"], error["field"]) for error in errors] == [
        ("syn-003", "TransactionAmt"),
        ("syn-011", "TransactionAmt"),
        ("syn-019", "TransactionAmt"),
    ]
    assert {error["problem"] for error in errors} == {"missing_required"}


def test_valid_batch_passes_validation():
    assert validate_batch(_payload_rows(), _api_contract()) == []


def test_contract_requires_transaction_amount():
    assert "TransactionAmt" in _api_contract()["required"]


def test_run_rejects_invalid_batch_with_no_posts(monkeypatch, tmp_path, capsys):
    from services.api.app import app

    spec = app.openapi()
    posts = {"count": 0}
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(_drop_amount({"syn-003", "syn-011", "syn-019"})))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def urlopen(request, timeout=30):
        if request.get_method() == "POST":
            posts["count"] += 1
            raise AssertionError("POST /predict must not be called")
        assert request.full_url.endswith("/openapi.json")
        return _Response(json.dumps(spec).encode("utf-8"))

    monkeypatch.setattr("prediction_batch_report.urllib.request.urlopen", urlopen)
    with pytest.raises(SystemExit) as caught:
        run("http://fraud-api:8000", invalid, None, out_dir)
    assert caught.value.code == 2
    assert posts["count"] == 0
    assert list(out_dir.iterdir()) == []
    error_text = capsys.readouterr().err
    assert "batch rejected: 3 of 24 rows invalid" in error_text
    assert "TransactionAmt missing in syn-003, syn-011, syn-019" in error_text


def test_http_422_message_keeps_type_and_loc_only():
    message = http_error_summary(
        "syn-003",
        422,
        {
            "detail": [
                {
                    "type": "missing",
                    "loc": ["body", "TransactionAmt"],
                    "msg": "Field required",
                    "input": {"card4": "do-not-print"},
                }
            ]
        },
    )
    assert message == (
        "POST /predict failed for syn-003: HTTP 422; "
        "type=missing loc=body.TransactionAmt"
    )
    assert "do-not-print" not in message
