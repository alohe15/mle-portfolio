#!/usr/bin/env python3
"""Score a fixed synthetic payload file and summarise the returned predictions.

Standard library only. This writes a local debugging artifact. It does not
log request bodies or scores.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

# Same four names as ACTION_NAMES in evals/cost_model.py. Imported nowhere:
# that module is not part of the standard library.
ACTION_NAMES = ("approve", "step_up", "review", "decline")
QUANTILES = (
    ("min", 0.0),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("max", 1.0),
)
SIMULATED_PERFORMANCE = "not computed (simulated labels, n too small)"


def is_valid_score(value: object) -> bool:
    """True when value is a real number on the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return 0.0 <= float(value) <= 1.0


def _quantile(sorted_values: list[float], probability: float) -> float:
    """Linear interpolation between nearest ranks. One value fills every quantile."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _action_counts(rows: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in ACTION_NAMES}
    counts["other"] = 0
    for row in rows:
        action = row.get("recommended_action")
        if action in counts and action != "other":
            counts[action] += 1
        else:
            counts["other"] += 1
    return counts


def record_from_response(example_id: str, body: dict) -> dict:
    """Keep the returned probability as its own field, separate from the action."""
    return {
        "example_id": example_id,
        "fraud_probability": body["fraud_probability"],
        "recommended_action": body["recommended_action"],
        "request_id": body["request_id"],
    }


def summarize(rows: list[dict], labels: dict | None = None) -> dict:
    """Aggregate prediction rows. Invalid scores stay in the row count.

    ``labels`` is either None or ``{"simulated": true, "labels": {example_id: 0 or 1}}``.
    Unknown labels are left absent. They are not filled with 0.
    """
    valid_scores = [
        float(row["fraud_probability"])
        for row in rows
        if is_valid_score(row.get("fraud_probability"))
    ]
    valid_scores.sort()
    if valid_scores:
        quantiles = {
            name: _quantile(valid_scores, probability)
            for name, probability in QUANTILES
        }
    else:
        quantiles = {name: None for name, _probability in QUANTILES}

    report = {
        "row_count": len(rows),
        "invalid_score_count": len(rows) - len(valid_scores),
        "score_quantiles": quantiles,
        "action_counts": _action_counts(rows),
        "scores": [row.get("fraud_probability") for row in rows],
        "label_status": "unavailable",
        "matched_label_count": None,
        "label_coverage": None,
        "joined_labels": None,
        "performance": None,
    }
    if labels is None:
        return report

    if labels.get("simulated") is not True:
        raise ValueError("labels must set simulated to true")
    label_map = labels.get("labels")
    if not isinstance(label_map, dict):
        raise ValueError("labels must contain a labels object")

    joined = {}
    for row in rows:
        example_id = row.get("example_id")
        if example_id in label_map:
            joined[example_id] = label_map[example_id]
    matched = len(joined)
    report["label_status"] = "simulated"
    report["matched_label_count"] = matched
    report["label_coverage"] = (matched / len(rows)) if rows else None
    report["joined_labels"] = joined
    report["performance"] = SIMULATED_PERFORMANCE
    return report


def _post_predict(base_url: str, request_body: dict) -> dict:
    url = base_url.rstrip("/") + "/predict"
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_labels(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def run(base_url: str, payloads_path: Path, labels_path: str | None, out_dir: Path) -> dict:
    payloads = json.loads(payloads_path.read_text())
    labels = _load_labels(labels_path)
    records = []
    for item in payloads:
        example_id = item["example_id"]
        try:
            body = _post_predict(base_url, item["request"])
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"POST /predict failed for {example_id}: HTTP {exc.code}"
            ) from None
        records.append(record_from_response(example_id, body))

    summary_rows = [
        {
            "example_id": record["example_id"],
            "fraud_probability": record["fraud_probability"],
            "recommended_action": record["recommended_action"],
        }
        for record in records
    ]
    report = summarize(summary_rows, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    report_path = out_dir / "report.json"
    text = json.dumps(report, indent=2) + "\n"
    report_path.write_text(text)
    print(text, end="")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise a synthetic /predict batch")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--payloads", default="tests/load/payloads.json")
    parser.add_argument("--labels", default=None)
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "fraud-evidence" / "batch"),
    )
    args = parser.parse_args()
    run(args.base_url, Path(args.payloads), args.labels, Path(args.out_dir))


if __name__ == "__main__":
    main()
