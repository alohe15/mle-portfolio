#!/usr/bin/env python3
"""Score a fixed synthetic payload file and summarise the returned predictions.

Standard library only. This writes a local debugging artifact. It does not
log request bodies or scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _deref(schema: dict, schemas: dict, depth: int = 0) -> dict:
    """Follow $ref. Sibling keys on the reference win over the target."""
    if depth > 12 or not isinstance(schema, dict):
        return schema
    if "$ref" not in schema:
        return schema
    name = str(schema["$ref"]).rsplit("/", 1)[-1]
    resolved = _deref(schemas[name], schemas, depth + 1)
    extra = {key: value for key, value in schema.items() if key != "$ref"}
    if extra and isinstance(resolved, dict):
        merged = dict(resolved)
        merged.update(extra)
        return merged
    return resolved


def _type_info(schema: dict, schemas: dict) -> tuple[list[str], bool]:
    """Concrete JSON types for a property, and whether null is allowed."""
    schema = _deref(schema, schemas)
    options = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(options, list):
        types: list[str] = []
        nullable = False
        for option in options:
            option_types, option_null = _type_info(option, schemas)
            nullable = nullable or option_null
            for name in option_types:
                if name not in types:
                    types.append(name)
        return types, nullable
    declared = schema.get("type")
    if isinstance(declared, list):
        nullable = "null" in declared
        return [name for name in declared if name != "null"], nullable
    if declared == "null":
        return [], True
    if isinstance(declared, str):
        return [declared], False
    if "properties" in schema:
        return ["object"], False
    return [], False


def contract_from_openapi(spec: dict) -> dict:
    """TransactionRequest as required names plus resolved property types.

    The schema is the POST /predict JSON body. Field names come from that
    schema, not from a list in this file.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    body = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    schema = _deref(body, schemas)
    properties = {}
    for name, prop in schema.get("properties", {}).items():
        types, nullable = _type_info(prop, schemas)
        properties[name] = {"types": types, "nullable": nullable}
    return {
        "required": list(schema.get("required") or []),
        "properties": properties,
    }


def fetch_contract(base_url: str) -> dict:
    """GET /openapi.json and return the TransactionRequest contract."""
    url = base_url.rstrip("/") + "/openapi.json"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        spec = json.loads(response.read().decode("utf-8"))
    return contract_from_openapi(spec)


def _value_matches(value: object, types: list[str]) -> bool:
    for name in types:
        if name == "string" and isinstance(value, str):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "object" and isinstance(value, dict):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "null" and value is None:
            return True
    return False


def validate_batch(rows: list[dict], contract: dict) -> list[dict]:
    """Check every row. Return all errors; do not stop at the first."""
    required = set(contract.get("required") or [])
    properties = contract.get("properties") or {}
    errors: list[dict] = []
    for row in rows:
        example_id = row.get("example_id")
        request = row.get("request")
        if not isinstance(request, dict):
            errors.append(
                {"example_id": example_id, "field": "request", "problem": "wrong_type"}
            )
            continue
        for field, spec in properties.items():
            types = list(spec.get("types") or [])
            nullable = bool(spec.get("nullable"))
            if field not in request:
                if field in required:
                    errors.append(
                        {
                            "example_id": example_id,
                            "field": field,
                            "problem": "missing_required",
                        }
                    )
                continue
            value = request[field]
            if value is None:
                if field in required and not nullable:
                    errors.append(
                        {
                            "example_id": example_id,
                            "field": field,
                            "problem": "null_required",
                        }
                    )
                elif not nullable and "null" not in types:
                    errors.append(
                        {
                            "example_id": example_id,
                            "field": field,
                            "problem": "wrong_type",
                        }
                    )
                continue
            if types and not _value_matches(value, types):
                errors.append(
                    {"example_id": example_id, "field": field, "problem": "wrong_type"}
                )
    return errors


def format_rejection(rows: list[dict], errors: list[dict]) -> str:
    """One stderr line. Names every bad row, not only the first."""
    invalid = []
    seen = set()
    for error in errors:
        example_id = error["example_id"]
        if example_id not in seen:
            seen.add(example_id)
            invalid.append(example_id)
    grouped: dict[tuple[str, str], list] = {}
    order: list[tuple[str, str]] = []
    words = {
        "missing_required": "missing",
        "null_required": "null",
        "wrong_type": "wrong type",
    }
    for error in errors:
        key = (str(error["field"]), words.get(error["problem"], error["problem"]))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        if error["example_id"] not in grouped[key]:
            grouped[key].append(error["example_id"])
    details = [
        f"{field} {word} in {', '.join(str(example_id) for example_id in ids)}"
        for (field, word), ids in ((key, grouped[key]) for key in order)
    ]
    return (
        f"batch rejected: {len(invalid)} of {len(rows)} rows invalid; "
        + "; ".join(details)
    )


def http_error_summary(example_id: str, status: int, body: object) -> str:
    """example_id plus 422 type and loc. Never the submitted values."""
    message = f"POST /predict failed for {example_id}: HTTP {status}"
    if status != 422 or not isinstance(body, dict):
        return message
    detail = body.get("detail")
    if not isinstance(detail, list):
        return message
    parts = []
    for item in detail:
        if not isinstance(item, dict):
            continue
        loc = item.get("loc")
        if isinstance(loc, list):
            loc_text = ".".join(str(piece) for piece in loc)
        else:
            loc_text = "" if loc is None else str(loc)
        parts.append(f"type={item.get('type')} loc={loc_text}")
    if not parts:
        return message
    return message + "; " + "; ".join(parts)


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


def _read_http_error(exc: urllib.error.HTTPError) -> object:
    try:
        raw = exc.read()
    except Exception:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def run(base_url: str, payloads_path: Path, labels_path: str | None, out_dir: Path) -> dict:
    payloads = json.loads(payloads_path.read_text())
    labels = _load_labels(labels_path)
    errors = validate_batch(payloads, fetch_contract(base_url))
    if errors:
        print(format_rejection(payloads, errors), file=sys.stderr)
        raise SystemExit(2)
    records = []
    for item in payloads:
        example_id = item["example_id"]
        try:
            body = _post_predict(base_url, item["request"])
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                http_error_summary(example_id, exc.code, _read_http_error(exc))
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
