"""Compare all model versions listed in models/registry.json.

Usage:
    python evals/compare_models.py

Writes a human-readable summary to evals/model_comparison_results.txt.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
OUTPUT_PATH = REPO_ROOT / "evals" / "model_comparison_results.txt"

TABLE_COLUMNS = [
    ("version", "version"),
    ("dataset_version", "dataset_version"),
    ("description", "description"),
    ("test_auc_pr", "test_auc_pr"),
    ("precision_at_budget", "precision_at_budget"),
    ("recall_at_budget", "recall_at_budget"),
    ("best_iteration", "best_iteration"),
    ("n_features", "n_features"),
]


def extract_test_auc_pr(metrics_doc: dict) -> float:
    """Return holdout test AUC-PR, normalizing across metric schema variants."""
    metrics = metrics_doc.get("metrics", metrics_doc)
    test_auc_pr = metrics.get("test_auc_pr")
    if test_auc_pr is not None:
        return float(test_auc_pr)
    auc_pr = metrics.get("auc_pr")
    if auc_pr is not None:
        return float(auc_pr)
    raise KeyError("No test_auc_pr or auc_pr found in metrics document")


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must contain a JSON array")
    return registry


def load_metrics(metrics_path: str) -> dict:
    path = repo_path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    return load_json(path)


def build_comparison_rows(registry: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in sorted(registry, key=lambda item: item["version"]):
        metrics_doc = load_metrics(entry["metrics_path"])
        rows.append(
            {
                "version": entry["version"],
                "dataset_version": entry["dataset_version"],
                "description": entry["description"],
                "test_auc_pr": extract_test_auc_pr(metrics_doc),
                "precision_at_budget": metrics_doc["metrics"]["precision_at_budget"],
                "recall_at_budget": metrics_doc["metrics"]["recall_at_budget"],
                "best_iteration": metrics_doc["metrics"]["best_iteration"],
                "n_features": metrics_doc["training"]["n_features"],
                "metrics_path": entry["metrics_path"],
            }
        )
    return rows


def format_table(rows: list[dict]) -> str:
    headers = [label for label, _ in TABLE_COLUMNS]
    keys = [key for _, key in TABLE_COLUMNS]

    str_rows = []
    for row in rows:
        str_rows.append(
            {
                "version": str(row["version"]),
                "dataset_version": str(row["dataset_version"]),
                "description": row["description"],
                "test_auc_pr": f"{row['test_auc_pr']:.4f}",
                "precision_at_budget": f"{row['precision_at_budget']:.4f}",
                "recall_at_budget": f"{row['recall_at_budget']:.4f}",
                "best_iteration": str(row["best_iteration"]),
                "n_features": str(row["n_features"]),
            }
        )

    widths = {header: len(header) for header in headers}
    for row in str_rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    description_width = max(widths["description"], 20)

    def format_row(row: dict[str, str]) -> str:
        return (
            f"{row['version']:>{widths['version']}}  "
            f"{row['dataset_version']:>{widths['dataset_version']}}  "
            f"{row['description']:<{description_width}}  "
            f"{row['test_auc_pr']:>{widths['test_auc_pr']}}  "
            f"{row['precision_at_budget']:>{widths['precision_at_budget']}}  "
            f"{row['recall_at_budget']:>{widths['recall_at_budget']}}  "
            f"{row['best_iteration']:>{widths['best_iteration']}}  "
            f"{row['n_features']:>{widths['n_features']}}"
        )

    header_line = (
        f"{'version':>{widths['version']}}  "
        f"{'dataset_version':>{widths['dataset_version']}}  "
        f"{'description':<{description_width}}  "
        f"{'test_auc_pr':>{widths['test_auc_pr']}}  "
        f"{'precision_at_budget':>{widths['precision_at_budget']}}  "
        f"{'recall_at_budget':>{widths['recall_at_budget']}}  "
        f"{'best_iteration':>{widths['best_iteration']}}  "
        f"{'n_features':>{widths['n_features']}}"
    )
    separator = "-" * len(header_line)
    body = [header_line, separator]
    body.extend(format_row(row) for row in str_rows)
    return "\n".join(body)


def build_report(rows: list[dict]) -> str:
    lines = [
        "IEEE-CIS Fraud Detection — Model Comparison",
        "=" * 64,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Registry: {REGISTRY_PATH.relative_to(REPO_ROOT)}",
        f"Versions: {len(rows)}",
        "",
        "Metrics sources:",
    ]
    for row in rows:
        lines.append(f"  v{row['version']}: {row['metrics_path']}")
    lines.extend(["", format_table(rows), ""])
    return "\n".join(lines)


def main() -> None:
    registry = load_registry()
    if not registry:
        raise ValueError("models/registry.json is empty")

    rows = build_comparison_rows(registry)
    report = build_report(rows)
    OUTPUT_PATH.write_text(report)

    print(f"Wrote comparison to {OUTPUT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
