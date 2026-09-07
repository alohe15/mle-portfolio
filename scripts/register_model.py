"""Register a frozen model as Candidate in models/registry.json.

Validation-first: every referenced artifact must exist on disk and the
entry commit must match `git rev-parse HEAD` before anything is written.

Usage:
    python scripts/register_model.py v9

The `commit` field records HEAD at registration time — i.e. the evaluation
evidence commit — not the later commit that lands the registry update.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

KNOWN_STATUSES = {"Experimental", "Candidate", "Production", "Rolled_back"}

# Decision-policy evaluation results (validation split) from Phase 3.
VALIDATION_ARGMIN_COST = 223278
VALIDATION_THRESHOLD_COST = 283082
COST_REDUCTION_PCT = 21.1
BOOTSTRAP_CI = [-62609, -57372]


def fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_version_arg(raw: str) -> int:
    text = raw.strip().lower()
    if text.startswith("v") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    fail(f"version argument must look like 'v9' or '9', got {raw!r}")
    raise AssertionError  # unreachable


def find_dataset_manifest(dataset_version: int) -> Path:
    matches = sorted(PROCESSED_DIR.glob(f"dataset_v{dataset_version}_*_manifest.json"))
    if not matches:
        fail(f"No dataset manifest found for dataset_v{dataset_version} in {PROCESSED_DIR}")
    if len(matches) > 1:
        fail(f"Multiple dataset manifests for dataset_v{dataset_version}: {matches}")
    return matches[0]


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def require_exists(label: str, relative: str) -> Path:
    path = REPO_ROOT / relative
    if not path.exists():
        fail(f"{label} does not exist on disk: {relative}")
    if not path.is_file():
        fail(f"{label} is not a file: {relative}")
    return path


def find_legacy_entry(registry: list[dict], version: int) -> dict:
    matches = [e for e in registry if e.get("version") == version and e.get("status") != "Candidate"]
    # Prefer the serving entry when present; else the unique non-Candidate entry.
    serving = [e for e in matches if e.get("is_serving") is True]
    if serving:
        return serving[0]
    if not matches:
        fail(f"No existing registry entry for version {version} to supplement as Candidate")
    if len(matches) > 1:
        # Multiple non-Candidate rows for same version shouldn't happen today.
        return matches[-1]
    return matches[0]


def ensure_no_candidate(registry: list[dict], version: int) -> None:
    dupes = [
        e
        for e in registry
        if e.get("version") == version and e.get("status") == "Candidate"
    ]
    if dupes:
        fail(
            f"Candidate entry already exists for version {version} "
            f"(registered_at={dupes[0].get('registered_at')})"
        )


def build_candidate_entry(
    version: int,
    legacy: dict,
    commit: str,
    registered_at: str,
) -> dict:
    config_path = legacy["config_path"]
    require_exists("config_path", config_path)
    model_config = load_json(REPO_ROOT / config_path)

    metrics_path = legacy["metrics_path"]
    require_exists("metrics_path", metrics_path)
    metrics_doc = load_json(REPO_ROOT / metrics_path)
    test_auc_pr = float(metrics_doc["metrics"]["test_auc_pr"])

    tuning_manifest_path = legacy["manifest_path"]
    require_exists("tuning_manifest_path / manifest_path", tuning_manifest_path)
    tuning_manifest = load_json(REPO_ROOT / tuning_manifest_path)

    fitted_transforms_path = tuning_manifest.get("fitted_transforms_path")
    if not fitted_transforms_path:
        fail(f"fitted_transforms_path missing from {tuning_manifest_path}")

    model_path = legacy["model_path"]
    # Prefer model_file from tuning manifest when present (same artifact).
    if "model_file" in tuning_manifest:
        expected = f"models/{tuning_manifest['model_file']}"
        if expected != model_path:
            fail(
                f"model_path mismatch: registry has {model_path!r}, "
                f"tuning manifest implies {expected!r}"
            )

    dataset_version = int(legacy["dataset_version"])
    if int(model_config.get("dataset_version", -1)) != dataset_version:
        fail(
            f"dataset_version mismatch: registry={dataset_version}, "
            f"config={model_config.get('dataset_version')}"
        )

    dataset_manifest = find_dataset_manifest(dataset_version)
    dataset_manifest_path = repo_relative(dataset_manifest)

    calibrator_path = f"models/lgbm_v{version}_calibrator.pkl"
    decision_policy_config = "configs/decision_policy_v1.json"

    compare_script = "evals/compare_policies.py"
    sensitivity_script = "evals/sensitivity_analysis.py"
    decision_doc = "docs/decision_policy.md"

    paths_to_check = {
        "model_path": model_path,
        "config_path": config_path,
        "dataset_manifest_path": dataset_manifest_path,
        "tuning_manifest_path": tuning_manifest_path,
        "fitted_transforms_path": fitted_transforms_path,
        "calibrator_path": calibrator_path,
        "decision_policy_config": decision_policy_config,
        "compare_policies_script": compare_script,
        "sensitivity_analysis_script": sensitivity_script,
        "decision_policy_doc": decision_doc,
        "metrics_path": metrics_path,
        "manifest_path": tuning_manifest_path,
        "dataset_config_path": legacy["dataset_config_path"],
    }
    for label, rel in paths_to_check.items():
        require_exists(label, rel)

    head = git_head()
    if commit != head:
        fail(f"commit field {commit!r} does not match git rev-parse HEAD {head!r}")

    # Same core schema as legacy entries, plus Candidate lifecycle fields.
    entry = {
        "version": version,
        "dataset_version": dataset_version,
        "description": legacy["description"],
        "model_path": model_path,
        "manifest_path": tuning_manifest_path,
        "metrics_path": metrics_path,
        "config_path": config_path,
        "dataset_config_path": legacy["dataset_config_path"],
        "created_at": legacy.get("created_at"),
        "is_serving": False,
        "status": "Candidate",
        "registered_at": registered_at,
        "commit": commit,
        "dataset_manifest_path": dataset_manifest_path,
        "tuning_manifest_path": tuning_manifest_path,
        "fitted_transforms_path": fitted_transforms_path,
        "calibrator_path": calibrator_path,
        "decision_policy_config": decision_policy_config,
        "evaluation_evidence": {
            "compare_policies_script": compare_script,
            "sensitivity_analysis_script": sensitivity_script,
            "decision_policy_doc": decision_doc,
            "test_auc_pr": round(test_auc_pr, 4),
            "validation_argmin_cost": VALIDATION_ARGMIN_COST,
            "validation_threshold_cost": VALIDATION_THRESHOLD_COST,
            "cost_reduction_pct": COST_REDUCTION_PCT,
            "bootstrap_ci": list(BOOTSTRAP_CI),
        },
        "promotion_requires": "mentor_approval",
        "promotion_owner": "Deepa",
    }
    return entry


def write_registry(registry: list[dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def print_summary(entry: dict, validated_paths: list[str]) -> None:
    print("=== Candidate registration successful ===")
    print(f"version              : {entry['version']}")
    print(f"status               : {entry['status']}")
    print(f"commit               : {entry['commit']}")
    print(f"registered_at        : {entry['registered_at']}")
    print(f"promotion_requires   : {entry['promotion_requires']}")
    print(f"promotion_owner      : {entry['promotion_owner']}")
    print(f"test_auc_pr          : {entry['evaluation_evidence']['test_auc_pr']}")
    print(
        "validation costs     : "
        f"argmin={entry['evaluation_evidence']['validation_argmin_cost']}  "
        f"threshold={entry['evaluation_evidence']['validation_threshold_cost']}  "
        f"reduction={entry['evaluation_evidence']['cost_reduction_pct']}%"
    )
    print("validated paths:")
    for rel in validated_paths:
        print(f"  OK  {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a Candidate model entry.")
    parser.add_argument("version", help="Model version, e.g. v9 or 9")
    args = parser.parse_args()

    version = parse_version_arg(args.version)
    if not REGISTRY_PATH.exists():
        fail(f"Registry not found: {REGISTRY_PATH}")

    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        fail("models/registry.json must be a JSON array")

    ensure_no_candidate(registry, version)
    legacy = find_legacy_entry(registry, version)

    commit = git_head()
    registered_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Snapshot existing entries for post-write integrity check.
    prior = json.dumps(registry, sort_keys=True)

    entry = build_candidate_entry(version, legacy, commit, registered_at)

    validated_paths = [
        entry["model_path"],
        entry["config_path"],
        entry["dataset_manifest_path"],
        entry["tuning_manifest_path"],
        entry["fitted_transforms_path"],
        entry["calibrator_path"],
        entry["decision_policy_config"],
        entry["evaluation_evidence"]["compare_policies_script"],
        entry["evaluation_evidence"]["sensitivity_analysis_script"],
        entry["evaluation_evidence"]["decision_policy_doc"],
    ]

    # Final commit check immediately before write.
    if git_head() != entry["commit"]:
        fail("HEAD changed during registration; refusing to write")

    registry.append(entry)
    # Ensure we did not mutate prior entries.
    if json.dumps(registry[:-1], sort_keys=True) != prior:
        fail("Internal error: existing registry entries would be modified")

    write_registry(registry)
    print_summary(entry, validated_paths)


if __name__ == "__main__":
    main()
