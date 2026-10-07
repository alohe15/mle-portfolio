"""Registry integrity tests for models/registry.json lifecycle entries."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"

KNOWN_STATUSES = {"Experimental", "Candidate", "Production", "Rolled_back"}
ISO8601_Z = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
COMMIT_HEX = re.compile(r"^[0-9a-f]{40}$")

PATH_FIELDS = (
    "model_path",
    "config_path",
    "dataset_manifest_path",
    "tuning_manifest_path",
    "fitted_transforms_path",
)


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    assert REGISTRY_PATH.exists(), f"Missing registry: {REGISTRY_PATH}"
    data = json.loads(REGISTRY_PATH.read_text())
    assert isinstance(data, list)
    return data


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).is_file()


def test_every_entry_has_version_and_status_or_is_serving(registry):
    for i, entry in enumerate(registry):
        assert "version" in entry, f"entry[{i}] missing version"
        has_status = "status" in entry
        has_serving = "is_serving" in entry
        assert has_status or has_serving, (
            f"entry[{i}] (version={entry.get('version')}) needs status or is_serving"
        )


def test_no_duplicate_version_status_pairs(registry):
    seen: set[tuple] = set()
    for entry in registry:
        version = entry["version"]
        status = entry.get("status")
        if status is None:
            # Legacy rows: distinguish serving vs non-serving experimental history.
            status = "Production" if entry.get("is_serving") else "Experimental"
        key = (version, status)
        assert key not in seen, f"Duplicate (version, status) pair: {key}"
        seen.add(key)


def test_path_fields_resolve_when_present(registry):
    missing = []
    for entry in registry:
        for field in PATH_FIELDS:
            if field not in entry:
                continue
            rel = entry[field]
            if not _exists(rel):
                missing.append((entry.get("version"), entry.get("status"), field, rel))
    assert not missing, f"Missing artifact files: {missing}"


def test_calibrator_path_exists_when_present(registry):
    for entry in registry:
        if "calibrator_path" not in entry:
            continue
        rel = entry["calibrator_path"]
        assert _exists(rel), f"calibrator_path missing: {rel}"


def test_decision_policy_config_exists_when_present(registry):
    for entry in registry:
        if "decision_policy_config" not in entry:
            continue
        rel = entry["decision_policy_config"]
        assert _exists(rel), f"decision_policy_config missing: {rel}"


def test_evaluation_evidence_script_paths_exist(registry):
    for entry in registry:
        evidence = entry.get("evaluation_evidence")
        if not evidence:
            continue
        for key, value in evidence.items():
            if not isinstance(value, str):
                continue
            if "/" not in value:
                continue
            # Only treat path-like strings as required files.
            if value.endswith((".py", ".md", ".json", ".txt", ".pkl", ".png")):
                assert _exists(value), (
                    f"evaluation_evidence.{key} path missing: {value}"
                )


def test_candidate_commit_is_40_char_hex(registry):
    candidates = [e for e in registry if e.get("status") == "Candidate"]
    assert candidates, "Expected at least one Candidate entry"
    for entry in candidates:
        commit = entry.get("commit", "")
        assert COMMIT_HEX.match(commit), f"Invalid commit hash: {commit!r}"


def test_candidate_registered_at_is_iso8601(registry):
    candidates = [e for e in registry if e.get("status") == "Candidate"]
    assert candidates, "Expected at least one Candidate entry"
    for entry in candidates:
        ts = entry.get("registered_at", "")
        assert ISO8601_Z.match(ts), f"Invalid registered_at: {ts!r}"
        # Also parseable by datetime.
        normalized = ts.replace("Z", "+00:00")
        datetime.fromisoformat(normalized)


def test_exactly_one_is_serving_true(registry):
    serving = [e for e in registry if e.get("is_serving") is True]
    assert len(serving) == 1, f"Expected exactly one is_serving=true, found {len(serving)}"


def test_candidate_version_matches_serving_version(registry):
    serving = [e for e in registry if e.get("is_serving") is True]
    assert len(serving) == 1
    candidates = [e for e in registry if e.get("status") == "Candidate"]
    assert candidates, "Expected at least one Candidate entry"
    serving_version = serving[0]["version"]
    for entry in candidates:
        assert entry["version"] == serving_version, (
            f"Candidate version {entry['version']} != serving version {serving_version}"
        )


def test_no_unknown_status_values(registry):
    for entry in registry:
        if "status" in entry:
            assert entry["status"] in KNOWN_STATUSES, (
                f"Unknown status {entry['status']!r} on version={entry.get('version')}"
            )
        else:
            # Legacy is_serving pattern.
            assert "is_serving" in entry
            assert isinstance(entry["is_serving"], bool)


def test_candidate_does_not_set_is_serving_true(registry):
    for entry in registry:
        if entry.get("status") == "Candidate":
            assert entry.get("is_serving") is False


def test_legacy_entries_untouched_count(registry):
    """Baseline: nine legacy training rows remain (versions 1–9 without Candidate status)."""
    legacy = [e for e in registry if e.get("status") != "Candidate"]
    assert len(legacy) == 9
    assert {e["version"] for e in legacy} == set(range(1, 10))
