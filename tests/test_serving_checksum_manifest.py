"""Guards for the serving-only checksum manifest."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_MANIFEST = REPO_ROOT / "artifacts.serving.sha256"
FULL_MANIFEST = REPO_ROOT / "artifacts.sha256"

GITIGNORED_MODEL_ARTIFACTS = {
    "models/lgbm_v9_optuna_tuning.txt",
    "models/lgbm_v9_optuna_tuning_fitted_transforms.pkl",
    "models/lgbm_v9_calibrator.pkl",
}


def _parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        digest, rel = raw.split(maxsplit=1)
        entries[rel.strip()] = digest
    return entries


def test_serving_checksum_manifest_has_expected_runtime_files():
    entries = _parse_manifest(SERVING_MANIFEST)
    expected = {
        "models/registry.json",
        "models/lgbm_v9_optuna_tuning_manifest.json",
        "models/lgbm_v9_optuna_tuning.txt",
        "models/lgbm_v9_optuna_tuning_fitted_transforms.pkl",
        "models/lgbm_v9_calibrator.pkl",
        "configs/lgbm_v9.json",
        "configs/dataset_v1.json",
        "configs/dataset_v2.json",
        "configs/dataset_v3.json",
        "configs/dataset_v5.json",
        "configs/decision_policy_v1.json",
    }
    assert set(entries) == expected


def test_committed_serving_checksum_paths_exist_in_fresh_clone():
    entries = _parse_manifest(SERVING_MANIFEST)
    missing = [
        rel
        for rel in entries
        if rel not in GITIGNORED_MODEL_ARTIFACTS and not (REPO_ROOT / rel).is_file()
    ]
    assert missing == []


def test_serving_model_artifact_digests_match_full_handoff_manifest():
    serving = _parse_manifest(SERVING_MANIFEST)
    full = _parse_manifest(FULL_MANIFEST)
    for rel in GITIGNORED_MODEL_ARTIFACTS:
        assert serving[rel] == full[rel]
