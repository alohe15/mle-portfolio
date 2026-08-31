"""Registry integrity tests.

Validates that registry entries have unique versions, required fields,
valid references to metadata files, and consistent lifecycle states.

Run: python -m pytest tests/test_registry.py -v
"""
import json
import math
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_registry():
    return json.load(open(PROJECT_ROOT / 'models' / 'registry.json'))

class TestRegistryStructure:
    """Basic structural integrity of registry.json."""

    def test_registry_is_list(self):
        reg = load_registry()
        assert isinstance(reg, list)

    def test_registry_not_empty(self):
        reg = load_registry()
        assert len(reg) > 0

    def test_unique_versions(self):
        reg = load_registry()
        versions = [e['version'] for e in reg]
        assert len(versions) == len(set(versions)), (
            f"Duplicate versions: {[v for v in versions if versions.count(v) > 1]}"
        )

    def test_versions_are_positive_integers(self):
        reg = load_registry()
        for e in reg:
            assert isinstance(e['version'], int) and e['version'] > 0

class TestRequiredFields:
    """Every entry must have the minimum required fields."""

    REQUIRED = ['version', 'dataset_version', 'model_path',
                'manifest_path', 'metrics_path', 'config_path']

    def test_all_entries_have_required_fields(self):
        reg = load_registry()
        for entry in reg:
            for field in self.REQUIRED:
                assert field in entry, (
                    f"v{entry['version']} missing required field: {field}"
                )

    def test_required_fields_not_empty(self):
        reg = load_registry()
        for entry in reg:
            for field in self.REQUIRED:
                val = entry.get(field)
                assert val is not None and val != '', (
                    f"v{entry['version']} has empty required field: {field}"
                )

class TestPathValidity:
    """Committed metadata files must exist on disk."""

    def test_config_paths_exist(self):
        reg = load_registry()
        for entry in reg:
            path = PROJECT_ROOT / entry['config_path']
            assert path.exists(), (
                f"v{entry['version']} config not found: {entry['config_path']}"
            )

    def test_metrics_paths_exist(self):
        reg = load_registry()
        for entry in reg:
            path = PROJECT_ROOT / entry['metrics_path']
            assert path.exists(), (
                f"v{entry['version']} metrics not found: {entry['metrics_path']}"
            )

    def test_manifest_paths_exist(self):
        reg = load_registry()
        for entry in reg:
            path = PROJECT_ROOT / entry['manifest_path']
            assert path.exists(), (
                f"v{entry['version']} manifest not found: {entry['manifest_path']}"
            )

    def test_dataset_config_paths_exist(self):
        reg = load_registry()
        for entry in reg:
            if 'dataset_config_path' in entry and entry['dataset_config_path']:
                path = PROJECT_ROOT / entry['dataset_config_path']
                assert path.exists(), (
                    f"v{entry['version']} dataset config not found: {entry['dataset_config_path']}"
                )

class TestServingConstraints:
    """At most one model may be serving."""

    def test_at_most_one_serving(self):
        reg = load_registry()
        serving = [e for e in reg if e.get('is_serving')]
        assert len(serving) <= 1, (
            f"Multiple serving entries: v{[e['version'] for e in serving]}"
        )

    def test_serving_model_has_metrics(self):
        reg = load_registry()
        serving = [e for e in reg if e.get('is_serving')]
        if serving:
            path = PROJECT_ROOT / serving[0]['metrics_path']
            assert path.exists()

class TestCandidateEntry:
    """Candidate entries must have extended traceability fields."""

    CANDIDATE_REQUIRED = ['git_commit', 'lifecycle_stage', 'registered_at',
                          'evaluation_evidence', 'test_metrics', 'promotion']

    def test_candidate_has_traceability(self):
        reg = load_registry()
        candidates = [e for e in reg if e.get('lifecycle_stage') == 'Candidate']
        for entry in candidates:
            for field in self.CANDIDATE_REQUIRED:
                assert field in entry, (
                    f"Candidate v{entry['version']} missing: {field}"
                )

    def test_candidate_git_commit_is_full_sha(self):
        reg = load_registry()
        candidates = [e for e in reg if e.get('lifecycle_stage') == 'Candidate']
        for entry in candidates:
            sha = entry.get('git_commit', '')
            assert len(sha) == 40, (
                f"Candidate v{entry['version']} git_commit is not a full SHA: {sha}"
            )

    def test_candidate_evidence_paths_exist(self):
        reg = load_registry()
        candidates = [e for e in reg if e.get('lifecycle_stage') == 'Candidate']
        for entry in candidates:
            evidence = entry.get('evaluation_evidence', {})
            for key, path in evidence.items():
                assert (PROJECT_ROOT / path).exists(), (
                    f"Candidate v{entry['version']} evidence missing: {key} → {path}"
                )

    def test_candidate_test_metrics_are_finite(self):
        reg = load_registry()
        candidates = [e for e in reg if e.get('lifecycle_stage') == 'Candidate']
        for entry in candidates:
            tm = entry.get('test_metrics', {})
            for key, val in tm.items():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val), (
                        f"Candidate v{entry['version']} {key}={val} is not finite"
                    )

    def test_candidate_promotion_checklist_exists(self):
        reg = load_registry()
        candidates = [e for e in reg if e.get('lifecycle_stage') == 'Candidate']
        for entry in candidates:
            checklist = entry.get('promotion', {}).get('promotion_checklist')
            assert checklist, f"Candidate v{entry['version']} has no promotion_checklist"
            assert (PROJECT_ROOT / checklist).exists(), (
                f"Candidate v{entry['version']} checklist not found: {checklist}"
            )

class TestVersionConsistency:
    """Registry version must match config and manifest versions."""

    def test_config_version_matches(self):
        reg = load_registry()
        for entry in reg:
            config_path = PROJECT_ROOT / entry['config_path']
            if config_path.exists():
                config = json.load(open(config_path))
                assert config.get('version') == entry['version'], (
                    f"Registry v{entry['version']} but config says v{config.get('version')}"
                )

    def test_dataset_version_matches(self):
        reg = load_registry()
        for entry in reg:
            config_path = PROJECT_ROOT / entry['config_path']
            if config_path.exists():
                config = json.load(open(config_path))
                assert config.get('dataset_version') == entry['dataset_version'], (
                    f"Registry dataset_v{entry['dataset_version']} but config says "
                    f"dataset_v{config.get('dataset_version')}"
                )
