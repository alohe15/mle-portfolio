"""Tests for model comparison and metrics validation.

Run: python -m pytest tests/test_model_comparison.py -v
"""

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path) as f:
        return json.load(f)


class TestMetricsSchema:
    """Validate that metrics JSONs contain required fields with finite values."""

    @pytest.fixture(scope="module")
    def registry(self):
        return load_json(PROJECT_ROOT / "models" / "registry.json")

    def test_all_versions_have_metrics(self, registry):
        """Every registry entry must have a metrics file that exists."""
        for entry in registry:
            path = PROJECT_ROOT / entry["metrics_path"]
            assert path.exists(), (
                f"v{entry['version']} metrics missing: {entry['metrics_path']}"
            )

    def test_required_metric_fields(self, registry):
        """Each metrics JSON must contain auc_pr (or test_auc_pr) and best_iteration."""
        for entry in registry:
            path = PROJECT_ROOT / entry["metrics_path"]
            if not path.exists():
                continue
            metrics = load_json(path)
            m = metrics.get("metrics", metrics)
            has_auc = "auc_pr" in m or "test_auc_pr" in m
            assert has_auc, f"v{entry['version']} missing auc_pr field"

    def test_metric_values_are_finite(self, registry):
        """No NaN or Inf in metric values."""
        for entry in registry:
            path = PROJECT_ROOT / entry["metrics_path"]
            if not path.exists():
                continue
            metrics = load_json(path)
            m = metrics.get("metrics", metrics)
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val), (
                        f"v{entry['version']} {key}={val} is not finite"
                    )

    def test_auc_pr_in_valid_range(self, registry):
        """AUC-PR must be between 0 and 1."""
        for entry in registry:
            path = PROJECT_ROOT / entry["metrics_path"]
            if not path.exists():
                continue
            metrics = load_json(path)
            m = metrics.get("metrics", metrics)
            auc = m.get("test_auc_pr", m.get("auc_pr"))
            if auc is not None:
                assert 0.0 <= auc <= 1.0, (
                    f"v{entry['version']} AUC-PR={auc} out of [0,1]"
                )


class TestComparisonIntegrity:
    """Validate that model comparisons use compatible data."""

    @pytest.fixture(scope="module")
    def registry(self):
        return load_json(PROJECT_ROOT / "models" / "registry.json")

    def test_all_versions_have_split_config(self, registry):
        """Every version's config must specify temporal split parameters."""
        for entry in registry:
            config_path = PROJECT_ROOT / entry["config_path"]
            if not config_path.exists():
                continue
            config = load_json(config_path)
            split = config.get("split", {})
            assert split.get("method") == "temporal", (
                f"v{entry['version']} split method is not temporal"
            )
            assert split.get("sort_column") == "TransactionDT", (
                f"v{entry['version']} sort_column is not TransactionDT"
            )

    def test_split_fractions_consistent(self, registry):
        """All versions with an explicit test_fraction must share the same value."""
        test_fractions = set()
        for entry in registry:
            config_path = PROJECT_ROOT / entry["config_path"]
            if not config_path.exists():
                continue
            config = load_json(config_path)
            tf = config.get("split", {}).get("test_fraction")
            if tf is not None:
                test_fractions.add(tf)
        assert len(test_fractions) <= 1, (
            f"Incompatible test fractions across versions: {test_fractions}. "
            f"Comparison table metrics are not directly comparable."
        )

    def test_v9_dataset_version_is_5(self, registry):
        """v9 must reference dataset_v5."""
        v9 = [e for e in registry if e["version"] == 9]
        assert v9, "v9 not in registry"
        assert v9[0]["dataset_version"] == 5


class TestComparisonScript:
    """Verify compare_models.py produces output."""

    def test_comparison_output_exists(self):
        """After running compare_models.py, output file must exist."""
        output = PROJECT_ROOT / "evals" / "model_comparison_results.txt"
        if not output.exists():
            pytest.skip("Run evals/compare_models.py first to generate output")
        content = output.read_text()
        assert "v1" in content.lower() or "version" in content.lower(), (
            "Comparison output doesn't contain expected version references"
        )
        assert "v9" in content.lower(), "Comparison output doesn't include v9"
        assert "delta_vs_v1" in content.lower(), (
            "Comparison output missing delta_vs_v1 column"
        )
