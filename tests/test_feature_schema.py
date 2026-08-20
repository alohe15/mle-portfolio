"""Feature schema validation tests for the locked model candidate (lgbm_v9).

Tests the contract between dataset_v5, lgbm_v9 config/manifest, and feature_list_v5.
Run: python -m pytest tests/test_feature_schema.py -v
"""
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)

# --- Fixtures ---

@pytest.fixture(scope="module")
def registry():
    return load_json(PROJECT_ROOT / "models" / "registry.json")

@pytest.fixture(scope="module")
def v9_entry(registry):
    entries = [e for e in registry if e["version"] == 9]
    assert len(entries) == 1, f"Expected exactly one v9 entry, found {len(entries)}"
    return entries[0]

@pytest.fixture(scope="module")
def v8_entry(registry):
    entries = [e for e in registry if e["version"] == 8]
    assert len(entries) == 1, f"Expected exactly one v8 entry, found {len(entries)}"
    return entries[0]

@pytest.fixture(scope="module")
def v9_manifest(v9_entry):
    return load_json(PROJECT_ROOT / v9_entry["manifest_path"])

@pytest.fixture(scope="module")
def v8_manifest(v8_entry):
    return load_json(PROJECT_ROOT / v8_entry["manifest_path"])

@pytest.fixture(scope="module")
def v9_metrics(v9_entry):
    return load_json(PROJECT_ROOT / v9_entry["metrics_path"])

@pytest.fixture(scope="module")
def v9_config():
    return load_json(PROJECT_ROOT / "configs" / "lgbm_v9.json")

@pytest.fixture(scope="module")
def v8_config():
    return load_json(PROJECT_ROOT / "configs" / "lgbm_v8.json")

@pytest.fixture(scope="module")
def dataset_config(v9_config):
    return load_json(PROJECT_ROOT / v9_config["dataset_config_path"])

@pytest.fixture(scope="module")
def feature_list(dataset_config):
    ds_version = dataset_config["dataset_version"]
    return load_json(
        PROJECT_ROOT / "data" / "processed" / f"feature_list_v{ds_version}.json"
    )

# --- Feature name and order tests ---

class TestFeatureNames:
    def test_feature_list_matches_manifest(self, v9_manifest, feature_list):
        """Feature list and manifest must contain the same features."""
        manifest_features = v9_manifest["feature_order"]
        assert set(manifest_features) == set(feature_list), (
            f"In manifest not feature_list: {set(manifest_features) - set(feature_list)}, "
            f"In feature_list not manifest: {set(feature_list) - set(manifest_features)}"
        )

    def test_no_duplicate_features_in_manifest(self, v9_manifest):
        """No duplicates in the manifest's feature_order."""
        features = v9_manifest["feature_order"]
        assert len(features) == len(set(features)), (
            f"Duplicates: {[f for f in features if features.count(f) > 1]}"
        )

    def test_no_duplicate_features_in_feature_list(self, feature_list):
        """No duplicates in the feature list."""
        assert len(feature_list) == len(set(feature_list)), (
            f"Duplicates: {[f for f in feature_list if feature_list.count(f) > 1]}"
        )

    def test_no_target_in_features(self, v9_manifest):
        """isFraud must not appear in the feature order."""
        assert "isFraud" not in v9_manifest["feature_order"]

    def test_no_id_columns_in_features(self, v9_manifest):
        """TransactionID and TransactionDT must not appear in feature order."""
        assert "TransactionID" not in v9_manifest["feature_order"]
        assert "TransactionDT" not in v9_manifest["feature_order"]

    def test_no_raw_d_columns_in_features(self, v9_manifest):
        """Raw D1-D15 must not appear (Dn-only is the winning representation)."""
        raw_d_cols = [f"D{i}" for i in range(1, 16)]
        present = [c for c in raw_d_cols if c in v9_manifest["feature_order"]]
        assert len(present) == 0, f"Raw D columns found in features: {present}"

class TestFeatureCounts:
    def test_manifest_count_matches_feature_order_length(self, v9_manifest):
        """n_features must match the actual length of feature_order."""
        assert v9_manifest["n_features"] == len(v9_manifest["feature_order"])

    def test_metrics_count_matches_manifest(self, v9_metrics, v9_manifest):
        """Training n_features must match manifest."""
        n_train = v9_metrics.get("training", {}).get("n_features")
        if n_train is not None:
            assert n_train == v9_manifest["n_features"]

    def test_feature_list_count_matches_manifest(self, v9_manifest, feature_list):
        """Feature list length must match manifest n_features."""
        assert len(feature_list) == v9_manifest["n_features"]

    def test_feature_count_decreased(self, v9_manifest, v8_manifest):
        """v9 n_features should be v8 n_features minus 15 (dropped raw D columns)."""
        assert v9_manifest["n_features"] == v8_manifest["n_features"] - 15

class TestFeatureOrder:
    def test_manifest_order_matches_feature_list_order(self, v9_manifest, feature_list):
        """Feature order must be identical (not just same set)."""
        assert v9_manifest["feature_order"] == feature_list

    def test_v9_order_is_v8_minus_raw_d(self, v9_manifest, v8_manifest):
        """v9 feature order equals v8 with raw D1–D15 removed (relative order preserved)."""
        raw_d = {f"D{i}" for i in range(1, 16)}
        expected = [c for c in v8_manifest["feature_order"] if c not in raw_d]
        assert v9_manifest["feature_order"] == expected

class TestFeatureDtypes:
    def test_all_features_have_dtypes(self, v9_manifest):
        """Every feature in feature_order must have a dtype in feature_dtypes."""
        if "feature_dtypes" in v9_manifest:
            for feat in v9_manifest["feature_order"]:
                assert feat in v9_manifest["feature_dtypes"], (
                    f"Feature '{feat}' missing from feature_dtypes"
                )

class TestConfigConsistency:
    def test_dataset_version_matches(self, v9_config, v9_manifest, dataset_config):
        """Config, manifest, and dataset config must agree on dataset version."""
        assert v9_config["dataset_version"] == v9_manifest["dataset_version"]
        assert v9_config["dataset_version"] == dataset_config["dataset_version"]

    def test_v9_params_match_v8(self, v9_config, v8_config):
        """v9 lgbm_params match v8 aside from post-training seed for reproducibility."""
        v9_params = dict(v9_config["lgbm_params"])
        v8_params = dict(v8_config["lgbm_params"])
        # seed was added post-training; current v9 binary was trained without it
        assert v9_params.pop("seed", None) == 42
        v8_params.pop("seed", None)
        assert v9_params == v8_params

    def test_split_is_temporal_70_15_15(self, v9_config):
        """Split must be temporal 70/15/15."""
        split = v9_config["split"]
        assert split["method"] == "temporal"
        assert split["train_fraction"] == 0.70
        assert split["val_fraction"] == 0.15
        assert split["test_fraction"] == 0.15
        assert split["sort_column"] == "TransactionDT"

class TestRegistryIntegrity:
    def test_v9_in_registry(self, registry):
        """v9 must exist in the registry."""
        versions = [e["version"] for e in registry]
        assert 9 in versions

    def test_at_most_one_serving(self, registry):
        """At most one entry may have is_serving: true."""
        serving = [e for e in registry if e.get("is_serving")]
        assert len(serving) <= 1, f"Multiple serving entries: {[e['version'] for e in serving]}"

    def test_v9_has_all_required_paths(self, v9_entry):
        """v9 registry entry must have all required path fields."""
        required = ["model_path", "manifest_path", "metrics_path", "config_path"]
        for field in required:
            assert field in v9_entry, f"Missing '{field}' in v9 registry entry"
            assert v9_entry[field], f"Empty '{field}' in v9 registry entry"

    def test_v9_is_serving(self, v9_entry, registry):
        """v9 is the locked serving model after Phase 1."""
        assert v9_entry.get("is_serving") is True
        serving = [e["version"] for e in registry if e.get("is_serving")]
        assert serving == [9]

class TestConvergence:
    def test_natural_convergence(self, v9_metrics):
        """best_iteration must be < 14000 (natural convergence, not ceiling hit)."""
        best_iter = v9_metrics.get("metrics", {}).get("best_iteration")
        if best_iter is None:
            best_iter = v9_metrics.get("best_iteration")
        assert best_iter is not None, "best_iteration not found in metrics"
        assert best_iter < 14000, (
            f"best_iteration={best_iter} >= 14000 — model may have hit the ceiling again"
        )
