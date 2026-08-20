"""Feature schema validation tests for the locked model candidate (lgbm_v8).

Tests the contract between dataset_v4, lgbm_v8 config/manifest, and feature_list_v4.
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
def v8_entry(registry):
    entries = [e for e in registry if e["version"] == 8]
    assert len(entries) == 1, f"Expected exactly one v8 entry, found {len(entries)}"
    return entries[0]

@pytest.fixture(scope="module")
def v8_manifest(v8_entry):
    return load_json(PROJECT_ROOT / v8_entry["manifest_path"])

@pytest.fixture(scope="module")
def v8_metrics(v8_entry):
    return load_json(PROJECT_ROOT / v8_entry["metrics_path"])

@pytest.fixture(scope="module")
def v8_config():
    return load_json(PROJECT_ROOT / "configs" / "lgbm_v8.json")

@pytest.fixture(scope="module")
def v7_config():
    return load_json(PROJECT_ROOT / "configs" / "lgbm_v7.json")

@pytest.fixture(scope="module")
def dataset_config(v8_config):
    return load_json(PROJECT_ROOT / v8_config["dataset_config_path"])

@pytest.fixture(scope="module")
def feature_list(dataset_config):
    ds_version = dataset_config["dataset_version"]
    return load_json(
        PROJECT_ROOT / "data" / "processed" / f"feature_list_v{ds_version}.json"
    )

# --- Feature name and order tests ---

class TestFeatureNames:
    def test_feature_list_matches_manifest(self, v8_manifest, feature_list):
        """Feature list and manifest must contain the same features."""
        manifest_features = v8_manifest["feature_order"]
        assert set(manifest_features) == set(feature_list), (
            f"In manifest not feature_list: {set(manifest_features) - set(feature_list)}, "
            f"In feature_list not manifest: {set(feature_list) - set(manifest_features)}"
        )

    def test_no_duplicate_features_in_manifest(self, v8_manifest):
        """No duplicates in the manifest's feature_order."""
        features = v8_manifest["feature_order"]
        assert len(features) == len(set(features)), (
            f"Duplicates: {[f for f in features if features.count(f) > 1]}"
        )

    def test_no_duplicate_features_in_feature_list(self, feature_list):
        """No duplicates in the feature list."""
        assert len(feature_list) == len(set(feature_list)), (
            f"Duplicates: {[f for f in feature_list if feature_list.count(f) > 1]}"
        )

    def test_no_target_in_features(self, v8_manifest):
        """isFraud must not appear in the feature order."""
        assert "isFraud" not in v8_manifest["feature_order"]

    def test_no_id_columns_in_features(self, v8_manifest):
        """TransactionID and TransactionDT must not appear in feature order."""
        assert "TransactionID" not in v8_manifest["feature_order"]
        assert "TransactionDT" not in v8_manifest["feature_order"]

    def test_no_raw_d_columns_in_features(self, v8_manifest):
        """Raw D1-D15 must not appear (Dn-only is the winning representation)."""
        raw_d_cols = [f"D{i}" for i in range(1, 16)]
        present = [c for c in raw_d_cols if c in v8_manifest["feature_order"]]
        assert len(present) == 0, f"Raw D columns found in features: {present}"

class TestFeatureCounts:
    def test_manifest_count_matches_feature_order_length(self, v8_manifest):
        """n_features must match the actual length of feature_order."""
        assert v8_manifest["n_features"] == len(v8_manifest["feature_order"])

    def test_metrics_count_matches_manifest(self, v8_metrics, v8_manifest):
        """Training n_features must match manifest."""
        n_train = v8_metrics.get("training", {}).get("n_features")
        if n_train is not None:
            assert n_train == v8_manifest["n_features"]

    def test_feature_list_count_matches_manifest(self, v8_manifest, feature_list):
        """Feature list length must match manifest n_features."""
        assert len(feature_list) == v8_manifest["n_features"]

class TestFeatureOrder:
    def test_manifest_order_matches_feature_list_order(self, v8_manifest, feature_list):
        """Feature order must be identical (not just same set)."""
        assert v8_manifest["feature_order"] == feature_list

    def test_v8_order_matches_v7_order(self, v8_manifest):
        """v8 and v7 must have identical feature order (same dataset)."""
        v7_entries = [
            e for e in load_json(PROJECT_ROOT / "models" / "registry.json")
            if e["version"] == 7
        ]
        if v7_entries:
            v7_manifest = load_json(PROJECT_ROOT / v7_entries[0]["manifest_path"])
            assert v8_manifest["feature_order"] == v7_manifest["feature_order"]

class TestFeatureDtypes:
    def test_all_features_have_dtypes(self, v8_manifest):
        """Every feature in feature_order must have a dtype in feature_dtypes."""
        if "feature_dtypes" in v8_manifest:
            for feat in v8_manifest["feature_order"]:
                assert feat in v8_manifest["feature_dtypes"], (
                    f"Feature '{feat}' missing from feature_dtypes"
                )

class TestConfigConsistency:
    def test_dataset_version_matches(self, v8_config, v8_manifest, dataset_config):
        """Config, manifest, and dataset config must agree on dataset version."""
        assert v8_config["dataset_version"] == v8_manifest["dataset_version"]
        assert v8_config["dataset_version"] == dataset_config["dataset_version"]

    def test_v8_params_match_v7_except_ceiling(self, v8_config, v7_config):
        """v8 lgbm_params must match v7 except n_estimators and early_stopping_rounds."""
        v8_params = v8_config["lgbm_params"].copy()
        v7_params = v7_config["lgbm_params"].copy()
        # Remove the two fields that should differ
        for key in ["n_estimators", "early_stopping_rounds"]:
            v8_params.pop(key, None)
            v7_params.pop(key, None)
        assert v8_params == v7_params, (
            f"Params differ beyond ceiling fields: {set(v8_params.items()) ^ set(v7_params.items())}"
        )

    def test_split_is_temporal_70_15_15(self, v8_config):
        """Split must be temporal 70/15/15."""
        split = v8_config["split"]
        assert split["method"] == "temporal"
        assert split["train_fraction"] == 0.70
        assert split["val_fraction"] == 0.15
        assert split["test_fraction"] == 0.15
        assert split["sort_column"] == "TransactionDT"

class TestRegistryIntegrity:
    def test_v8_in_registry(self, registry):
        """v8 must exist in the registry."""
        versions = [e["version"] for e in registry]
        assert 8 in versions

    def test_at_most_one_serving(self, registry):
        """At most one entry may have is_serving: true."""
        serving = [e for e in registry if e.get("is_serving")]
        assert len(serving) <= 1, f"Multiple serving entries: {[e['version'] for e in serving]}"

    def test_v8_has_all_required_paths(self, v8_entry):
        """v8 registry entry must have all required path fields."""
        required = ["model_path", "manifest_path", "metrics_path", "config_path"]
        for field in required:
            assert field in v8_entry, f"Missing '{field}' in v8 registry entry"
            assert v8_entry[field], f"Empty '{field}' in v8 registry entry"

class TestConvergence:
    def test_natural_convergence(self, v8_metrics):
        """best_iteration must be < 14000 (natural convergence, not ceiling hit)."""
        best_iter = v8_metrics.get("metrics", {}).get("best_iteration")
        if best_iter is None:
            best_iter = v8_metrics.get("best_iteration")
        assert best_iter is not None, "best_iteration not found in metrics"
        assert best_iter < 14000, (
            f"best_iteration={best_iter} >= 14000 — model may have hit the ceiling again"
        )
