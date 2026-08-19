"""Feature schema validation for the locked serving model (v8).

Run: python -m pytest tests/test_feature_schema.py -v
"""
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def registry():
    return load_json(PROJECT_ROOT / "models" / "registry.json")

@pytest.fixture(scope="module")
def v8_entry(registry):
    entries = [e for e in registry if e["version"] == 8]
    assert len(entries) == 1, f"Expected one v8 entry, found {len(entries)}"
    return entries[0]

@pytest.fixture(scope="module")
def v7_entry(registry):
    entries = [e for e in registry if e["version"] == 7]
    assert len(entries) == 1, f"Expected one v7 entry, found {len(entries)}"
    return entries[0]

@pytest.fixture(scope="module")
def v8_manifest(v8_entry):
    return load_json(PROJECT_ROOT / v8_entry["manifest_path"])

@pytest.fixture(scope="module")
def v7_manifest(v7_entry):
    return load_json(PROJECT_ROOT / v7_entry["manifest_path"])

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
    return load_json(PROJECT_ROOT / "data" / "processed" / f"feature_list_v{ds_version}.json")

class TestFeatureNames:
    def test_no_duplicates_in_manifest(self, v8_manifest):
        features = v8_manifest["feature_order"]
        dupes = [f for f in set(features) if features.count(f) > 1]
        assert not dupes, f"Duplicates: {dupes}"

    def test_no_duplicates_in_feature_list(self, feature_list):
        dupes = [f for f in set(feature_list) if feature_list.count(f) > 1]
        assert not dupes, f"Duplicates: {dupes}"

    def test_no_target_in_features(self, v8_manifest, feature_list):
        assert "isFraud" not in v8_manifest["feature_order"]
        assert "isFraud" not in feature_list

    def test_no_id_columns_in_features(self, v8_manifest, feature_list):
        for col in ["TransactionID", "TransactionDT"]:
            assert col not in v8_manifest["feature_order"], f"{col} in manifest"
            assert col not in feature_list, f"{col} in feature_list"

    def test_manifest_matches_feature_list(self, v8_manifest, feature_list):
        assert set(v8_manifest["feature_order"]) == set(feature_list)

class TestFeatureOrder:
    def test_exact_order_match(self, v8_manifest, feature_list):
        assert v8_manifest["feature_order"] == feature_list

    def test_v8_order_matches_v7(self, v8_manifest, v7_manifest):
        """v8 is a hyperparameter-only change — feature order must be identical to v7."""
        assert v8_manifest["feature_order"] == v7_manifest["feature_order"]

class TestFeatureCounts:
    def test_manifest_n_features_matches_order(self, v8_manifest):
        assert v8_manifest["n_features"] == len(v8_manifest["feature_order"])

    def test_metrics_n_features_matches_manifest(self, v8_metrics, v8_manifest):
        n = v8_metrics.get("training", {}).get("n_features", v8_metrics.get("n_features"))
        assert n == v8_manifest["n_features"]

    def test_feature_list_count_matches_manifest(self, feature_list, v8_manifest):
        assert len(feature_list) == v8_manifest["n_features"]

    def test_v8_count_matches_v7(self, v8_manifest, v7_manifest):
        assert v8_manifest["n_features"] == v7_manifest["n_features"]

class TestFeatureDtypes:
    def test_all_features_have_dtypes(self, v8_manifest):
        if "feature_dtypes" in v8_manifest:
            missing = [f for f in v8_manifest["feature_order"] if f not in v8_manifest["feature_dtypes"]]
            assert not missing, f"Missing dtypes: {missing}"

class TestConfigConsistency:
    def test_dataset_version_matches(self, v8_config, v8_manifest):
        assert v8_config["dataset_version"] == v8_manifest["dataset_version"]

    def test_split_sums_to_one(self, v8_config):
        s = v8_config["split"]
        total = s["train_fraction"] + s["val_fraction"] + s["test_fraction"]
        assert abs(total - 1.0) < 1e-9

    def test_temporal_split(self, v8_config):
        assert v8_config["split"]["method"] == "temporal"

    def test_v8_only_differs_from_v7_by_ceiling(self, v8_config, v7_config):
        """Only n_estimators, early_stopping_rounds, version, description, parent_version may differ."""
        v8p = v8_config.get("lgbm_params", {})
        v7p = v7_config.get("lgbm_params", {})
        param_diffs = {k for k in set(v8p) | set(v7p) if v8p.get(k) != v7p.get(k)}
        allowed_param_diffs = {"n_estimators", "early_stopping_rounds"}
        unexpected = param_diffs - allowed_param_diffs
        assert not unexpected, f"Unexpected lgbm_params differences: {unexpected}"

class TestRegistryIntegrity:
    def test_exactly_one_v8(self, registry):
        assert len([e for e in registry if e["version"] == 8]) == 1

    def test_v8_has_required_paths(self, v8_entry):
        for key in ["model_path", "manifest_path", "metrics_path", "config_path", "dataset_config_path"]:
            assert key in v8_entry, f"Missing {key}"

    def test_committed_files_exist(self, v8_entry):
        for key in ["manifest_path", "metrics_path", "config_path", "dataset_config_path"]:
            assert (PROJECT_ROOT / v8_entry[key]).exists(), f"{key}: {v8_entry[key]} missing"

class TestConvergence:
    def test_natural_convergence(self, v8_metrics):
        """v8 must have converged naturally — best_iteration well below the 15000 ceiling."""
        bi = v8_metrics.get("training", {}).get("n_estimators_used",
             v8_metrics.get("metrics", {}).get("best_iteration",
             v8_metrics.get("best_iteration")))
        assert bi is not None, "Cannot find best_iteration in metrics"
        assert bi < 14000, f"best_iteration={bi}, ceiling may have been hit again"
