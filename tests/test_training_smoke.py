"""Training smoke test — verifies train.py runs end-to-end on a tiny data slice.

Uses throwaway version 9999. Cleans up all artifacts after the test.
Run: python -m pytest tests/test_training_smoke.py -v
"""
import json
import shutil
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestTrainingSmoke:
    SMOKE_VERSION = 9999
    SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "lgbm_v9999.json"

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a throwaway config before the test, clean up after."""
        # Load v8 config as base
        v8_config = json.loads(
            (PROJECT_ROOT / "configs" / "lgbm_v8.json").read_text()
        )

        # Create smoke config with minimal trees for speed
        smoke_config = v8_config.copy()
        smoke_config["version"] = self.SMOKE_VERSION
        smoke_config["description"] = "Smoke test — throwaway, do not keep"
        smoke_config["parent_version"] = 8
        smoke_config["lgbm_params"] = v8_config["lgbm_params"].copy()
        smoke_config["lgbm_params"]["n_estimators"] = 10
        smoke_config["lgbm_params"]["early_stopping_rounds"] = 5

        self.SMOKE_CONFIG_PATH.write_text(json.dumps(smoke_config, indent=2))

        yield  # test runs here

        # Cleanup: remove all smoke artifacts
        self.SMOKE_CONFIG_PATH.unlink(missing_ok=True)
        for f in (PROJECT_ROOT / "models").glob(f"lgbm_v{self.SMOKE_VERSION}_*"):
            f.unlink(missing_ok=True)

        # Remove smoke entry from registry if added
        registry_path = PROJECT_ROOT / "models" / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
            registry = [e for e in registry if e.get("version") != self.SMOKE_VERSION]
            registry_path.write_text(json.dumps(registry, indent=2))

    def test_train_completes(self):
        """train.py should complete without errors on the real dataset with minimal trees."""
        result = subprocess.run(
            ["python", "scripts/train.py", "--config", str(self.SMOKE_CONFIG_PATH)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )
        assert result.returncode == 0, (
            f"train.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_artifacts_created(self):
        """After training, model, metrics, and manifest must exist."""
        # Run training first
        subprocess.run(
            ["python", "scripts/train.py", "--config", str(self.SMOKE_CONFIG_PATH)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )
        model_files = list(
            (PROJECT_ROOT / "models").glob(f"lgbm_v{self.SMOKE_VERSION}_*.txt")
        )
        metrics_files = list(
            (PROJECT_ROOT / "models").glob(f"lgbm_v{self.SMOKE_VERSION}_*_metrics.json")
        )
        manifest_files = list(
            (PROJECT_ROOT / "models").glob(f"lgbm_v{self.SMOKE_VERSION}_*_manifest.json")
        )
        assert len(model_files) >= 1, "Model .txt not created"
        assert len(metrics_files) >= 1, "Metrics .json not created"
        assert len(manifest_files) >= 1, "Manifest .json not created"
