"""Training pipeline smoke test.

Verifies train.py runs on a 10-tree config and produces all required artifacts.
Run: python -m pytest tests/test_training_smoke.py -v
"""
import json
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_VERSION = 9999

@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("smoke")
    v8_config = json.loads((PROJECT_ROOT / "configs" / "lgbm_v8.json").read_text())

    smoke = {**v8_config, "version": SMOKE_VERSION, "description": "Smoke test"}
    smoke["lgbm_params"] = {**v8_config["lgbm_params"], "n_estimators": 10, "early_stopping_rounds": 5, "verbose": -1}

    config_path = tmp_dir / "smoke.json"
    config_path.write_text(json.dumps(smoke, indent=2))

    result = subprocess.run(
        ["python", "scripts/train.py", "--config", str(config_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
    )
    return {"rc": result.returncode, "out": result.stdout, "err": result.stderr}

class TestSmoke:
    def test_exits_zero(self, smoke_run):
        assert smoke_run["rc"] == 0, f"Failed:\n{smoke_run['out']}\n{smoke_run['err']}"

    def test_model_file(self, smoke_run):
        assert list(PROJECT_ROOT.glob(f"models/lgbm_v{SMOKE_VERSION}_*.txt"))

    def test_metrics_json(self, smoke_run):
        files = list(PROJECT_ROOT.glob(f"models/lgbm_v{SMOKE_VERSION}_*_metrics.json"))
        assert files
        m = json.loads(files[0].read_text())
        assert "metrics" in m or "auc_pr" in m

    def test_manifest_json(self, smoke_run):
        files = list(PROJECT_ROOT.glob(f"models/lgbm_v{SMOKE_VERSION}_*_manifest.json"))
        assert files
        assert "feature_order" in json.loads(files[0].read_text())

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    for f in PROJECT_ROOT.glob(f"models/lgbm_v{SMOKE_VERSION}_*"):
        f.unlink(missing_ok=True)
    reg_path = PROJECT_ROOT / "models" / "registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        reg = [e for e in reg if e.get("version") != SMOKE_VERSION]
        reg_path.write_text(json.dumps(reg, indent=2))
