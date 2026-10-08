"""Guard: synthetic fixture models must never land in models/ or the registry."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def test_no_fixture_model_files_in_models_dir():
    leaked = sorted(p.name for p in MODELS_DIR.glob("fixture_model_*"))
    assert leaked == []


def test_serving_registry_model_is_not_a_fixture():
    registry = json.loads(REGISTRY_PATH.read_text())
    serving = [entry for entry in registry if entry.get("is_serving") is True]
    assert len(serving) == 1
    model_name = Path(serving[0]["model_path"]).name
    assert not model_name.startswith("fixture_")
