"""Load JSON configs from the configs/ directory."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.json"
    return json.loads(path.read_text())


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative
