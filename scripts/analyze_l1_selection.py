"""Analyze L1 (reg_alpha) feature selection from a trained LightGBM model.

Usage:
    python scripts/analyze_l1_selection.py --model-version 3

Loads the model and manifest via models/registry.json, extracts gain/split
importance, identifies features L1 zeroed out (gain == 0), and writes
evals/l1_selection_v{N}.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
EVALS_DIR = REPO_ROOT / "evals"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        raise TypeError("models/registry.json must contain a JSON array")
    return registry


def get_registry_entry(registry: list[dict], version: int) -> dict:
    matches = [entry for entry in registry if entry["version"] == version]
    if not matches:
        raise ValueError(f"Version {version} not found in models/registry.json")
    if len(matches) > 1:
        raise ValueError(f"Multiple registry entries found for version {version}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze L1 feature selection from a trained LightGBM model."
    )
    parser.add_argument(
        "--model-version",
        type=int,
        required=True,
        help="Model version number in models/registry.json (e.g. 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.model_version

    entry = get_registry_entry(load_registry(), version)
    model_path = repo_path(entry["model_path"])
    manifest_path = repo_path(entry["manifest_path"])
    config_path = repo_path(entry["config_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    model_config = load_json(config_path)
    if not isinstance(model_config, dict):
        raise TypeError(f"Expected dict model config: {config_path}")
    reg_alpha = model_config.get("lgbm_params", {}).get("reg_alpha")
    if reg_alpha is None:
        raise ValueError(
            f"Model v{version} config has no lgbm_params.reg_alpha — "
            "this analysis is for L1-regularized models"
        )

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected dict manifest: {manifest_path}")
    feature_order = manifest["feature_order"]

    booster = lgb.Booster(model_file=str(model_path))
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    if len(gain) != len(feature_order):
        raise ValueError(
            f"Importance length ({len(gain)}) != feature_order length "
            f"({len(feature_order)})"
        )

    feature_importances: dict[str, dict[str, float | int]] = {}
    active_feature_names: list[str] = []
    zeroed_feature_names: list[str] = []

    paired = []
    for name, g, s in zip(feature_order, gain, split):
        g_val = float(g)
        s_val = int(s)
        feature_importances[name] = {"gain": g_val, "split_count": s_val}
        paired.append((name, g_val, s_val))
        if g_val == 0.0:
            zeroed_feature_names.append(name)
        else:
            active_feature_names.append(name)

    paired.sort(key=lambda x: x[1], reverse=True)

    report = {
        "model_version": version,
        "reg_alpha": float(reg_alpha),
        "total_features": len(feature_order),
        "active_features": len(active_feature_names),
        "zeroed_features": len(zeroed_feature_names),
        "active_feature_names": active_feature_names,
        "zeroed_feature_names": zeroed_feature_names,
        "feature_importances": feature_importances,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVALS_DIR / f"l1_selection_v{version}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"L1 Feature Selection Results (v{version}, reg_alpha={reg_alpha}):")
    print(
        f"  Active features (gain > 0):  "
        f"{len(active_feature_names)} / {len(feature_order)}"
    )
    print(
        f"  Zeroed features (gain = 0):  "
        f"{len(zeroed_feature_names)} / {len(feature_order)}"
    )
    print()
    print("Top 20 by gain:")
    for i, (name, g_val, s_val) in enumerate(paired[:20], start=1):
        print(f"  {i}. {name:<24} gain={g_val:.1f}  splits={s_val}")
    print()
    if zeroed_feature_names:
        preview = ", ".join(zeroed_feature_names[:40])
        if len(zeroed_feature_names) > 40:
            preview += f", ... (+{len(zeroed_feature_names) - 40} more)"
        print(f"Zeroed features:\n  {preview}")
    else:
        print("Zeroed features:\n  (none)")
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
