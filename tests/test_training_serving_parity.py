"""Training-serving parity test — proves the API produces the same scores as training.

Takes 5 fixed rows from the validation split, runs them through both:
    Path A: the training pipeline (dataset → features → model.predict)
    Path B: the serving pipeline (raw fields → API → model.predict)

Scores must match within 1e-6 tolerance.

Run: python -m pytest tests/test_training_serving_parity.py -v -s
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from services.api.model_loader import load_bundle
from services.api.predictor import (
    apply_transformation,
    build_model_input,
    collect_dataset_transformations,
)

PARITY_INDICES = [0, 100, 500, 1000, 5000]
TOLERANCE = 1e-6


class TestTrainingServingParity:
    @pytest.fixture(scope="class")
    def training_artifacts(self):
        reg = json.load(open(PROJECT_ROOT / "models" / "registry.json"))
        v9 = [e for e in reg if e["version"] == 9][0]

        model = lgb.Booster(model_file=str(PROJECT_ROOT / v9["model_path"]))
        manifest = json.load(open(PROJECT_ROOT / v9["manifest_path"]))
        config = json.load(open(PROJECT_ROOT / v9["config_path"]))
        ds_config = json.load(open(PROJECT_ROOT / config["dataset_config_path"]))

        df = pd.read_parquet(PROJECT_ROOT / ds_config["output_path"])
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        n = len(df)
        train_end = int(n * config["split"]["train_fraction"])
        val_end = train_end + int(n * config["split"]["val_fraction"])
        val = df.iloc[train_end:val_end].reset_index(drop=True)

        fitted_path = PROJECT_ROOT / manifest["fitted_transforms_path"]
        fitted_transforms = pickle.loads(fitted_path.read_bytes())

        return {
            "model": model,
            "manifest": manifest,
            "config": config,
            "ds_config": ds_config,
            "val": val,
            "fitted_transforms": fitted_transforms,
            "bundle": load_bundle(),
            "v9": v9,
            "training_scores": [],
            "serving_scores": [],
        }

    @pytest.fixture(scope="class")
    def api_client(self):
        from services.api.app import app

        with TestClient(app) as client:
            yield client

    @pytest.fixture(scope="class")
    def raw_data(self):
        raw_path = PROJECT_ROOT / "data" / "raw" / "train_merged.parquet"
        if not raw_path.exists():
            raw_candidates = glob.glob(str(PROJECT_ROOT / "data" / "raw" / "*.parquet"))
            if raw_candidates:
                raw_path = Path(raw_candidates[0])
            else:
                pytest.skip("Raw data not available for parity test")

        df = pd.read_parquet(raw_path)
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        return df

    def _score_training_row(self, row_df: pd.DataFrame, artifacts: dict) -> float:
        ds_config = artifacts["ds_config"]
        fitted_transforms = artifacts["fitted_transforms"]
        bundle = artifacts["bundle"]
        model = artifacts["model"]

        for transform in collect_dataset_transformations(ds_config):
            if transform.get("requires_fit"):
                row_df = apply_transformation(row_df, transform, fitted_transforms)

        model_input = build_model_input(row_df, bundle)
        return float(model.predict(model_input)[0])

    def test_parity_training_path(self, training_artifacts):
        val = training_artifacts["val"]
        scores = []
        for idx in PARITY_INDICES:
            if idx >= len(val):
                continue
            row_df = val.iloc[[idx]].copy()
            scores.append(self._score_training_row(row_df, training_artifacts))

        training_artifacts["training_scores"] = scores

    def test_parity_serving_path(self, training_artifacts, api_client, raw_data):
        config = training_artifacts["config"]
        raw_data_sorted = raw_data.sort_values("TransactionDT").reset_index(drop=True)
        n = len(raw_data_sorted)
        train_end = int(n * config["split"]["train_fraction"])
        val_end = train_end + int(n * config["split"]["val_fraction"])
        raw_val = raw_data_sorted.iloc[train_end:val_end].reset_index(drop=True)

        serving_scores = []
        for idx in PARITY_INDICES:
            if idx >= len(raw_val):
                continue
            row = raw_val.iloc[idx]

            request = {}
            for col in row.index:
                val = row[col]
                if col in ("isFraud", "TransactionID"):
                    continue
                if pd.isna(val):
                    continue
                if isinstance(val, (np.integer,)):
                    request[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    request[col] = float(val)
                else:
                    request[col] = str(val)

            resp = api_client.post("/predict", json=request)
            assert resp.status_code == 200, f"Row {idx} failed: {resp.json()}"
            serving_scores.append(resp.json()["raw_score"])

        training_artifacts["serving_scores"] = serving_scores

    def test_scores_match(self, training_artifacts):
        training_scores = training_artifacts.get("training_scores", [])
        serving_scores = training_artifacts.get("serving_scores", [])

        assert len(training_scores) == len(serving_scores), (
            f"Score count mismatch: training={len(training_scores)}, "
            f"serving={len(serving_scores)}"
        )

        print(
            f"\n  Training-serving parity ({len(training_scores)} rows, "
            f"tolerance={TOLERANCE}):"
        )
        for i, (t_score, s_score) in enumerate(zip(training_scores, serving_scores)):
            delta = abs(t_score - s_score)
            status = "OK" if delta < TOLERANCE else "FAIL"
            print(
                f"    Row {PARITY_INDICES[i]:>5}: training={t_score:.8f}  "
                f"serving={s_score:.8f}  delta={delta:.2e} {status}"
            )

        for i, (t_score, s_score) in enumerate(zip(training_scores, serving_scores)):
            assert abs(t_score - s_score) < TOLERANCE, (
                f"PARITY FAILURE at row {PARITY_INDICES[i]}: "
                f"training={t_score:.8f} serving={s_score:.8f} "
                f"delta={abs(t_score - s_score):.2e} exceeds tolerance {TOLERANCE}"
            )
