"""Training vs serving numerical parity on validation rows.

Tolerance rationale: both paths run the same booster file on feature vectors
built from the same dataset-config transforms and the same frozen fitted
state. IEEE-754 double arithmetic through that shared graph should match;
1e-6 absorbs minor numpy/pandas float-path differences without hiding real
column-order or transform bugs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Absolute tolerance for raw scores and calibrated probabilities.
PARITY_ATOL = 1e-6

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVALS_DIR = REPO_ROOT / "evals"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from train import (  # noqa: E402
    find_dataset_manifest,
    load_json,
    parquet_path_from_manifest,
    repo_path,
    temporal_split,
)
from cost_model import optimal_action  # noqa: E402

from services.api.model_loader import load_serving_artifacts  # noqa: E402
from services.api.predictor import (  # noqa: E402
    apply_transformation_pipeline,
    build_model_input,
    predict as serving_predict,
)


N_ROWS = 5


@pytest.fixture(scope="module")
def artifacts():
    return load_serving_artifacts()


@pytest.fixture(scope="module")
def parity_rows(artifacts):
    """Validation rows: processed+fit path (training) vs raw→full pipeline (serving)."""
    entry = artifacts["registry_entry"]
    model_config = load_json(repo_path(entry["config_path"]))

    dataset_manifest_path = find_dataset_manifest(int(entry["dataset_version"]))
    processed = pd.read_parquet(parquet_path_from_manifest(dataset_manifest_path))
    _train_df, val_df, _test_df = temporal_split(
        processed, model_config["split"], config_path=entry["config_path"]
    )
    # The split copies the frame. Drop the copies this test does not score
    # before reading the raw table, or the container (8GB) is OOM-killed.
    del processed, _train_df, _test_df

    # Training-time feature path: parquet already has non-fit transforms;
    # apply frozen requires_fit state only, then the same model-input builder.
    fit_only = [s for s in artifacts["transformation_pipeline"] if s["requires_fit"]]
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    cost_params = artifacts["cost_params"]

    raw = pd.read_parquet(REPO_ROOT / "data" / "raw" / "train_merged.parquet")
    raw = raw.set_index("TransactionID", drop=False)

    rows = []
    for i in range(N_ROWS):
        val_row = val_df.iloc[[i]].copy()
        txn_id = int(val_row["TransactionID"].iloc[0])
        transformed = apply_transformation_pipeline(val_row, fit_only)
        model_input = build_model_input(transformed, artifacts)
        train_raw_score = float(model.predict(model_input)[0])
        train_cal = float(calibrator.predict([train_raw_score])[0])
        amount = float(val_row["TransactionAmt"].iloc[0])
        train_action, _ = optimal_action(train_cal, amount, cost_params)

        raw_row = raw.loc[txn_id]
        request = {
            k: (None if pd.isna(v) else v)
            for k, v in raw_row.drop(labels=["isFraud"], errors="ignore").items()
        }
        request["TransactionAmt"] = float(request["TransactionAmt"])

        rows.append(
            {
                "index": i,
                "TransactionID": txn_id,
                "train_raw": train_raw_score,
                "train_cal": train_cal,
                "train_action": str(train_action),
                "request": request,
            }
        )
    return rows


def test_training_serving_parity(artifacts, parity_rows, capsys):
    print("\n=== Training vs serving parity ===")
    print(
        f"{'idx':>4} {'txn':>10} {'train_raw':>12} {'serve_raw':>12} "
        f"{'abs_diff':>10} {'action_match':>12}"
    )
    for row in parity_rows:
        served = serving_predict(
            row["request"], artifacts, return_raw_score=True
        )
        diff = abs(served["raw_score"] - row["train_raw"])
        action_match = served["recommended_action"] == row["train_action"]
        print(
            f"{row['index']:4d} {row['TransactionID']:10d} "
            f"{row['train_raw']:12.8f} {served['raw_score']:12.8f} "
            f"{diff:10.3e} {str(action_match):>12}"
        )
        assert diff <= PARITY_ATOL, (
            f"raw score mismatch row {row['index']}: "
            f"train={row['train_raw']} serve={served['raw_score']} diff={diff}"
        )
        assert abs(served["fraud_probability"] - row["train_cal"]) <= PARITY_ATOL
        assert action_match

    captured = capsys.readouterr()
    # capsys consumes the table; write it back so `pytest -s` shows the rows.
    sys.stdout.write(captured.out)
    assert "Training vs serving parity" in captured.out
