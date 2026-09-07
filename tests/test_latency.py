"""Single-request latency benchmark for services.api.predictor.predict.

Benchmarks the ML pipeline only (no HTTP). Asserts warm P99 < 100ms.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from services.api.model_loader import load_serving_artifacts
from services.api.predictor import (
    apply_transformation_pipeline,
    build_model_input,
    predict,
)
from services.api.schemas import TransactionRequest

P99_LIMIT_MS = 100.0
N_WARMUP = 10
N_BENCH = 200


@pytest.fixture(scope="module")
def artifacts():
    return load_serving_artifacts()


@pytest.fixture(scope="module")
def sample_request(artifacts) -> dict[str, Any]:
    # Synthetic but complete enough for the transform chain.
    payload = {
        "TransactionAmt": 87.42,
        "ProductCD": "W",
        "card1": 12695,
        "card2": 490.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 325.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "dist1": 8.0,
        "dist2": None,
        "TransactionDT": 100000.0,
        "D1": 14.0,
        "D2": None,
        "D10": 3.0,
        "D15": 7.0,
        "C1": 1.0,
        "C2": 1.0,
        "DeviceType": "mobile",
        "DeviceInfo": "SM-G950F",
        "id_30": "Android",
        "id_31": "chrome",
    }
    # Validate against schema then flatten.
    req = TransactionRequest(**payload)
    return req.feature_dict()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def test_predict_latency_p99(artifacts, sample_request, capsys):
    for _ in range(N_WARMUP):
        predict(sample_request, artifacts)

    times_ms: list[float] = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        predict(sample_request, artifacts)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "min": min(times_ms),
        "mean": float(np.mean(times_ms)),
        "std": float(np.std(times_ms)),
        "p50": _percentile(times_ms, 50),
        "p95": _percentile(times_ms, 95),
        "p99": _percentile(times_ms, 99),
        "max": max(times_ms),
    }
    print("\n=== predict() latency (ms) ===")
    for k, v in stats.items():
        print(f"  {k:>4}: {v:.3f}")

    # Stage breakdown (P50 over a smaller sample).
    transform_ms: list[float] = []
    model_ms: list[float] = []
    cal_policy_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        import pandas as pd
        from services.api.predictor import _request_to_frame

        df = _request_to_frame(sample_request)
        df = apply_transformation_pipeline(df, artifacts["transformation_pipeline"])
        model_input = build_model_input(df, artifacts)
        t1 = time.perf_counter()
        raw = float(artifacts["model"].predict(model_input)[0])
        t2 = time.perf_counter()
        cal = float(artifacts["calibrator"].predict([raw])[0])
        import sys
        from pathlib import Path

        evals = str(Path(__file__).resolve().parent.parent / "evals")
        if evals not in sys.path:
            sys.path.insert(0, evals)
        from cost_model import optimal_action

        optimal_action(cal, float(sample_request["TransactionAmt"]), artifacts["cost_params"])
        t3 = time.perf_counter()
        transform_ms.append((t1 - t0) * 1000.0)
        model_ms.append((t2 - t1) * 1000.0)
        cal_policy_ms.append((t3 - t2) * 1000.0)

    print("=== stage P50 (ms) ===")
    print(f"  transforms+prep: {_percentile(transform_ms, 50):.3f}")
    print(f"  model.predict  : {_percentile(model_ms, 50):.3f}")
    print(f"  calibrate+policy: {_percentile(cal_policy_ms, 50):.3f}")

    assert stats["p99"] < P99_LIMIT_MS, (
        f"P99 {stats['p99']:.2f}ms exceeds limit {P99_LIMIT_MS}ms"
    )
