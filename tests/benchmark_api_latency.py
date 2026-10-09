"""Repeatable local latency / startup baselines for the fraud API.

Measures:
  - load_serving_artifacts() wall time (full startup load)
  - in-process predict() P50/P95/P99 (200 iters after 10 warmup)
  - in-process stage breakdown: transform / model.predict / calibrate+policy
  - optional HTTP predict() baselines if a server is reachable
  - optional Docker image size via `docker image inspect`

These are **local baselines**, not production guarantees.

Usage:
    python tests/benchmark_api_latency.py
    python tests/benchmark_api_latency.py --base-url http://127.0.0.1:8000
    python tests/benchmark_api_latency.py --image fraud-api:latest
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_WARMUP = 10
N_BENCH = 200
P99_LIMIT_MS = 100.0

SAMPLE_REQUEST: dict[str, Any] = {
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
    "D10": 3.0,
    "D15": 7.0,
    "C1": 1.0,
    "C2": 1.0,
    "DeviceType": "mobile",
    "DeviceInfo": "SM-G950F",
    "id_30": "Android",
    "id_31": "chrome",
}


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _summarize(values_ms: list[float]) -> dict[str, float]:
    return {
        "min": min(values_ms),
        "mean": float(statistics.fmean(values_ms)),
        "p50": _pct(values_ms, 50),
        "p95": _pct(values_ms, 95),
        "p99": _pct(values_ms, 99),
        "max": max(values_ms),
    }


def docker_image_size_mb(image: str) -> float | None:
    try:
        out = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(out) / (1024.0 * 1024.0)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def bench_http(base_url: str) -> dict[str, float] | None:
    import urllib.request

    predict_url = base_url.rstrip("/") + "/predict"
    body = json.dumps(SAMPLE_REQUEST).encode("utf-8")

    def once() -> float:
        req = urllib.request.Request(
            predict_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return (time.perf_counter() - t0) * 1000.0

    try:
        for _ in range(N_WARMUP):
            once()
    except Exception as exc:  # noqa: BLE001
        print(f"HTTP benchmark skipped (server not available): {exc}")
        return None

    times: list[float] = []
    for _ in range(N_BENCH):
        times.append(once())
    return _summarize(times)


def bench_stages(artifacts: dict[str, Any]) -> dict[str, float]:
    """Time transform / model.predict / calibrate+policy separately."""
    from cost_model import optimal_action
    from services.api.predictor import (
        _request_to_frame,
        apply_transformation_pipeline,
        build_model_input,
    )

    amount = float(SAMPLE_REQUEST["TransactionAmt"])
    transform_ms: list[float] = []
    model_ms: list[float] = []
    policy_ms: list[float] = []

    for _ in range(N_WARMUP):
        df = _request_to_frame(SAMPLE_REQUEST)
        df = apply_transformation_pipeline(df, artifacts["transformation_pipeline"])
        model_input = build_model_input(df, artifacts)
        raw = float(artifacts["model"].predict(model_input)[0])
        cal = float(artifacts["calibrator"].predict([raw])[0])
        optimal_action(cal, amount, artifacts["cost_params"])

    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        df = _request_to_frame(SAMPLE_REQUEST)
        df = apply_transformation_pipeline(df, artifacts["transformation_pipeline"])
        model_input = build_model_input(df, artifacts)
        transform_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        raw = float(artifacts["model"].predict(model_input)[0])
        model_ms.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        cal = float(artifacts["calibrator"].predict([raw])[0])
        optimal_action(cal, amount, artifacts["cost_params"])
        policy_ms.append((time.perf_counter() - t2) * 1000.0)

    return {
        "transform_p50": _pct(transform_ms, 50),
        "model_p50": _pct(model_ms, 50),
        "policy_p50": _pct(policy_ms, 50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud API latency baselines")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", default="fraud-api:latest")
    parser.add_argument("--skip-http", action="store_true")
    args = parser.parse_args()

    from services.api.model_loader import load_serving_artifacts
    from services.api.predictor import predict

    t0 = time.perf_counter()
    artifacts = load_serving_artifacts()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(N_WARMUP):
        predict(SAMPLE_REQUEST, artifacts)

    inproc: list[float] = []
    for _ in range(N_BENCH):
        t2 = time.perf_counter()
        predict(SAMPLE_REQUEST, artifacts)
        inproc.append((time.perf_counter() - t2) * 1000.0)
    inproc_stats = _summarize(inproc)
    stage_stats = bench_stages(artifacts)

    http_stats = None if args.skip_http else bench_http(args.base_url)
    image_mb = docker_image_size_mb(args.image)

    print()
    print("=== Local Baselines (not production guarantees) ===")
    if image_mb is None:
        print("Image size:          n/a  (docker image not found / docker unavailable)")
    else:
        print(f"Image size:          {image_mb:.1f} MB")
    print(f"Startup time:        {startup_ms:.1f} ms")
    print(f"In-process P50:      {inproc_stats['p50']:.3f} ms")
    print(f"In-process P99:      {inproc_stats['p99']:.3f} ms")
    if http_stats is None:
        print("HTTP P50:            skipped — no server running")
        print("HTTP P99:            skipped — no server running")
    else:
        print(f"HTTP P50:            {http_stats['p50']:.3f} ms")
        print(f"HTTP P99:            {http_stats['p99']:.3f} ms")
    print(f"Transform P50:       {stage_stats['transform_p50']:.3f} ms")
    print(f"Model predict P50:   {stage_stats['model_p50']:.3f} ms")
    print(f"Calibrate+policy P50:{stage_stats['policy_p50']:.3f} ms")
    print()
    print("In-process detail:", {k: round(v, 3) for k, v in inproc_stats.items()})
    if http_stats:
        print("HTTP detail:       ", {k: round(v, 3) for k, v in http_stats.items()})

    if inproc_stats["p99"] >= P99_LIMIT_MS:
        raise SystemExit(
            f"FAIL: in-process P99 {inproc_stats['p99']:.3f} ms >= {P99_LIMIT_MS} ms"
        )
    print(f"PASS: in-process P99 < {P99_LIMIT_MS} ms")


if __name__ == "__main__":
    main()
