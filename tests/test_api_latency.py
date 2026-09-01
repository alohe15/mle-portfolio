"""API latency benchmark — verifies p95 response time < 200ms.

Sends 100 sequential requests to the running API and reports percentiles.
Uses TestClient (in-process) for reproducible, network-free benchmarks.

Run: python -m pytest tests/test_api_latency.py -v -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestLatency:
    N_REQUESTS = 100
    P95_TARGET_MS = 200

    def test_p95_under_target(self, client):
        """p95 response time must be under 200ms."""
        latencies = []

        for _ in range(5):
            client.post("/predict", json={"TransactionAmt": 50.0})

        for i in range(self.N_REQUESTS):
            start = time.perf_counter()
            resp = client.post(
                "/predict",
                json={
                    "TransactionAmt": 75.0 + i,
                    "card1": 1000 + (i % 100),
                    "ProductCD": "W",
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert resp.status_code == 200
            latencies.append(elapsed_ms)

        latencies_arr = np.array(latencies)
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
        mean = float(np.mean(latencies_arr))

        print(f"\n  Latency benchmark ({self.N_REQUESTS} requests):")
        print(f"    Mean:  {mean:.1f}ms")
        print(f"    p50:   {p50:.1f}ms")
        print(f"    p95:   {p95:.1f}ms  (target: <{self.P95_TARGET_MS}ms)")
        print(f"    p99:   {p99:.1f}ms")
        print(f"    Min:   {min(latencies):.1f}ms")
        print(f"    Max:   {max(latencies):.1f}ms")

        assert p95 < self.P95_TARGET_MS, (
            f"p95 latency {p95:.1f}ms exceeds target {self.P95_TARGET_MS}ms"
        )

    def test_minimal_vs_full_request_latency(self, client):
        """Compare latency between minimal and full requests."""
        minimal_latencies = []
        full_latencies = []

        for _ in range(20):
            start = time.perf_counter()
            client.post("/predict", json={"TransactionAmt": 75.0})
            minimal_latencies.append((time.perf_counter() - start) * 1000)

        for _ in range(20):
            start = time.perf_counter()
            client.post(
                "/predict",
                json={
                    "TransactionAmt": 75.0,
                    "TransactionDT": 86400,
                    "card1": 1000,
                    "card4": "visa",
                    "ProductCD": "W",
                    "P_emaildomain": "gmail.com",
                    "R_emaildomain": "gmail.com",
                    "D1": 14.0,
                    "D2": 28.0,
                    "D3": 7.0,
                    "C1": 1.0,
                    "C2": 1.0,
                },
            )
            full_latencies.append((time.perf_counter() - start) * 1000)

        print(
            f"\n  Minimal request p50: {np.percentile(minimal_latencies, 50):.1f}ms"
        )
        print(f"  Full request p50:    {np.percentile(full_latencies, 50):.1f}ms")
        print(
            "  Overhead:            "
            f"{np.percentile(full_latencies, 50) - np.percentile(minimal_latencies, 50):.1f}ms"
        )
