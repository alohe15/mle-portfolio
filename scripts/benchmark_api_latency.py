"""Run the fraud API endpoint repeatedly and report latency statistics.

Usage:
    uvicorn services.api.app:app --reload --app-dir .
    python scripts/benchmark_api_latency.py

Optional env vars:
    API_URL   default http://127.0.0.1:8000/predict
    NUM_RUNS  default 1000
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path
from urllib import error

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from test_api_endpoint import DEFAULT_API_URL, build_test_payload, call_predict_endpoint


def timed_call(payload: dict, url: str) -> float:
    start = time.perf_counter()
    call_predict_endpoint(payload, url)
    return time.perf_counter() - start


def main() -> None:
    api_url = os.getenv("API_URL", DEFAULT_API_URL)
    num_runs = int(os.getenv("NUM_RUNS", "1000"))

    payload = build_test_payload()
    latencies_ms: list[float] = []

    try:
        for _ in range(num_runs):
            latency_s = timed_call(payload, api_url)
            latencies_ms.append(latency_s * 1000)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"API error ({exc.code}) from {api_url}: {body}", file=sys.stderr)
        sys.exit(1)
    except error.URLError as exc:
        print(f"Could not reach API at {api_url}: {exc}", file=sys.stderr)
        print(
            "Start the server first:\n"
            "  uvicorn services.api.app:app --reload --app-dir .",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"POST {api_url} ({num_runs} runs)")
    print(f"  avg:   {statistics.mean(latencies_ms):.2f} ms")
    print(f"  stdev: {statistics.stdev(latencies_ms):.2f} ms")
    print(f"  max:   {max(latencies_ms):.2f} ms")
    print(f"  min:   {min(latencies_ms):.2f} ms")


if __name__ == "__main__":
    main()
