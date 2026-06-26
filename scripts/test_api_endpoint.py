"""Generate a test payload from training data and POST it to the fraud API.

Usage:
    uvicorn services.api.app:app --reload --app-dir .
    python scripts/test_api_endpoint.py

Optional env vars:
    API_URL  default http://127.0.0.1:8000/predict
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "train_merged.parquet"
PAYLOAD_PATH = REPO_ROOT / "test_payload.json"
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


def build_test_payload() -> dict:
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0].drop("isFraud").to_dict()

    # Convert NaN to None for valid JSON
    sample = {k: (None if pd.isna(v) else v) for k, v in sample.items()}
    return {"features": sample}


def save_test_payload(payload: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(payload, f)

    print(f"Saved {path}")


def call_predict_endpoint(payload: dict, url: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    api_url = os.getenv("API_URL", DEFAULT_API_URL)

    payload = build_test_payload()
    save_test_payload(payload, PAYLOAD_PATH)

    try:
        result = call_predict_endpoint(payload, api_url)
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

    print(f"POST {api_url}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
