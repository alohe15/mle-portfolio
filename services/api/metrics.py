"""Prometheus metrics for one API process. Uses the default registry.

The finite action set is ``ACTION_NAMES`` in ``evals/cost_model.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVALS_DIR = REPO_ROOT / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from cost_model import ACTION_NAMES  # noqa: E402

OTHER_ACTION = "other"
_KNOWN_ACTIONS = frozenset(ACTION_NAMES)

HTTP_REQUESTS = Counter(
    "fraud_api_http_requests_total",
    "Completed HTTP requests.",
    labelnames=("route", "method", "status_class"),
)
HTTP_DURATION = Histogram(
    "fraud_api_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("route", "method", "status_class"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "fraud_api_predictions_total",
    "Successful /predict calls by recommended action.",
    labelnames=("action",),
)
MODEL_INFO = Gauge(
    "fraud_api_model_info",
    "Loaded serving identity. Value is 1.",
    labelnames=("model_version", "dataset_version"),
)
MODEL_LOADED = Gauge(
    "fraud_api_model_loaded",
    "1 if serving artifacts are loaded, else 0.",
)

# A zero series for each known action exists before any traffic.
for _action in ACTION_NAMES:
    PREDICTIONS.labels(action=_action).inc(0)


def observe_request(
    *,
    route: str,
    method: str,
    status_class: str,
    duration_seconds: float,
) -> None:
    HTTP_REQUESTS.labels(
        route=route, method=method, status_class=status_class
    ).inc()
    HTTP_DURATION.labels(
        route=route, method=method, status_class=status_class
    ).observe(duration_seconds)


def record_prediction(action: object) -> None:
    """Count a successful prediction. Unknown actions share the other label."""
    label = action if isinstance(action, str) and action in _KNOWN_ACTIONS else OTHER_ACTION
    PREDICTIONS.labels(action=label).inc()


def set_model_loaded(
    loaded: bool,
    *,
    model_version: str | None = None,
    dataset_version: str | None = None,
) -> None:
    MODEL_LOADED.set(1 if loaded else 0)
    if loaded and model_version and dataset_version:
        MODEL_INFO.labels(
            model_version=model_version,
            dataset_version=dataset_version,
        ).set(1)
