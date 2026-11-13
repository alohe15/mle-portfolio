# API metrics

`GET /metrics` exposes the process registry in Prometheus text format.

| Metric | Type | Labels | Units | Meaning |
|---|---|---|---|---|
| `fraud_api_http_requests_total` | Counter | `route`, `method`, `status_class` | requests | Completed requests. `route` is the path template, or `unmatched`. `status_class` is `2xx`, `4xx`, or `5xx`. |
| `fraud_api_http_request_duration_seconds` | Histogram | `route`, `method`, `status_class` | seconds | Latency of those requests. Buckets are 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, and 5s. Access logs stay in milliseconds. |
| `fraud_api_predictions_total` | Counter | `action` | predictions | Successful `/predict` calls. `action` is `approve`, `step_up`, `review`, `decline`, or `other`. The four actions are initialised to 0 at import. |
| `fraud_api_model_info` | Gauge | `model_version`, `dataset_version` | 1 | Set to 1 after artifacts load. |
| `fraud_api_model_loaded` | Gauge | none | 1 or 0 | 1 when serving artifacts are loaded. |

Health probes (`/health`, `/livez`, `/readyz`) and `/metrics` scrapes are counted under their own route labels. Prediction panels must filter `route="/predict"`. Latency panels should filter `status_class="2xx"` so fast 4xx rejections do not pull the percentiles down.

The container runs a single uvicorn process. The Dockerfile `CMD` has no `--workers`. The ConfigMap `UVICORN_WORKERS` value is not read by the `CMD`. Multiple workers would split counters per process and need Prometheus multiprocess mode.
