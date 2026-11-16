# Operations — local API monitoring

## 1. Purpose

Logs explain one request. Metrics summarise many. The dashboard shows how those metrics change over time. The Phase 7 signal plan, including Tier 1 and Tier 2, is in [docs/monitoring_plan.md](monitoring_plan.md). This note is only the local API logs, Prometheus metrics, and Grafana dashboard.

## 2. Start

Bring up the cluster and the API with [docs/kubernetes_runbook.md](kubernetes_runbook.md). Then install monitoring from the repo root:

```bash
kubectl apply -f deploy/monitoring/namespace.yaml && kubectl apply -f deploy/monitoring/
kubectl -n monitoring create secret generic grafana-admin --from-literal=admin-password="$(openssl rand -base64 18)" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k deploy/monitoring/grafana
kubectl -n monitoring rollout status deployment/prometheus --timeout=180s
kubectl -n monitoring rollout status deployment/grafana --timeout=180s
```

The `openssl` command creates the Grafana admin password in the cluster Secret `grafana-admin`. No password is stored in Git.

## 3. Access

Run each port-forward in the foreground in its own terminal tab:

```bash
kubectl -n monitoring port-forward svc/prometheus 9090:9090
kubectl -n monitoring port-forward svc/grafana 3000:3000
```

Open http://localhost:3000/d/fraud-api. Anonymous access is a Viewer (`GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer`). A port-forward attaches to one Pod and dies when that Pod is replaced. Rerun it after any rollout.

## 4. Log fields

Each completed request writes one JSON line on the `services.api.access` logger.

| Field | Meaning |
|---|---|
| `event` | Always `request_completed`. |
| `timestamp` | UTC time the request finished, ISO 8601. |
| `request_id` | Server-generated UUID. The client `X-Request-ID` header is ignored. |
| `route` | Path template, or `unmatched` when no route matched. |
| `method` | HTTP method. |
| `status_class` | `2xx`, `4xx`, or `5xx`. |
| `status_code` | HTTP status code. |
| `duration_ms` | Handler time in milliseconds. |
| `model_version` | Loaded model label, or `none` when artifacts are not loaded. |
| `error_category` | `none`, `validation_error`, `not_found`, `method_not_allowed`, `model_not_loaded`, `prediction_error`, or `unhandled_exception`. |
| `recommended_action` | Action on a successful `/predict`. `null` otherwise. |

Never logged: request bodies, headers, tokens, `TransactionID`, customer identifiers, exception messages.

## 5. Find one failed request

Take `X-Request-ID` from the response header or body, then:

```bash
kubectl -n fraud-local logs deploy/fraud-api | grep <request_id>
```

The malformed check in [docs/evidence/phase10a/step5_checks.txt](evidence/phase10a/step5_checks.txt) used `1d728182-bb3b-4ed5-844c-ae36c642a86d`. That id is this line:

```text
{"event": "request_completed", "timestamp": "2026-11-15T23:10:10.065804+00:00", "request_id": "1d728182-bb3b-4ed5-844c-ae36c642a86d", "route": "/predict", "method": "POST", "status_class": "4xx", "status_code": 422, "duration_ms": 1.132, "model_version": "v9", "error_category": "validation_error", "recommended_action": null}
```

## 6. Metrics

Details and the single-process note are in [docs/metrics.md](metrics.md).

| Name | Type | Labels | Units |
|---|---|---|---|
| `fraud_api_http_requests_total` | Counter | `route`, `method`, `status_class` | requests |
| `fraud_api_http_request_duration_seconds` | Histogram | `route`, `method`, `status_class` | seconds |
| `fraud_api_predictions_total` | Counter | `action` | predictions |
| `fraud_api_model_info` | Gauge | `model_version`, `dataset_version` | 1 |
| `fraud_api_model_loaded` | Gauge | none | 1 or 0 |

`POST /predict` is created at import for `2xx`, `4xx`, and `5xx`, each at 0. The four actions on `fraud_api_predictions_total` are also created at 0. `rate()` needs a prior sample, so the first event on a new series is invisible without that 0.

## 7. Dashboard

The saved dashboard is `deploy/monitoring/grafana/dashboards/fraud-api.json`. Queries below are copied from that file.

| Panel | Saved query | Window | Units |
|---|---|---|---|
| Prediction volume | `sum(rate(fraud_api_http_requests_total{route="/predict",method="POST"}[1m]))` | 1m | requests per second |
| Client error ratio (4xx) | `(sum(rate(fraud_api_http_requests_total{route="/predict",status_class="4xx"}[1m])) or vector(0)) / (sum(rate(fraud_api_http_requests_total{route="/predict"}[1m])) > 0)` | 1m | fraction of requests, 0 to 1 |
| Server error ratio (5xx) | `(sum(rate(fraud_api_http_requests_total{route="/predict",status_class="5xx"}[1m])) or vector(0)) / (sum(rate(fraud_api_http_requests_total{route="/predict"}[1m])) > 0)` | 1m | fraction of requests, 0 to 1 |
| Latency, successful predictions | `histogram_quantile(0.5, sum by (le) (rate(fraud_api_http_request_duration_seconds_bucket{route="/predict",status_class="2xx"}[5m]))) and on() (sum(rate(fraud_api_http_request_duration_seconds_count{route="/predict",status_class="2xx"}[5m])) > 0)` | 5m | seconds |
| Latency, successful predictions | `histogram_quantile(0.95, sum by (le) (rate(fraud_api_http_request_duration_seconds_bucket{route="/predict",status_class="2xx"}[5m]))) and on() (sum(rate(fraud_api_http_request_duration_seconds_count{route="/predict",status_class="2xx"}[5m])) > 0)` | 5m | seconds |
| Loaded model | `max by (model_version, dataset_version) (fraud_api_model_info)` | none | gauge value 1, shown as the series name |
| Model loaded / scrape up | `max(fraud_api_model_loaded)` | none | 1 or 0 |
| Model loaded / scrape up | `max(up{job="fraud-api"})` | none | 1 or 0 |
| Prediction action mix | `sum by (action) (increase(fraud_api_predictions_total[5m]))` | 5m | predictions |

`/livez`, `/readyz`, and `/metrics` are excluded from these panels. "No traffic" means no `/predict` requests in the window, not that the API is healthy. Error ratios count a missing error series as 0 when traffic exists (`or vector(0)`). Panels aggregate every scraped API Pod, including the drill Pod while it runs. `increase()` is extrapolated, so use logs for exact counts. With few requests, p95 is effectively the slowest request. `scrape_up=1` only proves `/metrics` answered, not that `/predict` works.

Check every query without a browser, with Prometheus forwarded to port 9090:

```bash
python3 scripts/check_dashboard_queries.py
```

A saved run is [docs/evidence/phase10a/dashboard_queries.txt](evidence/phase10a/dashboard_queries.txt).

Dashboard screenshots:

- [step4_grafana_unreachable.png](evidence/phase10a/step4_grafana_unreachable.png): every panel errored. Cause: the port-forward died when Grafana was OOMKilled.
- [step4_dashboard_before_5xx_fix.png](evidence/phase10a/step4_dashboard_before_5xx_fix.png): the 5xx panel said "No traffic" during traffic (missing error series).
- [step5_first_error_invisible.png](evidence/phase10a/step5_first_error_invisible.png): the drill's first 500 did not appear (`rate()` needs a prior sample).
- [step5_dashboard_service_error.png](evidence/phase10a/step5_dashboard_service_error.png): after pre-initialising the series, drill 500s are visible as a 5xx spike.

## 8. Is it getting worse?

Read the 4xx and 5xx ratio panels next to Prediction volume. A ratio is a share of `/predict` requests, so it needs that denominator. Volume near 0 with "No traffic" on a ratio means the window had no `/predict` calls. Volume above 0 with a rising ratio means a larger share of those calls failed. A higher ratio is not, by itself, more traffic.

## 9. Three checks

The service error is `tests/support/failing_api.py`, a test double that raises before a normal prediction. It runs only in the drill Pod from [deploy/drills/fault-instance.yaml](../deploy/drills/fault-instance.yaml), using the test image. The fraud-api Service selects `app: fraud-api`, so it does not send traffic to the drill. There is no failure route in the API.

```bash
kubectl apply -f deploy/drills/fault-instance.yaml
kubectl -n fraud-local rollout status deployment/fraud-api-drill --timeout=180s
bash scripts/run_monitoring_checks.sh
kubectl delete -f deploy/drills/fault-instance.yaml
```

The script needs Prometheus at `localhost:9090`. It records counters from the Pod's `/metrics`, waits 35 seconds, then prints the Prometheus value as `prometheus_after`. Pass/fail uses the Pod counters.

| Request type | Expected outcome | Actual outcome | Request ID | Evidence |
|---|---|---|---|---|
| Normal golden `POST /predict` through the Service | 200, `error_category` `none`, `2xx+1`, `approve+1` | 200, `none`, `2xx+1 approve+1`, PASS | `efee5592-bdc4-4ee5-ae5c-731004c65ec7` | [docs/evidence/phase10a/step5_checks.txt](evidence/phase10a/step5_checks.txt) |
| Malformed `{"ProductCD":"W"}` through the Service | 422, `validation_error`, `4xx+1`, predictions unchanged | 422, `validation_error`, `4xx+1 predictions+0`, PASS | `1d728182-bb3b-4ed5-844c-ae36c642a86d` | [docs/evidence/phase10a/step5_checks.txt](evidence/phase10a/step5_checks.txt) |
| Golden request to `localhost:8000` inside the drill Pod | 500, `prediction_error`, `5xx+1`, predictions unchanged, injected message absent | 500, `prediction_error`, `5xx+1 predictions+0 marker_absent`, PASS | `6165e24a-f804-46df-aefc-3aa9caace7b2` | [docs/evidence/phase10a/step5_checks.txt](evidence/phase10a/step5_checks.txt) |

A readiness failure is a different experiment, since traffic stops before prediction (Phase 11).

## 10. Troubleshooting

| What we saw | Cause | Fix |
|---|---|---|
| Dashboard "failed to fetch" / red triangles. [step4_grafana_unreachable.png](evidence/phase10a/step4_grafana_unreachable.png) | The port-forward died after a Pod replacement. | Rerun the port-forward. |
| Grafana `OOMKilled` (exit 137) at 256Mi | The 256Mi limit was too small, and Grafana was installing plugins at startup. | Memory limit raised to 512Mi. `GF_PLUGINS_PREINSTALL_DISABLED=true`. |
| 5xx panel said "No traffic" during traffic. [step4_dashboard_before_5xx_fix.png](evidence/phase10a/step4_dashboard_before_5xx_fix.png) | The 5xx series did not exist yet, so the ratio was empty. | `or vector(0)` on the error-ratio queries. |
| First error on a new Pod invisible to `rate()` / `increase()`. [step5_first_error_invisible.png](evidence/phase10a/step5_first_error_invisible.png) | `rate()` needs a prior sample. A series that appears already at 1 hides that first event. | Pre-initialise the `/predict` `2xx`, `4xx`, and `5xx` series at 0. |
| Check script counted +2 | Prometheus lags up to one scrape interval behind the Pod. | `scripts/run_monitoring_checks.sh` reads Pod `/metrics` for pass/fail. |
| Prometheus history lost after a restart | The Prometheus Deployment stores data on an `emptyDir`. | By design for local use. |

## 11. Cleanup

```bash
kubectl delete -f deploy/drills/fault-instance.yaml
kubectl delete -k deploy/monitoring/grafana
kubectl -n monitoring delete secret grafana-admin
kubectl delete -f deploy/monitoring/
```

Stop the port-forwards in their terminal tabs. To remove the cluster as well:

```bash
kind delete cluster --name fraud-local
```

## 12. Limits

Traffic in this setup is synthetic. Metrics are one process: `UVICORN_WORKERS` in [deploy/k8s/local/configmap.yaml](../deploy/k8s/local/configmap.yaml) is not read by the image `CMD`. This stack is local only. Thresholds and alerting are not set up.
