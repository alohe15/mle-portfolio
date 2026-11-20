#!/usr/bin/env bash
# Run the normal, malformed, and drill checks. Evidence stays outside the repo.
set -u
set +H

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN="${ROOT}/tests/fixtures/golden_request.json"
EVIDENCE="${HOME}/fraud-evidence/step5_checks.txt"
mkdir -p "${HOME}/fraud-evidence"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

prom() {
  curl -sf "http://localhost:9090/api/v1/query" --data-urlencode "query=$1" \
    | python3 -c 'import json,sys
payload=json.load(sys.stdin)
rows=payload.get("data",{}).get("result") or []
print(sum(float(row["value"][1]) for row in rows) if rows else 0)'
}

pod_metric() {
  local deploy="$1"
  local expr="$2"
  kubectl -n fraud-local exec "deploy/${deploy}" -- curl -sS localhost:8000/metrics \
    | python3 -c '
import re
import sys

expr = sys.argv[1].strip()
if expr.startswith("sum(") and expr.endswith(")"):
    expr = expr[4:-1]
matched = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?$", expr)
name = matched.group(1)
wanted = {}
if matched.group(2):
    for part in matched.group(2)[1:-1].split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key == "app":
            continue
        wanted[key] = value.strip().strip("\"")
total = 0.0
found = False
line_re = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([0-9eE.+-]+)$")
for line in sys.stdin:
    sample = line_re.match(line.strip())
    if sample is None or sample.group(1) != name:
        continue
    have = {}
    if sample.group(3):
        for part in sample.group(3).split(","):
            key, value = part.split("=", 1)
            have[key.strip()] = value.strip().strip("\"")
    if all(have.get(key) == value for key, value in wanted.items()):
        total += float(sample.group(4))
        found = True
print(total if found else 0)
' "$expr"
}

run_check() {
  local name="$1" expected="$2" deploy="$3" mode="$4"
  local req_expr="$5" pred_expr="$6" want_status="$7" want_category="$8"
  local want_req="$9" want_pred="${10}" want_marker="${11}"
  local raw="${WORKDIR}/response.http"
  local before_req before_pred after_req after_pred prom_req prom_pred
  before_req="$(pod_metric "$deploy" "$req_expr")"
  before_pred="$(pod_metric "$deploy" "$pred_expr")"
  if [[ "$mode" == "golden" ]]; then
    kubectl -n fraud-local exec -i "deploy/${deploy}" -- \
      curl -si -X POST -H 'Content-Type: application/json' --data-binary @- \
      "http://fraud-api:8000/predict" < "$GOLDEN" > "$raw"
  elif [[ "$mode" == "malformed" ]]; then
    kubectl -n fraud-local exec "deploy/${deploy}" -- \
      curl -si -X POST -H 'Content-Type: application/json' \
      -d '{"ProductCD":"W"}' "http://fraud-api:8000/predict" > "$raw"
  else
    kubectl -n fraud-local exec -i "deploy/${deploy}" -- \
      curl -si -X POST -H 'Content-Type: application/json' --data-binary @- \
      http://localhost:8000/predict < "$GOLDEN" > "$raw"
  fi
  sleep 35
  after_req="$(pod_metric "$deploy" "$req_expr")"
  after_pred="$(pod_metric "$deploy" "$pred_expr")"
  prom_req="$(prom "$req_expr")"
  prom_pred="$(prom "$pred_expr")"
  local log_file="${WORKDIR}/pod.log"
  kubectl -n fraud-local logs "deploy/${deploy}" --tail=400 > "$log_file" || true
  python3 - "$name" "$expected" "$raw" "$log_file" "$want_status" "$want_category" \
    "$before_req" "$after_req" "$want_req" "$before_pred" "$after_pred" \
    "$want_pred" "$want_marker" "$prom_req" "$prom_pred" <<'PY'
import json
import sys

(
    name,
    expected,
    raw_path,
    log_path,
    want_status,
    want_category,
    before_req,
    after_req,
    want_req,
    before_pred,
    after_pred,
    want_pred,
    want_marker,
    prom_req,
    prom_pred,
) = sys.argv[1:]
text = open(raw_path, errors="replace").read()
status = ""
request_id = ""
for line in text.splitlines():
    if line.startswith("HTTP/") and not status:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = parts[1]
    if line.lower().startswith("x-request-id:"):
        request_id = line.split(":", 1)[1].strip()
logs = open(log_path, errors="replace").read().splitlines()
matched = [line for line in logs if request_id and request_id in line]
category = ""
for line in matched:
    start = line.find("{")
    if start < 0:
        continue
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        continue
    if payload.get("request_id") == request_id and "error_category" in payload:
        category = payload["error_category"]
marker = "drill-injected-failure-7f3a"
marker_clean = marker not in text and all(marker not in line for line in matched)

def show(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return f"{int(rounded):+d}"
    return f"{value:+.4f}"

req_delta = float(after_req) - float(before_req)
pred_delta = float(after_pred) - float(before_pred)
if name == "normal":
    deltas = f"2xx{show(req_delta)} approve{show(pred_delta)}"
elif name == "malformed":
    deltas = f"4xx{show(req_delta)} predictions{show(pred_delta)}"
else:
    state = "absent" if marker_clean else "present"
    deltas = f"5xx{show(req_delta)} predictions{show(pred_delta)} marker_{state}"
deltas = f"{deltas} prometheus_after={prom_req}/{prom_pred}"
ok = (
    status == want_status
    and category == want_category
    and abs(req_delta - float(want_req)) < 1e-6
    and abs(pred_delta - float(want_pred)) < 1e-6
    and (marker_clean if want_marker == "yes" else True)
    and bool(request_id)
)
print(
    "\t".join(
        [
            name,
            expected,
            status or "-",
            request_id or "-",
            category or "-",
            deltas,
            "PASS" if ok else "FAIL",
        ]
    )
)
sys.exit(0 if ok else 1)
PY
}

{
  printf '%s\n' "check	expected	actual status	request_id	error_category	metric deltas	result"
  failed=0
  run_check normal "200 none 2xx+1 approve+1" fraud-api golden \
    'fraud_api_http_requests_total{app="fraud-api",route="/predict",status_class="2xx"}' \
    'fraud_api_predictions_total{app="fraud-api",action="approve"}' \
    200 none 1 1 no || failed=1
  run_check malformed "422 validation_error 4xx+1 predictions+0" fraud-api malformed \
    'fraud_api_http_requests_total{app="fraud-api",route="/predict",status_class="4xx"}' \
    'sum(fraud_api_predictions_total{app="fraud-api"})' \
    422 validation_error 1 0 no || failed=1
  run_check "service error" "500 prediction_error 5xx+1 predictions+0" fraud-api-drill drill \
    'fraud_api_http_requests_total{app="fraud-api-drill",status_class="5xx"}' \
    'sum(fraud_api_predictions_total{app="fraud-api-drill"})' \
    500 prediction_error 1 0 yes || failed=1
  exit "$failed"
} | tee "$EVIDENCE"
exit "${PIPESTATUS[0]}"
