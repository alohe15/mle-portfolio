#!/usr/bin/env bash
# Run one k6 Job in the kind cluster and save the log outside Git.
set -euo pipefail
set +H

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: bash scripts/run_load.sh RUN_NAME VUS DURATION [WARMUP_DURATION]" >&2
  exit 2
fi

RUN_NAME="$1"
VUS="$2"
DURATION="$3"
WARMUP_DURATION="${4:-15s}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="${ROOT}/deploy/load/k6-job.template.yaml"
EVIDENCE="${HOME}/fraud-evidence/load"
mkdir -p "$EVIDENCE"

duration_to_seconds() {
  python3 -c 'import re, sys
value = sys.argv[1].strip()
match = re.fullmatch(r"(\d+)(ms|s|m|h)", value)
if not match:
    raise SystemExit("bad duration: " + value)
amount = int(match.group(1))
scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[match.group(2)]
print(max(1, int(amount * scale)))' "$1"
}

# kind load fails here: it imports --all-platforms, but the local image only has the pulled linux/arm64 digest.
preload_k6_image() {
  if docker exec fraud-local-control-plane crictl images | grep k6; then
    return 0
  fi
  docker save grafana/k6:2.4.0 | docker exec -i fraud-local-control-plane ctr --namespace=k8s.io images import -
}

preload_k6_image

TIMEOUT="$(( $(duration_to_seconds "$DURATION") + 180 ))s"

kubectl -n fraud-local create configmap k6-scripts \
  --from-file="predict.js=${ROOT}/tests/load/predict.js" \
  --from-file="payloads.json=${ROOT}/tests/load/payloads.json" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n fraud-local delete job "$RUN_NAME" --ignore-not-found --wait=true

sed \
  -e "s|__RUN_NAME__|${RUN_NAME}|g" \
  -e "s|__VUS__|${VUS}|g" \
  -e "s|__DURATION__|${DURATION}|g" \
  -e "s|__WARMUP_DURATION__|${WARMUP_DURATION}|g" \
  "$TEMPLATE" | kubectl apply -f -

if ! kubectl -n fraud-local wait --for=condition=complete "job/${RUN_NAME}" --timeout="$TIMEOUT"; then
  echo "=== job status"
  kubectl -n fraud-local get job "$RUN_NAME" -o wide || true
  echo "=== pod status"
  kubectl -n fraud-local get pods -l "job-name=${RUN_NAME}" -o wide || true
  exit 1
fi

LOG_FILE="${EVIDENCE}/${RUN_NAME}.log"
SUMMARY_FILE="${EVIDENCE}/${RUN_NAME}.summary.json"
kubectl -n fraud-local logs "job/${RUN_NAME}" | tee "$LOG_FILE"

python3 - "$LOG_FILE" "$SUMMARY_FILE" <<'PY'
import json
import sys

log_path, summary_path = sys.argv[1:]
found = None
for line in open(log_path, errors="replace"):
    marker = "K6_SUMMARY "
    index = line.find(marker)
    if index < 0:
        continue
    rest = line[index + len(marker) :].strip()
    source = rest.rfind('" source=')
    if source >= 0:
        rest = rest[:source]
    if '\\"' in rest:
        rest = rest.replace('\\"', '"')
    found = json.loads(rest)
    break
if found is None:
    raise SystemExit("K6_SUMMARY line not found")
text = json.dumps(found)
open(summary_path, "w").write(text + "\n")
print(text)
PY

API_POD="$(kubectl -n fraud-local get pod -l app=fraud-api -o jsonpath='{.items[0].metadata.name}')"
API_NODE="$(kubectl -n fraud-local get pod -l app=fraud-api -o jsonpath='{.items[0].spec.nodeName}')"
API_RESOURCES="$(kubectl -n fraud-local get deploy fraud-api -o jsonpath='{.spec.template.spec.containers[0].resources}')"
K6_POD="$(kubectl -n fraud-local get pod -l "job-name=${RUN_NAME}" -o jsonpath='{.items[0].metadata.name}')"
K6_NODE="$(kubectl -n fraud-local get pod -l "job-name=${RUN_NAME}" -o jsonpath='{.items[0].spec.nodeName}')"
K6_IMAGE="$(kubectl -n fraud-local get job "$RUN_NAME" -o jsonpath='{.spec.template.spec.containers[0].image}')"

{
  echo "API pod: ${API_POD}"
  echo "API resources: ${API_RESOURCES}"
  echo "k6 image: ${K6_IMAGE}"
  echo "API node: ${API_NODE}"
  echo "k6 pod: ${K6_POD}"
  echo "k6 node: ${K6_NODE}"
  if [[ "$API_NODE" == "$K6_NODE" ]]; then
    echo "k6 shares the same kind node (same machine) as the API"
  else
    echo "k6 node differs from the API node"
  fi
} | tee -a "$LOG_FILE"
