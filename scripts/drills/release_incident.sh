#!/usr/bin/env bash
# Release-incident drill. The fault is a live readiness probe patch only.
# Expected failures must not abort the script.
set -uo pipefail
set +H

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GOLDEN="${ROOT}/tests/fixtures/golden_request.json"
LOG="${HOME}/fraud-evidence/phase11/release_incident.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

RESTORE_CMD='kubectl apply -f deploy/k8s/local/deployment.yaml && kubectl -n fraud-local rollout status deployment/fraud-api --timeout=180s'
NS=fraud-local

section() {
  echo
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 ==="
}

running_pod() {
  kubectl -n "$NS" get pods -l app=fraud-api --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

golden_fields() {
  local exec_pod="$1"
  local raw
  raw="$(kubectl -n "$NS" exec -i "$exec_pod" -- curl -sS -D - -X POST \
    -H 'Content-Type: application/json' --data-binary @- http://fraud-api:8000/predict \
    < "$GOLDEN")" || true
  printf '%s' "$raw" | python3 -c '
import json, sys
text = sys.stdin.read()
parts = text.split("\r\n\r\n", 1)
if len(parts) == 1:
    parts = text.split("\n\n", 1)
if len(parts) != 2:
    sys.stderr.write("golden response was not headers plus a body\n")
    sys.exit(1)
head, body = parts
request_id = ""
for line in head.splitlines():
    if line.lower().startswith("x-request-id:"):
        request_id = line.split(":", 1)[1].strip()
data = json.loads(body)
if not request_id:
    request_id = str(data.get("request_id", ""))
print(data["fraud_probability"])
print(data["recommended_action"])
print(request_id)
'
}

section "BASELINE"
BASE_POD="$(kubectl -n "$NS" get pod -l app=fraud-api -o jsonpath='{.items[0].metadata.name}')"
BASE_IP="$(kubectl -n "$NS" get pod "$BASE_POD" -o jsonpath='{.status.podIP}')"
BASE_READY="$(kubectl -n "$NS" get pod "$BASE_POD" -o jsonpath='{.status.containerStatuses[0].ready}')"
BASE_IMAGE="$(kubectl -n "$NS" get pod "$BASE_POD" -o jsonpath='{.spec.containers[0].image}')"
BASE_IMAGE_ID="$(kubectl -n "$NS" get pod "$BASE_POD" -o jsonpath='{.status.containerStatuses[0].imageID}')"
BASE_REVISION="$(kubectl -n "$NS" get deploy fraud-api -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')"
BASE_RS="$(kubectl -n "$NS" get pod "$BASE_POD" -o jsonpath='{.metadata.ownerReferences[0].name}')"
echo "pod: ${BASE_POD}"
echo "pod IP: ${BASE_IP}"
echo "ready: ${BASE_READY}"
echo "image: ${BASE_IMAGE}"
echo "imageID: ${BASE_IMAGE_ID}"
echo "deployment revision: ${BASE_REVISION}"
echo "replica set: ${BASE_RS}"
kubectl -n "$NS" get endpoints fraud-api
echo "golden via the Service:"
BASE_GOLDEN="$(golden_fields "$BASE_POD")"
BASE_SCORE="$(printf '%s\n' "$BASE_GOLDEN" | sed -n '1p')"
BASE_ACTION="$(printf '%s\n' "$BASE_GOLDEN" | sed -n '2p')"
BASE_REQUEST_ID="$(printf '%s\n' "$BASE_GOLDEN" | sed -n '3p')"
echo "fraud_probability: ${BASE_SCORE}"
echo "recommended_action: ${BASE_ACTION}"
echo "X-Request-ID: ${BASE_REQUEST_ID}"
echo "restoration command: ${RESTORE_CMD}"

section "INJECT"
T0="$(date -u +%s)"
echo "T0: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
kubectl -n "$NS" patch deployment fraud-api --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path","value":"/readyz-does-not-exist"}]'

section "OBSERVE"
kubectl -n "$NS" rollout status deployment/fraud-api --timeout=90s
echo "rollout status exit code: $?"
echo "pods:"
kubectl -n "$NS" get pods -l app=fraud-api -o wide
echo "replica sets:"
kubectl -n "$NS" get rs -l app=fraud-api \
  -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas'
NEW_POD="$(kubectl -n "$NS" get pods -l app=fraud-api -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.ownerReferences[0].name}{"\n"}{end}' | awk -v base="$BASE_RS" '$2 != base {print $1; exit}')"
NEW_RS="$(kubectl -n "$NS" get pod "$NEW_POD" -o jsonpath='{.metadata.ownerReferences[0].name}')"
echo "old replica set: ${BASE_RS}"
echo "new pod: ${NEW_POD}"
echo "new replica set: ${NEW_RS}"
echo "events for ${NEW_POD}, sorted by creation time, Readiness or Unhealthy:"
kubectl -n "$NS" get events --sort-by=.metadata.creationTimestamp | grep "$NEW_POD" | grep -E 'Readiness|Unhealthy' || true
echo "service endpoints:"
kubectl -n "$NS" get endpoints fraud-api
ENDPOINTS="$(kubectl -n "$NS" get endpoints fraud-api -o jsonpath='{.subsets[*].addresses[*].ip}')"
echo "endpoint IPs: ${ENDPOINTS}"
if [[ "$ENDPOINTS" == "$BASE_IP" ]]; then
  echo "endpoints contain only the old pod IP"
else
  echo "endpoints are not only the old pod IP"
fi
EXEC_POD="$(running_pod)"
echo "golden via the Service, exec from ${EXEC_POD}:"
OBS_GOLDEN="$(golden_fields "$EXEC_POD")"
OBS_SCORE="$(printf '%s\n' "$OBS_GOLDEN" | sed -n '1p')"
OBS_ACTION="$(printf '%s\n' "$OBS_GOLDEN" | sed -n '2p')"
OBS_REQUEST_ID="$(printf '%s\n' "$OBS_GOLDEN" | sed -n '3p')"
echo "fraud_probability: ${OBS_SCORE}"
echo "recommended_action: ${OBS_ACTION}"
echo "X-Request-ID: ${OBS_REQUEST_ID}"
for pod in $BASE_POD $NEW_POD; do
  if kubectl -n "$NS" logs "$pod" | grep -F "$OBS_REQUEST_ID" >/dev/null; then
    echo "request_id ${OBS_REQUEST_ID} is in ${pod} logs; this pod served it"
  else
    echo "request_id ${OBS_REQUEST_ID} is not in ${pod} logs"
  fi
done
echo "The probe now hits an unmatched route, so the new pod's logs show route \"unmatched\" 404s."
kubectl -n "$NS" logs "$NEW_POD" | grep -F '"route": "unmatched"' | head -n 2 || true

section "RESTORE"
T1="$(date -u +%s)"
echo "T1: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "running: ${RESTORE_CMD}"
eval "$RESTORE_CMD"

section "VERIFY"
echo "waiting until the broken pod is gone"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  remaining="$(kubectl -n "$NS" get pods -l app=fraud-api -o json | python3 -c '
import json, sys
items = json.load(sys.stdin)["items"]
print(sum(1 for item in items if item["metadata"].get("deletionTimestamp")))
')"
  if [[ "$remaining" == "0" ]]; then
    break
  fi
  sleep 2
done
echo "pods:"
kubectl -n "$NS" get pods -l app=fraud-api -o wide
echo "replica sets:"
kubectl -n "$NS" get rs -l app=fraud-api \
  -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas'
READY_COUNT="$(kubectl -n "$NS" get pods -l app=fraud-api --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.status.containerStatuses[0].ready}{"\n"}{end}' | grep -c true || true)"
BROKEN_REPLICAS="$(kubectl -n "$NS" get rs "$NEW_RS" -o jsonpath='{.spec.replicas}')"
echo "running ready pods: ${READY_COUNT}"
echo "broken replica set ${NEW_RS} desired replicas: ${BROKEN_REPLICAS}"
NOW_POD="$(kubectl -n "$NS" get pods -l app=fraud-api -o json | python3 -c '
import json, sys
items = json.load(sys.stdin)["items"]
for item in items:
    if item["metadata"].get("deletionTimestamp"):
        continue
    statuses = item.get("status", {}).get("containerStatuses") or []
    if item.get("status", {}).get("phase") == "Running" and statuses and statuses[0].get("ready"):
        print(item["metadata"]["name"])
        break
')"
NOW_IP="$(kubectl -n "$NS" get pod "$NOW_POD" -o jsonpath='{.status.podIP}')"
NOW_IMAGE="$(kubectl -n "$NS" get pod "$NOW_POD" -o jsonpath='{.spec.containers[0].image}')"
NOW_IMAGE_ID="$(kubectl -n "$NS" get pod "$NOW_POD" -o jsonpath='{.status.containerStatuses[0].imageID}')"
NOW_PROBE="$(kubectl -n "$NS" get deploy fraud-api -o jsonpath='{.spec.template.spec.containers[0].readinessProbe.httpGet.path}')"
echo "pod: ${NOW_POD}"
echo "pod IP: ${NOW_IP}"
echo "image: ${NOW_IMAGE}"
echo "imageID: ${NOW_IMAGE_ID}"
echo "baseline imageID: ${BASE_IMAGE_ID}"
echo "readinessProbe path: ${NOW_PROBE}"
if [[ "$NOW_IMAGE" == "fraud-api:a7555b2" ]]; then
  echo "image matches fraud-api:a7555b2"
else
  echo "image does not match fraud-api:a7555b2"
fi
if [[ "$NOW_IMAGE_ID" == "$BASE_IMAGE_ID" ]]; then
  echo "imageID matches the baseline imageID"
else
  echo "imageID does not match the baseline imageID"
fi
if [[ "$NOW_PROBE" == "/readyz" ]]; then
  echo "readinessProbe path is /readyz"
else
  echo "readinessProbe path is not /readyz"
fi
DIFF_OUT="$(kubectl diff -f deploy/k8s/local/deployment.yaml 2>&1)"
DIFF_RC=$?
echo "kubectl diff exit code: ${DIFF_RC}"
if [[ -n "${DIFF_OUT}" ]]; then
  printf '%s\n' "$DIFF_OUT"
fi
echo "service endpoints:"
kubectl -n "$NS" get endpoints fraud-api
NOW_ENDPOINTS="$(kubectl -n "$NS" get endpoints fraud-api -o jsonpath='{.subsets[*].addresses[*].ip}')"
echo "endpoint IPs: ${NOW_ENDPOINTS}"
if [[ "$NOW_ENDPOINTS" == "$NOW_IP" ]]; then
  echo "endpoints list the ready pod IP"
else
  echo "endpoints do not list only the ready pod IP"
fi
VERIFY_POD="$(running_pod)"
VERIFY_GOLDEN="$(golden_fields "$VERIFY_POD")"
VERIFY_SCORE="$(printf '%s\n' "$VERIFY_GOLDEN" | sed -n '1p')"
VERIFY_ACTION="$(printf '%s\n' "$VERIFY_GOLDEN" | sed -n '2p')"
VERIFY_REQUEST_ID="$(printf '%s\n' "$VERIFY_GOLDEN" | sed -n '3p')"
echo "fraud_probability: ${VERIFY_SCORE}"
echo "baseline fraud_probability: ${BASE_SCORE}"
echo "recommended_action: ${VERIFY_ACTION}"
echo "baseline recommended_action: ${BASE_ACTION}"
echo "X-Request-ID: ${VERIFY_REQUEST_ID}"
python3 - "$BASE_SCORE" "$VERIFY_SCORE" "$BASE_ACTION" "$VERIFY_ACTION" <<'PY'
import sys
base = float(sys.argv[1])
now = float(sys.argv[2])
base_action = sys.argv[3]
now_action = sys.argv[4]
if abs(now - base) <= 1e-6 and now_action == base_action:
    print("RECOVERY VERIFIED")
else:
    print("RECOVERY FAILED")
PY
NOW_REVISION="$(kubectl -n "$NS" get deploy fraud-api -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')"
echo "deployment revision: ${NOW_REVISION}"
kubectl -n "$NS" rollout history deployment/fraud-api
T_DONE="$(date -u +%s)"
echo "verified at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "total seconds from T0 to verified recovery: $((T_DONE - T0))"
