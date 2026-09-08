#!/usr/bin/env bash
# Smoke-test a live fraud API (default http://localhost:8000).
# Exit 0 on success, 1 on any failure.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

ok() { echo "  PASS: $*"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $*"; FAIL=$((FAIL + 1)); }

json_has() {
  local body="$1"
  local key="$2"
  python3 - "$body" "$key" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
key = sys.argv[2]
sys.exit(0 if key in body else 1)
PY
}

echo "=== Smoke Test: Health (${BASE_URL}/health) ==="
HEALTH_CODE=$(curl -s -o /tmp/smoke_health.json -w "%{http_code}" "${BASE_URL}/health" || true)
HEALTH_BODY=$(cat /tmp/smoke_health.json 2>/dev/null || echo "")
echo "  HTTP ${HEALTH_CODE}"
echo "  body: ${HEALTH_BODY}"
if [[ "${HEALTH_CODE}" == "200" ]]; then ok "health HTTP 200"; else bad "health expected 200 got ${HEALTH_CODE}"; fi
if json_has "${HEALTH_BODY}" "model_version"; then ok "health has model_version"; else bad "health missing model_version"; fi
if json_has "${HEALTH_BODY}" "dataset_version"; then ok "health has dataset_version"; else bad "health missing dataset_version"; fi

echo
echo "=== Smoke Test: Predict (${BASE_URL}/predict) ==="
PRED_CODE=$(curl -s -o /tmp/smoke_predict.json -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -d '{
    "TransactionAmt": 125.50,
    "ProductCD": "W",
    "card1": 10000,
    "card2": 111,
    "card3": 150,
    "card4": "visa",
    "card5": 226,
    "card6": "debit",
    "addr1": 123,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "dist1": 10.0,
    "dist2": null,
    "TransactionDT": 86500,
    "D1": 5, "D2": null, "D3": 2, "D4": null, "D5": null,
    "D6": null, "D7": null, "D8": null, "D9": null, "D10": 1,
    "D11": null, "D12": null, "D13": null, "D14": null, "D15": 3,
    "C1": 1, "C2": 1,
    "DeviceType": "mobile",
    "DeviceInfo": "SyntheticOS",
    "id_30": "iOS",
    "id_31": "safari",
    "metadata": {"TransactionID": 999001, "notes": "smoke"}
  }' \
  "${BASE_URL}/predict" || true)
PRED_BODY=$(cat /tmp/smoke_predict.json 2>/dev/null || echo "")
echo "  HTTP ${PRED_CODE}"
echo "  body: ${PRED_BODY}"
if [[ "${PRED_CODE}" == "200" ]]; then ok "predict HTTP 200"; else bad "predict expected 200 got ${PRED_CODE}"; fi
for key in fraud_probability recommended_action action_costs model_version dataset_version request_id; do
  if json_has "${PRED_BODY}" "${key}"; then ok "predict has ${key}"; else bad "predict missing ${key}"; fi
done

echo
echo "=== Smoke Test: Validation Error (missing TransactionAmt) ==="
VAL_CODE=$(curl -s -o /tmp/smoke_val.json -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -d '{"ProductCD": "W"}' \
  "${BASE_URL}/predict" || true)
echo "  HTTP ${VAL_CODE}"
echo "  body: $(cat /tmp/smoke_val.json 2>/dev/null || true)"
if [[ "${VAL_CODE}" == "422" ]]; then ok "validation HTTP 422"; else bad "validation expected 422 got ${VAL_CODE}"; fi

echo
echo "=== Summary: ${PASS} passed, ${FAIL} failed ==="
if [[ "${FAIL}" -eq 0 ]]; then
  echo "=== All smoke tests passed ==="
  exit 0
fi
echo "=== Smoke tests FAILED ==="
exit 1
