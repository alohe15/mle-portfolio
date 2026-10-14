# Release evidence — lgbm_v9 container
 
Recorded 2026-10-08 after the in-container test run on this machine.
Branch `pr/phase7-validation-parity`. The Dockerfiles and Compose test
service were added as part of the Phase 6 and 7 corrections.
 
## Image
 
| Image | Tag | Size | Created |
|---|---|---|---|
| Serving | `fraud-api:latest` | 1.79GB | 2026-10-08 16:01:08 -0400 |
| Test (serving image + pytest/httpx + `tests/`) | `fraud-api:test` | 1.81GB | 2026-10-08 16:07:49 -0400 |
 
Serving id `1b6142ebe60a`. Test id `e00d12acaeca`.
 
## Registry identity
 
From `models/registry.json` (`is_serving: true`) and `configs/lgbm_v9.json`:
 
| Field | Value |
|---|---|
| Model version | 9 (`lgbm_v9_optuna_tuning.txt`) |
| Dataset version | 5 (`configs/dataset_v5.json`) |
| Config version | 9 (`configs/lgbm_v9.json`) |
 
## In-container tests
 
Command (builds `Dockerfile.test` from the serving image, mounts `data/` read-only):
 
```bash
docker compose run --rm test
```
 
That runs:
 
```bash
python -m pytest tests/test_training_serving_parity.py tests/test_api_endpoint.py -v
```
 
Result: **14 passed**, 2 warnings, 12.32s, inside the container (`platform linux -- Python 3.11.16`).
 
```
tests/test_training_serving_parity.py::test_training_serving_parity PASSED
tests/test_api_endpoint.py::test_successful_prediction PASSED
tests/test_api_endpoint.py::test_missing_transaction_amt_returns_422 PASSED
tests/test_api_endpoint.py::test_wrong_type_transaction_amt_returns_422 PASSED
tests/test_api_endpoint.py::test_extreme_near_zero_amount PASSED
tests/test_api_endpoint.py::test_extreme_large_amount PASSED
tests/test_api_endpoint.py::test_all_null_historical_features PASSED
tests/test_api_endpoint.py::test_unseen_categorical_productcd PASSED
tests/test_api_endpoint.py::test_health_endpoint PASSED
tests/test_api_endpoint.py::test_negative_amount_returns_422 PASSED
tests/test_api_endpoint.py::test_response_has_no_raw_input_fields PASSED
tests/test_api_endpoint.py::test_request_id_unique PASSED
tests/test_api_endpoint.py::test_action_costs_four_keys PASSED
tests/test_api_endpoint.py::test_dataset_version_present PASSED
======================= 14 passed, 2 warnings in 12.32s ========================
```
 
The warnings are Starlette deprecations (`httpx` vs `httpx2`, `anyio.abc.BlockingPortal`). They are not test failures.
 
### Parity rows
 
Same image, scores printed with `-s`. Tolerance `PARITY_ATOL = 1e-6` on raw scores and calibrated probabilities. All five validation rows matched with absolute difference 0 and the same action.
 
```bash
docker compose run --rm --no-deps test python -m pytest tests/test_training_serving_parity.py -vv -s
```
 
```
=== Training vs serving parity ===
 idx        txn    train_raw    serve_raw   abs_diff action_match
   0    3400378   0.00000029   0.00000029  0.000e+00         True
   1    3400379   0.97191350   0.97191350  0.000e+00         True
   2    3400380   0.00235601   0.00235601  0.000e+00         True
   3    3400381   0.00000003   0.00000003  0.000e+00         True
   4    3400382   0.00000002   0.00000002  0.000e+00         True
PASSED
============================== 1 passed in 11.48s ==============================
```
 
## Startup checksum verification
 
The correction after this evidence run makes the lean image copy
`artifacts.serving.sha256`. On the next container evidence run, verify that
startup checks every serving artifact listed there before loading the booster
or any pickle.
 
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO services.api.model_loader Artifact checksums verified (11 files) against /app/artifacts.serving.sha256
INFO services.api Serving artifacts loaded model_version=v9 dataset_version=5 n_features=464 n_transforms=7
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
 
Health body:
 
```json
{"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
```
 
On a checkout where the serving checksum manifest is present, startup hashes
every file in `artifacts.serving.sha256` before opening the booster.
`shasum -a 256 -c artifacts.serving.sha256` passed for all serving paths. A
one-byte flip of a **copy** of `models/lgbm_v9_calibrator.pkl` raised:
 
```
ArtifactChecksumError: Checksum mismatch for reports/monitoring/_checksum_probe.pkl: expected 1fabce507f509862cd02c1de92bf0141bd5a2f29718031df54ea909cce7213d1, actual ca48d535eafd5fd7d080be0e3f28d90842140694685ed88f6f61366c3a2c8e56
```
 
The real calibrator was not modified. The probe file was deleted after the check.
 
## Phase 8 — Docker Release Testing
 
### Docker version
 
```
Docker Engine 29.7.2, Docker Desktop 4.90.0
Docker Compose v5.5.1 (darwin/arm64)
```
 
### Artifact checksum verification
 
```
shasum -a 256 -c artifacts.serving.sha256
# All 11 files: OK
```
 
### Docker build
 
```
$ docker compose build fraud-api
[+] Building 2.3s (27/27) FINISHED
 ✔ Image fraud-api:latest Built  2.4s
```
 
### Liveness and readiness endpoints
 
```
$ curl http://localhost:8000/livez
{"status":"alive"}
 
$ curl http://localhost:8000/readyz
{"status":"ready","model_version":"v9","dataset_version":5,"model_loaded":true}
 
$ curl http://localhost:8000/health
{"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
```
 
### Smoke test
 
```
$ bash scripts/smoke_test_api.sh
 
=== Smoke Test: Health (http://localhost:8000/health) ===
  HTTP 200
  body: {"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
  PASS: health HTTP 200
  PASS: health has model_version
  PASS: health has dataset_version
 
=== Smoke Test: Liveness (http://localhost:8000/livez) ===
  HTTP 200
  body: {"status":"alive"}
  PASS: livez HTTP 200
  PASS: livez body contains alive
 
=== Smoke Test: Readiness (http://localhost:8000/readyz) ===
  HTTP 200
  body: {"status":"ready","model_version":"v9","dataset_version":5,"model_loaded":true}
  PASS: readyz HTTP 200
  PASS: readyz body contains ready
  PASS: readyz has model_version
 
=== Smoke Test: Predict (http://localhost:8000/predict) ===
  HTTP 200
  body: {"fraud_probability":0.005580357142857143,"recommended_action":"approve","action_costs":{"approve":1.4006696428571428,"step_up":6.660184151785715,"review":10.070033482142858,"decline":124.79966517857143},"model_version":"v9","dataset_version":5,"request_id":"cdb762c4-93f8-4673-9bac-8cd9a0a3e568"}
  PASS: predict HTTP 200
  PASS: predict has fraud_probability
  PASS: predict has recommended_action
  PASS: predict has action_costs
  PASS: predict has model_version
  PASS: predict has dataset_version
  PASS: predict has request_id
 
=== Smoke Test: Validation Error (missing TransactionAmt) ===
  HTTP 422
  body: {"detail":[{"type":"missing","loc":["body","TransactionAmt"],"msg":"Field required","input":{"ProductCD":"W"}}]}
  PASS: validation HTTP 422
 
=== Summary: 16 passed, 0 failed ===
=== All smoke tests passed ===
```
 
### Tests inside Docker
 
```
$ docker compose run --rm test python -m pytest \
    tests/test_monitoring_schema.py \
    tests/test_fixture_models.py \
    tests/test_serving_checksum_manifest.py \
    tests/test_training_serving_parity.py \
    -q
 
...........................                             [100%]
27 passed in 12.93s
```
 
### Failure experiment
 
Renamed the model artifact to simulate a missing file, then attempted to rebuild the Docker image.
 
**Break:**
 
```
$ mv models/lgbm_v9_optuna_tuning.txt models/lgbm_v9_optuna_tuning.txt.bak
$ docker compose build fraud-api
 
ERROR [17/20] COPY models/lgbm_v9_optuna_tuning.txt models/lgbm_v9_optuna_tuning.txt
------
failed to solve: failed to compute cache key: failed to calculate checksum of ref
  "/models/lgbm_v9_optuna_tuning.txt": not found
```
 
The build failed at step 17/20 — Docker could not COPY the model file because it was missing from disk. The image cannot be built without all required artifacts present.
 
**Restore:**
 
```
$ mv models/lgbm_v9_optuna_tuning.txt.bak models/lgbm_v9_optuna_tuning.txt
$ docker compose build fraud-api
[+] Building 2.3s (27/27) FINISHED
 ✔ Image fraud-api:latest Built  2.4s
 
$ docker compose up fraud-api
fraud-api-1  | INFO services.api.model_loader Artifact checksums verified (11 files)
fraud-api-1  | INFO services.api Serving artifacts loaded model_version=v9 dataset_version=5 n_features=464 n_transforms=7
fraud-api-1  | INFO:     Application startup complete.
 
$ curl http://localhost:8000/readyz
{"status":"ready","model_version":"v9","dataset_version":5,"model_loaded":true}
```
 
Restoring the model file and rebuilding produced a working image. The API started, checksums passed, and `/readyz` returned 200.
 