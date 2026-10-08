# Release evidence — lgbm_v9 container

Recorded 2026-10-08 after the in-container test run on this machine.
Branch `pr/phase7-monitoring`. `Dockerfile` was not modified.

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

## Startup log

`docker compose up -d fraud-api`, then `GET /health` (200). The lean image does not contain `artifacts.sha256` (`Dockerfile` does not copy it), so checksum verification logs a warning and continues. The model still loads from the baked artifacts.

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
WARNING services.api.model_loader artifacts.sha256 not found at /app/artifacts.sha256; skipping startup checksum verification
INFO services.api Serving artifacts loaded model_version=v9 dataset_version=5 n_features=464 n_transforms=7
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Health body:

```json
{"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
```

On a checkout where `artifacts.sha256` is present (host `uvicorn`, or a mount of that file at `/app/artifacts.sha256`), startup hashes every serving and Candidate artifact path before opening the booster. `shasum -a 256 -c artifacts.sha256` passed for all nine paths. A one-byte flip of a **copy** of `models/lgbm_v9_calibrator.pkl` raised:

```
ArtifactChecksumError: Checksum mismatch for reports/monitoring/_checksum_probe.pkl: expected 1fabce507f509862cd02c1de92bf0141bd5a2f29718031df54ea909cce7213d1, actual ca48d535eafd5fd7d080be0e3f28d90842140694685ed88f6f61366c3a2c8e56
```

The real calibrator was not modified. The probe file was deleted after the check.
