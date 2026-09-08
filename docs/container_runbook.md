# Container runbook — fraud-api (lgbm_v9 / dataset_v5)

Step-by-step for a clean checkout. Serving artifacts are **gitignored**; they must
exist on disk before `docker build` or the image will miss required COPY sources.

## Prerequisites

1. Docker installed and running — `docker info` must succeed.
2. From the repo root, these files must exist (gitignored artifacts will be missing after a fresh clone):

```bash
ls -lh \
  models/registry.json \
  models/lgbm_v9_optuna_tuning.txt \
  models/lgbm_v9_optuna_tuning_manifest.json \
  models/lgbm_v9_optuna_tuning_fitted_transforms.pkl \
  models/lgbm_v9_calibrator.pkl \
  configs/dataset_v1.json \
  configs/dataset_v2.json \
  configs/dataset_v3.json \
  configs/dataset_v5.json \
  configs/decision_policy_v1.json \
  scripts/train.py \
  scripts/feature_engineering.py \
  evals/cost_model.py
```

| Path | How to produce |
|------|----------------|
| `models/lgbm_v9_optuna_tuning.txt` (+ `_manifest.json`, `_fitted_transforms.pkl`) | `python scripts/train.py --config configs/lgbm_v9.json` (after dataset_v5 is built) |
| `models/lgbm_v9_calibrator.pkl` | `python evals/calibrate.py` (Phase 3) |
| `configs/decision_policy_v1.json` | Committed — Phase 3 |
| Dataset configs v1/v2/v3/v5 | Committed |

Approximate sizes: model `.txt` ~238 MB; fitted transforms `.pkl` ~1.2 MB; calibrator ~2 KB.

3. Python 3.11+ for host-side smoke/benchmark scripts (optional if you only use Docker).

## Build

```bash
cd /path/to/mle-portfolio
docker build -t fraud-api .
```

Expected: layers install pinned `requirements.txt`, then COPY source + explicit
model artifacts. Errors like `lstat models/lgbm_v9_optuna_tuning.txt: no such file`
mean a gitignored artifact is missing on the host before build. Pip failures usually
mean a bad pin in `requirements.txt`.

```bash
docker images fraud-api
```

## Run

Option A — Compose:

```bash
docker compose up --build -d
```

Option B — raw Docker:

```bash
docker run -d -p 8000:8000 --name fraud-test fraud-api
docker ps   # confirm fraud-test is Up
```

Expected startup log (via `docker logs fraud-test`):

```text
Serving artifacts loaded model_version=v9 dataset_version=5 n_features=464 n_transforms=7
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

If you see `FileNotFoundError` or `model_loaded: false`, a COPY line is missing
or an artifact was absent at build time.

## Health check

```bash
curl -s http://localhost:8000/health
```

Expected JSON:

```json
{"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
```

| Field | Meaning |
|-------|---------|
| `status` | `healthy` when artifacts loaded |
| `model_version` | Serving registry version label |
| `dataset_version` | Dataset config version used for transforms |
| `model_loaded` | `true` iff startup load succeeded; `false` means a startup artifact failed to load — check `docker logs` |

## Prediction smoke test

```bash
chmod +x scripts/smoke_test_api.sh
bash scripts/smoke_test_api.sh http://localhost:8000
```

Expected: health 200, predict 200 with all response fields, missing-amount → 422,
final line `All smoke tests passed`, exit code 0. On failure, read `docker logs fraud-test`.

## Logs

```bash
docker logs fraud-test
# or
docker compose logs -f fraud-api
```

Normal: `n_features=464 n_transforms=7`, then access logs.

Failed load: traceback with `FileNotFoundError` for a specific path during lifespan;
`/health` shows `model_loaded: false`; `/predict` returns 503.

## Latency baseline

```bash
python tests/benchmark_api_latency.py --base-url http://127.0.0.1:8000 --image fraud-api:latest
```

Paste the printed `=== Local Baselines (not production guarantees) ===` table into the
PR / this runbook. Asserts in-process P99 < 100 ms.

## Running tests inside the container

Endpoint tests (mount tests; image excludes `tests/`):

```bash
docker run --rm \
  -v "$(pwd)/tests:/app/tests:ro" \
  fraud-api \
  python -m pytest tests/test_api_endpoint.py -v
```

Parity test needs data mounted:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/tests:/app/tests:ro" \
  fraud-api \
  python -m pytest tests/test_training_serving_parity.py -v
```

## Shutdown

```bash
docker stop fraud-test && docker rm fraud-test
# or
docker compose down
docker ps   # confirm fraud-api / fraud-test is gone
```

## Common failures

| Failure | Cause | Fix |
|---------|-------|-----|
| Container exits immediately / COPY failed at build | Model artifact missing on host | Re-check prerequisites `ls`; obtain artifacts via train/calibrate |
| `/health` → `model_loaded: false` | Startup artifact missing in image | `docker logs` for FileNotFoundError; compare path to Dockerfile COPY lines |
| Port 8000 in use | Another process bound | `lsof -i :8000`; or `-p 8001:8000` |
| Build fails at pip install | Bad/unpinned dependency | Fix `requirements.txt` exact pins and rebuild |
| LightGBM shared-lib error | Missing `libgomp1` | Rebuild from this Dockerfile (apt installs it) |

## Local baselines

Measured 2026-09-08 with
`python tests/benchmark_api_latency.py --base-url http://127.0.0.1:8000 --image fraud-api:latest`
against the running container. **Local baselines only — not production guarantees.**

| Metric | Value |
|--------|-------|
| Image size (`docker images`) | 1.79 GB |
| Image size (`docker image inspect` Size) | 406.5 MB |
| Startup time | 453.2 ms |
| In-process P50 | 63.551 ms |
| In-process P99 | 86.149 ms |
| HTTP P50 | 61.697 ms |
| HTTP P99 | 94.779 ms |
| Transform P50 | 49.634 ms |
| Model predict P50 | 14.032 ms |
| Calibrate+policy P50 | 0.155 ms |
