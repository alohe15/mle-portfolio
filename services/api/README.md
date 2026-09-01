# Fraud Detection API

Serves the frozen lgbm_v9 model. Returns calibrated fraud probability, recommended action, and a request ID for tracing.

## Start

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --app-dir .
```

## Endpoints

### POST /predict

Score a transaction.

**Minimal request:**
```json
{"TransactionAmt": 75.00}
```

**Full request (with optional fields):**
```json
{
  "TransactionAmt": 75.00,
  "TransactionDT": 86400,
  "card1": 1000,
  "card4": "visa",
  "ProductCD": "W",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com",
  "D1": 14.0,
  "C1": 1.0
}
```

**Response:**
```json
{
  "fraud_probability": 0.0231,
  "raw_score": 0.0089,
  "recommended_action": "approve",
  "model_version": 9,
  "dataset_version": 5,
  "policy_version": 1,
  "request_id": "a1b2c3d4-...",
  "warnings": []
}
```

### GET /health

```json
{"status": "healthy", "model_version": 9, "dataset_version": 5, "policy_version": 1, "n_features": 464}
```

## What the API computes vs what you provide

| Category | Count | Examples | Source |
|----------|-------|----------|--------|
| **Raw (you provide)** | 416 | TransactionAmt, card1-6, V-features, id_ features | Request body |
| **Per-row (API computes)** | 38 | email_match, Dn normalization, missingness flags | Computed from your raw fields |
| **Fitted lookup (API applies)** | 10 | card1_freq, uid_tx_count, DeviceInfo_freq | Frozen training-time lookup tables |
| **Historical (optional)** | 0 | — | Not required for lgbm_v9 |

See `services/api/feature_classification.json` for the full mapping.

Extra request fields are accepted via Pydantic `extra="allow"`. Fields not in the manifest are ignored; V-features and id_ columns do not need explicit schema listing.

## Validation errors

| Error | Cause | HTTP status |
|-------|-------|-------------|
| `TransactionAmt` required | Missing from request body | 422 |
| Invalid type | e.g. card1 sent as string | 422 |
| Model not loaded | Binary missing or startup failed | 503 |

## Privacy

Logs contain: request_id, model_version, recommended_action, fraud_probability, latency_ms.
Logs do NOT contain: TransactionAmt, card numbers, email addresses, device info, or any raw input.

## Tests

```bash
python -m pytest tests/test_api_endpoint.py tests/test_api_latency.py tests/test_training_serving_parity.py -v
```

## Model loading

At startup, the API reads `models/registry.json`, finds the serving entry, and loads:
model binary, manifest (feature order), fitted transforms, calibrator, and decision policy.
All artifacts are loaded once and held in memory. No per-request I/O.

## Module layout

| Module | Responsibility |
|--------|----------------|
| `app.py` | FastAPI routes, lifespan, privacy-safe logging |
| `schemas.py` | Request/response Pydantic models |
| `model_loader.py` | Registry-driven artifact loading at startup |
| `predictor.py` | Transformation pipeline + inference |
| `feature_classification.json` | Per-feature source contract (464 features) |
