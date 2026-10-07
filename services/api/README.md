# Fraud Inference API (lgbm_v9 / dataset_v5)

Demo / batch inference service for the registry serving model. Historical
features (D*, V*, C*, device/id fields) are **caller-supplied** — there is no
integrated feature store. At request time the API replays the full
transformation pipeline from the serving model's dataset config per
§10 of `ENGINEERING_STANDARDS.md`.

## Start

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

On startup the process loads (from `models/registry.json` `is_serving: true`,
with Candidate fallbacks for calibrator / decision policy / fitted transforms):

| Artifact | Source |
|----------|--------|
| LightGBM booster | `model_path` |
| Dataset config + transform chain | `dataset_config_path` → `scripts/feature_engineering.py` |
| Fitted requires_fit state | `fitted_transforms_path` (never refit) |
| Feature contract | tuning `manifest_path` → `feature_order` (464 cols) |
| Isotonic calibrator | `calibrator_path` |
| Cost params | `decision_policy_config` |

If any file is missing or feature counts mismatch, startup logs the error and
`POST /predict` returns **503**.

## Example request

Synthetic payload (not a real dataset row):

```bash
curl -s http://127.0.0.1:8000/predict \
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
    "D1": 5,
    "D2": null,
    "D3": 2,
    "D4": null,
    "D5": null,
    "D6": null,
    "D7": null,
    "D8": null,
    "D9": null,
    "D10": 1,
    "D11": null,
    "D12": null,
    "D13": null,
    "D14": null,
    "D15": 3,
    "C1": 1,
    "C2": 1,
    "DeviceType": "mobile",
    "DeviceInfo": "SyntheticOS",
    "id_30": "iOS",
    "id_31": "safari",
    "metadata": {"TransactionID": 999001, "notes": "demo"}
  }'
```

Example response shape:

```json
{
  "fraud_probability": 0.0123,
  "recommended_action": "approve",
  "action_costs": {
    "approve": 3.09,
    "step_up": 6.12,
    "review": 10.15,
    "decline": 123.96
  },
  "model_version": "v9",
  "dataset_version": 5,
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Validation errors

Missing required field:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ProductCD": "W"}'
# → HTTP 422
```

Wrong type:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"TransactionAmt": "abc"}'
# → HTTP 422
```

## Transformation replay

The pipeline is built by walking the inherited + current `transformations`
list in the dataset config and resolving each `function` against
`feature_engineering.py`. Non-`requires_fit` steps run as pure functions;
`requires_fit` steps call `.transform(df, params)` on the frozen objects from
the fitted-transforms pickle (**never** `.fit()` at serve time).

Swapping `is_serving` to another registry entry with a different
`dataset_config_path` rebuilds the pipeline on the next process start — no
API code changes.

## Privacy-safe logging

| Logged | Not logged |
|--------|------------|
| `request_id` | `TransactionAmt` |
| `model_version`, `dataset_version` | card*, email*, addr* |
| `recommended_action` | raw feature values |
| `latency_ms` | SHAP / explanations |

## Latency expectations

From `tests/test_latency.py` (direct `predict()` calls, not HTTP):

- Target: **P99 < 100 ms** per request on a warm process
- Typical breakdown: transforms + DataFrame prep, `Booster.predict`, then
  calibration + argmin policy (usually sub-millisecond)

Re-run the benchmark after serving changes to catch regressions.

## Health

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"healthy","model_version":"v9","dataset_version":5,"model_loaded":true}
```

## Tests

```bash
python -m pytest tests/test_api_endpoint.py -v
python -m pytest tests/test_training_serving_parity.py -v
python -m pytest tests/test_latency.py -v
```
