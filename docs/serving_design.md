# Serving design — lgbm_v9 / dataset_v5

End-to-end flow for the demo inference API. Contract:
`ENGINEERING_STANDARDS.md` §10. Decision policy: Phase 3
(`configs/decision_policy_v1.json`). Lifecycle: Phase 4 Candidate registration.

## 1. Request

Caller `POST /predict` with raw transaction fields (same shape a payment event
would have). Two input categories:

| Category | Role |
|----------|------|
| Current-transaction | Fields available from the payment event itself |
| Historical / device | Fields that need a feature store in production |

This version has **no feature store**. Historical fields are caller-supplied
so the API can still demonstrate full transform replay and parity.

## 2. Validation

`TransactionRequest` (Pydantic) enforces:

- `TransactionAmt` required, `> 0` (negatives → 422)
- Known current-transaction fields typed when present
- Extra keys allowed (`extra="allow"`) so C/V/D/id/M/device columns can be
  sent without enumerating hundreds of optional declarations
- Optional `metadata.TransactionID` for caller correlation (never returned)

Invalid types / missing required fields → **422** with Pydantic `detail`.

## 3. Current-transaction features

| Field | Type | Source in production |
|-------|------|----------------------|
| `TransactionAmt` | float | Payment authorization |
| `ProductCD` | string | Payment authorization |
| `card1`–`card6` | float/string | Card token / BIN attributes |
| `addr1`, `addr2` | float | Billing / shipping codes |
| `P_emaildomain`, `R_emaildomain` | string | Checkout form |
| `dist1`, `dist2` | float | Distance features on the event |
| `TransactionDT` | float | Event timestamp (seconds from ref) |

## 4. Historical-feature lookup

| Field family | Type | Source in production |
|--------------|------|----------------------|
| `D1`–`D15` | float | Cardholder history deltas (feature store) |
| `C1`–`C14` | float | Velocity / count aggregates |
| `V1`–`V339` | float | Vesta V-block aggregates |
| `M1`–`M9` | string/category | Match flags from identity graph |
| `id_01`–`id_38` | mixed | Device / identity attributes |
| `DeviceType`, `DeviceInfo` | string | Device fingerprinting |

Production sketch: streaming job writes keyed features
`(card1, addr1)` / device id → online store; the API enriches the event
before predict. Requires-fit outputs (`*_freq`, `uid_*`) are **not**
looked up — they are computed at serve time from frozen training statistics.

## 5. Transformation pipeline

Built at startup from the serving entry's `dataset_config_path` by walking
`collect_all_transformations` (inherited chain + current), resolving each
`function` from `scripts/feature_engineering.py`:

| Order | Name | requires_fit | Function |
|-------|------|--------------|----------|
| 1 | email_match | no | `create_email_match` |
| 2 | transaction_amt_features | no | `create_transaction_amt_features` |
| 3 | missingness_indicators | no | `create_missingness_indicators` |
| 4 | frequency_encoding | **yes** | `FrequencyEncoder` + pkl state |
| 5 | uid_aggregations | **yes** | `UidAggregator` + pkl state |
| 6 | normalize_d_columns | no | `normalize_d_columns` |
| 7 | identity_frequency_encoding | **yes** | `FrequencyEncoder` + pkl state |

Non-fit steps: `fn(df, params)`. Fit steps: `fitted.transform(df, params)` —
**never** `.fit()` at inference (§2g / §10).

Config-driven: changing `is_serving` to another model/dataset rebuilds this
list on restart without API code edits.

## 6. Feature selection and ordering

After transforms, columns are selected and reordered to the tuning
manifest `feature_order` (**464** features for v9). Missing → `np.nan`.
Extras discarded. Categoricals encoded with `model.pandas_categorical`
and `"__MISSING__"` for nulls (training contract).

## 7. Model score

`lgb.Booster.predict(model_input)` → raw score (same path as eval scripts).

## 8. Calibration

Phase 3 isotonic calibrator (`models/lgbm_v9_calibrator.pkl`):
`calibrator.predict([raw_score])[0]` → calibrated probability in `[0, 1]`.

## 9. Decision policy

`optimal_action(p, TransactionAmt, cost_params)` from `evals/cost_model.py`
computes approve / step_up / review / decline expected costs and returns the
argmin action plus the full cost vector.

## 10. Response

Returned: `fraud_probability`, `recommended_action`, `action_costs`,
`model_version` (`"v9"`), `dataset_version` (`5`), `request_id`.

Excluded: SHAP, raw inputs, card/email/address/amount echoes, metadata.

## 11. Failure modes

| Condition | HTTP | Behavior |
|-----------|------|----------|
| Artifacts failed to load | 503 | `/predict` unavailable |
| Schema / type errors | 422 | Pydantic detail |
| Unexpected predict exception | 500 | `{request_id}` only |
| Missing historical columns | 200 | NaN-filled; model runs |

## 12. Missing-history behavior

Omitted D/V/C (and other) columns are NaN after selection. The model was
trained with nonzero null rates on many of these columns, so predict does not
crash. Accuracy degrades: predictions without historical features are
equivalent to using primarily current-transaction signals, which reduces
discrimination.

## 13. Training-serving parity

Enforced by:

- Same dataset config → same ordered transform list and FE callables
- Same fitted-transforms pickle (no serve-time fit)
- Same `feature_order` and booster file
- Same calibrator + cost params

`tests/test_training_serving_parity.py` scores validation rows through the
training feature path and the serving `predict()` path and asserts raw scores,
calibrated probabilities, and actions match within **1e-6**.

## 14. Latency profile

`tests/test_latency.py` times `predict()` directly (no HTTP):

| Stage | What is measured |
|-------|------------------|
| Transforms + frame prep | pipeline + `build_model_input` |
| Model predict | `Booster.predict` |
| Calibration + policy | isotonic + `optimal_action` |

Expectation: overall **P99 < 100 ms** on a warm process; model predict usually
dominates, with transforms second and calibration/policy negligible.

## Required input field table (summary)

| Field | Type | Category | Production source |
|-------|------|----------|-------------------|
| TransactionAmt | float (>0) | current | Payment event |
| ProductCD | string | current | Payment event |
| card1–card6 | float/string | current | Card attributes |
| addr1, addr2 | float | current | Address codes |
| P/R_emaildomain | string | current | Checkout |
| dist1, dist2 | float | current | Event geometry |
| TransactionDT | float | current | Event time |
| D1–D15 | float | historical | Feature store |
| C1–C14 | float | historical | Feature store |
| V1–V339 | float | historical | Feature store |
| M1–M9 | mixed | historical | Identity graph |
| id_* / Device* | mixed | historical / device | Device & identity store |
| metadata.TransactionID | int | optional | Caller logging only |
