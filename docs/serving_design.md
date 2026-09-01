# Serving Design — lgbm_v9

## Request-to-response flow

```
1. Request arrives at POST /predict
2. Pydantic validates against TransactionInput schema
   → 422 on missing required fields or invalid types
3. Convert to dict, drop None values
4. Per-row feature engineering (inherited chain v2→v3→v4→v5):
   a. email_match: P_emaildomain == R_emaildomain → 0/1
   b. TransactionAmt_log1p, TransactionAmt_cents from TransactionAmt
   c. V*_is_null missingness indicators
   d. Dn normalization: floor(TransactionDT/86400 - D_col) for each D column
   e. If inputs are missing → output columns set to NaN, warning logged
5. Fitted lookup transforms:
   a. Load frozen lookup tables from models/lgbm_v9_optuna_tuning_fitted_transforms.pkl
   b. Apply .transform() to current row values (frequency encodings, UID aggregations)
   c. If lookup key not in table → 0.0 or NaN per encoder rules
6. Select and reorder to manifest's feature_order (464 columns)
   → Missing columns filled with NaN
   → Extra columns dropped (raw D1-D15 are inputs only, not in feature_order)
7. Apply categorical encoding per manifest dtypes and model.pandas_categorical
8. model.predict() → raw_score (float)
9. Calibrate: isotonic_regression.transform([raw_score]) → calibrated_score
10. Map calibrated_score to action via decision_policy thresholds:
    → [0.0, 0.036)     approve
    → [0.036, 0.051)   step_up
    → [0.051, 0.820)   review
    → [0.820, 1.0]     decline
11. Return PredictionResponse with request_id
```

## Feature classification summary

| Category | Count | Description |
|----------|-------|-------------|
| raw | 416 | Arrive with the transaction request |
| per_row | 38 | Computed from raw fields per request |
| fitted_lookup | 10 | Frozen training-time lookup tables |
| historical | 0 | Requires customer history / feature store |

**Total: 464 features**

## Required inputs and their sources

### Category 1: Raw (416 features)

These features must be supplied in the request body (or filled with NaN if omitted). TransactionAmt is the only required field.

`TransactionAmt`, `ProductCD`, `card1`, `card2`, `card3`, `card4`, `card5`, `card6`
`addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain`, `C1`, `C2`
`C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C10`
`C11`, `C12`, `C13`, `C14`, `M1`, `M2`, `M3`, `M4`
`M5`, `M6`, `M7`, `M8`, `M9`, `V1`, `V2`, `V3`
`V4`, `V5`, `V6`, `V7`, `V8`, `V9`, `V10`, `V11`
`V12`, `V13`, `V14`, `V15`, `V16`, `V17`, `V18`, `V19`
`V20`, `V21`, `V22`, `V23`, `V24`, `V25`, `V26`, `V27`
`V28`, `V29`, `V30`, `V31`, `V32`, `V33`, `V34`, `V35`
`V36`, `V37`, `V38`, `V39`, `V40`, `V41`, `V42`, `V43`
`V44`, `V45`, `V46`, `V47`, `V48`, `V49`, `V50`, `V51`
`V52`, `V53`, `V54`, `V55`, `V56`, `V57`, `V58`, `V59`
`V60`, `V61`, `V62`, `V63`, `V64`, `V65`, `V66`, `V67`
`V68`, `V69`, `V70`, `V71`, `V72`, `V73`, `V74`, `V75`
`V76`, `V77`, `V78`, `V79`, `V80`, `V81`, `V82`, `V83`
`V84`, `V85`, `V86`, `V87`, `V88`, `V89`, `V90`, `V91`
`V92`, `V93`, `V94`, `V95`, `V96`, `V97`, `V98`, `V99`
`V100`, `V101`, `V102`, `V103`, `V104`, `V105`, `V106`, `V107`
`V108`, `V109`, `V110`, `V111`, `V112`, `V113`, `V114`, `V115`
`V116`, `V117`, `V118`, `V119`, `V120`, `V121`, `V122`, `V123`
`V124`, `V125`, `V126`, `V127`, `V128`, `V129`, `V130`, `V131`
`V132`, `V133`, `V134`, `V135`, `V136`, `V137`, `V138`, `V139`
`V140`, `V141`, `V142`, `V143`, `V144`, `V145`, `V146`, `V147`
`V148`, `V149`, `V150`, `V151`, `V152`, `V153`, `V154`, `V155`
`V156`, `V157`, `V158`, `V159`, `V160`, `V161`, `V162`, `V163`
`V164`, `V165`, `V166`, `V167`, `V168`, `V169`, `V170`, `V171`
`V172`, `V173`, `V174`, `V175`, `V176`, `V177`, `V178`, `V179`
`V180`, `V181`, `V182`, `V183`, `V184`, `V185`, `V186`, `V187`
`V188`, `V189`, `V190`, `V191`, `V192`, `V193`, `V194`, `V195`
`V196`, `V197`, `V198`, `V199`, `V200`, `V201`, `V202`, `V203`
`V204`, `V205`, `V206`, `V207`, `V208`, `V209`, `V210`, `V211`
`V212`, `V213`, `V214`, `V215`, `V216`, `V217`, `V218`, `V219`
`V220`, `V221`, `V222`, `V223`, `V224`, `V225`, `V226`, `V227`
`V228`, `V229`, `V230`, `V231`, `V232`, `V233`, `V234`, `V235`
`V236`, `V237`, `V238`, `V239`, `V240`, `V241`, `V242`, `V243`
`V244`, `V245`, `V246`, `V247`, `V248`, `V249`, `V250`, `V251`
`V252`, `V253`, `V254`, `V255`, `V256`, `V257`, `V258`, `V259`
`V260`, `V261`, `V262`, `V263`, `V264`, `V265`, `V266`, `V267`
`V268`, `V269`, `V270`, `V271`, `V272`, `V273`, `V274`, `V275`
`V276`, `V277`, `V278`, `V279`, `V280`, `V281`, `V282`, `V283`
`V284`, `V285`, `V286`, `V287`, `V288`, `V289`, `V290`, `V291`
`V292`, `V293`, `V294`, `V295`, `V296`, `V297`, `V298`, `V299`
`V300`, `V301`, `V302`, `V303`, `V304`, `V305`, `V306`, `V307`
`V308`, `V309`, `V310`, `V311`, `V312`, `V313`, `V314`, `V315`
`V316`, `V317`, `V318`, `V319`, `V320`, `V321`, `V322`, `V323`
`V324`, `V325`, `V326`, `V327`, `V328`, `V329`, `V330`, `V331`
`V332`, `V333`, `V334`, `V335`, `V336`, `V337`, `V338`, `V339`
`id_01`, `id_02`, `id_03`, `id_04`, `id_05`, `id_06`, `id_07`, `id_08`
`id_09`, `id_10`, `id_11`, `id_12`, `id_13`, `id_14`, `id_15`, `id_16`
`id_17`, `id_18`, `id_19`, `id_20`, `id_21`, `id_22`, `id_23`, `id_24`
`id_25`, `id_26`, `id_27`, `id_28`, `id_29`, `id_30`, `id_31`, `id_32`
`id_33`, `id_34`, `id_35`, `id_36`, `id_37`, `id_38`, `DeviceType`, `DeviceInfo`

### Category 2: Per-row engineered (38 features)

Computed by the API from raw inputs using `scripts/feature_engineering.py` functions:

| Feature | Transform | Inputs |
|---------|-----------|--------|
| `email_match` | email_match | P_emaildomain, R_emaildomain |
| `TransactionAmt_log1p` | transaction_amt_features | TransactionAmt |
| `TransactionAmt_cents` | transaction_amt_features | TransactionAmt |
| `V1_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V2_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V3_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V4_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V5_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V6_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V7_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V8_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V9_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V10_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V11_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V14_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V15_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V16_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V17_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V18_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V19_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V20_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V21_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `V22_is_null` | missingness_indicators | V1, V2, V3, V4, V5, V6, ... |
| `D1n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D2n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D3n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D4n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D5n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D6n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D7n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D8n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D9n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D10n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D11n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D12n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D13n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D14n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |
| `D15n` | normalize_d_columns | TransactionDT, D1, D2, D3, D4, D5, ... |

### Category 3: Fitted lookup (10 features)

Applied using objects serialized during training (`models/lgbm_v9_optuna_tuning_fitted_transforms.pkl`):

| Feature | Transform | Inputs |
|---------|-----------|--------|
| `card1_freq` | frequency_encoding | card1, addr1, P_emaildomain, card2 |
| `addr1_freq` | frequency_encoding | card1, addr1, P_emaildomain, card2 |
| `P_emaildomain_freq` | frequency_encoding | card1, addr1, P_emaildomain, card2 |
| `card2_freq` | frequency_encoding | card1, addr1, P_emaildomain, card2 |
| `uid_tx_count` | uid_aggregations | card1, addr1, TransactionAmt |
| `uid_amt_mean` | uid_aggregations | card1, addr1, TransactionAmt |
| `uid_amt_std` | uid_aggregations | card1, addr1, TransactionAmt |
| `DeviceInfo_freq` | identity_frequency_encoding | DeviceInfo, id_30, id_31 |
| `id_30_freq` | identity_frequency_encoding | DeviceInfo, id_30, id_31 |
| `id_31_freq` | identity_frequency_encoding | DeviceInfo, id_30, id_31 |

### Category 4: Historical (0 features)

No features require real-time customer history or trailing-window computation. The API can serve single transactions with full fidelity when all raw inputs are provided.

UID aggregation features (`uid_tx_count`, `uid_amt_mean`, `uid_amt_std`) use **frozen training-split statistics**, not live trailing windows — they are Category 3 (fitted lookup), not historical.

## Training-serving parity

The API guarantees identical transformations to training by:
1. Importing the SAME functions from `scripts/feature_engineering.py` (not re-implemented)
2. Replaying the full inherited transformation chain via `build_dataset.collect_inherited_transformations`
3. Using the SAME fitted transform objects serialized during training
4. Reordering features to the SAME `feature_order` from the manifest
5. Applying the SAME categorical encoding as training (`model.pandas_categorical`)
6. Verified by `tests/test_training_serving_parity.py` with tolerance 1e-6 on 5 validation rows

## Failure modes

| Failure | Behavior | HTTP status |
|---------|----------|-------------|
| Model binary missing | Startup fails, health returns 503 | 503 |
| Calibrator missing | Raw score returned as fraud_probability, warning added | 200 |
| Decision policy missing | Probability returned, action defaults to approve, warning added | 200 |
| TransactionDT missing | Dn features set to NaN | 200 + warning |
| D columns missing | Dn features set to NaN | 200 + warning |
| Unseen categorical value | Frequency encoding returns 0.0, LightGBM handles unseen categories | 200 |
| Extreme TransactionAmt | Model scores it normally | 200 |
| Fitted transforms pickle missing | Startup fails (required for v9) | 503 |

## Missing-history behavior

Category 4 (historical) count is **0** for lgbm_v9. The API can serve single transactions with full fidelity when callers provide the raw transaction fields. No external feature store is required.

When optional raw fields are omitted, the API fills corresponding model features with NaN. LightGBM handles missing values natively. Warnings are returned in the response `warnings` array.

## Latency budget

| Component | Target | Notes |
|-----------|--------|-------|
| Total per-request | < 200ms (p95) | Deepa's target |
| Schema validation | < 1ms | Pydantic |
| Feature engineering | < 5ms | Per-row computations |
| Fitted lookups | < 2ms | Dict lookups |
| model.predict() | < 10ms | Single-row LightGBM inference |
| Calibration + policy | < 1ms | Isotonic transform + threshold compare |
| Overhead | < 5ms | FastAPI, serialization |

Benchmarked in-process via `tests/test_api_latency.py` (TestClient, 100 requests).
