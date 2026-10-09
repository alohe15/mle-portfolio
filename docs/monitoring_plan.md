# Monitoring plan — lgbm_v9 / dataset_v5

Status of all numeric thresholds: **proposed**. None are confirmed production
limits. Config: `evals/monitoring_config.json`. Reference baseline:
`evals/reference_profile.json` (training split of dataset_v5, first 70% by
`TransactionDT`). Serving contract: `TransactionRequest` /
`TransactionResponse` in `services/api/schemas.py`.

## Overview

The serving model is `lgbm_v9` (`is_serving: true`) on `dataset_v5`. Every
scored transaction produces a calibrated `fraud_probability` and a
`recommended_action` in `{approve, step_up, review, decline}` via the Phase 3
argmin cost policy. Monitoring exists to notice when live traffic stops
looking like the traffic the model was trained on, and — once labels arrive —
whether the model and policy still catch fraud at the reference rate.

Two tiers:

| Tier | When it fires | Data required | What it can catch |
|------|---------------|---------------|-------------------|
| **Tier 1 — immediate** | Same day as the comparison window (batch of scored traffic) | Serving inputs + model outputs only | Broken pipelines, schema changes, missingness shocks, gross covariate shift, score/action-volume shocks |
| **Tier 2 — delayed** | After `isFraud` labels are available | Serving inputs + outputs + labels | Real performance degradation: PR-AUC, precision/recall, fraud dollars captured, false-positive / friction rate |

Tier 1 is a **tripwire**. It is allowed to miss subtle attacks. Tier 2 is the
**authoritative** signal that the model is still doing its job.

Comparison window for the offline prototype: a single batch (the temporal
test split, simulating production traffic after the model was frozen). In
production the same script should run on a rolling window (daily batch of
scored transactions; weekly/monthly labeled windows).

Reference window: the **training split** (70% earliest `TransactionDT`). That
is the population the fitted transforms, booster, and calibrator were fit
against. The monitoring code loads those artifacts through
`services/api/model_loader.py` (registry-driven, §10) and never refits
`requires_fit` transforms (§2g).

## Why low drift does not guarantee good predictions

Tier 1 measures whether the **input distribution** and **score/action mix**
moved. It does not measure whether the **mapping from features to fraud**
still holds.

Concrete example. A new fraud ring targets a niche merchant category that is
~0.2% of volume but high ticket size. Feature histograms barely move: PSI on
the top-20 numeric features stays under 0.05, `TransactionAmt` PSI is quiet
because the new pattern is a thin tail, calibrated-score PSI is quiet because
those rows were never well-represented in training, and action-band volumes
move by a couple of percentage points — under the 30% relative threshold.
The model assigns the new pattern scores similar to ordinary high-amount
e-commerce, so `recommended_action` stays `approve` or `step_up`.

Thirty to ninety days later, chargebacks land. Tier 2 shows a drop in fraud
dollars captured (the new ring's dollars were approved) and a drop in PR-AUC
on the labeled window. That is the first reliable alarm.

This is why **Tier 2 is authoritative** and why a green Tier 1 dashboard is
not a performance guarantee. Tier 1 catches gross failures (wrong schema,
null explosion, score collapse). It does not catch low-volume, high-value
concept drift.

## Signal reference table

| Signal | Tier | What it measures | Why it matters | Calculation | Comparison window | Proposed threshold | Owner | Investigation steps | Response |
|--------|------|------------------|----------------|-------------|-------------------|--------------------|-------|---------------------|----------|
| `schema_missing_columns` | 1 | Model-feature columns absent from the scored frame vs the 464-column serving contract | A dropped column is silently NaN-filled by the API; the model scores a different object than it was trained on | Count of `feature_order` names not present; threshold 0 | Same-day scored batch vs training reference schema | 0 missing → **critical** | ML Eng + Data Eng | 1. Diff comparison columns vs `models/lgbm_v9_optuna_tuning_manifest.json` `feature_order`. 2. Check upstream pipeline / feature-store job for renamed or dropped fields. 3. Confirm the API transform pipeline still emits `requires_fit` outputs (`*_freq`, `uid_*`). 4. Replay one failing row through `POST /predict` and inspect missing keys. | Page on-call. Do not promote a replacement model until the column is restored or a new dataset version is trained. Block silent NaN-fill in the logger if a previously-present feature disappears. |
| `schema_extra_columns` | 1 | Columns on the comparison payload that are not in the 464-feature contract (after ignoring `isFraud`, `TransactionID`, `TransactionDT`) | Extra fields are allowed by `TransactionRequest(extra="allow")` but a sudden new name often means a pipeline joined the wrong table | Count of unexpected names | Same-day batch | 0 extras → **warning** | Data Eng | 1. List the extra names. 2. Check the latest producer schema / dbt model / feature-store entity. 3. Confirm they are not a typo of a real feature (`card1` vs `Card1`). | Log and ticket. Do not page. Ignore known log-only fields after adding them to the ignore list; do not pass unknown fields into `feature_order`. |
| `schema_dtype_mismatch` | 1 | A contract column changed kind (numeric ↔ string/bool/datetime) | LightGBM and the calibrator assume numeric/category encodings from training; a stringified float becomes NaN after `to_numeric(..., errors="coerce")` and looks like a missingness shock | Count of columns whose canonical kind disagrees with the reference dtype | Same-day batch | 0 mismatches → **critical** | Data Eng | 1. Identify the column and the new pandas dtype. 2. Inspect a sample of raw values (CSV quoting, `"null"` strings, mixed types). 3. Trace the producer: warehouse CAST, JSON serialization, or API client. | Page. Fix the producer or add an explicit parse in a **new** dataset version — do not silently recast in `app.py`. Re-score the window after the fix and confirm missingness returns to baseline. |
| `missingness_rate_change` | 1 | Absolute change in null rate per model feature (464 features + `TransactionAmt`) | Identity/device fields already have high null rates; a jump usually means a join failed or a vendor feed stopped | `abs(null_rate_comparison − null_rate_reference)` per column; flag if any column exceeds the threshold | Same-day batch vs training null rates | 0.05 absolute → **warning** | Data Eng | 1. Sort flagged columns by absolute delta. 2. If a whole family moved together (`id_*`, `Device*`, `V*`), check that feature-store lookup. 3. If a single column moved, check its producer. 4. Confirm `TransactionDT` is still populated (Dn features depend on it). | Restore the feed. If the new null rate is real (vendor change), document it and watch Tier 2; do not retrain on a broken feed. |
| `feature_distribution_psi` | 1 | Population Stability Index on the top-20 numeric features by LightGBM gain (fallback: highest-variance numerics) | Large PSI means the model is scoring a different population; trees split on training cutpoints that may no longer exist | 10-bin PSI using **training** quantile edges stored in the reference profile; flag any selected feature with PSI > threshold | Same-day batch vs training histograms | 0.2 per feature → **warning** | ML Eng | 1. Read `detail.flagged` for the feature list. 2. Plot comparison vs reference histogram for each flagged feature. 3. Segment by `ProductCD` / amount quartile — is it mix shift or a true shape change? 4. Check for a product launch, geo expansion, or holiday. | If mix shift (new product mix): update dashboards, do not retrain yet. If a feature is corrupted (impossible range): treat as a pipeline incident. If shape change persists 2+ weekly windows **and** Tier 2 also degrades: schedule retraining on a new dataset version. |
| `categorical_new_levels` | 1 | Category values on serving categoricals (`ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, plus other model categoricals) that were not in training | New `ProductCD` or card brand is a contract change; thousands of new `DeviceInfo` strings are expected noise | Count of unseen levels vs the reference proportion map | Same-day batch | 0 new serving-schema levels → **warning** | ML Eng | 1. List new levels (report caps at 20 per column). 2. For `ProductCD` / `card4` / `card6`, confirm with product/payments that the code is real. 3. For email domains, check for a new consumer ISP vs a logging bug (`None` vs `"nan"`). 4. Confirm the API maps unseen categories to `__MISSING__` via `pandas_categorical`, not to a random code. | New legitimate level: accept the warning; the API already maps unseen cats to `__MISSING__`. If a training categorical was overwritten with integers, treat as dtype/schema incident. |
| `score_distribution_psi` | 1 | PSI on calibrated `fraud_probability` vs the training score histogram | The decision policy is a function of `p`; if scores shift, action volumes and expected cost shift even with stable features | 10-bin PSI on calibrator output using training edges | Same-day batch | 0.1 → **warning** | ML Eng | 1. Compare score mean/P50/P95 to the reference `scores` block. 2. Check raw vs calibrated means — calibrator clip can pile mass at 0/1. 3. Join with missingness and PSI flags; a null explosion will drag scores. 4. Confirm the loaded calibrator path is `models/lgbm_v9_calibrator.pkl`. | If features look stable but scores moved: suspect a model-file / calibrator swap. If features moved too: follow feature PSI. Do not retune cost params to “fix” volume until you know why `p` moved. |
| `action_band_volume_change` | 1 | Relative change in the share of `approve` / `step_up` / `review` / `decline` vs training | Review capacity is 200/day (`review_capacity_per_day`); a spike in `review` or `decline` is an ops incident even if PR-AUC is unknown | `abs(p_obs − p_ref) / p_ref` per action; flag any band above threshold | Same-day batch | 0.30 relative → **warning** | Fraud Ops + ML Eng | 1. Read flagged bands and the observed proportions. 2. Overlay amount mix — argmin sends more high-`a` rows to review/decline. 3. Check score PSI. 4. Compare review count / day against 200. 5. Sample declined-legit candidates if labels exist. | Ops: add review headcount or temporarily raise friction only with a written exception. Eng: if the shift is a score bug, roll back the serving artifact. Do not change `cost_params` in place — that is a new policy version. |
| `amount_distribution_psi` | 1 | PSI on `TransactionAmt` vs training | Amount enters the cost model linearly; a shift in `a` changes action mix even at a fixed `p` | 10-bin PSI using training amount edges | Same-day batch | 0.2 → **warning** | Fraud Ops | 1. Compare amount P50/P95 to reference. 2. Segment by `ProductCD`. 3. Check for FX, denomination, or unit errors (cents vs dollars). 4. Confirm `TransactionAmt > 0` (API rejects non-positive). | Unit errors: page and halt scoring until fixed. Real mix shift: expect action-band movement; wait for Tier 2 before retraining. |
| `pr_auc_drop` | 2 | Absolute drop in PR-AUC vs training reference | Primary ranking metric under ~3.5% prevalence; a drop means the model is ranking fraud worse | `average_precision_score(isFraud, calibrated_p)` on the labeled window; flag if `ref − obs ≥ threshold` | Labeled window (chargebacks typically 30–90 days after the transaction) vs training PR-AUC | 0.05 absolute → **critical** | ML Eng | 1. Confirm label completeness (chargeback lag — do not judge a 7-day window). 2. Check fraud-rate shift; prevalence change moves PR-AUC. 3. Slice PR-AUC by amount quartile and `ProductCD`. 4. Review error analysis on the missed high-amount fraud. | If lag-adjusted drop holds for **two consecutive labeled windows**: open a retrain. Do not use the test split to tune. Freeze serving until a Candidate model beats v9 on a new validation window. |
| `precision_drop` | 2 | Precision at the frozen training operating point (prevalence-matched calibrated-score threshold stored in the reference profile) | Precision is the fraction of flagged traffic that is actually fraud; a drop burns reviewer time | Precision of `score ≥ operating_threshold` vs training | Labeled window | 0.10 absolute → **warning** | Fraud Ops + ML Eng | 1. Confirm the operating threshold was not recomputed on the comparison window. 2. Check FPR and fraud-rate shift. 3. Sample false positives in the flagged set. | If precision falls but recall holds: reviewers see more junk — tighten only after a policy review. If both fall: treat as model degradation (see PR-AUC). |
| `recall_drop` | 2 | Recall at the same frozen operating point | Recall is the share of fraud the threshold still catches | Recall of `score ≥ operating_threshold` vs training | Labeled window | 0.10 absolute → **warning** | ML Eng | 1. List missed fraud (false negatives) by amount. 2. Check whether they sit in a new `ProductCD` / email domain / device cluster. 3. Cross-check fraud-dollars-captured — recall can look stable while dollars drop. | Low-volume pattern: gather examples for a new dataset version. Do not add ad-hoc rules inside `app.py`. |
| `fraud_value_captured_drop` | 2 | Relative drop in the share of fraud dollars caught by the argmin policy, using the same catch rates as serving (`approve=0`, `step_up=step_up_stop_rate`, `review=review_catch_rate`, `decline=1`) | Dollar-weighted catch is what the cost model optimizes; count-based recall can hide missed high-amount fraud. Compared as **capture rate** (caught / total fraud dollars in the window) so a daily batch is comparable to the 413k-row training reference | `(ref_rate − obs_rate) / ref_rate` where rate = `sum(isFraud × catch_rate(action) × TransactionAmt) / sum(isFraud × TransactionAmt)` | Labeled window | 0.15 relative → **critical** | Fraud Ops + ML Eng | 1. Break down `fraud_dollars_by_action`. 2. If `approve` fraud dollars rose, the policy is waving through high-`a` fraud. 3. Inspect those rows' scores vs amounts. 4. Confirm cost params still match `configs/decision_policy_v1.json`. | Critical. If confirmed on a mature labeled window: Candidate retrain + policy review. Short-term: raise manual review on the affected merchant/category **outside** the model, with an expiry date. |
| `false_positive_rate_increase` | 2 | Increase in FPR at the frozen operating point (and customer friction = non-approve rate on legit traffic) | Friction is the cost paid by legitimate customers; FPR up 2pp is a lot of extra step-ups/reviews/declines | `P(score ≥ t \| isFraud=0)` comparison vs training; flag absolute increase | Labeled window | 0.02 absolute → **warning** | Fraud Ops | 1. Convert FPR to extra declined/reviewed legit txns per day. 2. Check amount mix (high `a` legit is more often declined). 3. Confirm labels are not still filling in (early chargeback windows look like extra FPs). | If friction is real: policy sensitivity run (`evals/sensitivity_analysis.py`) on validation-like traffic, not a silent threshold edit. Communicate to CX before any decline-rate change. |
| `fraud_rate_shift` | 2 | Absolute change in labeled fraud prevalence vs training (~3.5%) | Prevalence shift moves PR-AUC, precision, and action values even with a perfect model | `abs(fraud_rate_obs − fraud_rate_ref)` | Labeled window | 0.01 absolute → **warning** (default; not a production-confirmed cut) | Fraud Ops | 1. Confirm the window is lag-complete. 2. Check for a known attack wave or a labeling pipeline gap. 3. Recompute Tier 2 metrics prevalence-adjusted before calling a model failure. | Prevalence-only change: do not retrain solely on this. Attack wave: keep the model, add ops capacity, and watch fraud-dollars-captured. Labeling gap: fix labels before judging the model. |

## Tier 1 signals (immediate)

Tier 1 runs on every comparison sample. It does **not** need `isFraud`.

Schema checks use the **serving model-feature contract** (`feature_order`, 464
columns) after the same frozen transform replay the API uses. They do **not**
require training-only columns (`isFraud`, `TransactionID`). `TransactionRequest`
allows extra keys; extras are a warning, not a page. Missing historical
fields (C/V/D/id) are valid at serve time and are NaN-filled — they show up
as missingness, not as `schema_missing_columns`.

Missingness is computed on the transformed frame, per feature, including
`TransactionAmt` (already in `feature_order`). The proposed 5pp absolute cut
is large enough to ignore ordinary sampling noise on 0/1 identity fields and
small enough to catch a broken join.

Feature PSI uses LightGBM **gain** importance from the serving booster and
the training quantile edges stored in the reference profile, so the
comparison is not free to pick friendlier bins. Score PSI uses calibrated
probabilities (`TransactionResponse.fraud_probability`), not raw LightGBM
margins. Action bands use `optimal_action` from `evals/cost_model.py` with
`configs/decision_policy_v1.json` — the same function as the API.

Amount PSI is separate from feature PSI because `TransactionAmt` is both a
model feature and a cost-model input. A silent unit change (cents vs dollars)
can look like a modest feature PSI and a violent action-band shift.

## Tier 2 signals (delayed)

Chargebacks on card-not-present traffic typically arrive **30–90 days** after
the transaction. A labeled window that is younger than that is incomplete:
fraud rate is understated, precision looks high, and FPR looks high because
true fraud has not been tagged yet. Do not page on Tier 2 until the window
is lag-mature, or explicitly score a “partial labels” caveat.

Precision and recall in this plan are measured at the **frozen training
operating point**: the prevalence-matched calibrated-score threshold stored
in `reference_profile.json` → `labels.operating_threshold`. The comparison
window must not pick a new threshold. That matches how we would notice
production degradation at a fixed policy.

Fraud value captured is **not** “dollars with `isFraud=1` and action ≠
approve”. It uses the same catch-rate weights as `evals/per_transaction_policy.py`
because that is how serving claims to stop fraud (step-up does not catch
100%; review does not catch 100%; approve catches 0%). The alert compares
**capture rate** (caught dollars / total fraud dollars in the window), not
raw dollar totals, so a 1-day batch can be compared to the training
reference.

## Alert routing

All thresholds remain **proposed** until business review.

| Severity | Who | Channel | Timeline |
|----------|-----|---------|----------|
| **Warning** (Tier 1) | Data Eng (schema/missingness/amount) or ML Eng (PSI/scores/actions) | Ticket + daily digest, not a page | Acknowledge next business day. If the same signal fires **3 consecutive daily windows**, escalate to critical-severity review even if the metric is still “warning”. |
| **Warning** (Tier 2) | ML Eng + Fraud Ops | Ticket with labeled-window id and lag note | 5 business days to confirm lag-complete labels and either close or promote to a retrain discussion. |
| **Critical** (schema / dtype) | On-call Data Eng, secondary ML Eng | Page immediately | 15 minutes to confirm; 1 hour to disable scoring or pin a known-good producer. Model file is not swapped without a Candidate registry entry. |
| **Critical** (PR-AUC / fraud dollars) | ML Eng + Fraud Ops + the promotion owner (Deepa, per the v9 Candidate entry) | Page during business hours; ticket after hours if the window is still filling | 1 business day to confirm lag. If confirmed: start the retrain trigger below. Serving stays on v9 until a new Candidate beats it. |

Escalation: warning → 3 daily repeats → treat as critical investigation.
Critical schema → if unfixed in 4 hours → halt `/predict` behind the load
balancer rather than NaN-filling a hollow payload.

## Retraining triggers

Retrain (new **dataset** version if features/feeds changed, new **model**
version always) when **all** of the following hold:

1. The labeled window is lag-complete (chargebacks 30–90 days in).
2. A Tier 2 critical signal (`pr_auc_drop` or `fraud_value_captured_drop`)
   fires on **two consecutive** mature windows.
3. Prevalence shift alone does not explain the drop (`fraud_rate_shift`
   investigated and either small or adjusted for).

Investigate **upstream, do not retrain**, when:

- Any Tier 1 schema or dtype critical fires — the data is wrong.
- Missingness jumps in a whole feature family — restore the join first.
- Amount PSI fires with a unit-change story.
- Action bands move but score PSI is flat — look at `TransactionAmt` mix and
  cost params, not the booster.

A single bad day of PSI without Tier 2 confirmation is **not** a retrain.
v9 was locked after Phase 2; casual retrains throw away that lock.

## Limitations

This monitoring does **not** catch:

- **Adversarial adaptation.** An attacker who stays inside historical
  feature ranges and below volume thresholds will not move PSI, scores, or
  action mix. Only labeled dollars (Tier 2) and offline case review catch
  that, and only after lag.
- **Gradual concept drift below the PSI cut.** A slow 0.15 PSI creep each
  month never trips 0.2. Watch rolling plots, not only binary alerts.
- **Correlated feature shifts.** PSI is computed **per feature**. A rotation
  that preserves each margin but changes the joint (e.g. `C13` and `C1`
  swapping roles) can have low per-feature PSI and still break the model.
- **Calibrator staleness without score PSI.** If raw scores drift but the
  isotonic map piles them into similar calibrated bins, score PSI can look
  calm while ranking quality dies — another reason Tier 2 is authoritative.
- **Policy-parameter error.** Wrong `cost_params` change actions without
  changing scores. Action-band monitoring sees the symptom; it cannot tell
  model drift from someone editing `decision_policy_v1.json`.
- **Incomplete labels.** Early windows look like FPR spikes and recall
  collapses. Treat 30–90 day lag as part of the signal definition.
- **Training-split ranking optimism.** The reference PR-AUC is computed on the
  same rows the booster was fit on, so it is close to 1.0. A later window
  (including the temporal test split used as a simulated production batch)
  will almost always show a large `pr_auc_drop`. Treat that drop as a
  reminder that the ranking baseline is in-sample; for production, prefer a
  frozen **out-of-sample** labeled window (validation, after lag) as a second
  reference once business agrees. Action-band and feature PSI are still
  meaningful against train because they do not use `isFraud`.

## How to run

```bash
python evals/build_reference_profile.py
python evals/generate_drift_report.py evals/reference_profile.json <comparison.parquet> --labels
```

Artifacts: `evals/reference_profile.json` (committed, aggregates only),
`evals/monitoring_config.json` (proposed thresholds), JSON reports under
`reports/monitoring/` (gitignored). The comparison sample for the Phase 7
offline proof is the temporal **test** split — used here as a simulated
production window, not for model selection.
