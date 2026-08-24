# Model Card — lgbm_v9

## Intended use

Score incoming e-commerce transactions with a fraud probability. Intended for use as one input to a fraud review pipeline — not as a sole automated decision-maker. Downstream systems apply thresholds and business rules.

## Not intended for

- Autonomous transaction blocking without human review
- Scoring non-e-commerce transactions (card-present, ATM, wire transfers)
- Regulatory compliance decisions requiring explainable per-feature attribution
- Real-time customer-facing fraud alerts without a secondary verification step

## Training data

- Source: IEEE-CIS Vesta Corporation e-commerce transactions
- Rows: ~590K transactions
- Fraud rate: ~3.5%
- Temporal range: ~182 days of TransactionDT
- Split: temporal 70/15/15 (train/val/test), sorted by TransactionDT
- Train rows: ~413K | Val rows: ~89K | Test rows: ~89K

## Evaluation data

- Validation split (middle 15%): used for early stopping, SHAP analysis, ablation, error analysis, and threshold selection (Phase 3)
- Test split (latest 15%): evaluated once after all model decisions frozen. Test AUC-PR: 0.5952

## Model and features

- Algorithm: LightGBM (gradient-boosted decision trees)
- Features: 464 (Dn-only — raw D1–D15 dropped, replaced with normalized counterparts)
- Key hyperparameters: learning_rate 0.012, num_leaves 224, feature_fraction 0.895, scale_pos_weight 32.2
- Trees: 9779 (out of 15000 ceiling, early stopping at 200 rounds patience)
- Preprocessing: `requires_fit` transforms fit on training split only

## Performance

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-PR | 0.7022 | 0.5952 |
| AUC-ROC | 0.9297 | 0.9067 |

Bootstrap 95% CI vs v1 (raw baseline, test): [+0.0461, +0.0675]
Bootstrap 95% CI vs v8 (pre-Dn-drop, test): [+0.0024, +0.0062]

## Threshold status

**No operating threshold is currently set.** The model outputs calibrated-ish probabilities. Phase 3 will select thresholds using validation data only, with precision/recall tradeoffs documented for business review.

## Limitations

1. Trained on Vesta e-commerce data only — may not generalize to other payment types or merchants
2. Labels are assumed final; production labels arrive with delay (chargebacks)
3. Temporal drift: performance degrades over time as fraud patterns evolve
4. Dn normalization encodes relative time deltas that may capture seasonal patterns
5. Limited behavioral features for singleton UIDs (~58.7% of UID groups)
6. V-features are anonymized — no domain interpretation available for debugging

## Customer risks

- **False negatives** (missed fraud): customer loses money, trust erodes. Higher cost.
- **False positives** (blocked legitimate): customer frustrated, potential revenue loss. Lower cost per incident but high volume impact.
- The threshold (Phase 3) determines the tradeoff. No threshold has been set yet.

## Monitoring needs

- Track AUC-PR on a rolling window of labeled transactions
- Alert if fraud rate in scored population shifts >1pp from training distribution
- Monitor feature drift: TransactionAmt distribution, missing-value rates, Dn value ranges
- Monitor prediction distribution: if median predicted probability shifts significantly, investigate

## Retraining triggers

- AUC-PR on rolling labeled window drops below a threshold defined in Phase 3
- Fraud rate shifts materially from the ~3.5% training distribution
- New product codes or card types appear that were not in training data
- Dn feature distributions shift (indicates change in D-column reference points)
- Regulatory or business rule changes affecting fraud definition
