# Decision Policy — lgbm_v9

**Date**: 2026-08-28
**Model**: lgbm_v9 on dataset_v5 (464 Dn-only features)
**Policy version**: 1
**Status**: Proposed — not production-ready until business inputs are confirmed

---

## 1. Why a score needs an action

A fraud probability is not a decision. The model outputs a number between 0 and 1. Downstream systems need a discrete action: approve the transaction, request extra authentication, route to manual review, or decline. The mapping from probability to action depends on business costs, operational capacity, and customer friction tolerance — none of which are determined by the model alone.

## 2. Calibration

### Why calibration is needed

lgbm_v9 uses `scale_pos_weight=32.2` to handle class imbalance during training. This inflates the model's raw output scores well above the true fraud probability. Before mapping scores to actions via an expected value framework, the scores must be calibrated to reflect actual fraud rates.

### Method

Isotonic regression with 5-fold cross-fitting on the validation split (middle 15% by TransactionDT). Cross-fitting prevents the calibrator from overfitting to the same data used for evaluation — each validation row's calibrated score comes from a calibrator that never saw that row during fitting.

A separate deployment calibrator is fit on the full validation split for inference-time use.

### Results

**Before calibration:**
- Mean raw score: 0.0198
- Actual fraud rate: 0.0343
- Calibration bins (predicted → observed, n):
  - 0.05 → 0.0148 (86,393)
  - 0.15 → 0.5205 (219)
  - 0.25 → 0.5410 (122)
  - 0.35 → 0.4186 (86)
  - 0.45 → 0.5294 (68)
  - 0.55 → 0.6184 (76)
  - 0.65 → 0.7162 (74)
  - 0.75 → 0.8434 (83)
  - 0.85 → 0.7265 (117)
  - 0.95 → 0.9337 (1,343)

**After calibration:**
- Mean calibrated score: 0.0344 (fraud rate: 0.0343; gap 0.0001)
- Calibration bins (predicted → observed, n):
  - 0.05 → 0.0103 (84,215)
  - 0.15 → 0.1328 (1,258)
  - 0.25 → 0.2149 (563)
  - 0.35 → 0.3817 (241)
  - 0.45 → 0.4072 (221)
  - 0.55 → 0.5027 (374)
  - 0.65 → 0.6788 (137)
  - 0.75 → 0.7631 (287)
  - 0.85 → 0.8640 (228)
  - 0.95 → 0.9546 (1,057)

Figures: `evals/figures/calibration_before_v9.png`, `evals/figures/calibration_after_v9.png`

## 3. Cost model and assumptions

Each action has an expected cost that depends on the transaction's calibrated fraud probability (p) and dollar amount (a).

| Action | Expected cost formula | What it represents |
|--------|----------------------|-------------------|
| Approve | p × a × fraud_cost_multiplier | Fraud loss if the transaction is fraudulent |
| Step-up | p × a × multiplier × (1 - stop_rate) + (1-p) × a × friction_rate | Residual fraud + customer abandonment |
| Review | p × a × multiplier × (1 - catch_rate) + review_cost | Residual fraud + fixed operational cost |
| Decline | (1-p) × a × revenue_loss | Lost legitimate revenue |

### Default assumptions

| Parameter | Value | Source |
|-----------|-------|--------|
| fraud_cost_multiplier | 2.0 | Assumed — chargeback + investigation + fines |
| step_up_stop_rate | 0.70 | Assumed — fraction of fraudsters blocked by extra auth |
| step_up_friction_rate | 0.05 | Assumed — fraction of legit customers who abandon |
| review_catch_rate | 0.95 | Assumed — reviewer accuracy |
| review_cost_per_txn | $10.00 | Assumed — operational cost per review |
| review_capacity_per_day | 200 | Assumed — team headcount constraint |
| decline_revenue_loss | 1.0 | Transaction amount lost when declining a legit txn |

**Every parameter marked "Assumed" must be confirmed by the relevant business stakeholder before this policy moves from "proposed" to "active."**

## 4. Optimal threshold derivation

The optimal action for each transaction is whichever minimizes expected cost. The thresholds are the calibrated probability levels where the optimal action switches — they are crossover points between the cost curves, not arbitrary cutoffs.

Method: two-pass grid search over threshold triplets (t1, t2, t3) on the validation split. Coarse pass at 0.01 resolution, fine pass at 0.005 within ±0.05 of the coarse optimum. Capacity constraint: if review volume exceeds 200 reviews/day, a penalty is added to the cost.

Bootstrap validation (500 resamples, 95% CI) compares optimized vs hypothesis thresholds. On the full validation set the optimized policy is substantially cheaper; bootstrap resampling shows high variance and the CI includes zero (statistically equivalent under resampling).

### Optimized thresholds

| Boundary | Threshold | Action below | Action at or above |
|----------|-----------|-------------|-------------------|
| t1 | 0.0360 | Approve | Step-up authentication |
| t2 | 0.0510 | Step-up | Manual review |
| t3 | 0.8200 | Manual review | Auto-decline |

## 5. Results per band — starting hypothesis (0.40 / 0.80 / 0.95)

```
Action     Boundaries               Txns      %   Fraud     F%       Fraud$    F$%   Prec Recall≥  FP/Fric         Cost
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
approve    [0.00, 0.4000)         86,277  97.4%   1,246  41.0% $   226,059  45.5% 0.014  1.000        0 $   412,311
step_up    [0.4000, 0.8000)        1,019   1.2%     590  19.4% $    99,609  20.0% 0.579  0.590      429 $    57,259
review     [0.8000, 0.9500)          569   0.6%     513  16.9% $    76,666  15.4% 0.902  0.396       56 $    12,925
decline    [0.9500, 1.00]            716   0.8%     693  22.8% $    94,573  19.0% 0.968  0.228       23 $     2,203
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Fraud caught: 1,796/3,042 (59.0%)  Reviews: 569  Declined legit: 23  Total cost: $484,698
```

## 6. Results per band — optimized thresholds

```
Action     Boundaries               Txns      %   Fraud     F%       Fraud$    F$%   Prec Recall≥  FP/Fric         Cost
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
approve    [0.00, 0.0360)         80,200  90.5%     594  19.5% $   107,999  21.7% 0.007  1.000        0 $   176,938
step_up    [0.0360, 0.0510)          934   1.1%      39   1.3% $     7,217   1.5% 0.042  0.805      895 $    12,163
review     [0.0510, 0.8200)        6,212   7.0%   1,248  41.0% $   217,733  43.8% 0.201  0.792    4,964 $    82,782
decline    [0.8200, 1.00]          1,235   1.4%   1,161  38.2% $   163,959  33.0% 0.940  0.382       74 $     8,542
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Fraud caught: 2,448/3,042 (80.5%)  Reviews: 6,212  Declined legit: 74  Total cost: $280,424
```

## 7. Comparison: hypothesis vs optimized

| Metric | Hypothesis | Optimized | Delta |
|--------|-----------|-----------|-------|
| Total expected cost | $484,698 | $280,424 | -$204,274 (-42.1%) |
| Fraud caught (count) | 59.0% | 80.5% | +21.5pp |
| Fraud caught (dollars) | 54.5% | 78.3% | +23.8pp |
| Reviews/day | 18 | 198 | +180 |
| Declined legit txns | 23 | 74 | +51 |
| Bootstrap CI (cost reduction) | — | -$2,432,126 to $216,011 | excludes zero: NO |

## 8. Customer friction analysis

**Hypothesis policy (0.40 / 0.80 / 0.95):**
- Legitimate customers hitting step-up authentication: 429 transactions ($57,560 in transaction value)
- Legitimate customers auto-declined: 23 transactions ($2,949 in transaction value)
- Step-up band is narrow (1.2% of volume) but high precision (57.9% fraud)

**Optimized policy (0.036 / 0.051 / 0.82):**
- Legitimate customers hitting step-up authentication: 895 transactions ($159,900 in transaction value)
- Legitimate customers auto-declined: 74 transactions ($6,219 in transaction value)
- Review band absorbs most elevated-risk volume (7.0% of transactions), shifting fraud capture earlier at the cost of more operational load

## 9. Review load analysis

- Reviews per day under hypothesis policy: 18 (569 reviews over 31.4 validation days)
- Reviews per day under optimized policy: 198 (6,212 reviews over 31.4 validation days)
- Stated capacity: 200 reviews/day
- If capacity is halved to 100/day, thresholds shift to t1=0.036 / t2=0.100 / t3=0.820 (review band narrows, more volume pushed to step-up)

## 10. Sensitivity analysis

```
fraud_cost_multiplier:
  Value          t1     t2     t3         Cost  Fraud%  Rev/day
  ────────────────────────────────────────────────────────
  1.0          0.045  0.050  0.860 $   176,946   79.8%     199
  1.5          0.046  0.051  0.845 $   229,131   79.8%     198
  2.0          0.036  0.051  0.820 $   280,424   80.5%     198
  3.0          0.026  0.046  0.760 $   375,346   82.6%     198
  5.0          0.011  0.041  0.730 $   526,408   89.9%     200

step_up_stop_rate:
  Value          t1     t2     t3         Cost  Fraud%  Rev/day
  ────────────────────────────────────────────────────────
  0.5          0.046  0.051  0.820 $   282,104   79.8%     198
  0.6          0.041  0.051  0.820 $   281,552   79.8%     198
  0.7          0.036  0.051  0.820 $   280,424   80.5%     198
  0.8          0.030  0.050  0.830 $   278,050   81.7%     198
  0.9          0.026  0.051  0.820 $   275,076   82.6%     198

review_cost_per_txn:
  Value          t1     t2     t3         Cost  Fraud%  Rev/day
  ────────────────────────────────────────────────────────
  5.0          0.036  0.051  0.860 $   249,284   80.5%     199
  10.0         0.036  0.051  0.820 $   280,424   80.5%     198
  20.0         0.036  0.155  0.810 $   327,565   80.5%      63
  50.0         0.036  0.610  0.655 $   354,654   80.5%       3

review_capacity_per_day:
  Value          t1     t2     t3         Cost  Fraud%  Rev/day
  ────────────────────────────────────────────────────────
  100          0.036  0.100  0.820 $   297,850   80.5%     100
  200          0.036  0.051  0.820 $   280,424   80.5%     198
  500          0.026  0.031  0.820 $   278,342   82.6%     258
  10000        0.026  0.031  0.820 $   278,342   82.6%     258
```

Figure: `evals/figures/sensitivity_thresholds_v9.png`

### Which assumptions matter most?

**Review cost per transaction** has the largest leverage on threshold placement. When review cost rises from $10 to $50, t2 (step-up → review boundary) shifts from 0.051 to 0.610 — a 0.56 absolute change — collapsing the review band and pushing more volume to step-up or decline. Operations should confirm this cost first.

**Fraud cost multiplier** is the second-most influential parameter: t1 drops from 0.045 to 0.011 and t3 from 0.86 to 0.73 as multiplier rises from 1.0 to 5.0, making the policy progressively more aggressive.

## 11. Missing business inputs

Before this policy can move from "proposed" to "active":

| Input needed | Who confirms | Impact on thresholds |
|-------------|-------------|---------------------|
| Fraud cost multiplier | Finance / Risk | Drives all thresholds — higher multiplier → more aggressive fraud catching |
| Review team capacity | Operations | Constrains how much fraud can be routed to review vs declined |
| Review cost per transaction | Operations | Higher cost → fewer reviews, more auto-decisions |
| Step-up effectiveness | Product / Auth team | If step-up rarely stops fraudsters, the step-up band shrinks |
| Customer friction tolerance | Product | Higher tolerance → more step-up, less approve |
| Regulatory constraints on auto-decline | Legal / Compliance | May prohibit auto-decline above a dollar threshold |

## 12. Recommendation

The optimized thresholds (0.036 / 0.051 / 0.820) reduce total expected cost by $204,274 (42.1%) vs the starting hypothesis on the validation split, while keeping review volume within the assumed capacity of 200 reviews/day (198 reviews/day observed).

**These thresholds are not production-ready.** They rest on assumed cost parameters that have not been confirmed by business stakeholders. The sensitivity analysis (Section 10) shows review cost per transaction has the most leverage — that should be confirmed first.

To activate: confirm the missing inputs in Section 11, update `configs/decision_policy_v1.json` with confirmed values, re-run `evals/optimize_thresholds.py` with the confirmed parameters, and change the policy status from `"proposed"` to `"active"`.
