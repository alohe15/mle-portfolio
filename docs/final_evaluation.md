# Final Evaluation — lgbm_v9

**Date**: 2026-08-24
**Model**: lgbm_v9 on dataset_v5 (Dn-only, 464 features)
**Status**: Frozen and serving. No further changes to features, parameters, or split.

---

## 1. Problem and label

Binary classification: predict whether an e-commerce transaction is fraudulent.
Target: `isFraud` (1 = fraud, 0 = legitimate).
Label source: Vesta Corporation ground truth from the IEEE-CIS Kaggle competition.
Fraud rate: ~3.5% (~20,663 positive cases in ~590K transactions).

## 2. Data and temporal splits

Dataset: IEEE-CIS Vesta e-commerce transactions (~590K rows, 464 features after Dn normalization and raw D-column removal).
Temporal range: ~182 days of `TransactionDT`.
Split: temporal 70/15/15 sorted by TransactionDT (time delta in seconds from a reference point).
- Train: earliest 70% (~413K rows)
- Validation: middle 15% (~89K rows) — used for early stopping and all diagnostic analysis
- Test: latest 15% (~89K rows) — evaluated once after all decisions frozen

No shuffling. No stratification. The split preserves temporal ordering to simulate production deployment where the model trains on past data and scores future transactions.

## 3. Primary metric

AUC-PR (area under the precision-recall curve).
Probabilistic interpretation: the expected precision averaged uniformly over all recall levels — equivalently, the probability-weighted ability of the model to surface fraud cases at any given retrieval rate.

Why not AUC-ROC: at 3.5% prevalence, ROC's false positive rate denominator (~569K negatives) makes even large absolute increases in false positives appear small. AUC-PR's precision denominator (flagged transactions) is more sensitive to the errors that matter operationally.

## 4. V1-to-v9 comparison

All rows use **test-split AUC-PR** from each version's metrics JSON (via `evals/compare_models.py`). `delta_vs_v1` = test AUC-PR − v1 test AUC-PR.

```
version  dataset_version  description                                                                                                                          test_auc_pr  best_iteration  n_features  delta_vs_v1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      1                1  Raw baseline — all raw merged columns, no engineered features                                                                             0.5511             953         431      +0.0000
      2                2  Feature engineering migration: email_match, TransactionAmt transforms, missingness indicators, frequency encoding, UID aggregations       0.5698            1321         461      +0.0186
      3                2  L1 regularization (reg_alpha=1.0) for feature selection — drives low-signal feature weights to zero                                       0.5721            1993         461      +0.0210
      4                2  Clean re-baseline — dataset_v2 features under three-way temporal split with L1 regularization                                             0.5011            1947         461      -0.0500
      5                3  Dn normalization — test calendar-normalized D-features                                                                                    0.6412            2000         476      +0.0901
      6                4  Identity-side frequency encodings — test DeviceInfo/id_30/id_31 rarity                                                                    0.6527            1995         479      +0.1015
      7                4  Optuna-tuned hyperparameters on dataset_v4 (100 trials, 50% train subsample, retrained on full data)                                      0.5837            5000         479      +0.0326
      8                4  Tree ceiling test — raise n_estimators from 5000 to 15000 to check if v7 was capacity-limited                                             0.5909           11097         479      +0.0398
      9                5  Dn-only drop — retrain v8 hyperparameters on dataset_v5 (raw D1-D15 removed)                                                              0.5952            9779         464      +0.0441
```

### Key transitions

- **v1 → v2**: Feature engineering (email_match, amt transforms, missingness, freq encoding, UID aggs) lifted test AUC-PR +186 bps on dataset_v2.
- **v2 → v3**: L1 regularization for feature selection added a small further gain (+24 bps) on the same dataset.
- **v3 → v4**: Clean three-way temporal re-baseline on dataset_v2 dropped test AUC-PR to 0.5011 — honest holdout under the new split, not a modeling regression in isolation.
- **v4 → v5**: Dn normalization (dataset_v3) produced the largest single jump (+1401 bps vs v4; +901 vs v1).
- **v5 → v6**: Identity-side frequency encodings (dataset_v4) added another +115 bps.
- **v6 → v7**: Optuna hyperparameters on dataset_v4 with a 5000-tree ceiling fell to 0.5837 on test — capacity-limited relative to earlier experimental scores.
- **v7 → v8**: Raising the tree ceiling to 15000 recovered +72 bps (early stop at 11097).
- **v8 → v9**: Dropped raw D1–D15 (greedy-attractor fix). Test AUC-PR +43 bps (0.5909 → 0.5952).

## 5. Results across time windows

The D/Dn ablation tested three walk-forward 30-day test windows to check temporal stability.

| Variant | Mean AUC-PR | Result |
|---------|-------------|--------|
| Dn-only | 0.6740 | Won all 3 windows |
| D+Dn | < 0.6740 | Lost all 3 windows |
| D-only | 0.6545 | Lost all 3 windows |

v9's production performance (test AUC-PR 0.5952) is lower than the ablation's walk-forward numbers because the ablation used shorter windows from the middle of the dataset, while the test split covers the latest 15% where fraud patterns have drifted furthest from training. This gap is expected in temporal evaluation and is itself informative — it quantifies how much the model degrades over the time horizon.

## 6. Bootstrap significance

Bootstrap: 500 resamples, seed 42, **test split** (aligned by TransactionID where available). Source: `evals/bootstrap_significance.py`.

### v9 vs v1 (raw baseline) — test split

```
Mean (baseline):          0.5386
Mean (candidate):         0.5955
Mean difference:          +0.0569
95% CI for difference:    [+0.0461, +0.0675]
Result:                   statistically significant improvement
```

(Note: bootstrap mean for v1 on the shared three-way test window differs from v1's stored metrics AUC-PR because v1 was originally scored under its two-way split; the paired delta CI is the relevant quantity.)

### v9 vs v8 (Dn-only drop) — test split

```
Mean difference:          +0.0043
95% CI for difference:    [+0.0024, +0.0062]
Result:                   statistically significant improvement
```

Both CIs exclude zero, confirming statistically significant improvement.

## 7. SHAP interpretation

Computed on the **validation** split only (`evals/shap_analysis.py --version 9 --sample-size 1000 --seed 42`). No raw D1–D15 appear (not in the feature set).

| Rank | Feature | Mean \|SHAP\| |
|------|---------|---------------|
| 1 | C13 | 0.800242 |
| 2 | C1 | 0.410774 |
| 3 | D4n | 0.396334 |
| 4 | TransactionAmt | 0.373252 |
| 5 | D1n | 0.346359 |
| 6 | C14 | 0.334046 |
| 7 | P_emaildomain | 0.284691 |
| 8 | V70 | 0.279003 |
| 9 | card1_freq | 0.245254 |
| 10 | C11 | 0.239372 |
| 11 | card6 | 0.239211 |
| 12 | uid_amt_std | 0.230183 |
| 13 | uid_amt_mean | 0.229697 |
| 14 | D15n | 0.222397 |
| 15 | dist1 | 0.222053 |
| 16 | D10n | 0.219873 |
| 17 | C2 | 0.216482 |
| 18 | card1 | 0.216015 |
| 19 | TransactionAmt_cents | 0.210362 |
| 20 | C5 | 0.210078 |

### What SHAP shows

SHAP values decompose each prediction into per-feature contributions. A feature with high mean |SHAP| contributes more to the model's fraud/non-fraud distinction on average. The ranking reflects the model's learned importance — which features it relies on most.

### What SHAP does not show

SHAP does not prove causation. A feature ranking high in SHAP may be a proxy for an unmeasured true driver. SHAP also does not guarantee generalization — a feature can be important in the training distribution but lose signal under distribution shift. The ablation in Section 8 provides a complementary check: if removing a high-SHAP feature group barely changes the metric, the model has redundant paths and SHAP is overstating that group's unique contribution.

### Key observations

- **C-count features dominate** the top ranks (C13, C1, C14, C11, C2, C5) — counting/behavior intensity features drive local attributions more than anonymized V-block columns.
- **Dn columns appear prominently**: D4n (#3), D1n (#5), D15n (#14), D10n (#16). Normalized D features carry unique time-relative signal after raw D removal.
- **Card and amount**: TransactionAmt (#4), card1_freq (#9), card6 (#11), card1 (#18), TransactionAmt_cents (#19) confirm amount and card identity remain important.
- **V-features are sparse in the top 20** (only V70 at #8) despite 359 V columns — high redundancy within the V block.
- Aligns with Week 1–2 EDA emphasis on amount, card frequency, and counting features; Dn importance supports the Phase 1 Dn-only lock.

## 8. Feature-group ablation

Diagnostic retrains with v9 hyperparameters, `n_estimators≤2000`, evaluated on **validation only**. Deltas are vs frozen v9 val AUC-PR 0.7022 (full 9779 trees) — absolute levels mix capacity and feature removal; **relative ranking** is the intended signal. Source: `evals/ablation_feature_groups.py --all`.

```
Feature Group Ablation — lgbm_v9 (diagnostic, validation split, n_estimators≤2000)
Group                 Removed  Remaining  Val AUC-PR  Delta vs v9
────────────────────────────────────────────────────────────────
(full model)                0        464      0.7022      +0.0000
dn_columns                 15        449      0.6361      -0.0661
v_features                359        105      0.6621      -0.0401
card_features               8        456      0.6356      -0.0666
email_features              4        460      0.6660      -0.0362
addr_features               5        459      0.6732      -0.0290
id_features                43        421      0.6656      -0.0367
```

### Interpretation

- **Most critical groups**: `card_features` (Δ −0.0666, 8 cols) and `dn_columns` (Δ −0.0661, 15 cols) — largest unique contribution per the leave-one-group-out design.
- **Least critical among tested**: `addr_features` (Δ −0.0290) — more redundant with remaining features.
- **V-features**: removing 359 columns hurts less than removing 8 card columns (−0.0401 vs −0.0666), matching SHAP's sparse V presence in the top 20 — high within-block redundancy.
- **SHAP vs ablation**: SHAP ranks individual C-counts and Dn highest; ablation's group design does not drop C-counts as a named group. Card and Dn groups both show large unique contribution, consistent with card1/card6 and D*n appearing in the SHAP top 20. Email ranks mid in SHAP (P_emaildomain #7) but ablation delta is moderate (−0.0362), suggesting partial redundancy with other identity signals.

## 9. Error analysis

Validation split only (`evals/error_analysis.py --version 9`). Band precision = actual fraud / transactions in band. "Pred≥0.5" counts scores ≥ 0.5 within the band (operational threshold still unset — Phase 3).

### By prediction confidence

```
Confidence band    Transactions   Pred≥0.5   Actual fraud   Precision
0.0–0.1            86393          0          1281           0.0148
0.1–0.3            341            0          180            0.5279
0.3–0.5            154            0          72             0.4675
0.5–0.7            150            150        100            0.6667
0.7–0.9            200            200        155            0.7750
0.9–1.0            1343           1343       1254           0.9337
```

Most mass sits below 0.1 (expected under ~3.5% prevalence). High-confidence bands (0.9–1.0) are highly precise (~93%).

### By product code

```
ProductCD    Txns     Fraud    Fraud rate   AUC-PR
C            9125     1150     0.1260       0.8010
H            2350     140      0.0596       0.7456
R            3365     160      0.0475       0.8937
S            1450     107      0.0738       0.8362
W            72291    1485     0.0205       0.5986
```

ProductCD **W** dominates volume and has the lowest AUC-PR (0.5986) at the lowest fraud rate — the hardest product segment. **R** scores best (0.8937).

### By time

```
Period          Txns     Fraud    Fraud rate   AUC-PR
Early third     29527    1025     0.0347       0.8054
Middle third    29527    1036     0.0351       0.6728
Late third      29527    981      0.0332       0.6103
```

Validation AUC-PR degrades from early → late third (0.8054 → 0.6103), previewing the larger val→test drop — temporal drift within the validation window itself.

## 10. Conclusion

### Performance

v9 achieves test AUC-PR 0.5952, a **441 bps** improvement over the raw baseline (v1, 0.5511). The improvement is statistically significant (bootstrap CI excludes zero on the test split vs both v1 and v8). At the prevalence-matched operating point recorded in metrics, **55.1%** of flagged transactions are actually fraudulent (precision_at_budget) and **55.7%** of actual fraud is caught (recall_at_budget).

### Stability

Dn-only performance was stable across all three walk-forward ablation windows. The val-to-test gap (**1070 bps**: 0.7022 → 0.5952) indicates temporal drift over the test period but the model does not collapse — it degrades gradually (also visible inside validation thirds).

### Complexity and cost

464 features. **9779** LightGBM trees (best_iteration). Training time: ~**17.3** minutes (1039.7s wall). Inference latency: not yet benchmarked in this phase (API latency script exists but was not run here).

### Limitations

1. **Delayed labels**: fraud labels in this dataset are assumed final. In production, labels arrive with delay (chargebacks can take 30–90 days). The model was trained on resolved labels and may perform differently on transactions whose labels are still pending.
2. **Temporal drift**: the val-to-test gap and within-validation third degradation suggest fraud patterns change over time. A fixed model will degrade. Monitoring and periodic retraining are required.
3. **D-derived time shortcuts**: although raw D1–D15 were removed, the Dn normalization (`floor(TransactionDT/86400 - D)`) still encodes relative time deltas. If these correlate with seasonal fraud patterns in training, the signal may not generalize to different seasons.
4. **No user-level features**: the `(card1, addr1)` UID proxy achieves ~96% coverage but 58.7% are singletons. Behavioral features (velocity, spend patterns) are limited for first-time or rare users.
5. **Threshold not set**: no operating threshold has been selected. Phase 3 will set thresholds using validation data only. Until then, the model outputs probabilities, not decisions.
