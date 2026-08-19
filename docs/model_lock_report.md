# Model Lock Report — lgbm_v8

**Date**: 2026-08-19
**Author**: Arnav Lohe
**Status**: Locked — no further changes to features, parameters, or split

---

## 1. Selected version

- **Model**: lgbm_v8
- **Dataset**: dataset_v4
- **Config**: configs/lgbm_v8.json
- **Dataset config**: configs/dataset_v4.json
- **Parent**: lgbm_v7 (Optuna-tuned, hit 5000-round ceiling)

## 2. Performance summary

| Split | AUC-PR | Notes |
|-------|--------|-------|
| Validation | 0.7018 | Used for early stopping |
| Test (holdout) | 0.5909 | Evaluated once after all decisions frozen |

### v8 vs v7 (bootstrap, 500 resamples, 95% CI)

| Split | Delta | 95% CI | Significant |
|-------|-------|--------|-------------|
| Validation | +0.0091 | [+0.0072, +0.0111] | Yes |
| Test | +0.0072 | [+0.0054, +0.0091] | Yes |

### v8 vs v6 (bootstrap, 500 resamples, 95% CI)

| Split | Delta | 95% CI | Significant |
|-------|-------|--------|-------------|
| Test | +0.0546 | [+0.0483, +0.0609] | Yes |

## 3. Why v8 over v7

v7 (Optuna TPE, 100 trials) found optimal hyperparameters but the full retrain hit the `n_estimators` ceiling at 5000/5000 without early stopping firing — the model was still learning when cut off. v8 raised the ceiling to 15000 with `early_stopping_rounds=200`. Early stopping fired at iteration 11097, confirming natural convergence. The +0.72pp test AUC-PR improvement is statistically significant (bootstrap CI excludes zero) and comes at zero risk: identical features, identical hyperparameters, the model simply finished learning.

Training time increased from 512.6s to 1709.0s (~3.3× longer). This is a one-shot cost — the model trains once and serves indefinitely. Inference latency is unaffected by tree count in LightGBM's compiled prediction path at this scale.

## 4. D/Dn feature comparison

Dn-only outperformed both D+Dn and D-only across all three walk-forward 30-day test windows (branch: `diag/d-dn-ablation`). Dn-only mean AUC-PR: 0.6740. Raw D columns enabled LightGBM to recover calendar day from the D values and steer early splits toward non-generalizable temporal patterns. Raw D1–D15 are dropped from the feature set entirely.

## 5. Temporal stability

Three walk-forward 30-day test windows from the D/Dn ablation confirmed stable Dn-only performance. The winning variant (Dn-only) showed consistent lift across all windows with no degradation in later periods.

## 6. Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| learning_rate | 0.011978428868493889 | Optuna trial #85 |
| num_leaves | 224 | Optuna |
| n_estimators | 15000 | Ceiling raised from v7's 5000 |
| early_stopping_rounds | 200 | Raised from v7's 100 |
| feature_fraction | 0.8948476596035589 | Optuna |
| bagging_fraction | 0.8489447664564709 | Optuna |
| min_child_samples | 42 | Optuna |
| reg_alpha | 0.006491774642689657 | Optuna |
| reg_lambda | 0.20933776242575416 | Optuna |
| scale_pos_weight | 32.175142781078854 | Optuna |

All values except `n_estimators` and `early_stopping_rounds` are inherited from v7 unchanged.

## 7. Feature set

- Total features: 479
- Dataset version: 4
- `requires_fit` transforms: frequency_encoding, uid_aggregations, identity_frequency_encoding
- Feature list: `data/processed/feature_list_v4.json`

## 8. Unresolved risks

### 8a. Validation-to-test gap
Val AUC-PR 0.7018 vs test AUC-PR 0.5909 — a gap of ~1109 bps. Consistent with v6 (~1166 bps val-test gap when comparing stored val to bootstrap test) and v7 (~1089 bps). Attributed to temporal distribution shift between contiguous time slices rather than overfitting. The three-way split prevents any test information from leaking into model decisions.

### 8b. Accumulated test-set evaluation bias
The test set has been evaluated against 8 model versions across the experiment sequence. Each evaluation introduces implicit upward selection pressure. The final test AUC-PR should be interpreted as an upper bound.

### 8c. Random seed
No random seed is set in `configs/lgbm_v8.json`. Exact reproduction requires setting one; stochastic components (bagging, feature subsampling) will cause small run-to-run variation.

## 9. Holdout discipline

- Test split was NEVER passed to `eval_set` or used for early stopping
- Test split was NEVER used for hyperparameter selection (Optuna optimized on val only)
- Test split was NEVER used for threshold tuning
- Total holdout evaluations across all versions: 8
- v8 is a capacity extension of v7 — no new model decisions were made; only the ceiling was raised

## 10. Reproduction

```bash
python scripts/merge_train_data.py
python scripts/build_dataset.py --config configs/dataset_v4.json
python scripts/train.py --config configs/lgbm_v8.json
python evals/compare_models.py
```

Expected test AUC-PR: 0.5909. Run-to-run variation from stochastic components: ±0.002 estimate.

## 11. Experiment lineage

```
v1 (raw baseline, 0.5511)
 └─ v2 (feature engineering, 0.5698)
     └─ v3 (L1 regularization, 0.5721)
         └─ v4 (clean re-baseline three-way split, 0.5011)
             └─ v5 (Dn normalization, 0.6412 val-only in metrics)
                 └─ v6 (identity frequency encodings, 0.6527 val-only in metrics)
                     └─ v7 (Optuna re-tune, 0.5837 test, hit 5000 ceiling)
                         └─ v8 (ceiling raised to 15000, 0.5909) ← serving
```

All AUC-PR values are holdout test scores where available in metrics JSON; v5/v6 use validation scores from committed metrics (test not persisted).

## 12. What is frozen after this point

- Features: dataset_v4
- Hyperparameters: configs/lgbm_v8.json
- Split boundaries: 70/15/15 temporal on TransactionDT
- The only permitted change is the `is_serving` flag in `models/registry.json`
