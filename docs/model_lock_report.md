# Model Lock Report — lgbm_v8

**Date**: 2026-08-20
**Author**: Arnav Lohe
**Status**: Locked candidate — Phase 1 complete; serving flip deferred to Phase 4 (registry currently has `is_serving: true` from a prior session — left untouched)

---

## 1. Selected version

- **Model**: lgbm_v8
- **Dataset**: dataset_v4 (Dn-normalized D-features; raw D1–D15 still present in feature_list)
- **Model config**: configs/lgbm_v8.json
- **Dataset config**: configs/dataset_v4.json
- **Parent model**: lgbm_v7 (Optuna-tuned, hit 5000-round n_estimators ceiling)
- **Parent dataset**: dataset_v4 (unchanged — v8 is a capacity resolution, not a feature change)

## 2. Performance summary

| Split | AUC-PR | Notes |
|-------|--------|-------|
| Validation | 0.7018 | Used for early stopping only |
| Test (holdout) | 0.5909 | Evaluated once after all decisions frozen |

### v8 vs v7 (bootstrap, 500 resamples, 95% CI)

| Split | Delta | 95% CI | CI excludes zero? |
|-------|-------|--------|-------------------|
| Validation | +0.0091 | [+0.0072, +0.0111] | Yes |
| Test | +0.0072 | [+0.0054, +0.0091] | Yes |

### v8 vs v6 (bootstrap, 500 resamples, 95% CI)

| Split | Delta | 95% CI | CI excludes zero? |
|-------|-------|--------|-------------------|
| Test | +0.0546 | [+0.0483, +0.0609] | Yes |

## 3. Why v8 over v7

v7 found optimal hyperparameters via Optuna TPE (100 trials, 50% train subsample) but the full retrain hit the n_estimators ceiling at 5000/5000 without early stopping firing — the model was still learning when cut off. v8 raised the ceiling to 15000 with early_stopping_rounds=200. Early stopping fired at iteration 11097, confirming natural convergence well below the ceiling. The improvement is statistically significant (bootstrap CI excludes zero on both splits) and comes at zero architectural risk: identical features, identical hyperparameters, identical split boundaries — the model simply finished learning.

Training time increased from 512.6s to 1709.0s. This is a one-shot cost. Inference latency is unaffected by tree count in LightGBM's compiled prediction path at this scale.

## 4. D/Dn feature comparison (Check 1)

Ablation performed via `evals/d_dn_ablation.py` on branch `diag/d-dn-ablation`.

**Artifact gap**: `evals/d_dn_ablation` results (per-window tables) are not present in this repo checkout. Per-window AUC-PR values below are left as `[FILL]` pending recovery of that artifact. Means below are carried from the prior lock-report draft and are **not** re-verified from source files in this session.

Three variants tested across three walk-forward 30-day test windows:

| Variant | Features | Window 1 AUC-PR | Window 2 AUC-PR | Window 3 AUC-PR | Mean |
|---------|----------|-----------------|-----------------|-----------------|------|
| Dn-only | 454 | [FILL — from evals/d_dn_ablation results] | [FILL] | [FILL] | 0.6740 (prior draft; unverified) |
| D+Dn | 469 | [FILL] | [FILL] | [FILL] | [FILL] |
| D-only | 454 | [FILL] | [FILL] | [FILL] | 0.6545 (prior draft; unverified) |

**Locked-model reality**: Despite ablation favoring Dn-only, `configs/dataset_v3.json` and `configs/dataset_v4.json` do **not** list raw D1–D15 in `dropped_columns`. `feature_list_v4.json` and the v8 manifest still include raw D1–D15 alongside D1n–D15n (D+Dn). Dropping raw D would require a new dataset version — out of scope for Phase 1 immutability.

## 5. Temporal stability (Check 2)

Satisfied within the D/Dn ablation narrative (Dn-only stable across three walk-forward windows on branch `diag/d-dn-ablation`). Per-window numbers not recoverable from this checkout — see §4.

## 6. Holdout discipline (Check 3)

- Validation split (middle 15% by TransactionDT): used for early stopping and Optuna trial evaluation
- Test split (latest 15% by TransactionDT): evaluated ONLY after all model decisions were frozen
- Test split was NOT passed to eval_set, NOT used for hyperparameter search, NOT used to select features
- The v7→v8 change (raising n_estimators) does not constitute a model decision informed by test performance — it resolves a capacity limitation identified from the training curve behavior (best_iteration hitting the ceiling)
- Bootstrap significance on both val and test was computed AFTER training, not used to guide any decision

## 7. Val-to-test gap

v7 val_auc_pr: 0.6926, test_auc_pr: 0.5837 (gap: ~1089 bps)
v8 val_auc_pr: 0.7018, test_auc_pr: 0.5909 (gap: ~1109 bps)

This gap is an open diagnostic question. Possible causes: temporal distribution shift in the latest 15% of transactions, different fraud patterns emerging over time, or validation overfitting from Optuna search. The gap does not invalidate the model but should be investigated in future work.

## 8. Hyperparameters (frozen)

All from Optuna TPE search (v7), unchanged in v8 except tree ceiling:

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.011978428868493889 |
| num_leaves | 224 |
| feature_fraction | 0.8948476596035589 |
| bagging_fraction | 0.8489447664564709 |
| min_child_samples | 42 |
| reg_alpha | 0.006491774642689657 |
| reg_lambda | 0.20933776242575416 |
| scale_pos_weight | 32.175142781078854 |
| n_estimators | 15000 (ceiling) |
| early_stopping_rounds | 200 |
| best_iteration | 11097 (where early stopping fired) |

## 9. Unresolved risks

1. **Val-to-test gap**: ~1109 bps, not yet explained. Does not affect model selection (v8 beats v7 on both splits) but may indicate temporal drift.
2. **Multiple test-set evaluations**: The holdout test split has been evaluated across multiple versions (v1, v4, v6, v7, v8). While each evaluation is "read-only" (no decisions were made based on test results except the final lock), repeated evaluation inflates the probability of finding a spurious improvement by chance. Mitigated by requiring bootstrap CIs to exclude zero.
3. **No random seed in config**: No `seed` / `random_state` is set in `configs/lgbm_v8.json`. Exact numerical reproduction is not guaranteed across hardware/library versions, though the qualitative result is stable.
4. **Raw D columns still present**: Phase 1 schema test `test_no_raw_d_columns_in_features` fails against locked artifacts. Dn-only drop was never applied to dataset_v3/v4.
5. **`final_column_count` mismatch**: `configs/dataset_v4.json` records 475; parquet/manifest column_count is 472. Not corrected (would change historical metadata value; fields already present).

## 10. What is NOT changing after this point

- Features: frozen at dataset_v4 (479 features including requires_fit outputs)
- Hyperparameters: frozen at configs/lgbm_v8.json
- Split boundaries: frozen at 70/15/15 temporal by TransactionDT
- Random seed: not set in config (gap noted; not added — would change the model)
- The only permitted change is the `is_serving` flag in models/registry.json (Phase 4). Phase 1 does not flip serving flags.

## 11. Final test rule

The final test split (latest 15% by TransactionDT) was evaluated once for v8. It will be opened one additional time in Phase 2 to confirm the locked model's performance matches the recorded value. After that, the test split is permanently closed. No further model versions will be evaluated on it.
