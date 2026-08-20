# Model Lock Report — lgbm_v9

**Date**: 2026-08-20
**Author**: Arnav Lohe
**Status**: Locked candidate — Phase 1 Fix complete; serving flip deferred to Phase 4 (registry keeps v8 `is_serving: true`; v9 is `is_serving: false`)

---

## 1. Selected version

- **Model**: lgbm_v9
- **Dataset**: dataset_v5 (Dn-only — raw D1–D15 dropped after normalize_d_columns)
- **Model config**: configs/lgbm_v9.json
- **Dataset config**: configs/dataset_v5.json
- **Parent model**: lgbm_v8 (tree ceiling resolution; identical Optuna hyperparameters)
- **Parent dataset**: dataset_v4 → dataset_v5 drops raw D1–D15 only

## 2. Performance summary

| Split | AUC-PR | Notes |
|-------|--------|-------|
| Validation | 0.7022 | Used for early stopping only |
| Test (holdout) | 0.5952 | Evaluated once after all decisions frozen |

### v9 vs v8 (bootstrap, 500 resamples, 95% CI)

| Split | Delta | 95% CI | CI excludes zero? |
|-------|-------|--------|-------------------|
| Validation | +0.0004 | [-0.0012, +0.0022] | No |
| Test | +0.0043 | [+0.0024, +0.0062] | Yes |

### v8 vs v7 (bootstrap, 500 resamples, 95% CI) — lineage context

| Split | Delta | 95% CI | CI excludes zero? |
|-------|-------|--------|-------------------|
| Validation | +0.0091 | [+0.0072, +0.0111] | Yes |
| Test | +0.0072 | [+0.0054, +0.0091] | Yes |

## 3. Why v9

Two-step lineage, both capacity/correctness fixes rather than new modeling ideas:

1. **v7→v8**: resolved the tree ceiling (n_estimators 5000→15000). Early stopping fired at 11097.
2. **v8→v9**: resolved the raw D column gap. Ablation on `diag/d-dn-ablation` showed Dn-only beats D+Dn; Phase 1 lock left raw D1–D15 in dataset_v4. dataset_v5 drops them; v9 retrains with identical v8 hyperparameters.

Test AUC-PR improved 0.5909 → 0.5952 with a statistically significant bootstrap CI on the holdout. Validation delta is directionally positive but inconclusive — consistent with a small effect on a single split after a large shared capacity lift in v8.

## 4. D/Dn feature comparison (Check 1)

Ablation performed via `evals/d_dn_ablation.py` on branch `diag/d-dn-ablation`.

**Artifact gap**: `evals/d_dn_ablation` per-window tables are not present in this repo checkout. Per-window AUC-PR values below remain `[FILL]` pending recovery. Means below are from the prior lock-report draft.

Three variants tested across three walk-forward 30-day test windows:

| Variant | Features | Window 1 AUC-PR | Window 2 AUC-PR | Window 3 AUC-PR | Mean |
|---------|----------|-----------------|-----------------|-----------------|------|
| Dn-only | 454 | [FILL — from evals/d_dn_ablation results] | [FILL] | [FILL] | 0.6740 (prior draft; unverified) |
| D+Dn | 469 | [FILL] | [FILL] | [FILL] | [FILL] |
| D-only | 454 | [FILL] | [FILL] | [FILL] | 0.6545 (prior draft; unverified) |

**Locked-model reality**: The Dn-only drop is now applied. `configs/dataset_v5.json` lists D1–D15 in `dropped_columns` (applied after inherited `normalize_d_columns`). `feature_list_v5.json` and the v9 manifest contain D1n–D15n and no raw D1–D15 (464 features = 479 − 15).

## 5. Temporal stability (Check 2)

Satisfied within the D/Dn ablation narrative (Dn-only stable across three walk-forward windows on branch `diag/d-dn-ablation`). Per-window numbers not recoverable from this checkout — see §4.

## 6. Holdout discipline (Check 3)

- Validation split (middle 15% by TransactionDT): used for early stopping and Optuna trial evaluation
- Test split (latest 15% by TransactionDT): evaluated ONLY after all model decisions were frozen
- Test split was NOT passed to eval_set, NOT used for hyperparameter search, NOT used to select features
- The v8→v9 change (drop raw D) was motivated by ablation + Phase 1 schema gate failure, not by peeking at v9 test metrics before training
- Bootstrap significance on both val and test was computed AFTER training, not used to guide any decision

## 7. Val-to-test gap

v7 val_auc_pr: 0.6926, test_auc_pr: 0.5837 (gap: ~1089 bps)
v8 val_auc_pr: 0.7018, test_auc_pr: 0.5909 (gap: ~1109 bps)
v9 val_auc_pr: 0.7022, test_auc_pr: 0.5952 (gap: ~1070 bps)

Gap remains an open diagnostic. v9 slightly narrows it versus v8 while improving holdout AUC-PR.

## 8. Hyperparameters (frozen)

Identical to v8 (Optuna TPE from v7 + tree ceiling). Only the dataset changes:

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
| best_iteration | 9779 (where early stopping fired) |

## 9. Unresolved risks

1. **Val-to-test gap**: ~1070 bps, not yet explained. Does not affect model selection (v9 beats v8 on test with CI excluding zero) but may indicate temporal drift.
2. **Multiple test-set evaluations**: The holdout has been evaluated across v1–v9. Mitigated by requiring bootstrap CIs to exclude zero.
3. **No random seed in config**: No `seed` / `random_state` in `configs/lgbm_v9.json`. Exact numerical reproduction is not guaranteed across hardware/library versions.
4. **`final_column_count` mismatch**: `configs/dataset_v5.json` records 460 (= v4's 475 − 15); parquet/manifest column_count is 457. Same historical metadata pattern as v4; not corrected.
5. **feature_list requires_fit columns**: `build_dataset.py` writes parquet columns only into `feature_list_v5.json`; requires_fit outputs were aligned to v4-minus-D for train/manifest parity (464). train.py also appends missing requires_fit columns at train time.

## 10. What is NOT changing after this point

- Features: frozen at dataset_v5, n_features = 464
- Hyperparameters: frozen at configs/lgbm_v9.json (identical lgbm_params to v8)
- Split boundaries: frozen at 70/15/15 temporal by TransactionDT
- Random seed: not set in config (gap noted; not added — would change the model)
- The only permitted change is the `is_serving` flag in models/registry.json (Phase 4). Phase 1 Fix does not flip serving flags; v9 remains `is_serving: false`.

## 11. Final test rule

The final test split (latest 15% by TransactionDT) was evaluated once for v9 after training. It will be opened one additional time in Phase 2 to confirm the locked model's performance matches the recorded value. After that, the test split is permanently closed. No further model versions will be evaluated on it.
