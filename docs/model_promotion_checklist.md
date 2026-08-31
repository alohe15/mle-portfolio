# Model Promotion Checklist — lgbm_v9

**Candidate version**: v9
**Dataset version**: v5
**Registered at**: 2026-08-31T17:13:58.435174+00:00
**Git commit**: ed189fe488e5c79bbf67a76c2bd8268582ab7dbf

This checklist must be completed before v9 can be promoted from Candidate to Production.
Promotion is Deepa's decision. Arnav prepares the evidence; Deepa reviews and signs off.

---

## 1. Model integrity

- [ ] **Reproducible**: a new person can reproduce the model from README.md commands
- [ ] **Deterministic**: reproduced AUC-PR matches recorded value (0.5952), with any difference explained
- [ ] **Immutable**: features, hyperparameters, and split have not changed since model lock (Phase 1)
- [ ] **Versioned**: registry entry traces to exact git commit, configs, dataset, and manifests

## 2. Evaluation completeness

- [ ] **Test AUC-PR**: 0.5952 — evaluated once on the held-out test split after all decisions were frozen
- [ ] **Bootstrap significance**: v9 vs v1 baseline — CI excludes zero
- [ ] **SHAP analysis**: top 20 features documented, no unexpected dominance patterns
- [ ] **Feature-group ablation**: 6 groups tested, results consistent with SHAP
- [ ] **Error analysis**: by confidence band, product code, and temporal thirds
- [ ] **Val-to-test gap**: documented and acknowledged as a known risk

## 3. Decision policy

- [ ] **Calibration**: isotonic regression verified — calibrated mean matches fraud rate (gap < 0.005)
- [ ] **Thresholds**: derived from expected value optimization on validation split (0.036/0.051/0.820)
- [ ] **Sensitivity analysis**: review cost and fraud cost multiplier identified as highest-leverage assumptions
- [ ] **Cost assumptions**: all marked as unconfirmed — policy status is "proposed"
- [ ] **Customer impact**: friction and decline volume documented
- [ ] **Review capacity**: fits within assumed 200/day capacity (198/day observed)

## 4. Holdout discipline

- [ ] **Training split**: first 70% by TransactionDT — used for model fitting only
- [ ] **Validation split**: middle 15% — used for early stopping, SHAP, ablation, calibration, threshold optimization
- [ ] **Test split**: last 15% — evaluated exactly once after all model decisions frozen
- [ ] **No leakage**: `requires_fit` transforms fit on training split only
- [ ] **No test-split influence**: no threshold, feature, or hyperparameter decision used test performance

## 5. Serving readiness

- [ ] **API loads from registry**: `app.py` reads `registry.json` → model + manifest + dataset config + decision policy
- [ ] **Calibration in serving path**: API loads calibrator and applies it before threshold mapping
- [ ] **Action assignment**: API returns `fraud_probability`, `raw_score`, `action`, `model_version`, `dataset_version`, `policy_version`
- [ ] **Backward compatibility**: API works without decision policy config (returns probability only)
- [ ] **Latency**: response time under 200ms (note if not yet benchmarked)

## 6. Documentation

- [ ] **README.md**: reproduction commands, expected outputs, approximate runtimes
- [ ] **Model lock report**: D/Dn comparison, temporal stability, version selection rationale
- [ ] **Final evaluation**: v1-to-v9 table, SHAP, ablation, error analysis, limitations
- [ ] **Model card**: intended use, training data, performance, limitations, monitoring needs
- [ ] **Decision policy**: calibration, cost model, thresholds, sensitivity, missing business inputs
- [ ] **Engineering standards**: all code follows ENGINEERING_STANDARDS.md

## 7. Tests

- [ ] **Feature schema tests**: feature names, order, duplicates, dtypes, no raw D columns
- [ ] **Training smoke test**: train.py completes on throwaway config
- [ ] **Metrics validation**: required fields, finite values, valid ranges
- [ ] **Registry validation**: unique versions, valid paths, required fields
- [ ] **Threshold policy tests**: boundary scores, empty bands, extreme samples, cost model, calibration, config schema
- [ ] **All tests passing**: 80 tests, 0 failures

---

## Promotion decision

| Field | Value |
|-------|-------|
| Decision | [ ] Promote to Production / [ ] Reject with feedback |
| Decided by | Deepa |
| Date | |
| Notes | |

If promoted:
```bash
python scripts/register_model.py --version 9 --stage Production --promoted-by "Deepa"
```

If rejected: document feedback in this checklist, address issues, re-register as Candidate with a new commit.
