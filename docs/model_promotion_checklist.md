# Model promotion checklist (Candidate → Production)

Owner: **Deepa** (`promotion_owner` in the Candidate registry entry).

Use this list before promoting any Candidate. Every item is a concrete,
verifiable check — do not check a box without evidence.

## Registry and artifacts

- [ ] Candidate entry exists in `models/registry.json` with `"status": "Candidate"`
- [ ] All artifact paths in the Candidate entry resolve to existing files
      (`model_path`, `config_path`, `dataset_manifest_path`,
      `tuning_manifest_path` / `manifest_path`, `fitted_transforms_path`,
      `calibrator_path`, `decision_policy_config`, `metrics_path`)
- [ ] `commit` in the Candidate entry matches the git commit that produced the
      evaluation evidence (Phase 3 decision-policy commit), not necessarily the
      registration bookkeeping commit — see `models/README.md`
- [ ] Model config at `config_path` matches the tuning manifest (`config_path`,
      `dataset_version`, description) — no post-hoc config edits after training

## Evaluation quality

- [ ] Validation / documented AUC-PR is present under
      `evaluation_evidence.test_auc_pr` and meets the current minimum
      (lgbm_v9: **0.5952**)
- [ ] Decision policy evaluation shows argmin total expected cost ≤
      fixed-threshold total expected cost on validation (mathematical invariant)
- [ ] Bootstrap 95% CI for (argmin − threshold) expected-cost difference
      excludes zero (`evaluation_evidence.bootstrap_ci`)
- [ ] Sensitivity analysis has been reviewed — cost assumptions are either
      confirmed by the business or flagged as unconfirmed with documented
      impact ranges (`evals/sensitivity_analysis.py`, `docs/decision_policy.md`)
- [ ] Review capacity (reviews/day under argmin) is within operational limits
      **or** a capacity management plan via `cost_gap` ranking is documented in
      `docs/decision_policy.md`
- [ ] `docs/decision_policy.md` is complete — no placeholder sections
- [ ] Test split has **not** been used for any decision in this evaluation cycle
- [ ] `python -m pytest tests/test_decision_policy.py -v` passes

## Approval and rollback

- [ ] Promotion owner (Deepa) has reviewed the PR and signed off
- [ ] Rollback plan: previous Production / serving version is identified
      (current: legacy `is_serving: true` entry before promotion) and the
      revert procedure is documented:
      1. Set previous entry `is_serving: true`
      2. Set failing entry `is_serving: false` and `status: "Rolled_back"`
      3. Do not delete model artifacts
