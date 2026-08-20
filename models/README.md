# Models Directory

Trained model artifacts. Binary files (.txt, .pkl) are gitignored; JSON metadata is committed.

## Locked candidate (not yet serving)

**lgbm_v8** — Tree ceiling resolution on Optuna-tuned v7 hyperparameters, dataset_v4 (Dn-normalized; raw D still present)
- Config: configs/lgbm_v8.json
- Test AUC-PR: 0.5909
- Lock report: docs/model_lock_report.md
- Status: Phase 1 complete. Will be registered as serving in Phase 4 after clean reproduction.

**Deviation note:** On this branch, `models/registry.json` already has v8 with `is_serving: true` from a prior lock session. Phase 1 did **not** flip serving flags. Treat Phase 4 as confirmation/cleanup if needed; do not re-flip here.

## Current serving model

**lgbm_v8** (`is_serving: true` in registry from prior session)
- Config: `configs/lgbm_v8.json`
- Test AUC-PR: 0.5909
- Converged at iteration 11097 / 15000
- Lock report: `docs/model_lock_report.md`
