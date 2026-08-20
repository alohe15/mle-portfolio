# Models Directory

Trained model artifacts. Binary files (.txt, .pkl) are gitignored; JSON metadata is committed.

## Locked candidate (not yet serving)

**lgbm_v9** — Dn-only drop: v8 hyperparameters retrained on dataset_v5 (raw D1–D15 removed)
- Config: configs/lgbm_v9.json
- Test AUC-PR: 0.5952
- Lock report: docs/model_lock_report.md
- Status: Phase 1 Fix complete. Will be registered as serving in Phase 4 after clean reproduction. `is_serving: false`.

## Current serving model

**lgbm_v8** (`is_serving: true` in registry — left untouched per Phase 1 Fix constraints)
- Config: `configs/lgbm_v8.json`
- Test AUC-PR: 0.5909
- Converged at iteration 11097 / 15000
- Lock report: `docs/model_lock_report.md` (now documents v9 as locked candidate; v8 remains serving until Phase 4)
