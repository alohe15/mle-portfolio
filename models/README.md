# Models Directory

Trained model artifacts. Binary files (.txt, .pkl) are gitignored; JSON metadata is committed.

## Current serving model

**lgbm_v8** — Tree ceiling resolution for Optuna-tuned v7, dataset_v4
- Config: `configs/lgbm_v8.json`
- Test AUC-PR: 0.5909
- Converged at iteration 11097 / 15000
- Lock report: `docs/model_lock_report.md`
