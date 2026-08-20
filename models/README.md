# Models Directory

Trained model artifacts. Binary files (.txt, .pkl) are gitignored; JSON metadata is committed.

## Current serving model

**lgbm_v9** — Dn-only features (dataset_v5), Optuna-tuned hyperparameters
- Config: `configs/lgbm_v9.json`
- Dataset config: `configs/dataset_v5.json`
- Test AUC-PR: 0.5952
- Registry: `models/registry.json` (`is_serving: true`)
- Lock report: `docs/model_lock_report.md`
