# Models Directory

Trained model artifacts live here. Binary files (`.txt`, `.pkl`) are gitignored;
JSON metadata (registry, manifests, metrics) is committed.

`models/registry.json` is the **single source of truth** for model lifecycle.
Do not maintain a parallel tracking system that can disagree with it.

## Lifecycle states

| State | Meaning | Serving? |
|-------|---------|----------|
| **Experimental** | Model under active training/tuning. No evaluation package required. | No |
| **Candidate** | Frozen model with full evaluation evidence registered. Eligible for production promotion pending mentor approval. | No (until promoted) |
| **Production** | Promoted Candidate that is actively serving. Requires mentor sign-off and a documented rollback plan. | Yes (`is_serving: true`) |

Legacy registry rows (versions 1–9 before Candidate registration) use the
`is_serving` flag instead of an explicit `status` field. Treat
`is_serving: true` as the current Production/serving model, and other legacy
rows without `status` as Experimental training history.

Additional terminal state:

- **Rolled_back** — a former Production/Candidate that was reverted. Artifacts
  are retained; nothing is deleted.

## Experimental → Candidate

Evidence required (all paths must exist on disk; registration is validation-first
via `scripts/register_model.py`):

1. Frozen model artifact (`model_path`)
2. Reproducible model config (`config_path`)
3. Dataset version with dataset manifest (`dataset_manifest_path`)
4. Tuning / model manifest (`tuning_manifest_path` / `manifest_path`)
5. Fitted transforms pickle (`fitted_transforms_path`)
6. Calibrator (`calibrator_path`)
7. Decision policy config (`decision_policy_config`)
8. Evaluation on the **validation** split only: AUC-PR, argmin vs threshold
   cost comparison, sensitivity analysis — referenced under
   `evaluation_evidence`
9. Exact git commit hash (`commit`) for the evaluation evidence

Register with:

```bash
python scripts/register_model.py v9
```

### Commit hash semantics

The Candidate entry's `commit` field is `git rev-parse HEAD` **at the moment
`register_model.py` runs**. That is the evaluation-evidence commit (the Phase 3
decision-policy commit), **not** the later commit that appends the Candidate
row to the registry. This is intentional: the hash traces the code and
artifacts that *produced* the evidence, not the bookkeeping commit that
*recorded* registration.

## Candidate → Production

1. Mentor (`promotion_owner`, currently Deepa) reviews evaluation evidence
2. Walk `docs/model_promotion_checklist.md` and sign off on the PR
3. On approval:
   - Set the previous serving entry's `is_serving` to `false`
   - Set the promoted entry's `is_serving` to `true` and `status` to `Production`
4. Document rollback expectations in the promotion PR

## Rollback expectations

If the Production model degrades:

1. Identify the previous Production / serving registry entry
2. Revert `is_serving: true` to that previous entry
3. Change the failing entry's `status` to `Rolled_back`
4. **Do not delete** model files, manifests, calibrators, or metrics

## Current serving model

**lgbm_v9** — Dn-only features (dataset_v5), Optuna-tuned hyperparameters

- Config: `configs/lgbm_v9.json`
- Dataset config: `configs/dataset_v5.json`
- Test AUC-PR: 0.5952
- Registry: `models/registry.json` (`is_serving: true` on the legacy v9 row)
- Candidate registration (when present): same `version: 9` with `status: Candidate`
- Lock report: `docs/model_lock_report.md`
- Decision policy: `docs/decision_policy.md`
- Promotion checklist: `docs/model_promotion_checklist.md`
