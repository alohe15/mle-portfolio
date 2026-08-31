# Models Directory

Trained model artifacts. Binary files (.txt, .pkl) are gitignored; JSON metadata is committed.

## Lifecycle states

| State | Meaning | Who sets it | Requirements |
|-------|---------|-------------|-------------|
| **Experimental** | Training run exists but not fully evaluated. May be superseded. | `scripts/train.py` (automatic) | Config, metrics, manifest exist |
| **Candidate** | Frozen, fully evaluated, decision policy designed. Ready for promotion review. | `scripts/register_model.py` | All Experimental requirements PLUS: model lock report, final evaluation, model card, decision policy, all tests passing, git commit recorded |
| **Production** | Approved by mentor for production deployment. | Deepa (promotion owner) | All Candidate requirements PLUS: promotion checklist completed and signed off |

## Required evidence by stage

### Experimental → Candidate

- [ ] Model lock report (`docs/model_lock_report.md`)
- [ ] Final evaluation report (`docs/final_evaluation.md`)
- [ ] Model card (`docs/model_card.md`)
- [ ] Decision policy (`docs/decision_policy.md`)
- [ ] SHAP analysis completed
- [ ] Feature-group ablation completed
- [ ] Bootstrap significance vs baseline
- [ ] Calibration applied and verified
- [ ] All tests passing
- [ ] Clean git commit (no uncommitted changes)

### Candidate → Production

- [ ] Promotion checklist completed (`docs/model_promotion_checklist.md`)
- [ ] Deepa reviews and approves
- [ ] Rollback plan documented

## Current state

| Version | Dataset | Stage | Serving | Test AUC-PR |
|---------|---------|-------|---------|-------------|
| v9 | v5 | **Candidate** | Yes | 0.5952 |
| v8 | v4 | Experimental | No | 0.5909 |
| v7 | v4 | Experimental | No | 0.5837 |

## Promotion owner

**Deepa** (krishnamurthy.deepa@gmail.com) is the sole promotion owner. Only Deepa can move a model from Candidate to Production. The promotion decision is documented in the promotion checklist.

## Rollback

If a Production model must be reverted:
1. Set the previous Production version's `is_serving: true` in `registry.json`
2. Set the failing version's `is_serving: false`
3. Restart the API — it reads `registry.json` on startup
4. Document the rollback reason in the failing version's registry entry
5. No code changes required — the API is model-agnostic (§10)
