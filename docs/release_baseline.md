# Release baseline

Recorded from `~/fraud-evidence/step0.log`.

| Item | Value |
|---|---|
| Commit | `9b5347598b6a0c6bc4be07552f459b00385c236f` |
| Model | v9 |
| Dataset | 5 |
| ConfigMap `LOG_LEVEL` | `INFO` |
| ConfigMap `UVICORN_WORKERS` | `1` |
| sha256 of `artifacts.serving.sha256` | `773fd9b381310a636b5941ab6c01bfc5dbf93daaaf4bdf8159cf048456c6d649` |
| sha256 of `artifacts.sha256` | `e841d4853312d1df3b3abfebeb77c9ed6f5dc4b96dd03024d2fe72cf544dbdce` |
| Image config digest | `sha256:2b47aa6c0be6973d5fe05ace29aba409827d139fd49d16f0ba996c0ab8ec66f0` |
| Pod imageID | `docker.io/library/import-2026-11-09@sha256:0ebc615b30276f2f08eeabb128fc41ed392c3540ebc31b0ccddc77322b72a506` |
| Golden `fraud_probability` | `0.005580357142857143` |
| Golden action | `approve` |
| Golden `action_costs` | see below |

Golden `action_costs`:

```json
{
  "approve": 1.4006696428571428,
  "step_up": 6.660184151785715,
  "review": 10.070033482142858,
  "decline": 124.79966517857143
}
```

Score tolerance: absolute difference <= 1e-6. Status: proposed, rationale in [Score tolerance: rationale](#score-tolerance-rationale), awaiting agreement with Deepa.

The Docker image ID changes on identical rebuilds, so the config digest is the identity.

## Score tolerance: rationale

Origin. `PARITY_ATOL` in `tests/test_training_serving_parity.py` is `1e-6`, for training versus serving on raw scores and calibrated probabilities.

Observed. Every cited golden `fraud_probability` is `0.005580357142857143` (absolute difference 0): fresh clone, rebuild, fresh-checkout build, re-pin, and repeat calls. Same-image training versus serving is absolute difference 0 (`docs/release_evidence.md`). The CPU-limit run (`docs/load_test.md`) does not record this probability. Fifty Service calls: max |diff| 0.0, 0 action mismatches. Two `tests/load/payloads.json` runs: max |diff| 0.0 versus each other and versus `~/fraud-evidence/batch/predictions.jsonl`, 0 action mismatches. Table: `docs/evidence/phase10a/score_stability.txt`.

Why not exact equality. Float results can change in the last digits across library versions, CPU instruction sets, or thread counts with no change in behaviour. Exact equality would make the check flaky. Float rounding is about 1e-16 relative.

Why not looser. At amount 125.5, the smallest increase in p that changes the action is 0.028902401477832525 (approve to step_up), 28902.401477832525 times `1e-6`. No decrease, down to p = 0, changes the action (approve is already the lowest-cost action). A real artifact, model, or transform change is expected to move scores far more than 1e-6.

Limit. `0.005580357142857143` equals 5/896, which suggests it sits on a flat step of the isotonic calibrator. Small raw-score changes may not move it at all, so the golden check is coarse. The parity test (raw scores, same 1e-6) and the startup checksums cover smaller changes.

Rule. Absolute difference <= 1e-6 on `fraud_probability`, and `recommended_action` must match exactly. Scope: a release and recovery check on a fixed request, not a model-quality metric.

Status: proposed, rationale above, awaiting agreement with Deepa.
