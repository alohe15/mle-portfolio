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

Score tolerance: absolute difference <= 1e-6. Status: proposed, pending mentor approval.

The Docker image ID changes on identical rebuilds, so the config digest is the identity.
