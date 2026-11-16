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

## Phase 11 recovery baseline

Recorded after the image was pinned and rolled out, and before any release fault.

| Item | Value |
|---|---|
| Code commit | `a7555b2` |
| Image tag | `fraud-api:a7555b2` |
| Image ID (`docker image inspect --format '{{.Id}}'`) | `sha256:4317228a5def4615ed4dbc642f8dc8e39722a5ff1453ad4200cf6a51f2e623cd` |
| Image config digest | `sha256:16ab9529bcfe8afa8027aff31ed1fd999c315b6267bb09e64870ca458bc9bf89` |
| Pod imageID | `docker.io/library/import-2026-11-16@sha256:d8de616b56cf92ef76ec0a5f08e4ef1ce84989bd864bbeb3c563f77d79aae66a` |
| Deployment revision | `6` |
| ConfigMap `LOG_LEVEL` | `INFO` |
| ConfigMap `UVICORN_WORKERS` | `1` |
| Resources | requests `250m` / `768Mi`, limits `500m` / `1Gi` |
| Golden `fraud_probability` | `0.005580357142857143` |
| Golden action | `approve` |

The image ID above is the OCI index digest. The config digest is the image config blob, and that is the identity that stays put across a retag. The pod imageID is the digest Kind stored when the image was imported.

Score tolerance: absolute difference <= 1e-6 on `fraud_probability`, and the action must match exactly. Status: proposed, pending mentor approval.

Restoration command, written before any fault:

```bash
kubectl apply -f deploy/k8s/local/deployment.yaml && kubectl -n fraud-local rollout status deployment/fraud-api --timeout=180s
```
