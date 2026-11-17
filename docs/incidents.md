# Incidents

Values below are taken from `docs/evidence/phase11/`. The release section uses `release_incident.log` (the corrected run), `release_incident_with_dashboard.log` (the rerun with dashboard snapshots and background traffic), and `drill-traffic.summary.json`. The data section uses `part1_invalid_batch.stderr`, `part1_invalid_batch.stdout`, `part1_drift.stdout`, and `data_incident_rerun.txt`.

## Release incident

### Injected fault

The live Deployment's readiness probe `httpGet` path was patched to `/readyz-does-not-exist`. The committed manifest was not edited. `kubectl rollout status` was bounded at 90s.

### First signal

`rollout status` printed `error: timed out waiting for the condition` and exited 1. The new pod then showed `Readiness probe failed: HTTP probe failed with statuscode: 404`.

### Timeline (UTC, from the logs)

Corrected run, `release_incident.log`:

- 2026-11-16T22:22:16Z baseline. Pod `fraud-api-89f7dff65-s55dn`, IP `10.244.0.23`, image `fraud-api:a7555b2`, revision 8. Golden through the Service: `0.005580357142857143`, action `approve`.
- 2026-11-16T22:22:17Z T0, fault injected.
- 90s later, rollout wait exit 1. New pod `fraud-api-75c44c8f45-5g92w` at `10.244.0.25` was `0/1`. Endpoints stayed `10.244.0.23:8000`. Request `657890db-3473-4c31-b528-5278a2b14644` was in the old pod's logs and absent from the new pod. The new pod logged two `route: unmatched` GET 404s.
- 2026-11-16T22:23:48Z T1, restore applied. Rollout succeeded.
- Verified recovery 94s after T0. Revision 10.

Dashboard rerun, `release_incident_with_dashboard.log`. One Grafana screenshot should cover T0 `2026-11-16T23:55:25Z` through T1 `2026-11-16T23:56:58Z` (verified at `2026-11-16T23:57:04Z`).

- 2026-11-16T23:55:22Z baseline. Same pod `fraud-api-89f7dff65-s55dn`, same image and imageID, revision 10. Golden `0.005580357142857143`, action `approve`, request `3a726666-9570-486c-89cd-84ce93b69b94`.
- 2026-11-16T23:55:25Z dashboard BASELINE, then T0.
- 2026-11-16T23:56:56Z dashboard DURING, while the fault was still live. Rollout wait exit 1. New pod `fraud-api-75c44c8f45-csfvg` at `10.244.0.27` was `0/1`. Endpoints stayed `10.244.0.23:8000`. Request `ff606fed-4800-4980-83fe-9d23dac69b57` was served by `fraud-api-89f7dff65-s55dn` and was absent from the new pod. The new pod logged two unmatched GET 404s (`b54da588-0a77-4850-aeac-e4938b5f92a9`, `d3ae760a-93c7-40b2-a70a-7cc1fcc08319`).
- 2026-11-16T23:56:58Z T1, restore. `deployment.apps/fraud-api configured`, then `successfully rolled out`.
- 2026-11-16T23:57:03Z dashboard AFTER. `RECOVERY VERIFIED` at `2026-11-16T23:57:04Z`. Total 99s. Revision 12. Rollout history listed revisions 1, 2, 4, 5, 11, 12.

### Impact

The old pod kept serving. Endpoints listed only `10.244.0.23`, and the Service golden request was served by that pod. Background k6 (`drill-traffic`, 1 VU, 300s measured, warmup 5s) recorded `http_failures` 0 and `semantic_failures` 0 across 1780 requests (`achieved_rps` 5.933333333333334). That single measured window includes the pre-fault traffic, the fault, and the restore, so both failure counts stayed 0 through the fault and the restore.

The dashboard did not show the incident. Snapshot values:

| Panel | BASELINE 23:55:25Z | DURING 23:56:56Z | AFTER 23:57:03Z |
| --- | --- | --- | --- |
| Prediction volume, POST /predict | 2.7553718640979494 | 6.044578768417076 | 6.044578768417076 |
| Client error ratio (4xx) | 0 | 0 | 0 |
| Server error ratio (5xx) | 0 | 0 | 0 |
| Latency p50 | 0.193 | 0.18242981640851852 | 0.1822064056939502 |
| Latency p95 | 0.6799999999999989 | 0.41470664461004136 | 0.41111111111111076 |
| Loaded model | dataset 5, model v9, value 1 | same | same |
| model_loaded | 1 | 1 | 1 |
| scrape_up | 1 | 1 | 1 |
| action approve | 16.842164358471436 | 81.05348477352393 | 81.05348477352393 |
| action review | 113.68460941968219 | 567.3743934146675 | 567.3743934146675 |
| action step_up, decline | 0 | 0 | 0 |

Volume and the action rates rose between BASELINE and DURING with the background load, and those DURING values were unchanged at AFTER. The 4xx and 5xx ratios stayed 0. The model panels stayed at v9 / dataset 5, loaded 1, scrape up 1. The probe 404s are logged as route `unmatched`, so the `/predict` panels had nothing to show for them.

### Investigation

During the fault the new ReplicaSet `fraud-api-75c44c8f45` had desired 1 and was not ready. The old ReplicaSet `fraud-api-89f7dff65` stayed desired 1 and ready 1. The Service endpoints contained only the old pod IP. A golden POST through the Service from the new pod was answered by the old pod.

### Cause

The readiness path `/readyz-does-not-exist` is not a route, so the new pod stayed unready and the rollout could not finish inside 90s. With one replica, the old pod remained the only endpoint.

### Recovery commands (exact)

Recorded before the fault and run at T1:

```bash
kubectl apply -f deploy/k8s/local/deployment.yaml && kubectl -n fraud-local rollout status deployment/fraud-api --timeout=180s
```

Restore is `kubectl apply` of the committed manifest. `rollout undo` would walk ReplicaSet history. That history is not the release: Kubernetes reuses the old ReplicaSet, so the revision number changes on restore while the pod spec returns to the file. On the first restore the revision moved from 6 to 8. `release_incident.log` records 8 to 10. The dashboard rerun records 10 to 12, and history omits the reused revisions. The revision is not a release identity.

The 94s total on the corrected run, and the 99s total on the dashboard rerun, are dominated by the deliberate 90s wait. Neither number is a recovery-time measurement.

The drill's first run read a terminating pod (`fraud-api-75c44c8f45-bp9nt`) and reported that the endpoints did not match. The drill was fixed to wait until that pod was gone and to select the ready pod. `release_incident.log` is that corrected run.

### Recovery proof (exact checks and results)

From `release_incident_with_dashboard.log` after the restore, one ready pod `fraud-api-89f7dff65-s55dn`:

- image `fraud-api:a7555b2` (`image matches fraud-api:a7555b2`)
- imageID `docker.io/library/import-2026-11-16@sha256:d8de616b56cf92ef76ec0a5f08e4ef1ce84989bd864bbeb3c563f77d79aae66a`, matching the baseline imageID
- `readinessProbe path: /readyz`
- `kubectl diff` exit code 0
- broken ReplicaSet `fraud-api-75c44c8f45` desired replicas 0
- endpoints `10.244.0.23:8000`, the ready pod
- golden `fraud_probability` `0.005580357142857143` against the same baseline, action `approve` against `approve`
- `RECOVERY VERIFIED`

The same image, imageID, probe path, diff exit 0, and golden score are in `release_incident.log`.

### Prevention measure

`tests/test_k8s_probe_paths.py` reads `deploy/k8s/local/deployment.yaml` as text, extracts the liveness and readiness `httpGet` paths, and asserts each path is a GET route on the API. A probe pointed at `/readyz-does-not-exist` fails that test before release. The test is in the `docker compose` test command.

### Observed facts vs hypotheses

Observed facts:

- The live probe path was patched; the file on disk was restored with `kubectl apply`, and `kubectl diff` exited 0.
- The rollout wait exited 1 at the 90s bound. The old pod stayed the only endpoint and served the golden request.
- k6 `http_failures` and `semantic_failures` were both 0 for the 300s window that covers the fault and the restore.
- Dashboard 4xx and 5xx stayed 0. Model, loaded, and scrape-up panels stayed at 1. Volume rose with the background traffic.
- After restore the image tag, imageID, and `/readyz` path matched the baseline, and the golden score matched `0.005580357142857143` / `approve`.

Hypotheses:

- A 4xx panel would show the readiness 404s. The snapshots stayed at 0, because those requests are route `unmatched` and the panels are `/predict`.
- The deployment revision identifies the release. The revision moved (6 to 8, then 8 to 10, then 10 to 12) while the restored pod spec matched the file.
- 94s or 99s measures recovery. Both totals are the 90s wait plus the checks around it.

## Data incident

### Injected fault

3 of 24 rows in the batch were missing `TransactionAmt`: `syn-003`, `syn-011`, and `syn-019`.

### First signal

Before the fix, stderr was one line, `POST /predict failed for syn-003: HTTP 422`, and stdout was empty. The message names one example and the status code. It does not name the field, `syn-011`, or `syn-019`.

### Timeline (UTC, from the logs)

Before the fix, captured in `part1_invalid_batch.stderr` and `part1_invalid_batch.stdout`: the process exited 1 on `syn-003`. `syn-001` and `syn-002` had already been scored. `syn-011` and `syn-019` were never sent. No report was written.

Phase 7, `part1_drift.stdout`, run at 2026-11-16T22:32:48Z: comparison rows 24, `Scoring 24 comparison rows`, overall status `CRITICAL`, and `Wrote /out/drift_report.json`. The process exited 0. `schema_missing_columns` was critical at observed 438. `missingness_rate_change` was a warning at observed 1.0, reference 0.0, threshold 0.05, listed with the other fired signals (extra columns, new categorical levels, three PSI checks, action-band volume). The stdout does not name `TransactionAmt`.

After the fix, `data_incident_rerun.txt`:

- 2026-11-16T22:55:39Z `/predict` counters before the invalid batch: 2xx `9.0`, 4xx `1.0`, 5xx `0.0`.
- 2026-11-16T22:55:39Z invalid batch: `batch rejected: 3 of 24 rows invalid; TransactionAmt missing in syn-003, syn-011, syn-019`, exit code 2. `invalid_run2` was empty.
- 2026-11-16T22:55:40Z the same three `/predict` counters, unchanged.
- 2026-11-16T22:55:40Z valid batch exit code 0, `row_count` 24, `invalid_score_count` 0, report keys match the Phase 10B report, max absolute `fraud_probability` difference `0.0`, action mismatches 0.
- 2026-11-16T22:55:51Z golden through the Service: `0.005580357142857143`, action `approve`, absolute difference from baseline `0.0`.

### Impact

Before the fix, two rows were scored and the run stopped on a non-specific 422, with no report. The Phase 7 path scored all 24 rows, exited 0, and wrote a critical report whose missingness warning does not name `TransactionAmt`. After the fix, the invalid batch produced no output and no new `/predict` requests.

### Investigation

The pre-fix error stopped at the first rejected row. The Phase 7 stdout shows the critical result coming from 438 missing columns and a pile of unrelated warnings. A missing value on three rows is not what that status names.

### Cause

The batch tool posted rows until the API returned HTTP 422 for `syn-003`. It did not check the whole batch against the contract first. `/openapi.json` requires `TransactionAmt` on `POST /predict`.

### Recovery commands (exact)

Invalid batch, after the validator change:

```bash
python3 scripts/prediction_batch_report.py \
  --payloads "$HOME/fraud-evidence/phase11/invalid_batch.json" \
  --out-dir "$HOME/fraud-evidence/phase11/invalid_run2"
```

Valid batch, same tool, default payloads:

```bash
python3 scripts/prediction_batch_report.py --out-dir "$HOME/fraud-evidence/phase11/valid_run"
```

The fix rejects the whole batch against `/openapi.json` before any request, exits 2, and writes nothing.

### Recovery proof (exact checks and results)

From `data_incident_rerun.txt`:

- invalid run exit code 2, stderr naming all three ids, `invalid_run2` empty (`total 0`)
- `/predict` counters unchanged at 2xx `9.0`, 4xx `1.0`, 5xx `0.0`, and the log line `predict counters unchanged`
- valid run exit code 0, `row_count` 24, `invalid_score_count` 0, `report keys match 10B True`, max absolute `fraud_probability` difference `0.0`, action mismatches 0
- golden `0.005580357142857143` / `approve`, absolute difference `0.0`

### Prevention measure

Regression tests in `tests/test_batch_report.py`: `test_validate_batch_reports_every_missing_amount`, `test_valid_batch_passes_validation`, `test_contract_requires_transaction_amount`, `test_run_rejects_invalid_batch_with_no_posts`, `test_http_422_message_keeps_type_and_loc_only`.

The Phase 7 path is a separate weakness and is not fixed here. `part1_drift.stdout` records it: the drift report scored all 24 rows, exited 0, and reported overall critical, with the `TransactionAmt` missingness buried among unrelated flags. A prevention for that path, not implemented in this change, is a required-field null check before scoring and a non-zero exit when the overall status is critical.

### Observed facts vs hypotheses

Observed facts:

- Pre-fix stderr was `POST /predict failed for syn-003: HTTP 422`. Stdout was empty. Two rows were scored. No report was written.
- Phase 7 scored 24 rows, exited 0, overall status `CRITICAL`, with `schema_missing_columns` at 438 and a missingness warning that does not name `TransactionAmt`.
- Post-fix, the invalid batch exited 2, named `syn-003`, `syn-011`, and `syn-019`, left `invalid_run2` empty, and left the `/predict` counters unchanged.
- The valid rerun matched the Phase 10B report with max difference `0.0` and 0 action mismatches. The golden score was unchanged.

Hypotheses:

- Exit 0 plus overall critical meant the three missing amounts were the finding. The critical line is 438 missing columns, and the stdout never names `TransactionAmt`.
- The first HTTP 422 rejected the batch as a whole. `syn-001` and `syn-002` had already been scored, and the error did not mention the other two bad rows.

## What a successful command does not prove

In both drills a command can finish cleanly while the release is still wrong, or look failed while traffic is fine. The release restore is `kubectl apply` of the committed file followed by `rollout status`. That status succeeded on the corrected run, and the revision still moved, because Kubernetes reuses the old ReplicaSet. The first run of the drill read a terminating pod after the rollout and reported the endpoints wrong. The checks that held were the image tag, the imageID, the probe path, `kubectl diff` exit 0, and the golden score within tolerance, read from the ready pod that was actually serving. The 90s wait dominates the wall clock, so the command's elapsed time is not a recovery time either. On the data side, the Phase 7 command exited 0 and wrote a report after scoring every row, including the three missing `TransactionAmt`. The batch tool's exit 2, the empty out-dir, the unchanged `/predict` counters, and the valid rerun's max difference of `0.0` are what showed the bad batch never reached the model. A rollback or a report command that exits 0 is the point where those checks start.
