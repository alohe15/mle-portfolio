# Fresh-checkout self-check

This is a SELF-CHECK by the author, not an independent handoff. The independent handoff by another person is PENDING.

## What was run

On 2026-11-17 the author deleted the kind cluster, cloned the repo to `~/mle-handoff-check`, checked out `pr/phase11-recovery` at `0a0ab35`, and followed the README literally. That log is [selfcheck.log](evidence/phase11/selfcheck.log).

On 2026-11-19 the same checkout was fast-forwarded to `0edf624` and only the sections changed by that commit were rerun. The cluster was already up. That log is [selfcheck_rerun.log](evidence/phase11/selfcheck_rerun.log).

## Findings

| # | doc and section | problem found | fix (commit 0edf624) | rerun result (from Part A) |
| --- | --- | --- | --- | --- |
| 1 | `docs/artifacts.md` Fresh clone | No `cp` command and no source path for the binaries or the parquets. | `SRC=` and explicit `cp` for the three binaries, then the two parquets, then both `shasum` checks. | PASS. `SRC=~/mle-portfolio`. `artifacts.sha256` and `artifacts.serving.sha256` both exited 0. |
| 2 | `docs/container_runbook.md` Build | The build step said `cd /path/to/mle-portfolio`. | "From the repo root", then `docker build -t fraud-api .` | PASS. Build from `~/mle-handoff-check` exited 0. |
| 3 | `docs/container_runbook.md` Run | `docker compose up` started the test service, and a curl issued immediately got HTTP 000. | `docker compose up -d --wait fraud-api`, then the smoke test. | PASS. `--wait` reached Healthy and started only `fraud-api`. Smoke: 16 passed, 0 failed. A container and a port-forward left from 2026-11-17 were stopped first so port 8000 was free. The documented commands needed no further change. |
| 4 | `docs/container_runbook.md` Running tests | `python -m pytest` inside the serving image failed: no module named pytest. | `docker compose run --rm test`. | PASS. 41 passed. |
| 5 | `docs/kubernetes_runbook.md` section 1 | `kubectl get nodes` printed NotReady, and the runbook did not wait. | `kubectl wait --for=condition=Ready node --all --timeout=120s`. | PASS. The cluster already existed. The node was Ready, and the wait returned immediately (16:54:34Z). |
| 6 | `docs/kubernetes_runbook.md` section 2 | Tagging `latest` as `fraud-api:a7555b2` retargeted the pin. A frozen check against that tag fails once `scripts/` changes, because `scripts/` is copied whole. | `TAG` is the last commit that changed an image input. The manifest must already say `fraud-api:$TAG` or the guard exits 1. | PASS. `TAG=9789b59`. The guard passed. `kind load` and `rollout status` exited 0. Apply reported the deployment unchanged, so the existing pod stayed. |
| 7 | `docs/kubernetes_runbook.md` section 5 | `kubectl run -it` printed "Unable to use a TTY". | Both curls use `-i`. | PASS. `/readyz` returned the ready JSON for v9 / dataset 5. `/livez` returned `{"status":"alive"}`. No TTY error. kubectl can miss attach when curl has already exited and then streams the container log. No extra flags were added. |
| 8 | `docs/operations.md` Access | One shell could not leave the port-forward running and also run the query script. | Port-forwards stay in a second tab. The query script runs in the first tab. | PASS. Prometheus was already the documented port-forward, left running. Grafana was forwarded in its own process, because one foreground tab cannot hold both. `python3 scripts/check_dashboard_queries.py` exited 0: model v9, dataset 5, loaded 1, scrape up 1. |
| 9 | `docs/incidents.md` How to start | Neither the README nor the incident page gave the command that starts a drill. | Release drill: `bash scripts/drills/release_incident.sh`. Data drill: the invalid batch, then the valid batch. | PASS. The release drill printed `RECOVERY VERIFIED`. It also printed `image does not match fraud-api:a7555b2`, because that script still checks the old tag. No workaround was used. Invalid batch exit 2 and the out-dir was empty. Valid batch exit 0, `row_count` 24. |
| 10 | `docs/container_runbook.md` Running tests, and the README release-checks item | Compose left port 8000 bound, and a later port-forward bound it too. | `docker compose down` before the Kubernetes stage. | PASS. `docker compose down` exited 0 and port 8000 was free before the cluster steps. |
| 11 | `docs/operations.md` Cleanup | Nothing said how to delete a finished k6 Job without deleting the cluster. | `kubectl -n fraud-local get jobs` and `kubectl -n fraud-local delete job <name>`, plus `kubectl delete -f deploy/drills/fault-instance.yaml --ignore-not-found`. | PASS. The fault-instance delete exited 0. `get jobs` found none, so there was no job name to delete. Monitoring and the cluster were left up. |

## Stage summary

All 11 rerun items passed. None still needs a workaround. The documented commands were run as written after the 2026-11-17 leftovers on port 8000 were cleared.

## What was NOT exercised

Kubernetes runbook sections 6 and 7 (delete a pod, and the harmless rollout / undo). The four non-smoke load runs. Grafana screenshots. The optional `DASHBOARD_SNAPSHOT=1` release-drill mode.

## How a reviewer should run the independent handoff

Follow the README from a clean clone. Get the gitignored binaries and parquets the way [docs/artifacts.md](artifacts.md) says, and do not commit binaries or data. Record any step that is unclear, including the command that was run and what happened. This page is the author's own check. It does not replace that review.
