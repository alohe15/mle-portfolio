# Load test — local kind cluster

These runs measure the local API under a fixed synthetic mix. The resource numbers are experiment settings, not capacity targets.

## Setup

k6 `grafana/k6:2.4.0` runs as a Job in `fraud-local` and posts to `http://fraud-api:8000`. The cluster has one node, `fraud-local-control-plane`. That node runs k6, the API, Prometheus, and Grafana on the same machine. The API Deployment has one replica and one uvicorn process. Its resources are requests `250m` CPU / `768Mi` memory and limits `500m` CPU / `1Gi` memory, except the one run that raised the CPU limit to `1000m`.

The payload file is the 24-row fixed mix in `tests/load/payloads.json`. Each iteration sends only that row's `request`:

```text
payloads[(__VU * 7919 + __ITER) % 24].request
```

A 15 second warm-up at 1 user is excluded from the summary. The measured scenario starts 5 seconds after the warm-up ends and runs for 60 seconds. The smoke run used a 5 second warm-up and a 10 second measured window instead.

Peak CPU cores is the highest `rate(process_cpu_seconds_total[1m])` for the API process during the run. Peak RSS MiB is the highest `process_resident_memory_bytes / 1024 / 1024`. Each compared run stores those two numbers, in that order, in its `resources.txt`.

## How to rerun

From the repo root, with the kind cluster up:

```bash
bash scripts/run_load.sh smoke 1 10s 5s
bash scripts/run_load.sh baseline 1 60s
bash scripts/run_load.sh high 5 60s
kubectl -n fraud-local patch deployment fraud-api --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/cpu","value":"1000m"}]'
kubectl -n fraud-local rollout status deployment/fraud-api
bash scripts/run_load.sh high-cpu1000 5 60s
kubectl apply -f deploy/k8s/local/deployment.yaml
kubectl -n fraud-local rollout status deployment/fraud-api
bash scripts/run_load.sh high-repeat 5 60s
```

The patch sets the CPU limit to `1000m` (Kubernetes records that as `1`). `kubectl apply` restores the committed manifest, including the `500m` limit. Wait for each rollout before the next measured run.

## Results

Latency is milliseconds. Throughput is measured requests divided by the measured duration. Approve share is `actions.approve / requests` from the summary file. Copied files are in [docs/evidence/phase10b/load/](evidence/phase10b/load/).

| Run | Users | CPU limit | Requests | Throughput (req/s) | HTTP failures | Semantic failures | p50 | p95 | p99 | max | Peak CPU cores | Peak RSS MiB | Approve share |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| smoke | 1 | 500m | 68 | 6.800 | 0 | 0 | 123.9 | 202.7 | 206.9 | 209.1 | not measured (functional smoke check) | not measured (functional smoke check) | 8/68 (11.8%) |
| baseline | 1 | 500m | 413 | 6.883 | 0 | 0 | 123.6 | 196.9 | 240.4 | 299.7 | 0.501 | 320.1 | 52/413 (12.6%) |
| high | 5 | 500m | 272 | 4.533 | 0 | 0 | 1102.3 | 1307.7 | 1415.7 | 1581.1 | 0.501 | 349.2 | 34/272 (12.5%) |
| high-cpu1000 | 5 | 1000m | 599 | 9.983 | 0 | 0 | 496.3 | 604.9 | 671.5 | 742.0 | 1.002 | 412.8 | 76/599 (12.7%) |
| high-repeat | 5 | 500m | 265 | 4.417 | 0 | 0 | 1123.3 | 1329.4 | 1625.8 | 1873.3 | 0.502 | 412.4 | 32/265 (12.1%) |

The smoke run is a 10-second functional check and is not part of the comparison.

## Findings

1. Peak CPU equalled the limit in every run, so the API was CPU-bound.
2. 1000m vs 500m at 5 users: throughput 4.5 -> 10.0 req/s, p50 1102 -> 496 ms, larger than the difference between the two 500m runs (about 2.5%), so the evidence supports an improvement from raising the limit.
3. Little's law check: concurrency / throughput approximately equals p50 at 5 users, so the latency is mostly queueing. At 500m, 5 / 4.533 ≈ 1.10 s against a p50 of 1.10 s, and 5 / 4.417 ≈ 1.13 s against a p50 of 1.12 s. At 1000m, 5 / 9.983 ≈ 0.50 s against a p50 of 0.50 s.
4. Hypothesis, not confirmed: 5 users give lower throughput than 1 user at the same limit, possibly CFS throttling or GIL contention; the warm-up progress lines suggest about 16 req/s at 1 user with 1000m, but that was not a measured run.

## Uncertainty

Each setting was run once, except two runs at 500m and 5 users. All of them shared one laptop with the API, k6, Prometheus, and Grafana. The payloads are synthetic.

## Decision

The committed manifest stays at `500m`. Changing the release limit is a separate decision.
