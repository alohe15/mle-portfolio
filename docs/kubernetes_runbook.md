# Kubernetes runbook — fraud-api (local Kind)

Step-by-step for running fraud-api in a local Kind cluster. Start from a clean checkout at the repo root.

## Prerequisites

- Docker Desktop is running.
- `kind` is installed (`brew install kind`).
- `kubectl` is installed (`brew install kubectl`).
- The working directory is the repo root.

## 1. Create the Kind cluster

```bash
kind create cluster --name fraud-local
kubectl cluster-info --context kind-fraud-local
kubectl get nodes
```

Expected: one node, `fraud-local-control-plane`, in the Ready state.

## 2. Build and load the Docker image

```bash
docker compose build fraud-api
kind load docker-image fraud-api:latest --name fraud-local
```

`kind load` is required because Kind runs its own containerd and cannot see images in Docker Desktop's local store. The image is about 426 MB.

## 3. Deploy to the cluster

Apply in this order. The namespace must exist before the other objects.

```bash
kubectl apply -f deploy/k8s/local/namespace.yaml
kubectl apply -f deploy/k8s/local/configmap.yaml
kubectl apply -f deploy/k8s/local/deployment.yaml
kubectl apply -f deploy/k8s/local/service.yaml
```

Wait for the rollout:

```bash
kubectl -n fraud-local rollout status deployment/fraud-api
```

Expected: `deployment "fraud-api" successfully rolled out`

## 4. Verify the pod is healthy

```bash
kubectl -n fraud-local get pods
```

Expected: `1/1 Running`. The pod takes about 15–20 seconds to become Ready because the readiness probe has `initialDelaySeconds: 15`, and the container must load the LightGBM booster, calibrator, and fitted transforms into memory at startup.

## 5. Test from inside the cluster

Use a temporary curl pod to call the Service by DNS name:

```bash
kubectl -n fraud-local run curl-client --rm -it --restart=Never \
  --image=curlimages/curl -- \
  curl http://fraud-api:8000/readyz
```

Expected response: `{"status":"ready","model_version":"v9","dataset_version":5,"model_loaded":true}`

Also test liveness:

```bash
kubectl -n fraud-local run curl-client --rm -it --restart=Never \
  --image=curlimages/curl -- \
  curl http://fraud-api:8000/livez
```

Expected response: `{"status":"alive"}`

## 6. Self-healing — delete a pod

```bash
kubectl -n fraud-local delete pod -l app=fraud-api
kubectl -n fraud-local get pods -w
```

The ReplicaSet controller detects the missing pod and creates a replacement. The new pod should reach `1/1 Running` within about 20 seconds.

## 7. Rollout and rollback

Trigger a harmless rollout:

```bash
kubectl -n fraud-local set env deployment/fraud-api RELEASE_ID=test-v1
kubectl -n fraud-local rollout status deployment/fraud-api
```

Then undo it:

```bash
kubectl -n fraud-local rollout undo deployment/fraud-api
kubectl -n fraud-local rollout status deployment/fraud-api
```

Check history:

```bash
kubectl -n fraud-local rollout history deployment/fraud-api
```

## 8. Teardown

```bash
kind delete cluster --name fraud-local
```

This removes the cluster, all pods, and all services. No local state persists.

## Troubleshooting

### OOMKilled (Exit Code 137, CrashLoopBackOff)

The container loads LightGBM artifacts into memory at startup. If memory limits are too low, the kernel OOM-kills the process. The deployment uses `requests: 768Mi` and `limits: 1Gi` to accommodate this. Diagnose with:

```bash
kubectl -n fraud-local describe pod -l app=fraud-api
```

Look for `Reason: OOMKilled`. Fix by increasing the memory limits in `deployment.yaml`.

### Rollout stuck at "0 of 1 updated replicas are available"

This is normal during startup. The readiness probe will not pass until the model finishes loading (about 15 seconds). Wait up to 60 seconds before investigating. If it never passes, check logs:

```bash
kubectl -n fraud-local logs -l app=fraud-api
```

### curl hangs when testing the Service

Use the Service port. The Service exposes port 8000 (`spec.ports[0].port: 8000`), which maps to the container's port 8000 (`targetPort: 8000`). Call `http://fraud-api:8000/readyz`, not port 80.
