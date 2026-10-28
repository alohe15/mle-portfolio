# Kubernetes evidence — local Kind

Recorded 2026-10-28 on branch `pr/phase9-local-kubernetes` after applying the corrected manifests in commit `77cefd0` (`77cefd0080a279260dd0ddd6b64ad5038ff59fe0`).

Image loaded into the cluster: `fraud-api:latest` `sha256:52a632748221d86d15557fafb47fe6e6e562877a3a3d751a9faf68cfea1ba58d`. `docker images` reports 1.79GB. `docker image inspect` reports `Size=426301259`. Created 2026-10-14 12:43:58 -0400. Pull policy on the pod is `Never`.

## Tool versions

```
kind v0.33.0 go1.27.0 darwin/amd64
kubectl Client Version: v1.36.1
Docker Engine 29.7.2, Docker Desktop 4.90.0
Docker Compose version v5.5.1
Kind node: kindest/node:v1.37.0
```

## Node

```
$ kubectl get nodes
NAME                        STATUS   ROLES           AGE   VERSION
fraud-local-control-plane   Ready    control-plane   20s   v1.37.0
```

## Apply and rollout

```
$ kubectl apply -f deploy/k8s/local/namespace.yaml
namespace/fraud-local created
$ kubectl apply -f deploy/k8s/local/configmap.yaml
configmap/fraud-api-config created
$ kubectl apply -f deploy/k8s/local/deployment.yaml
deployment.apps/fraud-api created
$ kubectl apply -f deploy/k8s/local/service.yaml
service/fraud-api created

$ kubectl -n fraud-local rollout status deployment/fraud-api
Waiting for deployment "fraud-api" rollout to finish: 0 of 1 updated replicas are available...
deployment "fraud-api" successfully rolled out
```

Live Service port is `8000` with `targetPort: 8000`. Live resources are requests `768Mi` / `250m` and limits `1Gi` / `500m`.

## Ready pod

```
$ kubectl -n fraud-local get pods
NAME                         READY   STATUS    RESTARTS   AGE
fraud-api-79574cd964-wbcwc   1/1     Running   0          15s
```

## In-cluster probes

```
$ kubectl -n fraud-local run curl-client --rm -i --restart=Never \
    --image=curlimages/curl -- \
    curl -sS http://fraud-api:8000/readyz
{"status":"ready","model_version":"v9","dataset_version":5,"model_loaded":true}

$ kubectl -n fraud-local run curl-client --rm -i --restart=Never \
    --image=curlimages/curl -- \
    curl -sS http://fraud-api:8000/livez
{"status":"alive"}
```

Each curl pod was deleted after it printed the body.

## Pod replacement

Deleted `fraud-api-79574cd964-wbcwc`. The ReplicaSet created `fraud-api-79574cd964-7x422`, which reached `1/1 Running` with 0 restarts.

```
$ kubectl -n fraud-local delete pod -l app=fraud-api
pod "fraud-api-79574cd964-wbcwc" deleted from fraud-local namespace

$ kubectl -n fraud-local get pods
NAME                         READY   STATUS    RESTARTS   AGE
fraud-api-79574cd964-7x422   1/1     Running   0          16s
```

## Rollout and rollback

```
$ kubectl -n fraud-local set env deployment/fraud-api RELEASE_ID=test-v1
deployment.apps/fraud-api env updated

$ kubectl -n fraud-local rollout status deployment/fraud-api
Waiting for deployment "fraud-api" rollout to finish: 1 old replicas are pending termination...
deployment "fraud-api" successfully rolled out
```

Rollout pod: `fraud-api-6bc4d9dd75-tvgb6` (`1/1 Running`). The previous pod `fraud-api-79574cd964-7x422` went to `Terminating`.

```
$ kubectl -n fraud-local rollout undo deployment/fraud-api
deployment.apps/fraud-api rolled back

$ kubectl -n fraud-local rollout status deployment/fraud-api
Waiting for deployment "fraud-api" rollout to finish: 1 old replicas are pending termination...
deployment "fraud-api" successfully rolled out
```

`kubectl` warned that rollback does not update the `last-applied-configuration` annotation. After undo, the running pod was `fraud-api-79574cd964-d7tsv` (`1/1 Running`, 0 restarts) and `RELEASE_ID` was gone. A follow-up in-cluster `/readyz` returned the same ready body.

```
$ kubectl -n fraud-local rollout history deployment/fraud-api
deployment.apps/fraud-api
REVISION  CHANGE-CAUSE
2         <none>
3         <none>
```

Revision 1 is no longer listed. The rollback reused the original ReplicaSet (`pod-template-hash=79574cd964`) and recorded it as revision 3, with no extra environment variables. Revision 2 is the `RELEASE_ID=test-v1` template (`pod-template-hash=6bc4d9dd75`), still at requests `768Mi` / limits `1Gi`.
