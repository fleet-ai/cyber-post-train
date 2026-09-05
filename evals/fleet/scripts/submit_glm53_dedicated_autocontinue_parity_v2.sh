#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/glm53-dedicated-autocontinue-live-parity-v2.yaml"
MANIFEST_SHA256=82dc1944a5dc069d28c0f249f9c4738ffcf6278a5a60f77d130718e339ab7931
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
CAMPAIGN_SHA256=sha256:63946f224a33eb0d2c2a6fdba34358ebf9ca379e5f0f156cd137c95a9b98097e
RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-primary-campaign-release-preview-v1.json"
TOMBSTONE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-autocontinue-parity-v1-failure-tombstone.json"
TOMBSTONE_SHA256=sha256:8386d9a730abc459f8ef6a21dab67a413ff24b8fe0cae00201e2280bf27cfa73
JOB=chris-cyber-glm53-dedicated-autocontinue-parity-v2
OUT=/mnt/sfs/jobs/chris-cyber-glm53-dedicated-autocontinue-parity-v2
IMAGE=ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@sha256:ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec

test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256"
test "$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign_sha256"])' "$CAMPAIGN")" = "$CAMPAIGN_SHA256"
test "$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["receipt_sha256"])' "$TOMBSTONE")" = "$TOMBSTONE_SHA256"
uv run python -c 'import json,sys; from evals.fleet import self_hosted; value=json.load(open(sys.argv[1])); assert value["receipt_sha256"]==self_hosted.digest_without(value,"receipt_sha256")' "$TOMBSTONE"
uv run python -m evals.fleet.autocontinue_campaign --campaign "$CAMPAIGN" --release "$RELEASE" >/dev/null

require_uid() {
  local kind=$1 name=$2 expected=$3
  test "$(kubectl -n "$NS" get "$kind" "$name" -o jsonpath='{.metadata.uid}')" = "$expected"
}

require_serving() {
  local rayjob=$1 rayjob_uid=$2 cluster=$3 cluster_uid=$4 service=$5 service_uid=$6 pod=$7 pod_uid=$8
  require_uid rayjob.ray.io "$rayjob" "$rayjob_uid"
  require_uid raycluster.ray.io "$cluster" "$cluster_uid"
  require_uid service "$service" "$service_uid"
  require_uid pod "$pod" "$pod_uid"
  test "$(kubectl -n "$NS" get rayjob.ray.io "$rayjob" -o jsonpath='{.status.jobStatus}')" = RUNNING
  test "$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].ready}')" = true
  test "$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].restartCount}')" = 0
  test "$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.spec.containers[0].image}')" = "$IMAGE"
  test "$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].imageID}')" = "$IMAGE"
  test "$(kubectl -n "$NS" exec "$pod" -- curl --max-time 20 -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200
}

require_uid job chris-cyber-glm53-dedicated-autocontinue-parity-v1 9b2d01a0-e1ef-4947-a558-eacbe58f046d
require_uid pod chris-cyber-glm53-dedicated-autocontinue-parity-v1-6428x 2826699c-f94d-44e3-b5ae-73368d0f50a3
test "$(kubectl -n "$NS" get job chris-cyber-glm53-dedicated-autocontinue-parity-v1 -o jsonpath='{.status.failed}')" = 1

require_serving ft-run-98c32208 ef7cb0f2-84d4-4017-ae31-bf34ebb70d0d ft-run-98c32208-5gpb2 5107d72e-ae6a-4582-a490-55b349421d06 ft-run-98c32208-5gpb2-head-svc 2c0e64de-c4a0-4f70-ae07-d15b1adad0b3 ft-run-98c32208-5gpb2-head-7qxwc 3f37ab91-4afa-4e4d-8f29-a3eeda70a774
require_serving ft-run-9e92209d 2bbfd34a-d74a-4497-ab28-48cada0c9753 ft-run-9e92209d-pppzg f7f7ae34-6a79-482a-b771-4639b786f66a ft-run-9e92209d-pppzg-head-svc 14ae906a-60ef-4a02-b6bf-814a00c2c89f ft-run-9e92209d-pppzg-head-rc9nd bfeb4c6a-9e4a-4f14-adb2-3f516dd25fdb

test -z "$(kubectl -n "$NS" get job "$JOB" --ignore-not-found -o name)"
test -z "$(kubectl -n "$NS" get pods -l "job-name=$JOB" -o name)"
kubectl -n "$NS" exec ft-run-98c32208-5gpb2-head-7qxwc -- test ! -e "$OUT"
kubectl apply --dry-run=server -f "$MANIFEST" -o name >/dev/null

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"mode":"preview","predecessor_tombstoned":true,"job_absent":true,"sfs_root_absent":true,"serving_replicas":2}'
  exit 0
fi

kubectl create -f "$MANIFEST" -o jsonpath='{.metadata.name}{" "}{.metadata.uid}{"\n"}'
