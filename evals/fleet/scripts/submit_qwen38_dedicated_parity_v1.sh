#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  printf '%s\n' "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST=$ROOT/evals/fleet/cluster/qwen38-dedicated-tp1-live-parity-v1.yaml
JOB=chris-cyber-q38-dedicated-tp1-v1-parity
OUT=/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v1/parity

require_uid() {
  local kind=$1 name=$2 expected=$3
  test "$(kubectl -n "$NS" get "$kind" "$name" -o jsonpath='{.metadata.uid}')" = "$expected"
}

require_uid rayjob.ray.io ft-run-c4c7d042 9f1ecf8a-a047-44ae-8d51-45af1f4ae9ac
require_uid raycluster.ray.io ft-run-c4c7d042-vh5wn 6516f5ef-0e62-42e0-9e3a-c455e8009536
require_uid service ft-run-c4c7d042-vh5wn-head-svc 4bb3c7a4-77cc-46cc-a3a6-e204b7f46b33
require_uid pod ft-run-c4c7d042-vh5wn-head-qcgc4 205a9faf-8ee5-494c-b27c-bfeaeda812d8
test "$(kubectl -n "$NS" get rayjob.ray.io ft-run-c4c7d042 -o jsonpath='{.status.jobStatus}')" = RUNNING
test "$(kubectl -n "$NS" get pod ft-run-c4c7d042-vh5wn-head-qcgc4 -o jsonpath='{.status.containerStatuses[0].restartCount}')" = 0
test "$(kubectl -n "$NS" exec ft-run-c4c7d042-vh5wn-head-qcgc4 -- curl --max-time 20 -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200
test -z "$(kubectl -n "$NS" get job "$JOB" --ignore-not-found -o name)"
test -z "$(kubectl -n "$NS" get pods -l "job-name=$JOB" -o name)"
kubectl -n "$NS" exec allie-dev -- test ! -e "$OUT"
kubectl apply --dry-run=server -f "$MANIFEST" -o name >/dev/null

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"mode":"preview","serving_ready":true,"job_absent":true,"sfs_root_absent":true,"scored_requests":0}'
  exit 0
fi

kubectl create -f "$MANIFEST" -o jsonpath='{.metadata.name}{" "}{.metadata.uid}{"\n"}'
