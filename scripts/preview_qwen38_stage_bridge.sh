#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
NAMESPACE=fleet-train-jobs
NAME=chris-cyber-qwen38-stage-bridge-1d4bf0f2-v1
PLAN=$ROOT/configs/qualification/qwen38-27b-stage-bridge-v1.json
LOCK=$ROOT/configs/models/qwen38-27b-1d4bf0f2.lock.json
CODE=$ROOT/training/model_stage_bridge.py
JOB=$ROOT/cluster/jobs/chris-cyber-qwen38-stage-bridge-1d4bf0f2-v1.yaml

if [[ ${1:-preview} != preview || $# -gt 1 ]]; then
  echo "This reviewed rail is dry-run only; execution requires a follow-up approval and change." >&2
  exit 2
fi

test "$(kubectl config current-context)" = "$CONTEXT"
python3 "$CODE" validate --plan "$PLAN" --model-lock "$LOCK" >/dev/null
expected_job_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution"]["job_manifest_sha256"])' "$PLAN")
actual_job_sha=sha256:$(shasum -a 256 "$JOB" | awk '{print $1}')
test "$actual_job_sha" = "$expected_job_sha"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$(kubectl -n "$NAMESPACE" get pvc sfs-shared -o jsonpath='{.status.phase}')" = Bound
verify_storage_binding() {
  local label=$1 expected namespace claim volume_name pvc_uid pv_uid pvc_json pv_json
  expected=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["storage_topology"][sys.argv[2]], separators=(",",":")))' "$PLAN" "$label")
  namespace=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["namespace"])' "$expected")
  claim=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["name"])' "$expected")
  volume_name=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["volume_name"])' "$expected")
  pvc_uid=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["uid"])' "$expected")
  pv_uid=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["pv_uid"])' "$expected")
  pvc_json=$(kubectl -n "$namespace" get pvc "$claim" -o json)
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["uid"])' <<<"$pvc_json")" = "$pvc_uid"
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"]["volumeName"])' <<<"$pvc_json")" = "$volume_name"
  pv_json=$(kubectl get pv "$volume_name" -o json)
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["uid"])' <<<"$pv_json")" = "$pv_uid"
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"]["claimRef"]["namespace"])' <<<"$pv_json")" = "$namespace"
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"]["claimRef"]["name"])' <<<"$pv_json")" = "$claim"
  test "$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"]["claimRef"]["uid"])' <<<"$pv_json")" = "$pvc_uid"
}
verify_storage_binding inference_claim
verify_storage_binding training_claim
inference_pvc_uid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["storage_topology"]["inference_claim"]["uid"])' "$PLAN")
training_pvc_uid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["storage_topology"]["training_claim"]["uid"])' "$PLAN")
inference_model_uid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_identity"]["uid"])' "$PLAN")
revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["revision"])' "$PLAN")
source_path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_identity"]["source_path"])' "$PLAN")
test "$inference_pvc_uid" != "$training_pvc_uid"
test "$(kubectl -n inference get inferencemodel qwen3.8-27b -o jsonpath='{.metadata.uid}')" = "$inference_model_uid"
test "$(kubectl -n inference get inferencemodel qwen3.8-27b -o jsonpath='{.spec.model.revision}')" = "$revision"
test "$(kubectl -n inference get inferencemodel qwen3.8-27b -o jsonpath='{.spec.model.sourcePath}')" = "$source_path"
test "$(kubectl -n inference get inferencemodel qwen3.8-27b -o jsonpath='{.status.phase}')" = ready

if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
  echo "Job $NAMESPACE/$NAME already exists; refusing even a misleading launch preview." >&2
  exit 1
fi
if kubectl -n "$NAMESPACE" get configmap "$NAME" >/dev/null 2>&1; then
  echo "ConfigMap $NAMESPACE/$NAME already exists; refusing even a misleading launch preview." >&2
  exit 1
fi

temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
kubectl -n "$NAMESPACE" create configmap "$NAME" \
  --from-file=model_stage_bridge.py="$CODE" \
  --from-file=plan.json="$PLAN" \
  --from-file=model-lock.json="$LOCK" \
  --dry-run=client -o json | python3 -c \
  'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)' \
  >"$temporary"
kubectl create --dry-run=server -f "$temporary" >/dev/null
kubectl create --dry-run=server -f "$JOB" >/dev/null
echo "Qwen3.8 stage bridge local and server dry-runs passed; no resources or model bytes were created"
