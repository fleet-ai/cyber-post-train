#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-stage-574bd7b3-v1
OBSERVER=chris-cyber-qwen36-sft-stage-observer-v1
NAMESPACE=inference
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-sft-inference-stage-job.yaml"
PLAN="$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
TOKENIZER_EVIDENCE="$ROOT/docs/evidence/post_sft/2026-08-31-tokenizer-equivalence.json"
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"

configmap() {
  local stage_input=$1
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=training__init__.py="$ROOT/evals/post_sft/runtime/training__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=training_post_sft_base_surface.py="$ROOT/training/post_sft_base_surface.py" \
    --from-file=training_post_sft_staging.py="$ROOT/training/post_sft_staging.py" \
    --from-file=stage-input.json="$stage_input" --dry-run=client -o json | \
    python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)'
}

require_all_absent() {
  local resource name
  while read -r resource name; do
    if kubectl -n "$NAMESPACE" get "$resource" "$name" >/dev/null 2>&1; then
      echo "Resource $NAMESPACE/$resource/$name already exists; refusing partial replacement" >&2
      exit 1
    fi
  done <<EOF
serviceaccount $OBSERVER
role.rbac.authorization.k8s.io $OBSERVER
rolebinding.rbac.authorization.k8s.io $OBSERVER
configmap $NAME
job.batch $NAME
EOF
}

case "$MODE" in
  preview)
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_staging validate-bundle \
      --plan "$PLAN" --root "$ROOT" >/dev/null
    configmap "$PLAN" | kubectl create --dry-run=server -f - >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    echo "create-only server dry-run passed; no resources created"
    ;;
  submit)
    test "$#" = 4
    observation=$2
    cast_receipt=$3
    cast_manifest=$4
    test "$(kubectl -n fleet-train-jobs get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    test "$(kubectl -n fleet-train-jobs get job chris-cyber-qwen36-sft-bf16-cast-v4 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
    require_all_absent
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_staging validate-bundle \
      --plan "$PLAN" --root "$ROOT" >/dev/null
    stage_input=$(mktemp)
    config_map=$(mktemp)
    trap 'rm -f "$stage_input" "$config_map"' EXIT
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_staging build-input \
      --plan "$PLAN" --observation "$observation" --cast-receipt "$cast_receipt" \
      --cast-manifest "$cast_manifest" \
      --tokenizer-evidence "$TOKENIZER_EVIDENCE" --output "$stage_input"
    configmap "$stage_input" > "$config_map"
    kubectl create --dry-run=server -f "$config_map" >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    require_all_absent
    kubectl create -f "$config_map"
    kubectl create -f "$JOB"
    ;;
  *) echo "usage: $0 [preview|submit OBSERVATION CAST_RECEIPT CAST_MANIFEST]" >&2; exit 2 ;;
esac
