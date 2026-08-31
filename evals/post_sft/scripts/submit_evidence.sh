#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-evidence-v1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-sft-evidence-job.yaml"
PLAN="$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=training__init__.py="$ROOT/training/__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=training_tokenizer_equivalence.py="$ROOT/training/tokenizer_equivalence.py" \
    --from-file=plan.json="$PLAN" --dry-run=client -o json | \
    python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)'
}

require_absent() {
  if kubectl -n "$NAMESPACE" get configmap "$NAME" >/dev/null 2>&1; then
    echo "ConfigMap $NAMESPACE/$NAME already exists; refusing to replace it" >&2
    exit 1
  fi
  if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
    echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
    exit 1
  fi
}

case "$MODE" in
  preview)
    configmap | kubectl create --dry-run=server -f - >/dev/null
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    echo "create-only server dry-run passed; no resources created"
    ;;
  submit)
    test "$(kubectl -n "$NAMESPACE" get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    require_absent
    config_map=$(mktemp)
    trap 'rm -f "$config_map"' EXIT
    configmap > "$config_map"
    kubectl create --dry-run=server -f "$config_map" >/dev/null
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    require_absent
    kubectl create -f "$config_map"
    kubectl create -f "$JOB"
    ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac
