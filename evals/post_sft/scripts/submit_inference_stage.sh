#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-stage-574bd7b3-v1
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
    --from-file=training__init__.py="$ROOT/training/__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=training_post_sft_staging.py="$ROOT/training/post_sft_staging.py" \
    --from-file=stage-input.json="$stage_input" --dry-run=client -o yaml
}

case "$MODE" in
  preview)
    configmap "$PLAN" | kubectl apply --dry-run=server -f - >/dev/null
    kubectl apply --dry-run=server -f "$JOB" >/dev/null
    echo "server dry-run passed; no resources created"
    ;;
  submit)
    test "$#" = 3
    observation=$2
    raw_manifest=$3
    test "$(kubectl -n fleet-train-jobs get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    test "$(kubectl -n fleet-train-jobs get job chris-cyber-qwen36-sft-evidence-v1 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
    if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
      echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
      exit 1
    fi
    stage_input=$(mktemp)
    trap 'rm -f "$stage_input"' EXIT
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_staging build-input \
      --plan "$PLAN" --observation "$observation" --raw-manifest "$raw_manifest" \
      --tokenizer-evidence "$TOKENIZER_EVIDENCE" --output "$stage_input"
    configmap "$stage_input" | kubectl apply -f -
    kubectl apply -f "$JOB"
    ;;
  *) echo "usage: $0 [preview|submit OBSERVATION RAW_MANIFEST]" >&2; exit 2 ;;
esac
