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
    --from-file=plan.json="$PLAN" --dry-run=client -o yaml
}

case "$MODE" in
  preview)
    configmap | kubectl apply --dry-run=server -f - >/dev/null
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" >/dev/null
    echo "server dry-run passed; no resources created"
    ;;
  submit)
    test "$(kubectl -n "$NAMESPACE" get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
      echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
      exit 1
    fi
    configmap | kubectl apply -f -
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" --execute
    ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac
