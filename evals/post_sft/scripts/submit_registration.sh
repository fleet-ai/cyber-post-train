#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-register-574bd7b3-v1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-sft-register-job.yaml"
MODE=${1:-preview}
RECEIPT=${2:-$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test -s "$RECEIPT"

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=training__init__.py="$ROOT/training/__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_register_post_sft.py="$ROOT/training/register_post_sft.py" \
    --from-file=serving-registration-receipt.json="$RECEIPT" --dry-run=client -o yaml
}

case "$MODE" in
  preview)
    configmap | kubectl apply --dry-run=server -f - >/dev/null
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" >/dev/null
    echo "server dry-run passed; no resources created"
    ;;
  submit)
    test "$#" = 2
    test "$(kubectl -n inference get job chris-cyber-qwen36-sft-stage-574bd7b3-v1 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
    PYTHONPATH="$ROOT" uv run python -c 'import json,sys; from training.register_post_sft import validate_registration_receipt; validate_registration_receipt(json.load(open(sys.argv[1])))' "$RECEIPT"
    if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
      echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
      exit 1
    fi
    configmap | kubectl apply -f -
    PYTHONPATH="$ROOT" uv run python -m training.cluster_submit "$JOB" --execute
    ;;
  *) echo "usage: $0 [preview|submit SERVING_REGISTRATION_RECEIPT]" >&2; exit 2 ;;
esac
