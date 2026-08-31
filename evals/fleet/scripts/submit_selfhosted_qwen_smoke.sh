#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-qcode-fleet-smoke-v1-r1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB=$ROOT/evals/fleet/cluster/qwen-code-selfhosted-smoke-job.yaml
CONFIG=$ROOT/evals/fleet/configs/qwen36-27b-qwen-code-selfhosted-smoke-v1.json
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$(kubectl -n "$NAMESPACE" get secret fleet-api -o json | jq -r '.data | has("FLEET_API_KEY")')" = true

fleet_api_key=$(kubectl -n "$NAMESPACE" get secret fleet-api \
  -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY=$fleet_api_key uv run python "$ROOT/evals/fleet/self_hosted.py" \
  preflight --config "$CONFIG" >/dev/null
unset fleet_api_key

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=Dockerfile.qwen-code="$ROOT/evals/fleet/Dockerfile.qwen-code" \
    --from-file=config.json="$CONFIG" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=run_selfhosted_qwen_smoke.sh="$ROOT/evals/fleet/scripts/run_selfhosted_qwen_smoke.sh" \
    --dry-run=client -o yaml
}

case "$MODE" in
  preview)
    configmap | kubectl apply --dry-run=server -f - >/dev/null
    kubectl apply --dry-run=server -f "$JOB"
    ;;
  submit)
    if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
      echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
      exit 1
    fi
    if kubectl -n "$NAMESPACE" get configmap "$NAME" >/dev/null 2>&1; then
      echo "ConfigMap $NAMESPACE/$NAME already exists; refusing to replace it" >&2
      exit 1
    fi
    configmap | kubectl apply -f -
    kubectl apply -f "$JOB"
    ;;
  *)
    echo "usage: $0 [preview|submit]" >&2
    exit 2
    ;;
esac
