#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen38-qcode-fleet-test20-base-v1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB=$ROOT/evals/fleet/cluster/qwen38-code-selfhosted-test20-job.yaml
CONFIG=$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-test20-base-v1.json
SPLIT=$ROOT/configs/data/fleet-a62-task-split-v1.json
MODE=${1:-preview}
tmp_dir=$(mktemp -d)
receipt_path=$tmp_dir/holdout-receipt.json
gate_receipt_path=$tmp_dir/gate-receipt.json
trap 'rm -rf "$tmp_dir"' EXIT

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$(kubectl -n "$NAMESPACE" get secret fleet-api -o json | jq -r '.data | has("FLEET_API_KEY")')" = true

fleet_api_key=$(kubectl -n "$NAMESPACE" get secret fleet-api \
  -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY=$fleet_api_key uv run python -m evals.fleet.holdout \
  preflight --config "$CONFIG" --split "$SPLIT" --receipt-out "$receipt_path" >/dev/null
unset fleet_api_key

if test "$MODE" = submit; then
  : "${Q38_CALIBRATION_RECEIPT:?set Q38_CALIBRATION_RECEIPT to the exact V3 frozen receipt}"
  : "${Q38_CALIBRATION_ROOT:?set Q38_CALIBRATION_ROOT to the exact V3 campaign root}"
  uv run python -m evals.fleet.qwen38_holdout_gate build \
    --config "$CONFIG" \
    --calibration-receipt "$Q38_CALIBRATION_RECEIPT" \
    --campaign-root "$Q38_CALIBRATION_ROOT" \
    --receipt-out "$gate_receipt_path" >/dev/null
else
  printf '%s\n' '{"preview_only":true,"launch_authorized":false}' >"$gate_receipt_path"
fi

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=Dockerfile.qwen-code="$ROOT/evals/fleet/Dockerfile.qwen-code" \
    --from-file=config.json="$CONFIG" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=gate-receipt.json="$gate_receipt_path" \
    --from-file=holdout.py="$ROOT/evals/fleet/holdout.py" \
    --from-file=qwen38_holdout_gate.py="$ROOT/evals/fleet/qwen38_holdout_gate.py" \
    --from-file=receipt.json="$receipt_path" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=split.json="$SPLIT" \
    --from-file=run_selfhosted_qwen38_test20.sh="$ROOT/evals/fleet/scripts/run_selfhosted_qwen38_test20.sh" \
    --dry-run=client -o yaml
}

if test "$MODE" = preview; then
  configmap | kubectl create --dry-run=server -f - >/dev/null
  kubectl create --dry-run=server -f "$JOB" >/dev/null
  jq -n \
    --arg name "$NAME" \
    --arg receipt_sha256 "$(jq -r '.receipt_sha256' "$receipt_path")" \
    '{ok:true,name:$name,planned_sessions:20,launch_authorized:false,holdout_receipt_sha256:$receipt_sha256}'
  exit 0
fi

if kubectl -n "$NAMESPACE" get job "$NAME" >/dev/null 2>&1; then
  echo "Job $NAMESPACE/$NAME already exists; refusing to replace it" >&2
  exit 1
fi
if kubectl -n "$NAMESPACE" get configmap "$NAME" >/dev/null 2>&1; then
  echo "ConfigMap $NAMESPACE/$NAME already exists; refusing to replace it" >&2
  exit 1
fi
configmap | kubectl create -f -
kubectl create -f "$JOB"
