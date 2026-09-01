#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CAMPAIGN_ID=chris-cyber-q38-qcode-reward-cal-p1-v2
OUT_PARENT=${FLEET_EVAL_OUT_PARENT:-$ROOT/artifacts/qwen38-fleet-calibration}
OUT_DIR=$OUT_PARENT/$CAMPAIGN_ID
CONFIG=$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v2.json
SPLIT=$ROOT/configs/data/fleet-a62-task-split-v1.json
RECEIPT=$OUT_PARENT/qwen38-27b-qwen-code-reward-calibration-pass1-v2-receipt.json
QWEN_IMAGE=chris/qwen-code:0.22.3-q38-cal-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

MODE=${1:-preview}
case "$MODE" in
  preview|run) ;;
  *) echo "usage: $0 [preview|run]" >&2; exit 2 ;;
esac

if test -z "${FLEET_API_KEY:-}"; then
  test "$(kubectl -n fleet-train-jobs get secret fleet-api -o json | jq -r '.data | has("FLEET_API_KEY")')" = true
  fleet_calibration_key=$(kubectl -n fleet-train-jobs get secret fleet-api \
    -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
  export FLEET_API_KEY=$fleet_calibration_key
fi

test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
docker info >/dev/null
mkdir -p "$OUT_PARENT"
if test "$MODE" = preview; then
  RECEIPT=$OUT_PARENT/qwen38-27b-qwen-code-reward-calibration-pass1-v2-preview-receipt.json
fi
test ! -e "$RECEIPT"
if test "$MODE" = run; then
  test ! -e "$OUT_DIR"
fi

uv run --no-project --with httpx==0.28.1 python -m evals.fleet.qwen38_calibration \
  preflight --config "$CONFIG" --split "$SPLIT" --receipt-out "$RECEIPT" \
  >"$OUT_PARENT/preflight.json"

if test "$MODE" = preview; then
  jq '{ok,campaign_id,planned_sessions,model_revision,sealed_test_tasks_selected,receipt_sha256}' \
    "$OUT_PARENT/preflight.json"
  exit 0
fi

docker build --pull --platform linux/amd64 \
  --tag "$QWEN_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.qwen-code" \
  "$ROOT/evals/fleet"
test "$(docker image inspect "$QWEN_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm --platform linux/amd64 "$QWEN_IMAGE" qwen --version)" = 0.22.3
docker pull --platform linux/amd64 "$PROXY_IMAGE"

capture_image_evidence() {
  if test -d "$OUT_DIR"; then
    docker image inspect "$QWEN_IMAGE" >"$OUT_DIR/qwen-code-image-inspect.json" || true
    docker image inspect "$PROXY_IMAGE" >"$OUT_DIR/proxy-image-inspect.json" || true
  fi
}
trap capture_image_evidence EXIT

export QWEN_CODE_IMAGE=$QWEN_IMAGE
export FIXED_PROXY_IMAGE=$PROXY_IMAGE
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.qwen38_calibration \
  run --config "$CONFIG" --split "$SPLIT" --receipt "$RECEIPT" \
  --out-dir "$OUT_DIR" --proxy-script "$ROOT/evals/fleet/fixed_proxy.py"

test -s "$OUT_DIR/summary.json"
jq -e '.planned_sessions == 20 and .model_outcomes == 20 and .infrastructure_errors == 0' \
  "$OUT_DIR/summary.json" >/dev/null
