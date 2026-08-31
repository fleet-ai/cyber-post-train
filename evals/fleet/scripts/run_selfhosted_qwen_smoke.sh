#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
RUN_ID=chris-cyber-qwen36-qwencode0223-fleet-smoke-v1
JOB_NAME=chris-cyber-qwen36-qcode-fleet-smoke-v1
OUT_PARENT=${FLEET_EVAL_OUT_PARENT:-/mnt/sfs/jobs/$JOB_NAME}
OUT_DIR=$OUT_PARENT/$RUN_ID
CONFIG=$ROOT/evals/fleet/configs/qwen36-27b-qwen-code-selfhosted-smoke-v1.json
QWEN_IMAGE=chris/qwen-code:0.22.3-fleet-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected through the cluster secret}"
test "$(uname -m)" = x86_64
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test ! -e "$OUT_DIR"

for _ in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker info >/dev/null

mkdir -p "$OUT_PARENT"
uv run --no-project --with httpx==0.28.1 python "$ROOT/evals/fleet/self_hosted.py" \
  preflight --config "$CONFIG" >"$OUT_PARENT/preflight.json"

docker build --pull --platform linux/amd64 \
  --tag "$QWEN_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.qwen-code" \
  "$ROOT/evals/fleet"
test "$(docker image inspect "$QWEN_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm "$QWEN_IMAGE" qwen --version)" = 0.22.3
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
uv run --no-project --with httpx==0.28.1 python "$ROOT/evals/fleet/self_hosted.py" \
  run \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --proxy-script "$ROOT/evals/fleet/fixed_proxy.py"

test -s "$OUT_DIR/result.json"
test -s "$OUT_DIR/cleanup.json"
python - "$OUT_DIR/cleanup.json" <<'PY'
import json
import sys

cleanup = json.load(open(sys.argv[1]))
assert cleanup["containers_removed"] is True
assert cleanup["instance_closed"] is True
PY
