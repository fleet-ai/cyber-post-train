#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
JOB_NAME=${FULL_JOB_NAME:?FULL_JOB_NAME is required}
PLAN_FILE=${FULL_PLAN_FILE:?FULL_PLAN_FILE is required}
OUT_PARENT=${FLEET_EVAL_OUT_PARENT:-/mnt/sfs/jobs/$JOB_NAME}
AGENT_IMAGE=chris/opencode:1.18.27-cyber-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected from the campaign Secret}"
test "$(uname -m)" = x86_64
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test ! -e "$OUT_PARENT"

for _ in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker info >/dev/null

docker build --pull --platform linux/amd64 \
  --tag "$AGENT_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" \
  "$ROOT/evals/fleet"
test "$(docker image inspect "$AGENT_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm "$AGENT_IMAGE" opencode --version)" = 1.18.27
docker pull --platform linux/amd64 "$PROXY_IMAGE"

export AGENT_HARNESS_IMAGE=$AGENT_IMAGE
export FIXED_PROXY_IMAGE=$PROXY_IMAGE
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.opencode_train_sweep_runner \
  --plan "$ROOT/evals/fleet/configs/$PLAN_FILE" \
  --out-dir "$OUT_PARENT" \
  --proxy-script "$ROOT/evals/fleet/fixed_proxy.py" \
  --parallel
