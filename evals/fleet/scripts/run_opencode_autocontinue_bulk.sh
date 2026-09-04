#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
MODE=${BULK_CONTROLLER_MODE:?BULK_CONTROLLER_MODE must be preflight or run}
JOB_NAME=${BULK_JOB_NAME:?BULK_JOB_NAME is required}
PLAN=${BULK_PLAN_PATH:?BULK_PLAN_PATH is required}
OUT_ROOT=${FLEET_EVAL_OUT_ROOT:-/mnt/sfs/jobs/$JOB_NAME}
AGENT_IMAGE=chris/opencode:1.18.27-cyber-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected from the campaign Secret}"
: "${JOB_UID:?JOB_UID must come from the downward API}"
: "${POD_UID:?POD_UID must come from the downward API}"
test "$MODE" = preflight -o "$MODE" = run
test "$(uname -m)" = x86_64

if test "$MODE" = preflight; then
  PREFLIGHT_OUT=${BULK_PREFLIGHT_OUT:?BULK_PREFLIGHT_OUT is required}
  PACKAGE=${BULK_PACKAGE_PATH:?BULK_PACKAGE_PATH is required}
  AUTHORIZATION=${BULK_AUTHORIZATION_PATH:?BULK_AUTHORIZATION_PATH is required}
  QWEN_OBSERVER=${BULK_QWEN_OBSERVER_PATH:?BULK_QWEN_OBSERVER_PATH is required}
  GLM_OBSERVER=${BULK_GLM_OBSERVER_PATH:?BULK_GLM_OBSERVER_PATH is required}
  : "${CONFIGMAP_UID:?CONFIGMAP_UID must bind the immutable preflight ConfigMap}"
  exec uv run --no-project --with httpx==0.28.1 python \
    -m evals.fleet.autocontinue_bulk_controller preflight \
    --plan "$PLAN" \
    --package "$PACKAGE" \
    --authorization "$AUTHORIZATION" \
    --qwen-observer "$QWEN_OBSERVER" \
    --glm-observer "$GLM_OBSERVER" \
    --out-dir "$OUT_ROOT" \
    --out "$PREFLIGHT_OUT" \
    --repo "$ROOT"
fi

RELEASE=${BULK_RELEASE_PATH:?BULK_RELEASE_PATH is required}
PREFLIGHT=${BULK_PREFLIGHT_PATH:?BULK_PREFLIGHT_PATH is required}
PREFLIGHT_OBSERVER=${BULK_PREFLIGHT_OBSERVER_PATH:?BULK_PREFLIGHT_OBSERVER_PATH is required}
PROXY=${BULK_FIXED_PROXY_PATH:-$ROOT/evals/fleet/fixed_proxy.py}
: "${DOCKER_HOST:?DOCKER_HOST is required for scored execution}"
test "$DOCKER_HOST" = unix:///var/run/docker.sock
test ! -e "$OUT_ROOT"

for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag "$AGENT_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker image inspect "$AGENT_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm "$AGENT_IMAGE" opencode --version)" = 1.18.27
docker pull --platform linux/amd64 "$PROXY_IMAGE"

export AGENT_HARNESS_IMAGE=$AGENT_IMAGE
export FIXED_PROXY_IMAGE=$PROXY_IMAGE
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_bulk_controller run \
  --plan "$PLAN" \
  --release "$RELEASE" \
  --preflight "$PREFLIGHT" \
  --preflight-observer "$PREFLIGHT_OBSERVER" \
  --out-dir "$OUT_ROOT" \
  --proxy "$PROXY" \
  --repo "$ROOT"
