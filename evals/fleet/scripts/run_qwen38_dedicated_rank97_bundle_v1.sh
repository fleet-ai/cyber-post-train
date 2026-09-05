#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
: "${JOB_UID:?JOB_UID is required}" "${POD_UID:?POD_UID is required}"
: "${QWEN_RANK97_RELEASE_PATH:?QWEN_RANK97_RELEASE_PATH is required}"
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen38_dedicated_rank97_bundle_v1
