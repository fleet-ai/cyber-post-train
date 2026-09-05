#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
SPEC_FILE=${GENERATION2_SPEC_FILE:?GENERATION2_SPEC_FILE is required}
SPEC_PATH="$ROOT/evals/fleet/configs/$SPEC_FILE"

# A non-self-digesting rendered plan cannot reach image preparation, route
# discovery, durable claims, Fleet sessions, or the model.
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation2_canary_v2 validate-spec \
  --spec "$SPEC_PATH" --repo "$ROOT"
if [[ "${GENERATION2_RENDERER_GATE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

JOB_NAME=${GENERATION2_JOB_NAME:?GENERATION2_JOB_NAME is required}
RELEASE_FILE=${GENERATION2_RELEASE_FILE:?GENERATION2_RELEASE_FILE is required}
LAUNCH_ROUTE_FILE=${GENERATION2_LAUNCH_ROUTE_FILE:?GENERATION2_LAUNCH_ROUTE_FILE is required}
PACKAGE_COMMIT=${PACKAGE_COMMIT:?PACKAGE_COMMIT is required}
OUT_ROOT=${FLEET_EVAL_OUT_ROOT:-/mnt/sfs/jobs/$JOB_NAME}
AGENT_IMAGE=chris/opencode:1.18.27-cyber-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected from the campaign Secret}"
: "${JOB_UID:?JOB_UID must come from the downward API}"
: "${POD_UID:?POD_UID must come from the downward API}"
test "$(uname -m)" = x86_64
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
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
  -m evals.fleet.autocontinue_generation2_canary_v2 run \
  --spec "$SPEC_PATH" \
  --release "$ROOT/evals/fleet/configs/$RELEASE_FILE" \
  --launch-route "$ROOT/evals/fleet/configs/$LAUNCH_ROUTE_FILE" \
  --out-dir "$OUT_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py" \
  --repo "$ROOT" \
  --package-commit "$PACKAGE_COMMIT"
