#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
PLAN_FILE=${HOSTED_PLAN_FILE:?HOSTED_PLAN_FILE is required}
PLAN_PATH="$ROOT/evals/fleet/configs/$PLAN_FILE"

validate_executable_renderer() {
  uv run --no-project --with httpx==0.28.1 python - "$1" <<'PY'
import json
import sys
from pathlib import Path

from evals.fleet import self_hosted


def pairs_no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise ValueError("canary executable plan is unsafe or absent")
plan = json.loads(path.read_text(), object_pairs_hook=pairs_no_duplicates)
if not isinstance(plan, dict):
    raise ValueError("canary executable plan root is not an object")
model = plan.get("model")
harness = plan.get("harness")
if not isinstance(model, dict) or not isinstance(harness, dict):
    raise ValueError("canary executable renderer inputs are malformed")
if harness.get("context_management") != self_hosted.OPENCODE_CONTEXT_MANAGEMENT:
    raise ValueError("canary executable context treatment drifted")
if harness.get("compaction_headroom_tokens") != 20000:
    raise ValueError("canary executable compaction headroom is absent or invalid")
settings = self_hosted.opencode_settings({"model": model, "harness": harness})
canonical = self_hosted.canonical_json(settings)
if (
    settings.get("compaction") != {"auto": True, "reserved": 20000}
    or "plugin" in settings
    or harness.get("settings_canonical_sha256") != self_hosted.sha256(canonical)
    or harness.get("settings_file_sha256") != self_hosted.sha256(canonical + b"\n")
):
    raise ValueError("canary executable renderer settings digest drifted")
PY
}

# This invariant stays ahead of environment preparation, route checks, durable
# claims, and the model/verifier path. Historical plan validation remains
# separate so frozen evidence can still be inspected.
validate_executable_renderer "$PLAN_PATH"
if [[ "${CANARY_RENDERER_GATE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

JOB_NAME=${HOSTED_JOB_NAME:?HOSTED_JOB_NAME is required}
RELEASE_FILE=${SCORING_RELEASE_FILE:?SCORING_RELEASE_FILE is required}
LAUNCH_ROUTE_FILE=${LAUNCH_ROUTE_FILE:?LAUNCH_ROUTE_FILE is required}
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
  -m evals.fleet.autocontinue_canary_hosted_runtime run-hosted \
  --plan "$PLAN_PATH" \
  --release "$ROOT/evals/fleet/configs/$RELEASE_FILE" \
  --launch-route "$ROOT/evals/fleet/configs/$LAUNCH_ROUTE_FILE" \
  --out-dir "$OUT_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py" \
  --repo "$ROOT" \
  --package-commit "$PACKAGE_COMMIT"
