#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${JOB_NAME:?}" "${JOB_UID:?}" "${POD_UID:?}" "${GENERATION13_SECRET_UID:?}" "${FLEET_API_KEY:?}"
ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/docs/evidence/qwen38-study"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$ROOT/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/runner.py "$ROOT/evals/fleet/opencode_train_sweep_runner.py"
install -m 0644 /bootstrap/g13.py "$ROOT/evals/fleet/autocontinue_generation13_simple_cell.py"
install -m 0644 /bootstrap/fixed_proxy.py "$ROOT/evals/fleet/fixed_proxy.py"
install -m 0644 /bootstrap/Dockerfile.opencode "$ROOT/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/spec.json "$ROOT/evals/fleet/configs/generation13-spec.json"
install -m 0644 /bootstrap/g12-tombstone.json \
  "$ROOT/$(python3 -c 'import json; print(json.load(open("/bootstrap/spec.json"))["predecessor_generation12"]["tombstone_path"])')"
cd "$ROOT"
stage() {
  uv run --no-project --with httpx==0.28.1 python \
    -m evals.fleet.autocontinue_generation13_simple_cell mark-stage \
    --spec "$ROOT/evals/fleet/configs/generation13-spec.json" --stage "$1"
}
stage 01-before-static-validation
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation13_simple_cell validate \
  --spec "$ROOT/evals/fleet/configs/generation13-spec.json" --repo "$ROOT"
stage 02-after-static-validation
stage 03-before-docker-ready
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
stage 04-docker-ready
stage 05-before-docker-build
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
stage 06-after-docker-build
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation13_simple_cell run \
  --spec "$ROOT/evals/fleet/configs/generation13-spec.json" --repo "$ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
