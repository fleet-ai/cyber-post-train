#!/usr/bin/env bash
set -euo pipefail
: "${GENERATION11_SPEC_KEY:?}" "${JOB_NAME:?}" "${JOB_UID:?}" "${POD_UID:?}" "${GENERATION11_SECRET_UID:?}" "${FLEET_API_KEY:?}"
ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/docs/evidence/qwen38-study" "$ROOT/docker-config"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
# ConfigMap projections are symlinks; strict loaders receive only regular copies.
install -m 0644 /bootstrap/self_hosted.py "$ROOT/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/runner.py "$ROOT/evals/fleet/opencode_train_sweep_runner.py"
install -m 0644 /bootstrap/g11.py "$ROOT/evals/fleet/generation11_simple_cell.py"
install -m 0644 /bootstrap/fixed_proxy.py "$ROOT/evals/fleet/fixed_proxy.py"
install -m 0644 /bootstrap/Dockerfile.opencode "$ROOT/evals/fleet/Dockerfile.opencode"
install -m 0644 "/bootstrap/${GENERATION11_SPEC_KEY}" "$ROOT/evals/fleet/configs/spec.json"
install -m 0644 /bootstrap/q-g10-tombstone.json "$ROOT/docs/evidence/qwen38-study/2026-09-05-qwen38-generation10-preclaim-preoutput-tombstone-v1.json"
install -m 0644 /bootstrap/g-g10-tombstone.json "$ROOT/docs/evidence/qwen38-study/2026-09-05-glm53-generation10-preclaim-preoutput-tombstone-v1.json"
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.generation11_simple_cell validate --spec "$ROOT/evals/fleet/configs/spec.json" --repo "$ROOT"
test "$(sha256sum /docker-cli/bin/docker | awk '{print $1}')" = 242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722
test "$(sha256sum /docker-cli/plugins/docker-buildx | awk '{print $1}')" = 8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
install -D -m 0755 /docker-cli/plugins/docker-buildx "$DOCKER_CONFIG/cli-plugins/docker-buildx"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.generation11_simple_cell run --spec "$ROOT/evals/fleet/configs/spec.json" --repo "$ROOT" --proxy "$ROOT/evals/fleet/fixed_proxy.py"
