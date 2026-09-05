#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${JOB_NAME:?}" "${JOB_UID:?}" "${POD_UID:?}" "${SECRET_UID:?}" "${FLEET_API_KEY:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/docs/evidence/qwen38-study"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$ROOT/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/runner.py "$ROOT/evals/fleet/opencode_train_sweep_runner.py"
install -m 0644 /bootstrap/endpoint_lease.py "$ROOT/evals/fleet/endpoint_lease.py"
install -m 0644 /bootstrap/bulk.py "$ROOT/evals/fleet/exact_pass4_bulk_v3.py"
install -m 0644 /bootstrap/bulk_runtime.py "$ROOT/evals/fleet/exact_pass4_bulk_runtime_v3.py"
install -m 0644 /bootstrap/universe.py "$ROOT/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/crypto.py "$ROOT/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/canary.py "$ROOT/evals/fleet/hosted_glm_exact_canary_v1.py"
install -m 0644 /bootstrap/fixed_proxy.py "$ROOT/evals/fleet/fixed_proxy.py"
install -m 0644 /bootstrap/Dockerfile.opencode "$ROOT/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/campaign.json "$ROOT/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$ROOT/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/glm-template.json "$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
install -m 0644 /bootstrap/qwen-template.json "$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
install -m 0644 /bootstrap/bulk-qwen-a.json "$ROOT/evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json"
install -m 0644 /bootstrap/bulk-qwen-b.json "$ROOT/evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json"
install -m 0644 /bootstrap/bulk-glm-a.json "$ROOT/evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json"
install -m 0644 /bootstrap/bulk-glm-b.json "$ROOT/evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json"
install -m 0644 /bootstrap/parity.json "$ROOT/docs/evidence/qwen38-study/2026-09-05-glm53-hosted-actual-opencode-parity-v1.json"
cd "$ROOT"
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.hosted_glm_exact_canary_v1 validate --repo "$ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.hosted_glm_exact_canary_v1 run --repo "$ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
