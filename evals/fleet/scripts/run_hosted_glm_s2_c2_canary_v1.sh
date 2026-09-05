#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for binding in \
  self_hosted.py:self_hosted.py \
  runner.py:opencode_train_sweep_runner.py \
  endpoint_lease.py:endpoint_lease.py \
  predecessor.py:exact_pass4_bulk_v3.py \
  engine.py:exact_pass4_bulk_runtime_v3.py \
  universe.py:exact_pass4_universe.py \
  crypto.py:exact_pass4_crypto.py \
  inventory.py:exact_pass4_task_inventory.py \
  bulk.py:hosted_glm_exact_bulk_v1.py \
  base_runtime.py:hosted_glm_exact_bulk_runtime_v1.py \
  original_release.py:hosted_glm_exact_bulk_release_v1.py \
  c2.py:hosted_glm_s2_c2_canary_v1.py \
  c2_release.py:hosted_glm_s2_c2_release_v1.py \
  c2_runtime.py:hosted_glm_s2_c2_runtime_v1.py \
  fixed_proxy.py:fixed_proxy.py; do
  install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/${binding#*:}"
done
for binding in \
  campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json \
  selection.json:opencode-easiest-train100-selection-v2.json \
  glm-template.json:glm53-opencode-autocontinue-canary1-v1.json \
  qwen-template.json:qwen38-opencode-autocontinue-canary1-v1.json \
  bulk-qwen-a.json:exact-pass4-bulk-qwen-a-v3.json \
  bulk-qwen-b.json:exact-pass4-bulk-qwen-b-v3.json \
  bulk-glm-a.json:exact-pass4-bulk-glm-a-v3.json \
  bulk-glm-b.json:exact-pass4-bulk-glm-b-v3.json; do
  install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/configs/${binding#*:}"
done
install -m 0644 /bootstrap/Dockerfile.opencode "$ROOT/evals/fleet/Dockerfile.opencode"
cd "$ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.hosted_glm_s2_c2_runtime_v1 --repo "$ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
