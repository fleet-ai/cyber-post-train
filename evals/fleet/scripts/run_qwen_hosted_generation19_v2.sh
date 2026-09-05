#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for file in self_hosted.py opencode_train_sweep_runner.py exact_pass4_bulk_v3.py \
  exact_pass4_bulk_runtime_v3.py exact_pass4_universe.py exact_pass4_crypto.py \
  endpoint_lease.py qwen_bulk_generation16.py qwen_hosted_generation18.py \
  qwen_hosted_generation19_bulk.py qwen_hosted_generation19_v2.py \
  qwen_hosted_generation19_v2_runtime.py fixed_proxy.py Dockerfile.opencode; do
  install -m 0644 "/bootstrap/$file" "$ROOT/evals/fleet/$file"
done
install -m 0644 /bootstrap/plan-v2-a.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-a-v2.json"
install -m 0644 /bootstrap/plan-v2-b.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-b-v2.json"
case "$G19_CONTROLLER" in
  qwen-a) RUNTIME_PLAN=/bootstrap/plan-v2-a.json ;;
  qwen-b) RUNTIME_PLAN=/bootstrap/plan-v2-b.json ;;
  *) exit 64 ;;
esac
install -m 0644 "$RUNTIME_PLAN" "$ROOT/evals/fleet/configs/runtime-plan.json"
cd "$ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_generation19_v2_runtime \
  --plan "$ROOT/evals/fleet/configs/runtime-plan.json" \
  --out "$G19_OUTPUT_ROOT" --diagnostic-root "$G19_DIAGNOSTIC_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
