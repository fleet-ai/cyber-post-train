#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
root=/workspace/cyber-post-train
state=/mnt/sfs/jobs/chris-cyber-q38-glm53-pass4-ledger-v1
mkdir -p \
  "$root/evals/fleet/configs" \
  "$root/evals/fleet" \
  "$root/docs/evidence/qwen38-study" \
  "$state"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"

for name in \
  rollout_worker.py rollout_campaign.py rollout_ledger.py opencode_self_hosted.py \
  fixed_proxy.py exact_pass4_crypto.py exact_pass4_universe.py Dockerfile.opencode; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
install -m 0644 /bootstrap/campaign.json \
  "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json \
  "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/snapshot.json \
  "$root/docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json"

cd "$root"
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.rollout_campaign build-plan \
  --campaign evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json \
  --selection evals/fleet/configs/opencode-easiest-train100-selection-v2.json \
  --snapshot docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json \
  --output "$state/plan.csv"

for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$root/evals/fleet/Dockerfile.opencode" "$root/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.rollout_worker \
  --repo-root "$root" \
  --database "$state/ledger.sqlite3" \
  --plan "$state/plan.csv" \
  --campaign "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json" \
  --selection "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json" \
  --snapshot "$root/docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json" \
  --output-root "$state" \
  --claim-root /mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1 \
  --worker-id chris-four-route-canary-v1 \
  --cells-per-block 1
