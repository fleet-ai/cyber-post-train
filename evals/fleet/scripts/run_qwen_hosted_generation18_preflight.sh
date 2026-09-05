#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/docs/evidence/qwen38-study"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for file in self_hosted.py exact_pass4_bulk_v3.py exact_pass4_bulk_runtime_v3.py \
  exact_pass4_universe.py exact_pass4_crypto.py endpoint_lease.py \
  qwen_hosted_generation18.py qwen_hosted_generation18_preflight.py; do
  install -m 0644 "/bootstrap/$file" "$ROOT/evals/fleet/$file"
done
install -m 0644 /bootstrap/plan.json "$ROOT/evals/fleet/configs/qwen-hosted-generation18-canary.json"
install -m 0644 /bootstrap/parity.json \
  "$ROOT/docs/evidence/qwen38-study/2026-09-05-qwen38-hosted-actual-opencode-parity-v1.json"
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_generation18_preflight \
  --plan evals/fleet/configs/qwen-hosted-generation18-canary.json \
  --parity docs/evidence/qwen38-study/2026-09-05-qwen38-hosted-actual-opencode-parity-v1.json \
  --output /mnt/sfs/jobs/chris-q38-ac-r005-a1-g18-v1-preflight/CLEAR.json
