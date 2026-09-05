#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for file in self_hosted.py exact_pass4_bulk_v3.py exact_pass4_bulk_runtime_v3.py \
  exact_pass4_universe.py exact_pass4_crypto.py endpoint_lease.py \
  qwen_bulk_generation16.py qwen_bulk_generation16_preflight.py \
  qwen_hosted_generation19_bulk.py qwen_hosted_generation19_preflight.py; do
  install -m 0644 "/bootstrap/$file" "$ROOT/evals/fleet/$file"
done
install -m 0644 /bootstrap/plan-a.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-a.json"
install -m 0644 /bootstrap/plan-b.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-b.json"
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_generation19_preflight \
  --output /mnt/sfs/jobs/chris-q38-ac-exact100-g19-preflight-v1/CLEAR.json
