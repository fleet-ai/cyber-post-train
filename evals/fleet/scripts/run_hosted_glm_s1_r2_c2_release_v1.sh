#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for binding in self_hosted.py:self_hosted.py runner.py:opencode_train_sweep_runner.py endpoint_lease.py:endpoint_lease.py predecessor.py:exact_pass4_bulk_v3.py engine.py:exact_pass4_bulk_runtime_v3.py universe.py:exact_pass4_universe.py crypto.py:exact_pass4_crypto.py inventory.py:exact_pass4_task_inventory.py bulk.py:hosted_glm_exact_bulk_v1.py bulk_runtime.py:hosted_glm_exact_bulk_runtime_v1.py original_release.py:hosted_glm_exact_bulk_release_v1.py successor.py:hosted_glm_s1_r2_c2_successor_v1.py successor_release.py:hosted_glm_s1_r2_c2_release_v1.py fixed_proxy.py:fixed_proxy.py; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/${binding#*:}"; done
for binding in campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json selection.json:opencode-easiest-train100-selection-v2.json glm-template.json:glm53-opencode-autocontinue-canary1-v1.json qwen-template.json:qwen38-opencode-autocontinue-canary1-v1.json bulk-qwen-a.json:exact-pass4-bulk-qwen-a-v3.json bulk-qwen-b.json:exact-pass4-bulk-qwen-b-v3.json bulk-glm-a.json:exact-pass4-bulk-glm-a-v3.json bulk-glm-b.json:exact-pass4-bulk-glm-b-v3.json; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/configs/${binding#*:}"; done
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python -c 'from pathlib import Path;from evals.fleet import hosted_glm_s1_r2_c2_release_v1 as r;from evals.fleet.exact_pass4_bulk_runtime_v3 import _write_once;x=r.build(Path("/workspace/cyber-post-train"));r.OUTPUT_PATH.parent.mkdir(parents=True,mode=0o700,exist_ok=False);_write_once(r.OUTPUT_PATH,x)'
