#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}" "${GLM_HOSTED_R30_SOURCE_SHA256:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for binding in self_hosted.py:self_hosted.py runner.py:opencode_train_sweep_runner.py endpoint_lease.py:endpoint_lease.py predecessor.py:exact_pass4_bulk_v3.py base_engine.py:exact_pass4_bulk_runtime_v3.py engine.py:hosted_glm_whole_task_engine_v1.py universe.py:exact_pass4_universe.py crypto.py:exact_pass4_crypto.py inventory.py:exact_pass4_task_inventory.py bulk.py:hosted_glm_exact_bulk_v1.py source_runtime.py:hosted_glm_exact_bulk_runtime_v1.py original_release.py:hosted_glm_exact_bulk_release_v1.py rank29_successor.py:hosted_glm_rank29_a3a4_c2_successor_v1.py rank29_runtime.py:hosted_glm_rank29_a3a4_c2_runtime_v1.py whole.py:hosted_glm_whole_task_successor_v1.py single_slot_v1.py:hosted_glm_rank30_single_slot_v1.py single_slot_v3.py:hosted_glm_rank30_single_slot_v3.py single_slot_v4.py:hosted_glm_rank30_single_slot_v4.py successor.py:hosted_glm_rank30_single_slot_v5.py kube.py:hosted_glm_rank29_a3a4_c2_release_v2.py release_v1.py:hosted_glm_rank30_single_slot_release_v1.py release_v3.py:hosted_glm_rank30_single_slot_release_v3.py release.py:hosted_glm_rank30_single_slot_release_v4.py; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/${binding#*:}"; done
for binding in campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json selection.json:opencode-easiest-train100-selection-v2.json glm-template.json:glm53-opencode-autocontinue-canary1-v1.json qwen-template.json:qwen38-opencode-autocontinue-canary1-v1.json bulk-qwen-a.json:exact-pass4-bulk-qwen-a-v3.json bulk-qwen-b.json:exact-pass4-bulk-qwen-b-v3.json bulk-glm-a.json:exact-pass4-bulk-glm-a-v3.json bulk-glm-b.json:exact-pass4-bulk-glm-b-v3.json; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/configs/${binding#*:}"; done
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.hosted_glm_rank30_single_slot_release_v4
