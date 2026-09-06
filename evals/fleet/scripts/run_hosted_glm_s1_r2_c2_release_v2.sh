#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for binding in self_hosted.py:self_hosted.py runner.py:opencode_train_sweep_runner.py endpoint_lease.py:endpoint_lease.py predecessor.py:exact_pass4_bulk_v3.py engine.py:exact_pass4_bulk_runtime_v3.py universe.py:exact_pass4_universe.py crypto.py:exact_pass4_crypto.py inventory.py:exact_pass4_task_inventory.py bulk.py:hosted_glm_exact_bulk_v1.py bulk_runtime.py:hosted_glm_exact_bulk_runtime_v1.py original_release.py:hosted_glm_exact_bulk_release_v1.py prior_successor.py:hosted_glm_s1_r2_c2_successor_v1.py prior_release.py:hosted_glm_s1_r2_c2_release_v1.py successor.py:hosted_glm_s1_r2_c2_successor_v2.py successor_release.py:hosted_glm_s1_r2_c2_release_v2.py; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/${binding#*:}"; done
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python -c 'import os;from pathlib import Path;from evals.fleet import hosted_glm_s1_r2_c2_release_v2 as r;from evals.fleet import self_hosted;x=r.build(Path("/workspace/cyber-post-train"));x["job_uid"]=os.environ["JOB_UID"];x["pod_uid"]=os.environ["POD_UID"];x["receipt_sha256"]=self_hosted.digest_without(x,"receipt_sha256");r.OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=False);r.OUTPUT_PATH.write_bytes(self_hosted.canonical_json(x)+b"\n")'
