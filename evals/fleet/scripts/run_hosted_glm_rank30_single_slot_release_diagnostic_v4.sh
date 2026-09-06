#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${JOB_UID:?}" "${POD_UID:?}"
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for binding in self_hosted.py:self_hosted.py runner.py:opencode_train_sweep_runner.py endpoint_lease.py:endpoint_lease.py predecessor.py:exact_pass4_bulk_v3.py base_engine.py:exact_pass4_bulk_runtime_v3.py engine.py:hosted_glm_whole_task_engine_v1.py universe.py:exact_pass4_universe.py crypto.py:exact_pass4_crypto.py inventory.py:exact_pass4_task_inventory.py bulk.py:hosted_glm_exact_bulk_v1.py source_runtime.py:hosted_glm_exact_bulk_runtime_v1.py original_release.py:hosted_glm_exact_bulk_release_v1.py rank29_successor.py:hosted_glm_rank29_a3a4_c2_successor_v1.py rank29_runtime.py:hosted_glm_rank29_a3a4_c2_runtime_v1.py whole.py:hosted_glm_whole_task_successor_v1.py single_slot_v1.py:hosted_glm_rank30_single_slot_v1.py successor.py:hosted_glm_rank30_single_slot_v3.py kube.py:hosted_glm_rank29_a3a4_c2_release_v2.py release_v1.py:hosted_glm_rank30_single_slot_release_v1.py release.py:hosted_glm_rank30_single_slot_release_v3.py diagnostic.py:hosted_glm_rank30_single_slot_release_diagnostic_v4.py; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/${binding#*:}"; done
for binding in campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json selection.json:opencode-easiest-train100-selection-v2.json glm-template.json:glm53-opencode-autocontinue-canary1-v1.json qwen-template.json:qwen38-opencode-autocontinue-canary1-v1.json bulk-qwen-a.json:exact-pass4-bulk-qwen-a-v3.json bulk-qwen-b.json:exact-pass4-bulk-qwen-b-v3.json bulk-glm-a.json:exact-pass4-bulk-glm-a-v3.json bulk-glm-b.json:exact-pass4-bulk-glm-b-v3.json; do install -m 0644 "/bootstrap/${binding%%:*}" "$ROOT/evals/fleet/configs/${binding#*:}"; done
cd "$ROOT"
set +e
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.hosted_glm_rank30_single_slot_release_diagnostic_v4
status=$?
set -e
OUTPUT=/mnt/sfs/jobs/chris-glm53-exact100-hosted-r030-single-slot-release-diagnostic-v4/DIAGNOSTIC.json
if (( status != 0 )) && [[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]; then
  python - "$status" "$OUTPUT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

body = {
    "schema_version": "fleet-hosted-glm-rank30-release-phase-diagnostic-bootstrap-v4",
    "status": "FAILED",
    "diagnostic_job": "chris-glm53-exact100-hosted-r030-single-slot-release-diagnostic-v4",
    "diagnostic_configmap": "chris-glm53-exact100-hosted-r030-single-slot-release-diagnostic-v4-run",
    "observer_job_uid": os.environ["JOB_UID"],
    "observer_pod_uid": os.environ["POD_UID"],
    "failed_phase": "00-bootstrap-runtime",
    "error_type_class": "bootstrap-runtime-error",
    "error_sha256": "sha256:" + hashlib.sha256(canonical({"exit_code": int(sys.argv[1])})).hexdigest(),
    "session_boundary_executed": False,
    "model_calls": 0,
    "task_calls": 0,
    "session_calls": 0,
    "verifier_calls": 0,
    "scoring_calls": 0,
    "api_mutation_calls": 0,
    "diagnostic_receipt_writes": 1,
    "scores_read": False,
    "prompts_traces_flags_read": False,
    "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
}
body["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical(body)).hexdigest()
path = Path(sys.argv[2])
path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(canonical(body) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
fi
exit "$status"
