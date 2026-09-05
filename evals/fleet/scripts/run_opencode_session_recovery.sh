#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/sfs/jobs/chris-cyber-opencode-fleet-smokes-v2}
OUT_ROOT=${OUT_ROOT:-/mnt/sfs/jobs/chris-cyber-opencode-fleet-smokes-v2-session-recovery-v1}

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected from the existing campaign Secret}"
test -d "$SOURCE_ROOT/qwen38-smoke"
test -d "$SOURCE_ROOT/glm53-smoke"
test ! -e "$OUT_ROOT"
mkdir "$OUT_ROOT"
chmod 700 "$OUT_ROOT"

runner=(uv run --no-project --with httpx==0.28.1 python -m evals.fleet.self_hosted recover-session)

"${runner[@]}" \
  --config "$ROOT/evals/fleet/configs/qwen38-opencode-train-sweep-smoke-v2.json" \
  --source-dir "$SOURCE_ROOT/qwen38-smoke" \
  --out-dir "$OUT_ROOT/qwen38-smoke" >/dev/null &
qwen_pid=$!
"${runner[@]}" \
  --config "$ROOT/evals/fleet/configs/glm53-opencode-train-sweep-smoke-v2.json" \
  --source-dir "$SOURCE_ROOT/glm53-smoke" \
  --out-dir "$OUT_ROOT/glm53-smoke" >/dev/null &
glm_pid=$!

set +e
wait "$qwen_pid"
qwen_rc=$?
wait "$glm_pid"
glm_rc=$?
set -e
test "$qwen_rc" = 0
test "$glm_rc" = 0

python - "$OUT_ROOT/qwen38-smoke" "$OUT_ROOT/glm53-smoke" "$OUT_ROOT/ACCEPTED.json" <<'PY'
import hashlib
import json
import sys
import uuid
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


summary = []
for root_name in sys.argv[1:3]:
    root = Path(root_name)
    recovered = json.loads((root / "RECOVERED.json").read_text())
    digest = recovered.pop("receipt_sha256")
    if digest != "sha256:" + hashlib.sha256(canonical(recovered)).hexdigest():
        raise RuntimeError(f"{root.name}: recovered receipt digest mismatch")
    if recovered.get("recovered") is not True:
        raise RuntimeError(f"{root.name}: trace was not recovered")
    if recovered.get("scores_included") is not False:
        raise RuntimeError(f"{root.name}: recovered receipt exposed a score")
    if recovered.get("prompts_or_traces_included") is not False:
        raise RuntimeError(f"{root.name}: recovered receipt exposed private content")
    uuid.UUID(recovered["session_id"])
    uuid.UUID(recovered["verifier_execution_id"])
    summary.append(
        {
            "run_id": recovered["run_id"],
            "session_id": recovered["session_id"],
            "verifier_execution_id": recovered["verifier_execution_id"],
            "trace_recovered": True,
        }
    )

accepted = {
    "schema_version": "fleet-opencode-session-recovery-acceptance-v1",
    "accepted": True,
    "runs": summary,
    "scores_included": False,
    "prompts_or_traces_included": False,
}
accepted["acceptance_sha256"] = "sha256:" + hashlib.sha256(canonical(accepted)).hexdigest()
with open(sys.argv[3], "x") as handle:
    json.dump(accepted, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
