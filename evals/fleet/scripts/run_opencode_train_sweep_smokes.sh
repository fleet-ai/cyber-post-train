#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
JOB_NAME=chris-cyber-opencode-fleet-smokes-v1
OUT_PARENT=${FLEET_EVAL_OUT_PARENT:-/mnt/sfs/jobs/$JOB_NAME}
QWEN_OUT=$OUT_PARENT/qwen38-smoke
GLM_OUT=$OUT_PARENT/glm53-smoke
ACCEPTED=$OUT_PARENT/ACCEPTED.json
AGENT_IMAGE=chris/opencode:1.18.27-cyber-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected from the campaign Secret}"
test "$(uname -m)" = x86_64
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test ! -e "$OUT_PARENT"

for _ in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker info >/dev/null
mkdir -p "$OUT_PARENT"

docker build --pull --platform linux/amd64 \
  --tag "$AGENT_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" \
  "$ROOT/evals/fleet"
test "$(docker image inspect "$AGENT_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm "$AGENT_IMAGE" opencode --version)" = 1.18.27
docker pull --platform linux/amd64 "$PROXY_IMAGE"

export AGENT_HARNESS_IMAGE=$AGENT_IMAGE
export FIXED_PROXY_IMAGE=$PROXY_IMAGE
runner=(uv run --no-project --with httpx==0.28.1 python -m evals.fleet.self_hosted run)

"${runner[@]}" \
  --config "$ROOT/evals/fleet/configs/qwen38-opencode-train-sweep-smoke-v1.json" \
  --out-dir "$QWEN_OUT" \
  --proxy-script "$ROOT/evals/fleet/fixed_proxy.py" >/dev/null &
qwen_pid=$!
"${runner[@]}" \
  --config "$ROOT/evals/fleet/configs/glm53-opencode-train-sweep-smoke-v1.json" \
  --out-dir "$GLM_OUT" \
  --proxy-script "$ROOT/evals/fleet/fixed_proxy.py" >/dev/null &
glm_pid=$!

set +e
wait "$qwen_pid"
qwen_rc=$?
wait "$glm_pid"
glm_rc=$?
set -e
test "$qwen_rc" = 0
test "$glm_rc" = 0

python - "$QWEN_OUT" "$GLM_OUT" "$ACCEPTED" <<'PY'
import hashlib
import json
import sys
import uuid
from pathlib import Path

roots = [Path(sys.argv[1]), Path(sys.argv[2])]
summary = []
for root in roots:
    result = json.loads((root / "result.json").read_text())
    cleanup = json.loads((root / "cleanup.json").read_text())
    ingest = json.loads((root / "session-ingest.json").read_text())
    assert result["agent_exit_code"] == 0
    assert result["agent_termination"] == "completed"
    assert result["session_ingest_status"] == "completed"
    assert cleanup == {
        "instance_created": True,
        "instance_closed": True,
        "containers_removed": True,
    }
    assert ingest["status"] == "completed"
    uuid.UUID(result["instance_id"])
    uuid.UUID(result["evidence_run_id"])
    uuid.UUID(result["verifier_execution_id"])
    uuid.UUID(result["session_id"])
    summary.append(
        {
            "run_id": result["run_id"],
            "harness": result["harness"],
            "authoritative_verifier_execution": True,
            "session_ingest_completed": True,
            "cleanup_completed": True,
        }
    )
accepted = {
    "schema_version": "fleet-opencode-train-sweep-smokes-acceptance-v1",
    "accepted": True,
    "runs": summary,
    "scores_included": False,
    "prompts_or_traces_included": False,
}
raw = json.dumps(accepted, sort_keys=True, separators=(",", ":")).encode()
accepted["acceptance_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
with open(sys.argv[3], "x") as handle:
    json.dump(accepted, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
