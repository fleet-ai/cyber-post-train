#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
bootstrap_stage() {
  QWEN_HOSTED_BOOTSTRAP_STAGE="$1" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

stage = os.environ["QWEN_HOSTED_BOOTSTRAP_STAGE"]
if stage not in {
    "00-started",
    "01-package-source-validated",
    "02-sources-installed",
    "03-docker-ready",
    "04-image-built",
    "05-harness-version-validated",
    "06-runtime-exec",
}:
    raise SystemExit("invalid bootstrap stage")
body = {
    "schema_version": "fleet-qwen38-hosted-whole-task-bootstrap-stage-v1",
    "status": "REACHED",
    "stage": stage,
    "job_uid": os.environ["JOB_UID"],
    "pod_uid": os.environ["POD_UID"],
    "release_receipt_sha256": os.environ["QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256"],
    "package_source_receipt_sha256": os.environ[
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
    ],
    "scores_included": False,
    "prompts_or_traces_included": False,
    "credentials_included": False,
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
body["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
root = Path(os.environ["QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT"]) / "bootstrap-stages"
root.mkdir(mode=0o700, parents=True, exist_ok=True)
path = root / f"{stage}.json"
payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
PY
}
bootstrap_stage 00-started
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path("/bootstrap")
receipt = json.loads(root.joinpath("package-source.json").read_text())
body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
if receipt.get("receipt_sha256") != expected:
    raise SystemExit("package source receipt self-digest drifted")
if expected != os.environ.get("QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"):
    raise SystemExit("package source environment binding drifted")
for name, digest in receipt.get("files", {}).items():
    path = root / name
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"package source file drifted: {name}")
PY
bootstrap_stage 01-package-source-validated
for file in self_hosted.py opencode_train_sweep_runner.py exact_pass4_bulk_v3.py \
  exact_pass4_bulk_runtime_v3.py exact_pass4_universe.py exact_pass4_crypto.py \
  endpoint_lease.py qwen_bulk_generation16.py qwen_hosted_generation18.py \
  qwen_hosted_generation19_bulk.py qwen_hosted_generation19_v2.py \
  qwen_hosted_generation19_v2_runtime.py qwen_hosted_generation19_v3.py \
  qwen_hosted_generation19_v4.py qwen_hosted_generation19_v4_runtime.py \
  qwen_hosted_whole_task_successor_v1.py \
  qwen_hosted_whole_task_successor_v1_runtime.py fixed_proxy.py Dockerfile.opencode; do
  install -m 0644 "/bootstrap/$file" "$ROOT/evals/fleet/$file"
done
install -m 0644 /bootstrap/source-plan-v4-a.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-a-v4.json"
install -m 0644 /bootstrap/source-plan-v4-b.json \
  "$ROOT/evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json"
install -m 0644 /bootstrap/plan.json "$ROOT/evals/fleet/configs/runtime-plan.json"
bootstrap_stage 02-sources-installed
cd "$ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
bootstrap_stage 03-docker-ready
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
bootstrap_stage 04-image-built
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
bootstrap_stage 05-harness-version-validated
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
bootstrap_stage 06-runtime-exec
CANARY_ARGS=()
if [[ "${QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY:-false}" == "true" ]]; then
  CANARY_ARGS=(--runtime-gate-canary-receipt \
    "$QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT/RUNTIME-GATE-CANARY.json")
fi
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_whole_task_successor_v1_runtime \
  --plan "$ROOT/evals/fleet/configs/runtime-plan.json" \
  --out "$QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT" \
  --diagnostic-root "$QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py" \
  "${CANARY_ARGS[@]}"
