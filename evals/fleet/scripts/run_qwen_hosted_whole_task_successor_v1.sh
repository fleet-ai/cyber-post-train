#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
mkdir -p "$ROOT/evals/fleet/configs"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
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
cd "$ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_whole_task_successor_v1_runtime \
  --plan "$ROOT/evals/fleet/configs/runtime-plan.json" \
  --out "$QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT" \
  --diagnostic-root "$QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
