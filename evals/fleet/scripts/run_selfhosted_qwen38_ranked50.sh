#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
CAMPAIGN_ID=chris-cyber-q38-qcode-fleet-ranked50-base-v1
JOB_NAME=chris-cyber-qwen38-qcode-fleet-ranked50-base-v1
OUT_PARENT=${FLEET_EVAL_OUT_PARENT:-/mnt/sfs/jobs/$JOB_NAME}
OUT_DIR=$OUT_PARENT/$CAMPAIGN_ID
CONFIG=$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-ranked50-base-v1.json
SPLIT=$ROOT/configs/data/fleet-a62-task-split-v1.json
RECEIPT=$OUT_PARENT/qwen38-27b-qwen-code-ranked50-base-v1.receipt.json
RUNTIME_IMAGES=$OUT_PARENT/qwen38-27b-qwen-code-ranked50-base-v1.runtime-images.json
QWEN_IMAGE=chris/qwen-code:0.22.3-q38-fleet-v1
PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

: "${FLEET_API_KEY:?FLEET_API_KEY must be injected through the cluster secret}"
test "$(uname -m)" = x86_64
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test ! -e "$OUT_DIR"
test ! -e "$RECEIPT"
test ! -e "$OUT_PARENT/preflight.json"
test ! -e "$RUNTIME_IMAGES"

for _ in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker info >/dev/null

mkdir -p "$OUT_PARENT"
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.holdout \
  preflight --config "$CONFIG" --split "$SPLIT" --receipt-out "$RECEIPT" \
  --preflight-out "$OUT_PARENT/preflight.json" >/dev/null

docker build --pull --platform linux/amd64 \
  --tag "$QWEN_IMAGE" \
  --file "$ROOT/evals/fleet/Dockerfile.qwen-code" \
  "$ROOT/evals/fleet"
test "$(docker image inspect "$QWEN_IMAGE" --format '{{.Architecture}}')" = amd64
test "$(docker run --rm "$QWEN_IMAGE" qwen --version)" = 0.22.3
docker pull --platform linux/amd64 "$PROXY_IMAGE"

python - "$RUNTIME_IMAGES" "$QWEN_IMAGE" "$PROXY_IMAGE" \
  "$(docker image inspect "$QWEN_IMAGE" --format '{{.Id}}')" \
  "$(docker image inspect "$QWEN_IMAGE" --format '{{.Architecture}}')" \
  "$(docker image inspect "$PROXY_IMAGE" --format '{{.Id}}')" \
  "$(docker image inspect "$PROXY_IMAGE" --format '{{.Architecture}}')" <<'PY'
import hashlib
import json
import sys

path, qwen_ref, proxy_ref, qwen_id, qwen_arch, proxy_id, proxy_arch = sys.argv[1:]
receipt = {
    "schema_version": "fleet-eval-runtime-images-v1",
    "images": {
        "qwen_code": {"requested_ref": qwen_ref, "runtime_id": qwen_id, "architecture": qwen_arch},
        "fixed_proxy": {
            "requested_ref": proxy_ref,
            "runtime_id": proxy_id,
            "architecture": proxy_arch,
        },
    },
}
assert qwen_id.startswith("sha256:") and proxy_id.startswith("sha256:")
assert qwen_arch == "amd64" and proxy_arch == "amd64"
raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
with open(path, "x") as handle:
    json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY

export QWEN_CODE_IMAGE=$QWEN_IMAGE
export FIXED_PROXY_IMAGE=$PROXY_IMAGE
export FLEET_EVAL_RUNTIME_IMAGES_RECEIPT=$RUNTIME_IMAGES
uv run --no-project --with httpx==0.28.1 python -m evals.fleet.holdout \
  run --config "$CONFIG" --split "$SPLIT" --receipt "$RECEIPT" \
  --out-dir "$OUT_DIR" --proxy-script "$ROOT/evals/fleet/fixed_proxy.py"

test -s "$OUT_DIR/summary.json"
test -s "$OUT_DIR/ACCEPTED.json"
python - "$OUT_DIR/summary.json" "$OUT_DIR/ACCEPTED.json" <<'PY'
import hashlib
import json
import sys

summary = json.load(open(sys.argv[1]))
accepted = json.load(open(sys.argv[2]))
assert summary["planned_sessions"] == 50
assert summary["model_outcomes"] == 50
assert summary["authoritative_verifier_backed_outcomes"] == 50
assert summary["infrastructure_errors"] == 0
assert summary["first_task_release_gate_satisfied"] is True
assert accepted["accepted"] is True
claimed = accepted.pop("acceptance_sha256")
raw = json.dumps(accepted, sort_keys=True, separators=(",", ":")).encode()
assert claimed == "sha256:" + hashlib.sha256(raw).hexdigest()
PY
