#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/cyber-post-train
STAGE_ROOT=/mnt/sfs/jobs/chris-q38-ac-r005-a1-g18-v1-preflight-v2
mark() {
  python3 - "$STAGE_ROOT/$1.json" "$1" <<'PY'
import hashlib, json, os, sys
path, stage = sys.argv[1:]
body = {
    "schema_version": "fleet-qwen-generation18-preflight-stage-v1",
    "stage": stage,
    "prompts_traces_flags_scores_or_credentials_included": False,
}
raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
value = {**body, "receipt_sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(payload)
PY
}
mark 01-shell-started
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/docs/evidence/qwen38-study"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
for file in self_hosted.py exact_pass4_bulk_v3.py exact_pass4_bulk_runtime_v3.py \
  exact_pass4_universe.py exact_pass4_crypto.py endpoint_lease.py \
  qwen_hosted_generation18.py qwen_hosted_generation18_preflight.py; do
  install -m 0644 "/bootstrap/$file" "$ROOT/evals/fleet/$file"
done
install -m 0644 /bootstrap/plan.json "$ROOT/evals/fleet/configs/qwen-hosted-generation18-canary.json"
install -m 0644 /bootstrap/parity.json \
  "$ROOT/docs/evidence/qwen38-study/2026-09-05-qwen38-hosted-actual-opencode-parity-v1.json"
mark 02-package-hydrated
cd "$ROOT"
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_hosted_generation18_preflight \
  --plan evals/fleet/configs/qwen-hosted-generation18-canary.json \
  --parity docs/evidence/qwen38-study/2026-09-05-qwen38-hosted-actual-opencode-parity-v1.json \
  --output /mnt/sfs/jobs/chris-q38-ac-r005-a1-g18-v1-preflight-v2/CLEAR.json
