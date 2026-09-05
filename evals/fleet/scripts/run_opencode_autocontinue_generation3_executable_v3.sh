#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
BOOTSTRAP=${GENERATION3_BOOTSTRAP:-/bootstrap}
: "${GENERATION3_PACKAGE_AGGREGATE_SHA256:?}"

python3 - "$BOOTSTRAP/package-manifest.json" "$BOOTSTRAP" "$ROOT" \
  "$GENERATION3_PACKAGE_AGGREGATE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, bootstrap_path, destination_path, expected = sys.argv[1:]
bootstrap = Path(bootstrap_path)
destination = Path(destination_path)
manifest = json.loads(Path(manifest_path).read_text())
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":")
).encode()
sha = lambda value: "sha256:" + hashlib.sha256(value).hexdigest()
if (
    manifest.get("schema_version")
    != "fleet-opencode-autocontinue-generation3-split-package-v2"
    or manifest.get("launch_authorized") is not False
    or manifest.get("aggregate_sha256") != expected
    or manifest.get("aggregate_sha256") != sha(canonical(manifest.get("objects")))
):
    raise SystemExit("generation-3 semantic mounted package manifest drifted")
for obj in manifest["objects"]:
    unsigned = {key: value for key, value in obj.items() if key != "payload_sha256"}
    if obj.get("payload_sha256") != sha(canonical(unsigned)):
        raise SystemExit("generation-3 semantic mounted object manifest drifted")
    for entry in obj["entries"]:
        source = bootstrap / entry["data_key"]
        raw = source.read_bytes()
        if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
            raise SystemExit("generation-3 semantic mounted payload drifted")
        target = destination / entry["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
PY

touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation3_executable_package_v2 verify-mounted \
  --manifest "$BOOTSTRAP/package-manifest.json" --bootstrap "$BOOTSTRAP"
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation3_canary validate-held \
  --held "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation3-canaries-held-v1.json" \
  --repo "$ROOT"

printf '%s\n' 'generation-3 semantic split package verified; no scoring release is installed' >&2
exit 3
