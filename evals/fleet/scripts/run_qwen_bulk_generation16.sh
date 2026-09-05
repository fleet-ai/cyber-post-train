#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
BOOTSTRAP=${BULK_BOOTSTRAP:-/bootstrap}
: "${BULK_PLAN:?BULK_PLAN is required}" "${BULK_OUTPUT_ROOT:?BULK_OUTPUT_ROOT is required}"
: "${BULK_PACKAGE_AGGREGATE_SHA256:?BULK_PACKAGE_AGGREGATE_SHA256 is required}"
: "${JOB_UID:?JOB_UID is required}" "${POD_UID:?POD_UID is required}"
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"

python3 - "$BOOTSTRAP/package-manifest.json" "$BOOTSTRAP" "$ROOT" \
  "$BULK_PACKAGE_AGGREGATE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest_path, bootstrap_path, destination_path, expected = sys.argv[1:]
bootstrap, destination = Path(bootstrap_path), Path(destination_path)
manifest = json.loads(Path(manifest_path).read_text())
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
sha = lambda raw: "sha256:" + hashlib.sha256(raw).hexdigest()
if (manifest.get("schema_version") != "fleet-qwen-generation16-bulk-split-package-v1"
    or manifest.get("launch_authorized") is not True
    or manifest.get("release_included") is not True
    or manifest.get("aggregate_sha256") != expected
    or manifest.get("aggregate_sha256") != sha(canonical(manifest.get("objects")))):
    raise SystemExit("Generation-16 mounted package manifest drifted")
for obj in manifest["objects"]:
    body = {k: v for k, v in obj.items() if k != "payload_sha256"}
    if obj.get("payload_sha256") != sha(canonical(body)):
        raise SystemExit("Generation-16 mounted object manifest drifted")
    for entry in obj["entries"]:
        raw = (bootstrap / entry["data_key"]).read_bytes()
        if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
            raise SystemExit("Generation-16 mounted payload drifted")
        target = destination / entry["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
PY

touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
cd "$ROOT"
test "$(command -v docker)" = /docker-cli/bin/docker
test "$(sha256sum /docker-cli/bin/docker | awk '{print $1}')" = \
  242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722
test "$(sha256sum /docker-cli/plugins/docker-buildx | awk '{print $1}')" = \
  8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
test ! -e "$DOCKER_CONFIG"
install -D -m 0755 /docker-cli/plugins/docker-buildx \
  "$DOCKER_CONFIG/cli-plugins/docker-buildx"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7

exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.qwen_bulk_generation16_runtime run \
  --plan "$BULK_PLAN" --out "$BULK_OUTPUT_ROOT" \
  --proxy "$ROOT/evals/fleet/fixed_proxy.py"
