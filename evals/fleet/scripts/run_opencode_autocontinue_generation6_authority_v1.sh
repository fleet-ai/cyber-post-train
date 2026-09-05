#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
BOOTSTRAP=${GENERATION6_BOOTSTRAP:-/bootstrap}
: "${GENERATION6_PACKAGE_AGGREGATE_SHA256:?}" "${GENERATION6_RELEASE_FILE_SHA256:?}"
: "${GENERATION6_PACKAGE_COMMIT:?}" "${GENERATION6_MODEL:?}" "${GENERATION6_JOB_NAME:?}"

python3 - "$BOOTSTRAP/package-manifest.json" "$BOOTSTRAP" "$ROOT" \
  "$GENERATION6_PACKAGE_AGGREGATE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, bootstrap_path, destination_path, expected = sys.argv[1:]
bootstrap = Path(bootstrap_path)
destination = Path(destination_path)
manifest = json.loads(Path(manifest_path).read_text())
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
sha = lambda value: "sha256:" + hashlib.sha256(value).hexdigest()
if (
    manifest.get("schema_version")
    != "fleet-opencode-autocontinue-generation6-authority-split-package-v1"
    or manifest.get("launch_authorized") is not False
    or manifest.get("release_included") is not False
    or manifest.get("aggregate_sha256") != expected
    or manifest.get("aggregate_sha256") != sha(canonical(manifest.get("objects")))
):
    raise SystemExit("generation-6 mounted package manifest drifted")
for obj in manifest["objects"]:
    unsigned = {key: value for key, value in obj.items() if key != "payload_sha256"}
    if obj.get("payload_sha256") != sha(canonical(unsigned)):
        raise SystemExit("generation-6 mounted object manifest drifted")
    for entry in obj["entries"]:
        raw = (bootstrap / entry["data_key"]).read_bytes()
        if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
            raise SystemExit("generation-6 mounted payload drifted")
        target = destination / entry["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
PY

touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"
cd "$ROOT"
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation6_authority_package_v1 verify-mounted \
  --manifest "$BOOTSTRAP/package-manifest.json" --bootstrap "$BOOTSTRAP"

case "$GENERATION6_MODEL" in
  qwen3.8-27b) SPEC_PATH=evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation6-v1.json ;;
  glm-5.3) SPEC_PATH=evals/fleet/configs/glm53-opencode-autocontinue-canary-generation6-v1.json ;;
  *) printf '%s\n' 'unsupported generation-6 model' >&2; exit 2 ;;
esac
install -m 0644 "$BOOTSTRAP/release.json" "$ROOT/evals/fleet/configs/generation6-release.json"
install -m 0644 "$BOOTSTRAP/launch-route.json" "$ROOT/evals/fleet/configs/generation6-launch-route.json"
test "sha256:$(sha256sum "$ROOT/evals/fleet/configs/generation6-release.json" | awk '{print $1}')" = \
  "$GENERATION6_RELEASE_FILE_SHA256"

uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation6_authority_v1 validate-release \
  --spec "$ROOT/$SPEC_PATH" \
  --authority "$ROOT/docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation6-root-authorization-v1.json" \
  --release "$ROOT/evals/fleet/configs/generation6-release.json" \
  --repo "$ROOT" --package-commit "$GENERATION6_PACKAGE_COMMIT"

: "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
OUT_ROOT=${FLEET_EVAL_OUT_ROOT:-/mnt/sfs/jobs/${GENERATION6_JOB_NAME}}
test ! -e "$OUT_ROOT" && test ! -L "$OUT_ROOT"
test "$(command -v docker)" = /docker-cli/bin/docker
test "$(sha256sum /docker-cli/bin/docker | awk '{print $1}')" = \
  242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722
test "$(sha256sum /docker-cli/plugins/docker-buildx | awk '{print $1}')" = \
  8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
test ! -e "$DOCKER_CONFIG"
install -D -m 0755 /docker-cli/plugins/docker-buildx \
  "$DOCKER_CONFIG/cli-plugins/docker-buildx"
test "$(sha256sum "$DOCKER_CONFIG/cli-plugins/docker-buildx" | awk '{print $1}')" = \
  8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
test "$(docker --version)" = 'Docker version 27.5.1, build 9f9e405'
docker buildx version | grep -F 'v0.20.1' >/dev/null
for _ in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then break; fi
  sleep 1
done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation6_authority_v1 run \
  --spec "$ROOT/$SPEC_PATH" \
  --authority "$ROOT/docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation6-root-authorization-v1.json" \
  --release "$ROOT/evals/fleet/configs/generation6-release.json" \
  --launch-route "$ROOT/evals/fleet/configs/generation6-launch-route.json" \
  --out-dir "$OUT_ROOT" --proxy "$ROOT/evals/fleet/fixed_proxy.py" \
  --repo "$ROOT" --package-commit "$GENERATION6_PACKAGE_COMMIT"
