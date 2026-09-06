#!/usr/bin/env bash
set -euo pipefail

test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test -n "${QUALIFICATION_PACKAGE_COMMIT:-}"
test -n "${QUALIFICATION_OUTPUT_ROOT:-}"
test ! -e "$QUALIFICATION_OUTPUT_ROOT"

mkdir -p /work/evals/fleet/configs "$QUALIFICATION_OUTPUT_ROOT/gpu-observer"
for source in /bootstrap/*__SLASH__*; do
  target="/work/$(basename "$source" | sed 's,__SLASH__,/,g')"
  install -D -m 0444 "$source" "$target"
done
test "$(cat /bootstrap/package_commit)" = "$QUALIFICATION_PACKAGE_COMMIT"

for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null

docker build --pull --platform linux/amd64 \
  --tag chris/opencode:1.18.27-cyber-v1 \
  --file /work/evals/fleet/Dockerfile.opencode /work/evals/fleet
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27

export PYTHONPATH=/work
uv run --with httpx==0.28.1 --with pyyaml==6.0.2 \
  python -m evals.fleet.glm53_dedicated_v22_concurrency_qualification_v1 run \
  --root /work \
  --authorization /authorization/authorization.json \
  --gpu-observer-root "$QUALIFICATION_OUTPUT_ROOT/gpu-observer" \
  --out "$QUALIFICATION_OUTPUT_ROOT/RAW.json"
uv run --with httpx==0.28.1 --with pyyaml==6.0.2 \
  python -m evals.fleet.glm53_dedicated_v22_concurrency_qualification_v1 validate \
  --root /work \
  --raw "$QUALIFICATION_OUTPUT_ROOT/RAW.json" \
  --out "$QUALIFICATION_OUTPUT_ROOT/QUALIFIED.json"
