#!/usr/bin/env bash
set -euo pipefail

: "${PREBULK_PACKAGE_AGGREGATE_SHA256:?}" "${PREBULK_PACKAGE_COMMIT:?}"
: "${PREBULK_RELEASE_FILE_SHA256:?}"
: "${PREBULK_MODE:?}" "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
BOOTSTRAP=${PREBULK_BOOTSTRAP:-/bootstrap}
OUT=${PREBULK_OUTPUT_ROOT:-/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3}

uv run --no-project --with httpx==0.28.1 python "$BOOTSTRAP/package.py" materialize \
  --projected "$BOOTSTRAP" --destination "$ROOT" \
  --expected "$PREBULK_PACKAGE_AGGREGATE_SHA256"
cd "$ROOT"
release="$ROOT/docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-prebulk-reconciliation-release-v3.json"
install -D -m 0600 "$BOOTSTRAP/prebulk-release.json" "$release"
test "sha256:$(sha256sum "$release" | awk '{print $1}')" = "$PREBULK_RELEASE_FILE_SHA256"

case "$PREBULK_MODE" in
  source)
    exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.exact_pass4_prebulk_reconciliation_v3 \
      source --root "$ROOT" --output-root "$OUT" --evidence-root /mnt/sfs
    ;;
  accept)
    exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.exact_pass4_prebulk_reconciliation_v3 \
      accept --root "$ROOT" --output-root "$OUT" \
      --package-commit "$PREBULK_PACKAGE_COMMIT" --evidence-root /mnt/sfs
    ;;
  *)
    printf '%s\n' 'unsupported prebulk reconciliation mode' >&2
    exit 2
    ;;
esac
