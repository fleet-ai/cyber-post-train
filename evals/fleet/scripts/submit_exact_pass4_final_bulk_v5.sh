#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  printf '%s\n' \
    'usage: submit_exact_pass4_final_bulk_v5.sh preview|submit RELEASE INVENTORY PREBULK FRESH_DUPLICATE' >&2
  exit 2
}

[[ $# -eq 5 ]] || usage
MODE=$1
RELEASE=$(realpath "$2")
INVENTORY=$(realpath "$3")
PREBULK=$(realpath "$4")
FRESH_DUPLICATE=$(realpath "$5")
[[ "$MODE" == preview || "$MODE" == submit ]] || usage
SOURCE_ROOT=$(git rev-parse --show-toplevel)
PACKAGE_COMMIT=$(jq -er '.package_commit | select(test("^[0-9a-f]{40}$"))' "$RELEASE")
WORK_DIR=$(mktemp -d)
trap 'rm -rf -- "$WORK_DIR"' EXIT
SNAPSHOT=$WORK_DIR/package-commit
RENDERED=$WORK_DIR/rendered

python3 -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$SOURCE_ROOT" --expected-head "$PACKAGE_COMMIT"
python3 -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$SOURCE_ROOT" --commit "$PACKAGE_COMMIT" --destination "$SNAPSHOT"

ARGS=(
  --release "$RELEASE" --inventory "$INVENTORY" --prebulk "$PREBULK"
  --fresh-duplicate "$FRESH_DUPLICATE"
  --repo "$SNAPSHOT" --output-dir "$RENDERED"
)
[[ -z "${FINAL_QUALIFIER_LAUNCH_RELEASE:-}" ]] || ARGS+=(--qualifier-launch-release "$FINAL_QUALIFIER_LAUNCH_RELEASE")
[[ -z "${FINAL_QUALIFIER_MODEL:-}" ]] || ARGS+=(--qualifier-model "$FINAL_QUALIFIER_MODEL")
[[ -z "${FINAL_QUALIFIER_TERMINAL:-}" ]] || ARGS+=(--qualifier-terminal "$FINAL_QUALIFIER_TERMINAL")
[[ -z "${FINAL_QUALIFIER_JOB:-}" ]] || ARGS+=(--qualifier-job "$FINAL_QUALIFIER_JOB")
[[ -z "${FINAL_QUALIFIER_PODS:-}" ]] || ARGS+=(--qualifier-pods "$FINAL_QUALIFIER_PODS")
[[ -z "${FINAL_DEDICATED_PARITY:-}" ]] || ARGS+=(--dedicated-parity "$FINAL_DEDICATED_PARITY")
[[ -z "${FINAL_DEDICATED_CANARY:-}" ]] || ARGS+=(--dedicated-canary "$FINAL_DEDICATED_CANARY")
[[ -z "${FINAL_DEDICATED_RUNTIME:-}" ]] || ARGS+=(--dedicated-runtime "$FINAL_DEDICATED_RUNTIME")

(
  cd "$SNAPSHOT"
  PYTHONPATH="$SNAPSHOT" python3 -m evals.fleet.exact_pass4_final_bulk_renderer_v5 "${ARGS[@]}"
)
uv run --project "$SNAPSHOT" python \
  -m evals.fleet.kubernetes_create_relay validate \
  --manifest "$RENDERED/manifest.json" --allowlist "$RENDERED/allowlist.json"
if [[ "$MODE" == preview ]]; then
  printf '%s\n' 'preview only; no objects created'
  exit 0
fi

: "${FINAL_BULK_CREATE_RECEIPT:?FINAL_BULK_CREATE_RECEIPT must be an unused absolute path}"
[[ "$FINAL_BULK_CREATE_RECEIPT" == /* ]] || {
  printf '%s\n' 'FINAL_BULK_CREATE_RECEIPT must be absolute' >&2
  exit 2
}
test ! -e "$FINAL_BULK_CREATE_RECEIPT" && test ! -L "$FINAL_BULK_CREATE_RECEIPT"
uv run --project "$SNAPSHOT" python \
  -m evals.fleet.kubernetes_create_relay create \
  --manifest "$RENDERED/manifest.json" --allowlist "$RENDERED/allowlist.json" \
  --transport auto --receipt "$FINAL_BULK_CREATE_RECEIPT"
