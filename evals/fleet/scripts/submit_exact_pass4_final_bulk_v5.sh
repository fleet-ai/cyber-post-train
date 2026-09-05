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
MANIFEST=$WORK_DIR/released.yaml

SECRET_UID=$(kubectl -n fleet-train-jobs get secret chris-cyber-opencode-evals-v2 \
  -o jsonpath='{.metadata.uid}')
test "$SECRET_UID" = e0febd8e-94a2-46b0-a0bf-dd6b3154187b
SECRET_KEY=$(kubectl -n fleet-train-jobs get secret chris-cyber-opencode-evals-v2 \
  -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$SECRET_KEY" uv run --with httpx==0.28.1 python - <<'PY'
import os
import httpx

response = httpx.get(
    "https://orchestrator.fleetai.com/v1/account",
    headers={"Authorization": f"Bearer {os.environ['FLEET_API_KEY']}"},
    timeout=30,
)
response.raise_for_status()
value = response.json()
if value.get("team_id") != "a1025f0b-ad67-49fc-a023-51800ab43e84" or value.get("team_name") != "fleet":
    raise SystemExit("exact evaluator Secret does not resolve to Fleet team")
PY
unset SECRET_KEY

python3 -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$SOURCE_ROOT" --expected-head "$PACKAGE_COMMIT"
python3 -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$SOURCE_ROOT" --commit "$PACKAGE_COMMIT" --destination "$SNAPSHOT"

ARGS=(
  --release "$RELEASE" --inventory "$INVENTORY" --prebulk "$PREBULK"
  --fresh-duplicate "$FRESH_DUPLICATE"
  --repo "$SNAPSHOT" --output "$MANIFEST"
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
kubectl create --dry-run=server -f "$MANIFEST" -o name
if [[ "$MODE" == preview ]]; then
  printf '%s\n' 'preview only; no objects created'
  exit 0
fi

while read -r resource; do
  if kubectl -n fleet-train-jobs get "$resource" >/dev/null 2>&1; then
    printf '%s\n' "create-once collision: $resource" >&2
    exit 1
  fi
done < <(kubectl create --dry-run=client -f "$MANIFEST" -o name)
kubectl create -f "$MANIFEST"
