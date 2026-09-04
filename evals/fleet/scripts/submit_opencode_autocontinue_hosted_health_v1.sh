#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAMESPACE=fleet-train-jobs
NAME=chris-cyber-opencode-ac-hosted-health-v1
SECRET=chris-cyber-opencode-evals-v2
SECRET_UID=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
MODE=${1:-preview}
KUBECTL=(kubectl -n "$NAMESPACE")

if [[ "$MODE" != preview && "$MODE" != launch ]]; then
  echo "usage: $0 [preview|launch]" >&2
  exit 2
fi

test "$("${KUBECTL[@]}" get secret "$SECRET" -o jsonpath='{.metadata.uid}')" = "$SECRET_UID"
! "${KUBECTL[@]}" get configmap "$NAME" >/dev/null 2>&1
! "${KUBECTL[@]}" get job "$NAME" >/dev/null 2>&1

render_configmap() {
  "${KUBECTL[@]}" create configmap "$NAME" \
    --from-file=probe.py="$ROOT/evals/fleet/autocontinue_hosted_health.py" \
    --dry-run=client -o json | jq '.immutable = true'
}

render_configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
"${KUBECTL[@]}" create --dry-run=server \
  -f "$ROOT/evals/fleet/cluster/opencode-autocontinue-hosted-health-v1.yaml" >/dev/null

if [[ "$MODE" == preview ]]; then
  echo "hosted health preflight preview passed; no objects created"
  exit 0
fi

test -z "$(git -C "$ROOT" status --porcelain)"
render_configmap | "${KUBECTL[@]}" create -f - >/dev/null
"${KUBECTL[@]}" create \
  -f "$ROOT/evals/fleet/cluster/opencode-autocontinue-hosted-health-v1.yaml" >/dev/null
echo "$NAME"
