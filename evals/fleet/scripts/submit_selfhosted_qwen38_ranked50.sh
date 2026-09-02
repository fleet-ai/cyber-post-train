#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen38-qcode-fleet-ranked50-base-v1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")
JOB=$ROOT/evals/fleet/cluster/qwen38-code-selfhosted-ranked50-job.yaml
CONFIG=$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-ranked50-base-v1.json
SPLIT=$ROOT/configs/data/fleet-a62-task-split-v1.json
MODE=${1:-preview}
tmp_dir=$(mktemp -d)
secret_metadata_path=$tmp_dir/secret-metadata.json
trap 'rm -rf "$tmp_dir"' EXIT

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
: "${FLEET_CREDENTIAL_ROTATION_RECEIPT:?set to a sanitized post-incident rotation receipt}"
uv run python - "$secret_metadata_path" "$EXPECTED_CONTEXT" <<'PY'
import json
import sys
from pathlib import Path

from evals.credential_rotation import read_secret_metadata

Path(sys.argv[1]).write_text(
    json.dumps(read_secret_metadata(context=sys.argv[2]), sort_keys=True) + "\n"
)
PY
rotation_not_before=$(jq -r '.credential_gate.rotation_not_before' "$CONFIG")
uv run python -m evals.fleet.qwen38_fleet50 credential-gate \
  --credential-receipt "$FLEET_CREDENTIAL_ROTATION_RECEIPT" \
  --secret-metadata "$secret_metadata_path" \
  --rotation-not-before "$rotation_not_before"
uv run python -m evals.fleet.qwen38_fleet50 validate \
  --split "$SPLIT" \
  --exclusions "$ROOT/evals/fleet/configs/qwen38-prior-attempt-exclusions-v1.json" \
  --selection "$ROOT/evals/fleet/configs/qwen38-27b-ranked50-selection-v1.json"
privacy_status=$(jq -r '.privacy_gate.status' "$CONFIG")
duplicate_status=$(jq -r '.duplicate_gate.status' "$CONFIG")
if test "$MODE" = submit; then
  for status in "$privacy_status" "$duplicate_status"; do
    if test "$status" != ready; then
      echo "ranked-50 launch blocked: $status" >&2
      exit 1
    fi
  done
fi

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=Dockerfile.qwen-code="$ROOT/evals/fleet/Dockerfile.qwen-code" \
    --from-file=config.json="$CONFIG" \
    --from-file=credential-rotation-receipt.json="$FLEET_CREDENTIAL_ROTATION_RECEIPT" \
    --from-file=credential_rotation.py="$ROOT/evals/credential_rotation.py" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=holdout.py="$ROOT/evals/fleet/holdout.py" \
    --from-file=qwen38_fleet50.py="$ROOT/evals/fleet/qwen38_fleet50.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=split.json="$SPLIT" \
    --from-file=v2-incident.json="$ROOT/docs/evidence/qwen38-study/2026-09-01-fleet-calibration-v2-infrastructure-incident.json" \
    --from-file=v3-terminal.json="$ROOT/docs/evidence/qwen38-study/2026-09-01-fleet-calibration-v3-terminal-valid-zero.json" \
    --from-file=v2-treatment-plan.json="$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v2.json" \
    --from-file=v3-treatment-plan.json="$ROOT/evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v3.json" \
    --from-file=run_selfhosted_qwen38_ranked50.sh="$ROOT/evals/fleet/scripts/run_selfhosted_qwen38_ranked50.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

if test "$MODE" = preview; then
  configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  "${KUBECTL[@]}" create --dry-run=server -f "$JOB" >/dev/null
  jq -n \
    --arg name "$NAME" \
    '{ok:true,name:$name,planned_sessions:50,launch_authorized:false,credential_rotation_verified:true,first_task_gate:"authoritative_model_outcome_with_verifier_uuid_and_cleanup"}'
  exit 0
fi

while read -r resource; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get "$resource" "$NAME" >/dev/null 2>&1; then
    echo "Resource $NAMESPACE/$resource/$NAME already exists; refusing to replace it" >&2
    exit 1
  fi
done <<EOF
configmap
serviceaccount
role.rbac.authorization.k8s.io
rolebinding.rbac.authorization.k8s.io
job.batch
EOF
configmap | "${KUBECTL[@]}" create -f -
"${KUBECTL[@]}" create -f "$JOB"
