#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-opencode-fleet-smokes-v2-session-recovery-v1
SECRET=chris-cyber-opencode-evals-v2
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")
JOB=$ROOT/evals/fleet/cluster/opencode-session-recovery-job.yaml
MODE=${1:-preview}

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.metadata.name}')" = "$SECRET"

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=qwen-config.json="$ROOT/evals/fleet/configs/qwen38-opencode-train-sweep-smoke-v2.json" \
    --from-file=glm-config.json="$ROOT/evals/fleet/configs/glm53-opencode-train-sweep-smoke-v2.json" \
    --from-file=run-recovery.sh="$ROOT/evals/fleet/scripts/run_opencode_session_recovery.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

if test "$MODE" = preview; then
  configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  "${KUBECTL[@]}" create --dry-run=server -f "$JOB" >/dev/null
  jq -n '{ok:true,model_rollouts:0,session_recoveries:2,create_once:true}'
  exit 0
fi

for resource in configmap job.batch; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get "$resource" "$NAME" >/dev/null 2>&1; then
    echo "Resource $NAMESPACE/$resource/$NAME already exists; refusing to replace it" >&2
    exit 1
  fi
done

configmap | "${KUBECTL[@]}" create -f - >/dev/null
"${KUBECTL[@]}" create -f "$JOB" >/dev/null
jq -n --arg name "$NAME" '{created:true,name:$name,model_rollouts:0,session_recoveries:2}'
