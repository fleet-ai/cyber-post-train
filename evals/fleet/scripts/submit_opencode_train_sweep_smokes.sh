#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
GENERATION=${2:-v1}
case "$GENERATION" in
  v1) JOB_FILE=opencode-train-sweep-smokes-job.yaml ;;
  v2) JOB_FILE=opencode-train-sweep-smokes-job-v2.yaml ;;
  *) echo "generation must be v1 or v2" >&2; exit 2 ;;
esac
ATTEMPT=${GENERATION#v}
NAME=chris-cyber-opencode-fleet-smokes-$GENERATION
SECRET=chris-cyber-opencode-evals-$GENERATION
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")
JOB=$ROOT/evals/fleet/cluster/$JOB_FILE
MODE=${1:-preview}
PYTHON_BIN=$(command -v python3 || command -v python)

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit] [v1|v2]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=qwen-config.json="$ROOT/evals/fleet/configs/qwen38-opencode-train-sweep-smoke-v${ATTEMPT}.json" \
    --from-file=glm-config.json="$ROOT/evals/fleet/configs/glm53-opencode-train-sweep-smoke-v${ATTEMPT}.json" \
    --from-file=run-smokes.sh="$ROOT/evals/fleet/scripts/run_opencode_train_sweep_smokes.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

if test "$MODE" = preview; then
  configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  "${KUBECTL[@]}" create --dry-run=server -f "$JOB" >/dev/null
  jq -n --arg generation "$GENERATION" \
    '{ok:true,generation:$generation,paid_smoke_sessions:2,models:["qwen3.8-27b","glm-5.3"],pass_k:1}'
  exit 0
fi

for resource in configmap job.batch; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get "$resource" "$NAME" >/dev/null 2>&1; then
    echo "Resource $NAMESPACE/$resource/$NAME already exists; refusing to replace it" >&2
    exit 1
  fi
done
if "${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" >/dev/null 2>&1; then
  echo "Secret $NAMESPACE/$SECRET already exists; refusing to replace it" >&2
  exit 1
fi

"$PYTHON_BIN" - "$EXPECTED_CONTEXT" "$NAMESPACE" "$SECRET" <<'PY'
import json
import os
import subprocess
import sys

context, namespace, name = sys.argv[1:]
manifest = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {
        "name": name,
        "namespace": namespace,
        "labels": {
            "cyber-post-train.fleet.ai/experiment": "opencode-qwen38-glm53-fleet-smokes-v1",
            "cyber-post-train.fleet.ai/owner": "chris",
        },
    },
    "type": "Opaque",
    "stringData": {"FLEET_API_KEY": os.environ["FLEET_API_KEY"]},
}
subprocess.run(
    ["kubectl", "--context", context, "create", "-f", "-"],
    input=json.dumps(manifest),
    text=True,
    check=True,
    stdout=subprocess.DEVNULL,
)
PY
configmap | "${KUBECTL[@]}" create -f - >/dev/null
"${KUBECTL[@]}" create -f "$JOB" >/dev/null
jq -n --arg name "$NAME" '{created:true,name:$name,paid_smoke_sessions:2}'
