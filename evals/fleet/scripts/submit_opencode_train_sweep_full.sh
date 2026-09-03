#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
CONFIGMAP=chris-cyber-opencode-train-sweep-full-v3
SECRET=chris-cyber-opencode-evals-v2
QWEN_JOB=chris-cyber-opencode-q38-train50-p4-v3
GLM_JOB=chris-cyber-opencode-glm53-train100-p4-v3
JOB_FILE=$ROOT/evals/fleet/cluster/opencode-train-sweep-full-jobs.yaml
PREFLIGHT_JOB=chris-cyber-opencode-full-sfs-preflight-v3
PREFLIGHT_FILE=$ROOT/evals/fleet/cluster/opencode-train-sweep-full-preflight-job.yaml
QWEN_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json
GLM_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-train100-pass4-v3.json
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

uv run python - "$QWEN_PLAN" "$GLM_PLAN" <<'PY'
import json
import sys
from pathlib import Path

from evals.fleet.opencode_train_sweep_runner import validate_plan

expected = [(50, 199, 200), (100, 399, 400)]
for raw_path, shape in zip(sys.argv[1:], expected, strict=True):
    plan = json.loads(Path(raw_path).read_text())
    validate_plan(plan)
    actual = (plan["task_count"], plan["new_session_count"], plan["total_session_count"])
    if actual != shape:
        raise RuntimeError("full-plan launch arithmetic drifted")
PY

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$CONFIGMAP" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=qwen-plan-v3.json="$QWEN_PLAN" \
    --from-file=glm-plan-v3.json="$GLM_PLAN" \
    --from-file=run-full.sh="$ROOT/evals/fleet/scripts/run_opencode_train_sweep_full.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

for job in "$PREFLIGHT_JOB" "$QWEN_JOB" "$GLM_JOB"; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get job "$job" >/dev/null 2>&1; then
    echo "Job $NAMESPACE/$job already exists; refusing duplicate" >&2
    exit 1
  fi
done
if "${KUBECTL[@]}" -n "$NAMESPACE" get configmap "$CONFIGMAP" >/dev/null 2>&1; then
  echo "ConfigMap $NAMESPACE/$CONFIGMAP already exists; refusing replacement" >&2
  exit 1
fi
if test "$MODE" = preview; then
  configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  "${KUBECTL[@]}" create --dry-run=server -f "$PREFLIGHT_FILE" >/dev/null
  "${KUBECTL[@]}" create --dry-run=server -f "$JOB_FILE" >/dev/null
  jq -n '{ok:true,jobs:2,qwen_new_sessions:199,glm_new_sessions:399,total_new_sessions:598}'
  exit 0
fi

"${KUBECTL[@]}" create -f "$PREFLIGHT_FILE" >/dev/null
"${KUBECTL[@]}" -n "$NAMESPACE" wait --for=condition=complete \
  --timeout=600s "job/$PREFLIGHT_JOB" >/dev/null
configmap | "${KUBECTL[@]}" create -f - >/dev/null
"${KUBECTL[@]}" create -f "$JOB_FILE" >/dev/null
jq -n --arg qwen "$QWEN_JOB" --arg glm "$GLM_JOB" \
  '{created:true,jobs:[$qwen,$glm],qwen_new_sessions:199,glm_new_sessions:399}'
