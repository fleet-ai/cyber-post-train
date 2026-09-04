#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
TARGET=${2:-glm}
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
QWEN_CONFIGMAP=chris-cyber-opencode-train-sweep-full-v4-qwen
GLM_CONFIGMAP=chris-cyber-opencode-train-sweep-full-v4-glm
SECRET=chris-cyber-opencode-evals-v2
QWEN_JOB=chris-cyber-opencode-q38-train50-p4-v4
GLM_JOB=chris-cyber-opencode-glm53-train100-p4-v4
JOB_FILE=$ROOT/evals/fleet/cluster/opencode-train-sweep-full-jobs.yaml
QWEN_PREFLIGHT_JOB=chris-cyber-opencode-q38-sfs-preflight-v4
GLM_PREFLIGHT_JOB=chris-cyber-opencode-glm53-sfs-preflight-v4
PREFLIGHT_FILE=$ROOT/evals/fleet/cluster/opencode-train-sweep-full-preflight-job.yaml
QWEN_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-train50-pass4-v4.json
GLM_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-train100-pass4-v4.json
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")

case "$MODE" in
  preview|submit) ;;
  *) echo "usage: $0 [preview|submit] [qwen|glm|both]" >&2; exit 2 ;;
esac
case "$TARGET" in
  qwen|glm|both) ;;
  *) echo "usage: $0 [preview|submit] [qwen|glm|both]" >&2; exit 2 ;;
esac

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

plans=()
shapes=()
if test "$TARGET" = qwen || test "$TARGET" = both; then
  plans+=("$QWEN_PLAN")
  shapes+=("50,197,200,2")
fi
if test "$TARGET" = glm || test "$TARGET" = both; then
  plans+=("$GLM_PLAN")
  shapes+=("100,399,400,0")
fi
uv run python - "${plans[@]}" -- "${shapes[@]}" <<'PY'
import json
import sys
from pathlib import Path

from evals.fleet.opencode_train_sweep_runner import validate_parallel_plan

separator = sys.argv.index("--")
paths = sys.argv[1:separator]
expected = [tuple(map(int, raw.split(","))) for raw in sys.argv[separator + 1 :]]
for raw_path, shape in zip(paths, expected, strict=True):
    plan = json.loads(Path(raw_path).read_text())
    validate_parallel_plan(plan)
    actual = (
        plan["task_count"],
        plan["new_session_count"],
        plan["total_session_count"],
        len(plan["prior_accepted"]),
    )
    if actual != shape:
        raise RuntimeError("full-plan launch arithmetic drifted")
PY

configmap() {
  local model=$1
  local name plan_key plan_path
  if test "$model" = qwen; then
    name=$QWEN_CONFIGMAP
    plan_key=qwen-plan-v4.json
    plan_path=$QWEN_PLAN
  else
    name=$GLM_CONFIGMAP
    plan_key=glm-plan-v4.json
    plan_path=$GLM_PLAN
  fi
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$name" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file="$plan_key=$plan_path" \
    --from-file=run-full.sh="$ROOT/evals/fleet/scripts/run_opencode_train_sweep_full.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

selected_jobs=()
selected_configmaps=()
selected_preflights=()
if test "$TARGET" = qwen || test "$TARGET" = both; then
  selected_jobs+=("$QWEN_JOB")
  selected_configmaps+=("$QWEN_CONFIGMAP")
  selected_preflights+=("$QWEN_PREFLIGHT_JOB")
fi
if test "$TARGET" = glm || test "$TARGET" = both; then
  selected_jobs+=("$GLM_JOB")
  selected_configmaps+=("$GLM_CONFIGMAP")
  selected_preflights+=("$GLM_PREFLIGHT_JOB")
fi
for job in "${selected_jobs[@]}"; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get job "$job" >/dev/null 2>&1; then
    echo "Job $NAMESPACE/$job already exists; refusing duplicate" >&2
    exit 1
  fi
done
for job in "${selected_preflights[@]}"; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get job "$job" >/dev/null 2>&1; then
    echo "Preflight Job $NAMESPACE/$job already exists; refusing stale reuse" >&2
    exit 1
  fi
done
for config in "${selected_configmaps[@]}"; do
  if "${KUBECTL[@]}" -n "$NAMESPACE" get configmap "$config" >/dev/null 2>&1; then
    echo "ConfigMap $NAMESPACE/$config already exists; refusing replacement" >&2
    exit 1
  fi
done
job_document() {
  if test "$1" = qwen; then
    awk '/^---$/{exit} {print}' "$JOB_FILE"
  else
    awk 'found{print} /^---$/{found=1}' "$JOB_FILE"
  fi
}
preflight_document() {
  if test "$1" = qwen; then
    awk '/^---$/{exit} {print}' "$PREFLIGHT_FILE"
  else
    awk 'found{print} /^---$/{found=1}' "$PREFLIGHT_FILE"
  fi
}
if test "$MODE" = preview; then
  if test "$TARGET" = qwen || test "$TARGET" = both; then
    configmap qwen | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
    job_document qwen | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  fi
  if test "$TARGET" = glm || test "$TARGET" = both; then
    configmap glm | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
    job_document glm | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  fi
  if test "$TARGET" = qwen || test "$TARGET" = both; then
    preflight_document qwen | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  fi
  if test "$TARGET" = glm || test "$TARGET" = both; then
    preflight_document glm | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  fi
  jq -n --arg target "$TARGET" '{ok:true,target:$target}'
  exit 0
fi

if test "$TARGET" = qwen || test "$TARGET" = both; then
  preflight_document qwen | "${KUBECTL[@]}" create -f - >/dev/null
  "${KUBECTL[@]}" -n "$NAMESPACE" wait --for=condition=complete \
    --timeout=600s "job/$QWEN_PREFLIGHT_JOB" >/dev/null
  configmap qwen | "${KUBECTL[@]}" create -f - >/dev/null
  job_document qwen | "${KUBECTL[@]}" create -f - >/dev/null
fi
if test "$TARGET" = glm || test "$TARGET" = both; then
  preflight_document glm | "${KUBECTL[@]}" create -f - >/dev/null
  "${KUBECTL[@]}" -n "$NAMESPACE" wait --for=condition=complete \
    --timeout=600s "job/$GLM_PREFLIGHT_JOB" >/dev/null
  configmap glm | "${KUBECTL[@]}" create -f - >/dev/null
  job_document glm | "${KUBECTL[@]}" create -f - >/dev/null
fi
jq -n --arg target "$TARGET" '{created:true,target:$target}'
