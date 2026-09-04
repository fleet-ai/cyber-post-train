#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
TARGET=${2:-both}
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
QWEN_CONFIGMAP=chris-cyber-q38-hosted-complete49-p4-v5
GLM_CONFIGMAP=chris-cyber-glm53-hosted-complete99-p4-v5
QWEN_JOB=chris-cyber-q38-opencode11827-hosted-complete49-p4-v5
GLM_JOB=chris-cyber-glm53-opencode11827-hosted-complete99-p4-v5
QWEN_PREFLIGHT=chris-cyber-q38-hosted49-p4-v5-preflight
GLM_PREFLIGHT=chris-cyber-glm53-hosted99-p4-v5-preflight
QWEN_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-complete49-pass4-v5.json
GLM_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-complete99-pass4-v5.json
JOB_FILE=$ROOT/evals/fleet/cluster/opencode-hosted-successor-jobs-v5.yaml
PREFLIGHT_FILE=$ROOT/evals/fleet/cluster/opencode-hosted-successor-preflight-v5.yaml
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

uv run python - "$QWEN_PLAN" "$GLM_PLAN" <<'PY'
import json
import sys
from pathlib import Path

from evals.fleet.hosted_sweep_controller import validate_plan

expected = ((49, 196, 196, 0), (99, 394, 396, 2))
for raw_path, shape in zip(sys.argv[1:], expected, strict=True):
    plan = json.loads(Path(raw_path).read_text())
    validate_plan(plan)
    actual = (
        plan["task_count"],
        plan["new_session_count"],
        plan["total_session_count"],
        len(plan["credited_sessions"]),
    )
    if actual != shape:
        raise RuntimeError("hosted successor launch arithmetic drifted")
    if plan["execution"]["score_blind_concurrency_schedule"] != [
        {
            "accepted_outcomes_at_least": 0,
            "workers": 1,
            "headroom_gate": "fixed_at_launch_no_automatic_widening",
        }
    ]:
        raise RuntimeError("hosted successor worker cap drifted")
PY

document() {
  local file=$1 model=$2
  if test "$model" = qwen; then
    awk '/^---$/{exit} {print}' "$file"
  else
    awk 'found{print} /^---$/{found=1}' "$file"
  fi
}

configmap() {
  local model=$1 name plan
  if test "$model" = qwen; then
    name=$QWEN_CONFIGMAP
    plan=$QWEN_PLAN
  else
    name=$GLM_CONFIGMAP
    plan=$GLM_PLAN
  fi
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$name" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=plan.json="$plan" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_hosted_successor.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

set_model_values() {
  if test "$1" = qwen; then
    job=$QWEN_JOB
    preflight=$QWEN_PREFLIGHT
    config=$QWEN_CONFIGMAP
    plan=$QWEN_PLAN
  else
    job=$GLM_JOB
    preflight=$GLM_PREFLIGHT
    config=$GLM_CONFIGMAP
    plan=$GLM_PLAN
  fi
}

assert_source_terminal() {
  local plan=$1 source_name source_uid observed_uid active terminal
  source_name=$(jq -r '.source.source_job_name' "$plan")
  source_uid=$(jq -r '.source.source_job_uid' "$plan")
  observed_uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" \
    -o jsonpath='{.metadata.uid}')
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" \
    -o jsonpath='{.status.active}')
  terminal=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')
  test "$observed_uid" = "$source_uid"
  test -z "$active"
  test "$terminal" = True
}

assert_no_active_peer() {
  local active
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg q "$QWEN_JOB" --arg g "$GLM_JOB" '
      [.items[]
       | select((.status.active // 0) > 0)
       | .metadata.name
       | select(test("(opencode.*(q38|glm53)|(q38|glm53).*opencode)"))
       | select(. != $q and . != $g)] | join(",")')
  test -z "$active" || {
    echo "active hosted OpenCode peer exists: $active" >&2
    return 1
  }
}

selected=()
if test "$TARGET" = qwen || test "$TARGET" = both; then selected+=(qwen); fi
if test "$TARGET" = glm || test "$TARGET" = both; then selected+=(glm); fi

assert_no_active_peer
for model in "${selected[@]}"; do
  set_model_values "$model"
  assert_source_terminal "$plan"
  for kind_name in "job:$job" "job:$preflight" "configmap:$config"; do
    kind=${kind_name%%:*}
    name=${kind_name#*:}
    if "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1; then
      echo "$kind $NAMESPACE/$name already exists; refusing create-once reuse" >&2
      exit 1
    fi
  done
  configmap "$model" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  document "$PREFLIGHT_FILE" "$model" | \
    "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  document "$JOB_FILE" "$model" | \
    "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
done

if test "$MODE" = preview; then
  jq -n --arg target "$TARGET" '{ok:true,target:$target,workers_per_model:1}'
  exit 0
fi

for model in "${selected[@]}"; do
  set_model_values "$model"
  configmap "$model" | "${KUBECTL[@]}" create -f - >/dev/null
  document "$PREFLIGHT_FILE" "$model" | "${KUBECTL[@]}" create -f - >/dev/null
  "${KUBECTL[@]}" -n "$NAMESPACE" wait --for=condition=complete \
    --timeout=900s "job/$preflight" >/dev/null
  "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$preflight"
  assert_source_terminal "$plan"
  assert_no_active_peer
  document "$JOB_FILE" "$model" | "${KUBECTL[@]}" create -f - >/dev/null
  uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$job" -o jsonpath='{.metadata.uid}')
  jq -n --arg model "$model" --arg job "$job" --arg uid "$uid" \
    --arg plan_sha256 "$(jq -r '.plan_sha256' "$plan")" \
    '{created:true,model:$model,job:$job,uid:$uid,plan_sha256:$plan_sha256,workers:1}'
done
