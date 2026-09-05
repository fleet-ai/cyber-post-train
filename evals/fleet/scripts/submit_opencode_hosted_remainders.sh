#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
TARGET=${2:-both}
GENERATION=${3:-v1}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
KUBECTL=(kubectl --context "$CONTEXT")
Q_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-complete48-pass4-v6.json
G_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-odd47-pass4-v8.json
Q_JOB=chris-cyber-q38-opencode11827-hosted-complete48-p4-v6
G_JOB=chris-cyber-glm53-opencode11827-hosted-odd47-p4-v8
Q_PREFLIGHT=chris-cyber-q38-hosted48-p4-v6-preflight
G_PREFLIGHT=chris-cyber-glm53-hosted-odd47-p4-v8-preflight
Q_CONFIG=chris-cyber-q38-hosted-complete48-p4-v6
G_CONFIG=chris-cyber-glm53-hosted-odd47-p4-v8

if test "$GENERATION" = v2; then
  Q_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json
  G_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-odd46-pass4-v9.json
  Q_JOB=chris-cyber-q38-opencode11827-hosted-complete47-p4-v7
  G_JOB=chris-cyber-glm53-opencode11827-hosted-odd46-p4-v9
  Q_PREFLIGHT=chris-cyber-q38-hosted47-p4-v7-preflight
  G_PREFLIGHT=chris-cyber-glm53-hosted-odd46-p4-v9-preflight
  Q_CONFIG=chris-cyber-q38-hosted-complete47-p4-v7
  G_CONFIG=chris-cyber-glm53-hosted-odd46-p4-v9
fi
if test "$GENERATION" = v3; then
  Q_PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json
  G_PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json
  Q_JOB=chris-cyber-q38-opencode11827-hosted-complete47-p4-v7
  G_JOB=chris-cyber-glm53-opencode11827-hosted-odd45-p4-v10
  Q_PREFLIGHT=chris-cyber-q38-hosted47-p4-v7-preflight
  G_PREFLIGHT=chris-cyber-glm53-hosted-odd45-p4-v10-preflight
  Q_CONFIG=chris-cyber-q38-hosted-complete47-p4-v7
  G_CONFIG=chris-cyber-glm53-hosted-odd45-p4-v10
fi

case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
case "$TARGET" in qwen|glm|both) ;; *) exit 2 ;; esac
case "$GENERATION" in v1|v2|v3) ;; *) exit 2 ;; esac
test "$(kubectl config current-context)" = "$CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

uv run python - "$Q_PLAN" "$G_PLAN" <<'PY'
import json
import sys
from pathlib import Path
from evals.fleet.hosted_sweep_controller import validate_plan

expected = {
    "qwen38_remainder": (48, 192, set(range(3, 51))),
    "glm53_remainder": (47, 188, set(range(7, 100, 2))),
    "qwen38_remainder2": (47, 188, set(range(4, 51))),
    "glm53_remainder2": (46, 184, set(range(9, 100, 2))),
    "glm53_remainder3": (45, 180, set(range(11, 100, 2))),
}
for path in sys.argv[1:]:
    plan = json.loads(Path(path).read_text())
    validate_plan(plan)
    tasks, sessions, source_ranks = expected[plan["shard_key"]]
    if (plan["task_count"], plan["new_session_count"]) != (tasks, sessions):
        raise RuntimeError("hosted remainder arithmetic drifted")
    if {row["source_rank"] for row in plan["tasks"]} != source_ranks:
        raise RuntimeError("hosted remainder ownership drifted")
PY

set_model() {
  if test "$1" = qwen; then
    plan=$Q_PLAN; job=$Q_JOB; preflight=$Q_PREFLIGHT; config=$Q_CONFIG
    peer_pattern='(opencode.*q38|q38.*opencode)'
  else
    plan=$G_PLAN; job=$G_JOB; preflight=$G_PREFLIGHT; config=$G_CONFIG
    peer_pattern='(opencode.*glm53.*hosted|glm53.*opencode.*hosted)'
  fi
}

render_job() {
  if test "$1" = qwen; then
    replacement=complete48-p4-v6
    test "$GENERATION" = v1 || replacement=complete47-p4-v7
    awk '/^---$/{exit} {print}' "$ROOT/evals/fleet/cluster/opencode-hosted-successor-jobs-v5.yaml" |
      sed -e "s/complete49-p4-v5/$replacement/g" |
      awk '{if ($0 == "      restartPolicy: Never") print "      priorityClassName: fleet-infra-quiet"; print}'
  else
    replacement=odd47-p4-v8
    test "$GENERATION" = v1 || replacement=odd46-p4-v9
    test "$GENERATION" != v3 || replacement=odd45-p4-v10
    sed -e "s/odd49-p4-v7/$replacement/g" \
      "$ROOT/evals/fleet/cluster/opencode-glm53-hosted-odd-job-v7.yaml" |
      awk '{if ($0 == "      restartPolicy: Never") print "      priorityClassName: fleet-infra-quiet"; print}'
  fi
}

render_preflight() {
  if test "$1" = qwen; then
    replacement=complete48-p4-v6
    preflight_token=q38-hosted48-p4-v6-preflight
    if test "$GENERATION" = v2; then
      replacement=complete47-p4-v7
      preflight_token=q38-hosted47-p4-v7-preflight
    fi
    awk '/^---$/{exit} {print}' "$ROOT/evals/fleet/cluster/opencode-hosted-successor-preflight-v5.yaml" |
      sed -e "s/q38-hosted49-p4-v5-preflight/$preflight_token/g" \
        -e "s/complete49-p4-v5/$replacement/g"
  else
    replacement=odd47-p4-v8
    test "$GENERATION" = v1 || replacement=odd46-p4-v9
    test "$GENERATION" != v3 || replacement=odd45-p4-v10
    sed -e "s/odd49-p4-v7/$replacement/g" \
      "$ROOT/evals/fleet/cluster/opencode-glm53-hosted-odd-preflight-v7.yaml"
  fi
}

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$config" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=plan.json="$plan" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_hosted_successor.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

assert_terminal_and_idle() {
  local source_name source_uid active
  source_name=$(jq -r '.source.source_job_name' "$plan")
  source_uid=$(jq -r '.source.source_job_uid' "$plan")
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" -o jsonpath='{.metadata.uid}')" = "$source_uid"
  test -z "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" -o jsonpath='{.status.active}')"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')" = True
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg job "$job" --arg pattern "$peer_pattern" '[.items[]
      | select((.status.active // 0) > 0) | .metadata.name
      | select(test($pattern)) | select(. != $job)] | join(",")')
  test -z "$active" || { echo "active model peer exists: $active" >&2; return 1; }
}

selected=()
if test "$TARGET" = qwen || test "$TARGET" = both; then selected+=(qwen); fi
if test "$TARGET" = glm || test "$TARGET" = both; then selected+=(glm); fi
for model in "${selected[@]}"; do
  set_model "$model"
  assert_terminal_and_idle
  for kind_name in "job:$job" "job:$preflight" "configmap:$config"; do
    kind=${kind_name%%:*}; name=${kind_name#*:}
    ! "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1
  done
  configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  render_preflight "$model" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
  render_job "$model" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
done
if test "$MODE" = preview; then jq -n --arg target "$TARGET" '{ok:true,target:$target,workers:1}'; exit; fi

for model in "${selected[@]}"; do
  set_model "$model"
  configmap | "${KUBECTL[@]}" create -f - >/dev/null
  render_preflight "$model" | "${KUBECTL[@]}" create -f - >/dev/null
  for _ in $(seq 1 180); do
    succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.succeeded}')
    failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.failed}')
    test "$succeeded" = 1 && break
    if test "$failed" = 1; then "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$preflight" >&2; exit 1; fi
    sleep 5
  done
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.succeeded}')" = 1
  "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$preflight"
  assert_terminal_and_idle
  render_job "$model" | "${KUBECTL[@]}" create -f - >/dev/null
  uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$job" -o jsonpath='{.metadata.uid}')
  jq -n --arg model "$model" --arg job "$job" --arg uid "$uid" \
    --arg plan_sha256 "$(jq -r '.plan_sha256' "$plan")" \
    '{created:true,model:$model,job:$job,uid:$uid,plan_sha256:$plan_sha256,workers:1}'
done
