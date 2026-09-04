#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
CONFIG=chris-cyber-glm53-hosted-primary46-p4-v12
PREFLIGHT=${CONFIG}-preflight
JOB=chris-cyber-glm53-opencode11827-hosted-primary46-p4-v12
PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-http500-primary46-pass4-v12.json
RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-http500-hosted-primary-scoring-release-v1.json
MANIFEST=$ROOT/evals/fleet/cluster/opencode-glm53-hosted-http500-primary46-v12.yaml
OLD_ORIGINAL=chris-cyber-glm53-opencode11827-hosted-odd45-p4-v10
OLD_ORIGINAL_UID=a4812f3d-d76a-49ba-a2cf-fec575ca071d
OLD_REPLACEMENT=chris-cyber-glm53-opencode11827-hosted-replacement-r106-p4-v1
OLD_REPLACEMENT_UID=e4a31855-907e-43a4-86f9-3ba0019e3fe4
KUBECTL=(kubectl --context "$CONTEXT")

case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
test "$(kubectl config current-context)" = "$CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

uv run python - "$PLAN" "$RELEASE" <<'PY'
import sys
from pathlib import Path
from evals.fleet.hosted_sweep_controller import (
    load_object,
    validate_plan,
    validate_glm_http500_scoring_release,
)

plan = load_object(Path(sys.argv[1]))
release = load_object(Path(sys.argv[2]))
validate_plan(plan)
validate_glm_http500_scoring_release(plan, release)
if (
    plan["task_count"] != 46
    or plan["new_session_count"] != 184
    or plan["dedicated_a_task_count"] != 27
    or plan["dedicated_b_task_count"] != 27
    or plan["primary_estimator_task_count"] != 100
    or plan["primary_estimator_cell_count"] != 400
    or plan["execution"]["required_priority_class"] != "fleet-train-high"
):
    raise RuntimeError("GLM HTTP500 hosted primary arithmetic drifted")
PY

assert_terminal_predecessors() {
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$OLD_ORIGINAL" \
    -o jsonpath='{.metadata.uid}')" = "$OLD_ORIGINAL_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$OLD_ORIGINAL" \
    -o jsonpath='{.status.failed}')" = 1
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$OLD_REPLACEMENT" \
    -o jsonpath='{.metadata.uid}')" = "$OLD_REPLACEMENT_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$OLD_REPLACEMENT" \
    -o jsonpath='{.status.failed}')" = 1
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg self "$JOB" '[.items[]
      | select((.status.active // 0) > 0)
      | .metadata.name
      | select(test("glm53.*opencode.*hosted"))
      | select(. != $self)] | join(",")')
  test -z "$active" || { echo "unexpected active hosted GLM peer: $active" >&2; return 1; }
}

render_config() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$CONFIG" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=plan.json="$PLAN" \
    --from-file=release.json="$RELEASE" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_hosted_successor.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}

render_preflight() { awk '/^---$/{exit} {print}' "$MANIFEST"; }
render_job() { awk 'found{print} /^---$/{found=1; next}' "$MANIFEST"; }

assert_terminal_predecessors
for kind_name in "job:$JOB" "job:$PREFLIGHT" "configmap:$CONFIG"; do
  kind=${kind_name%%:*}
  name=${kind_name#*:}
  ! "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1
done
render_config | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
render_preflight | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
render_job | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
if test "$MODE" = preview; then
  jq -n --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
    '{ok:true,model:"glm53",new_tasks:46,new_sessions:184,workers:1,
      priority_class:"fleet-train-high",plan_sha256:$plan}'
  exit
fi

render_config | "${KUBECTL[@]}" create -f - >/dev/null
render_preflight | "${KUBECTL[@]}" create -f - >/dev/null
for _ in $(seq 1 180); do
  succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" \
    -o jsonpath='{.status.succeeded}')
  failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" \
    -o jsonpath='{.status.failed}')
  test "$succeeded" = 1 && break
  if test "$failed" = 1; then
    "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT" >&2
    exit 1
  fi
  sleep 5
done
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" \
  -o jsonpath='{.status.succeeded}')" = 1
"${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT"
assert_terminal_predecessors
render_job | "${KUBECTL[@]}" create -f - >/dev/null
uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.metadata.uid}')
jq -n --arg job "$JOB" --arg uid "$uid" \
  --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
  '{created:true,job:$job,uid:$uid,plan_sha256:$plan,new_tasks:46,new_sessions:184,
    workers:1,priority_class:"fleet-train-high"}'
