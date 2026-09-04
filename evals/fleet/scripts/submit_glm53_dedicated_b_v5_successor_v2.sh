#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
CONFIG=chris-cyber-glm53-dedicated-b-v5-successor27-p4-v2
PREFLIGHT=chris-cyber-glm53-ded-bv5-s27-p4-v2-preflight
JOB=chris-cyber-glm53-ded-bv5-s27-p4-v2
PLAN=$ROOT/evals/fleet/configs/glm53-opencode-dedicated-b-v5-successor27-pass4-v2.json
RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-b-v5-scoring-release-v1.json
MANIFEST=$ROOT/evals/fleet/cluster/opencode-glm53-dedicated-b-v5-successor27-v2.yaml
SERVICE=ft-run-9e92209d-pppzg-head-svc
SERVICE_UID=14ae906a-60ef-4a02-b6bf-814a00c2c89f
RAYJOB_UID=2bbfd34a-d74a-4497-ab28-48cada0c9753
RAYCLUSTER=ft-run-9e92209d-pppzg
RAYCLUSTER_UID=f7f7ae34-6a79-482a-b771-4639b786f66a
HEAD_POD=ft-run-9e92209d-pppzg-head-rc9nd
HEAD_POD_UID=bfeb4c6a-9e4a-4f14-adb2-3f516dd25fdb
OLD_JOB=chris-cyber-glm53-opencode11827-dedicated-b-even27-p4-v1
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
    validate_glm_dedicated_b_v5_scoring_release,
    validate_plan,
)

plan = load_object(Path(sys.argv[1]))
release = load_object(Path(sys.argv[2]))
validate_plan(plan)
validate_glm_dedicated_b_v5_scoring_release(plan, release)
if (
    plan["task_count"] != 27
    or plan["new_session_count"] != 108
    or plan["execution"]["required_priority_class"] != "fleet-train-high"
    or plan["treatment_block"]["serving_generation"] != "v5"
):
    raise RuntimeError("GLM dedicated B v5 arithmetic or treatment drifted")
PY

assert_live_treatment() {
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get service "$SERVICE" \
    -o jsonpath='{.metadata.uid}')" = "$SERVICE_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get raycluster "$RAYCLUSTER" \
    -o jsonpath='{.metadata.uid}')" = "$RAYCLUSTER_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" \
    -o jsonpath='{.metadata.uid}')" = "$HEAD_POD_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" \
    -o jsonpath='{.status.containerStatuses[0].ready}')" = true
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" \
    -o jsonpath='{.status.containerStatuses[0].restartCount}')" = 0
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get rayjobs -o json | jq \
    --arg uid "$RAYJOB_UID" '[.items[] | select(.metadata.uid == $uid)] | length')" = 1
  ! "${KUBECTL[@]}" -n "$NAMESPACE" get job "$OLD_JOB" >/dev/null 2>&1
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

assert_live_treatment
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
    '{ok:true,model:"glm53",treatment:"dedicated-b-v5",new_tasks:27,
      new_sessions:108,workers:1,priority_class:"fleet-train-high",plan_sha256:$plan}'
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
assert_live_treatment
render_job | "${KUBECTL[@]}" create -f - >/dev/null
uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.metadata.uid}')
jq -n --arg job "$JOB" --arg uid "$uid" \
  --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
  '{created:true,job:$job,uid:$uid,plan_sha256:$plan,new_tasks:27,
    new_sessions:108,workers:1,priority_class:"fleet-train-high"}'
