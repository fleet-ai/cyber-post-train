#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
CONFIG=chris-cyber-glm53-ded-a-v5-gap-s6a4-p4-v1
PREFLIGHT=${CONFIG}-preflight
JOB=chris-cyber-glm53-opencode11827-ded-a-v5-gap-s6a4-p4-v1
PLAN=$ROOT/evals/fleet/configs/glm53-opencode-dedicated-a-v5-source6-attempt4-gap-pass4-v1.json
RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-a-source6-attempt4-gap-scoring-release-v1.json
MANIFEST=$ROOT/evals/fleet/cluster/opencode-glm53-dedicated-a-v5-source6-attempt4-gap-v1.yaml
PREDECESSOR=chris-cyber-glm53-ded-av5-s27-p4-v2
PREDECESSOR_UID=87e2ca85-d684-4d87-9f05-f4399b7906d0
HEAD_POD=ft-run-98c32208-5gpb2-head-7qxwc
HEAD_POD_UID=3f37ab91-4afa-4e4d-8f29-a3eeda70a774
RAY_JOB=ft-run-98c32208
RAY_JOB_UID=ef7cb0f2-84d4-4017-ae31-bf34ebb70d0d
RAY_CLUSTER=ft-run-98c32208-5gpb2
RAY_CLUSTER_UID=5107d72e-ae6a-4582-a490-55b349421d06
SERVICE=ft-run-98c32208-5gpb2-head-svc
SERVICE_UID=2c0e64de-c4a0-4f70-ae07-d15b1adad0b3
case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
KUBECTL=(kubectl --context "$CONTEXT")
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
    validate_dedicated_a_completed_exit1_gap_release,
    validate_plan,
)
plan = load_object(Path(sys.argv[1]))
release = load_object(Path(sys.argv[2]))
validate_plan(plan)
validate_dedicated_a_completed_exit1_gap_release(plan, release)
if plan["task_count"] != 1 or plan["new_session_count"] != 1:
    raise RuntimeError("dedicated A source6/a4 gap arithmetic drifted")
if len(plan["credited_sessions"]) != 3 or plan["total_session_count"] != 4:
    raise RuntimeError("dedicated A source6 credited cell arithmetic drifted")
PY
assert_live_partition() {
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.metadata.uid}')" = "$PREDECESSOR_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.status.failed}')" = 1
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.status.active}')" != 1
  predecessor_pods=$("${KUBECTL[@]}" -n "$NAMESPACE" get pods -l "job-name=$PREDECESSOR" -o json)
  test "$(jq -r --arg uid "$(jq -r '.source.predecessor_pod_uid' "$PLAN")" \
    '[.items[] | select(.metadata.uid == $uid and (.status.phase == "Failed" or .status.phase == "Succeeded"))] | length' \
    <<<"$predecessor_pods")" = 1
  test "$(jq -r '[.items[] | select(.status.phase == "Pending" or .status.phase == "Running")] | length' \
    <<<"$predecessor_pods")" = 0
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" -o jsonpath='{.metadata.uid}')" = "$HEAD_POD_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" -o jsonpath='{.status.containerStatuses[0].ready}')" = true
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$HEAD_POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')" = 0
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" exec "$HEAD_POD" -c ray-head -- sh -c \
    'curl --silent --output /dev/null --write-out "%{http_code}" http://127.0.0.1:8000/health')" = 200
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get rayjob.ray.io "$RAY_JOB" -o jsonpath='{.metadata.uid}')" = "$RAY_JOB_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get raycluster.ray.io "$RAY_CLUSTER" -o jsonpath='{.metadata.uid}')" = "$RAY_CLUSTER_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get service "$SERVICE" -o jsonpath='{.metadata.uid}')" = "$SERVICE_UID"
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg self "$JOB" '[.items[] | select((.status.active // 0) > 0) | .metadata.name
      | select(test("glm53.*ded-a.*gap")) | select(. != $self)] | join(",")')
  test -z "$active" || { echo "unexpected active dedicated-A gap peer: $active" >&2; return 1; }
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
assert_live_partition
for kind_name in "job:$JOB" "job:$PREFLIGHT" "configmap:$CONFIG"; do
  kind=${kind_name%%:*}; name=${kind_name#*:}
  ! "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1
done
render_config | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
render_preflight | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
render_job | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
if test "$MODE" = preview; then
  jq -n --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
    '{ok:true,tasks:1,new_cells:1,credited_cells:3,workers:1,priority_class:"fleet-train-high",plan_sha256:$plan}'
  exit
fi
render_config | "${KUBECTL[@]}" create -f - >/dev/null
render_preflight | "${KUBECTL[@]}" create -f - >/dev/null
for _ in $(seq 1 180); do
  succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')
  failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.failed}')
  test "$succeeded" = 1 && break
  if test "$failed" = 1; then "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT" >&2; exit 1; fi
  sleep 5
done
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')" = 1
"${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT"
assert_live_partition
render_job | "${KUBECTL[@]}" create -f - >/dev/null
uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.metadata.uid}')
jq -n --arg job "$JOB" --arg uid "$uid" --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
  '{created:true,job:$job,uid:$uid,plan_sha256:$plan,tasks:1,new_cells:1,credited_cells:3,workers:1}'
