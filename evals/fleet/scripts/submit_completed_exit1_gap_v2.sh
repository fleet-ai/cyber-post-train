#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TARGET=${1:-}
MODE=${2:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2

case "$TARGET" in
  qwen)
    CONFIG=chris-cyber-q38-hosted-gap-q6q7-p4-v2
    JOB=chris-cyber-q38-opencode11827-hosted-gap-q6q7-p4-v2
    PLAN=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-completed-exit1-gap-pass4-v2.json
    RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-completed-exit1-gap-scoring-release-v2.json
    MANIFEST=$ROOT/evals/fleet/cluster/opencode-qwen38-hosted-completed-exit1-gap-v2.yaml
    PREDECESSOR=chris-cyber-q38-opencode11827-hosted-successor49-p4-v8
    PREDECESSOR_UID=84727b82-5187-425b-ba4b-9598f46691c4
    PEER_PATTERN='q38.*opencode.*hosted'
    ;;
  glm)
    CONFIG=chris-cyber-glm53-hosted-gap-g13g15-p4-v2
    JOB=chris-cyber-glm53-opencode11827-hosted-gap-g13g15-p4-v2
    PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-completed-exit1-gap-pass4-v2.json
    RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-completed-exit1-gap-scoring-release-v2.json
    MANIFEST=$ROOT/evals/fleet/cluster/opencode-glm53-hosted-completed-exit1-gap-v2.yaml
    PREDECESSOR=chris-cyber-glm53-opencode11827-hosted-primary46-p4-v12
    PREDECESSOR_UID=0e25db24-8700-478f-8862-d1210511393c
    PEER_PATTERN='glm53.*opencode.*hosted'
    ;;
  *) exit 2 ;;
esac
case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
PREFLIGHT=${CONFIG}-preflight
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
    validate_completed_exit1_gap_scoring_release,
    validate_plan,
)

plan = load_object(Path(sys.argv[1]))
release = load_object(Path(sys.argv[2]))
validate_plan(plan)
validate_completed_exit1_gap_scoring_release(plan, release)
if plan["task_count"] != 2 or plan["new_session_count"] != 5:
    raise RuntimeError("completed exit-1 gap arithmetic drifted")
if len(plan["credited_sessions"]) != 3 or plan["total_session_count"] != 8:
    raise RuntimeError("completed exit-1 credited cell arithmetic drifted")
PY

assert_live_partition() {
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.metadata.uid}')" = "$PREDECESSOR_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.status.active}')" = 1
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg predecessor "$PREDECESSOR" --arg self "$JOB" --arg pattern "$PEER_PATTERN" \
    '[.items[] | select((.status.active // 0) > 0) | .metadata.name
      | select(test($pattern)) | select(. != $predecessor and . != $self)] | join(",")')
  test -z "$active" || { echo "unexpected active same-model hosted peer: $active" >&2; return 1; }
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
  jq -n --arg target "$TARGET" --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
    '{ok:true,target:$target,tasks:2,new_cells:5,credited_cells:3,workers:1,priority_class:"fleet-train-high",plan_sha256:$plan}'
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
  '{created:true,job:$job,uid:$uid,plan_sha256:$plan,tasks:2,new_cells:5,credited_cells:3,workers:1}'
