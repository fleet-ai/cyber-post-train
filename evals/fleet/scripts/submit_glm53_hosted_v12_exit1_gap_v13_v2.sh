#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
CONFIG=chris-cyber-glm53-gap-19212529-v13-v2
JOB=chris-cyber-glm53-gap-19212529-v13-v2
PREFLIGHT=${CONFIG}-preflight
PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-v12-exit1-gap4-pass4-v13.json
RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-hosted-v12-exit1-gap-scoring-release-v2.json
TOMBSTONE=$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-hosted-v12-exit1-gap-preflight-v1-failure.json
MANIFEST=$ROOT/evals/fleet/cluster/opencode-glm53-hosted-v12-exit1-gap-v13-v2.yaml
PREDECESSOR=chris-cyber-glm53-opencode11827-hosted-primary46-p4-v12
PREDECESSOR_UID=0e25db24-8700-478f-8862-d1210511393c
PREDECESSOR_POD=chris-cyber-glm53-opencode11827-hosted-primary46-p4-v12-6qnjj
PREDECESSOR_POD_UID=3a04238e-497e-4d21-a8d9-f33c50085879
KUBECTL=(kubectl --context "$CONTEXT")

case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
test "$(kubectl config current-context)" = "$CONTEXT"

uv run python - "$PLAN" "$RELEASE" "$TOMBSTONE" <<'PY'
import sys
from pathlib import Path
from evals.fleet.hosted_sweep_controller import (
    load_object,
    validate_glm_hosted_v12_exit1_gap_release_v2,
    validate_glm_hosted_v12_gap_preflight_failure,
    validate_plan,
)
plan = load_object(Path(sys.argv[1]))
release = load_object(Path(sys.argv[2]))
validate_plan(plan)
tombstone = load_object(Path(sys.argv[3]))
validate_glm_hosted_v12_gap_preflight_failure(tombstone)
validate_glm_hosted_v12_exit1_gap_release_v2(plan, release, tombstone)
assert plan["task_count"] == 4
assert plan["new_session_count"] == 11
assert len(plan["credited_sessions"]) == 5
PY

assert_live_partition() {
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.metadata.uid}')" = "$PREDECESSOR_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREDECESSOR" -o jsonpath='{.status.active}')" = 1
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$PREDECESSOR_POD" -o jsonpath='{.metadata.uid}')" = "$PREDECESSOR_POD_UID"
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get pod "$PREDECESSOR_POD" -o jsonpath='{.status.containerStatuses[0].ready}')" = true
  active=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
    --arg predecessor "$PREDECESSOR" --arg self "$JOB" \
    '[.items[] | select((.status.active // 0) > 0) | .metadata.name
      | select(test("glm53.*opencode.*hosted"))
      | select(. != $predecessor and . != $self)] | join(",")')
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
    '{ok:true,tasks:4,new_cells:11,credited_cells:5,workers:1,priority_class:"fleet-train-high",plan_sha256:$plan}'
  exit
fi

render_config | "${KUBECTL[@]}" create -f - >/dev/null
render_preflight | "${KUBECTL[@]}" create -f - >/dev/null
for _ in $(seq 1 180); do
  succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')
  failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.failed}')
  test "$succeeded" = 1 && break
  test "$failed" = 1 && exit 1
  sleep 5
done
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')" = 1
assert_live_partition
render_job | "${KUBECTL[@]}" create -f - >/dev/null
uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.metadata.uid}')
jq -n --arg job "$JOB" --arg uid "$uid" --arg plan "$(jq -r '.plan_sha256' "$PLAN")" \
  '{created:true,job:$job,uid:$uid,plan_sha256:$plan,tasks:4,new_cells:11,credited_cells:5,workers:1}'
