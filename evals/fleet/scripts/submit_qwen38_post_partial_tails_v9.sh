#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
MANIFEST=$ROOT/evals/fleet/cluster/opencode-qwen38-hosted-post-partial-tail-v9.yaml
RELEASE=$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-post-partial-tail-scoring-release-v1.json
RUNNER=$ROOT/evals/fleet/scripts/run_opencode_hosted_successor.sh
PLAN_A=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-post-partial-tail-a22-pass4-v9.json
PLAN_B=$ROOT/evals/fleet/configs/qwen38-opencode-hosted-post-partial-tail-b22-pass4-v9.json
CONFIG_A=chris-cyber-q38-hosted-tail-a22-p4-v9
CONFIG_B=chris-cyber-q38-hosted-tail-b22-p4-v9
PREFLIGHT_A=${CONFIG_A}-preflight
PREFLIGHT_B=${CONFIG_B}-preflight
JOB_A=chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9
JOB_B=chris-cyber-q38-opencode11827-hosted-tail-b22-p4-v9
KUBECTL=(kubectl --context "$CONTEXT")

case "$MODE" in preview|submit) ;; *) exit 2 ;; esac
test "$(kubectl config current-context)" = "$CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" get priorityclass fleet-train-high -o jsonpath='{.value}')" = 10000
test "$("${KUBECTL[@]}" get priorityclass fleet-train-high -o jsonpath='{.preemptionPolicy}')" = PreemptLowerPriority
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

uv run python - "$PLAN_A" "$PLAN_B" "$RELEASE" <<'PY'
import sys
from pathlib import Path
from evals.fleet.hosted_sweep_controller import load_object, validate_plan, validate_qwen_post_partial_tail_release

plans = [load_object(Path(path)) for path in sys.argv[1:3]]
release = load_object(Path(sys.argv[3]))
for plan in plans:
    validate_plan(plan)
    validate_qwen_post_partial_tail_release(plan, release)
cells = [
    {(int(row["source_rank"]), int(row["attempt"])) for row in plan["attempts"]}
    for plan in plans
]
if cells[0] & cells[1] or len(cells[0] | cells[1]) != 176:
    raise RuntimeError("Qwen tail plan overlap or arithmetic drifted")
PY

render_doc() {
  local target=$1
  awk -v target="$target" 'BEGIN{doc=1} /^---$/{doc++;next} doc==target{print}' "$MANIFEST"
}

render_config() {
  local config=$1 plan=$2
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$config" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=plan.json="$plan" \
    --from-file=release.json="$RELEASE" \
    --from-file=run.sh="$RUNNER" \
    --dry-run=client -o json | jq '.immutable = true'
}

assert_no_qwen_streams() {
  local count
  count=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq '[.items[] | select((.status.active // 0) > 0) | .metadata.name | select(test("q38.*opencode.*hosted"))] | length')
  test "$count" = 0
}

for object in "configmap:$CONFIG_A" "configmap:$CONFIG_B" "job:$PREFLIGHT_A" "job:$PREFLIGHT_B" "job:$JOB_A" "job:$JOB_B"; do
  kind=${object%%:*}; name=${object#*:}
  ! "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1
done
assert_no_qwen_streams
render_config "$CONFIG_A" "$PLAN_A" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
render_config "$CONFIG_B" "$PLAN_B" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
for document in 1 2 3 4; do render_doc "$document" | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null; done
if test "$MODE" = preview; then
  jq -n --arg a "$(jq -r '.plan_sha256' "$PLAN_A")" --arg b "$(jq -r '.plan_sha256' "$PLAN_B")" \
    '{ok:true,plans:[$a,$b],tasks_per_shard:22,cells_per_shard:88,workers_per_shard:1,maximum_qwen_hosted_streams:2,priority_class:"fleet-train-high",preemption_immunity:false}'
  exit
fi

render_config "$CONFIG_A" "$PLAN_A" | "${KUBECTL[@]}" create -f - >/dev/null
render_config "$CONFIG_B" "$PLAN_B" | "${KUBECTL[@]}" create -f - >/dev/null
render_doc 1 | "${KUBECTL[@]}" create -f - >/dev/null
render_doc 2 | "${KUBECTL[@]}" create -f - >/dev/null
for preflight in "$PREFLIGHT_A" "$PREFLIGHT_B"; do
  for _ in $(seq 1 180); do
    succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.succeeded}')
    failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.failed}')
    test "$succeeded" = 1 && break
    test "$failed" = 1 && exit 1
    sleep 5
  done
  test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.succeeded}')" = 1
  test -z "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$preflight" -o jsonpath='{.status.failed}')"
done
assert_no_qwen_streams
render_doc 3 | "${KUBECTL[@]}" create -f - >/dev/null
render_doc 4 | "${KUBECTL[@]}" create -f - >/dev/null
jq -n \
  --arg job_a "$JOB_A" --arg uid_a "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB_A" -o jsonpath='{.metadata.uid}')" \
  --arg job_b "$JOB_B" --arg uid_b "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB_B" -o jsonpath='{.metadata.uid}')" \
  '{created:true,jobs:[{name:$job_a,uid:$uid_a},{name:$job_b,uid:$uid_b}],tasks_per_shard:22,cells_per_shard:88,workers_per_shard:1,priority_class:"fleet-train-high",preemption_immunity:false}'
