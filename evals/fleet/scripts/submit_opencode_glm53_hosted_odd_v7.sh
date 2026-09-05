#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
SECRET=chris-cyber-opencode-evals-v2
CONFIGMAP=chris-cyber-glm53-hosted-odd49-p4-v7
JOB=chris-cyber-glm53-opencode11827-hosted-odd49-p4-v7
PREFLIGHT=chris-cyber-glm53-hosted-odd49-p4-v7-preflight
PLAN=$ROOT/evals/fleet/configs/glm53-opencode-hosted-odd49-pass4-v7.json
JOB_FILE=$ROOT/evals/fleet/cluster/opencode-glm53-hosted-odd-job-v7.yaml
PREFLIGHT_FILE=$ROOT/evals/fleet/cluster/opencode-glm53-hosted-odd-preflight-v7.yaml
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")

case "$MODE" in preview|submit) ;; *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;; esac
test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq \
  -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get secret "$SECRET" \
  -o jsonpath='{.data.FLEET_API_KEY}' | wc -c | tr -d ' ')" -gt 0

uv run python - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path
from evals.fleet.hosted_sweep_controller import validate_plan

plan = json.loads(Path(sys.argv[1]).read_text())
validate_plan(plan)
if (plan["task_count"], plan["new_session_count"], plan["total_session_count"],
        len(plan["credited_sessions"])) != (49, 196, 196, 0):
    raise RuntimeError("GLM hosted odd-shard arithmetic drifted")
if {row["source_rank"] for row in plan["tasks"]} != set(range(3, 100, 2)):
    raise RuntimeError("GLM hosted odd-shard ownership drifted")
if {row["source_rank"] for row in plan["reserved_tasks"]} != set(range(4, 101, 2)):
    raise RuntimeError("GLM dedicated even-shard reservation drifted")
PY

source_name=$(jq -r '.source.source_job_name' "$PLAN")
source_uid=$(jq -r '.source.source_job_uid' "$PLAN")
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" -o jsonpath='{.metadata.uid}')" = "$source_uid"
test -z "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" -o jsonpath='{.status.active}')"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$source_name" \
  -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')" = True

active_peer=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
  --arg job "$JOB" '[.items[] | select((.status.active // 0) > 0)
    | .metadata.name | select(test("(opencode.*glm53|glm53.*opencode)"))
    | select(. != $job)] | join(",")')
test -z "$active_peer" || { echo "active GLM OpenCode peer exists: $active_peer" >&2; exit 1; }
for kind_name in "job:$JOB" "job:$PREFLIGHT" "configmap:$CONFIGMAP"; do
  kind=${kind_name%%:*}; name=${kind_name#*:}
  if "${KUBECTL[@]}" -n "$NAMESPACE" get "$kind" "$name" >/dev/null 2>&1; then
    echo "$kind $NAMESPACE/$name already exists; refusing create-once reuse" >&2
    exit 1
  fi
done

configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$CONFIGMAP" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=plan.json="$PLAN" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_hosted_successor.sh" \
    --dry-run=client -o json | jq '.immutable = true'
}
configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
"${KUBECTL[@]}" create --dry-run=server -f "$PREFLIGHT_FILE" >/dev/null
"${KUBECTL[@]}" create --dry-run=server -f "$JOB_FILE" >/dev/null
if test "$MODE" = preview; then
  jq -n '{ok:true,model:"glm53",hosted_source_ranks:"odd 3-99",workers:1}'
  exit 0
fi

configmap | "${KUBECTL[@]}" create -f - >/dev/null
"${KUBECTL[@]}" create -f "$PREFLIGHT_FILE" >/dev/null
for _ in $(seq 1 180); do
  succeeded=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')
  failed=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.failed}')
  if test "$succeeded" = 1; then break; fi
  if test "$failed" = 1; then
    "${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT" >&2
    exit 1
  fi
  sleep 5
done
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$PREFLIGHT" -o jsonpath='{.status.succeeded}')" = 1
"${KUBECTL[@]}" -n "$NAMESPACE" logs "job/$PREFLIGHT"
active_peer=$("${KUBECTL[@]}" -n "$NAMESPACE" get jobs -o json | jq -r \
  --arg job "$JOB" '[.items[] | select((.status.active // 0) > 0)
    | .metadata.name | select(test("(opencode.*glm53|glm53.*opencode)"))
    | select(. != $job)] | join(",")')
test -z "$active_peer"
"${KUBECTL[@]}" create -f "$JOB_FILE" >/dev/null
uid=$("${KUBECTL[@]}" -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.metadata.uid}')
jq -n --arg job "$JOB" --arg uid "$uid" --arg plan_sha256 "$(jq -r '.plan_sha256' "$PLAN")" \
  '{created:true,model:"glm53",job:$job,uid:$uid,plan_sha256:$plan_sha256,workers:1}'
