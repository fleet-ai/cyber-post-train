#!/usr/bin/env bash
set -euo pipefail
umask 077
usage(){ echo 'usage: submit_opencode_generation11_simple_cells_v1.sh preview|submit qwen|glm|both' >&2; exit 2; }
[[ $# == 2 ]] || usage
MODE=$1 TARGET=$2
[[ "$MODE" == preview || "$MODE" == submit ]] || usage
[[ "$TARGET" == qwen || "$TARGET" == glm || "$TARGET" == both ]] || usage
ROOT=$(git rev-parse --show-toplevel); NS=fleet-train-jobs
TEMPLATE=$ROOT/evals/fleet/cluster/opencode-generation11-simple-cell-template-v1.yaml
SECRET=chris-cyber-opencode-evals-v2; SECRET_UID=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
WORK=$(mktemp -d); trap 'rm -rf -- "$WORK"' EXIT
models=(); [[ "$TARGET" == qwen || "$TARGET" == both ]] && models+=(qwen); [[ "$TARGET" == glm || "$TARGET" == both ]] && models+=(glm)
set_model() {
  if [[ "$1" == qwen ]]; then
    JOB=chris-q38-ac-r004-a1-g11-v1; CM=chris-q38-ac-g11-simple-v1
    SPEC=$ROOT/evals/fleet/configs/qwen38-opencode-generation11-simple-r004-a1-v1.json
    OLD_JOB=chris-q38-ac-r004-a1-g10-v1; OLD_UID=eed2d8d4-e638-4af4-98d3-113f5b76ce56
    OLD_POD=chris-q38-ac-r004-a1-g10-v1-r2gbb; OLD_POD_UID=ca02a8e7-61bd-43e6-b6c7-2ee533d48556
    OLD_EXEC=84df092e1904306b308326188609a520f9ffd870a45fad26f65ad1eb60f3aa0b
    NEW_EXEC=b9ca7eeb281f106ae4e47c2779f8af452b04cf123e6f0a5a863eaeb21a2958c9
  else
    JOB=chris-glm53-ac-r013-a1-g11-v1; CM=chris-glm53-ac-g11-simple-v1
    SPEC=$ROOT/evals/fleet/configs/glm53-opencode-generation11-simple-r013-a1-v1.json
    OLD_JOB=chris-glm53-ac-r013-a1-g10-v1; OLD_UID=c367aa54-2171-40c2-9deb-7890027a267f
    OLD_POD=chris-glm53-ac-r013-a1-g10-v1-vlxxn; OLD_POD_UID=de3d3cc3-b8b3-4c23-8c04-0e9988d2dfd2
    OLD_EXEC=fa6623334f96957d0879a97d3d08f1fae226719f080043e274d067a73f0fe0e2
    NEW_EXEC=28b8bac1ba9f51442d65238f0ca3c4345651c55b13e5b0062b449f687a8af215
  fi
}
test "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.metadata.uid}')" = "$SECRET_UID"
test "$(kubectl -n "$NS" get secret "$SECRET" -o go-template='{{range $k,$v := .data}}{{$k}}{{"\n"}}{{end}}')" = FLEET_API_KEY
: "${SFS_OBSERVER_UID:?bind exact allie-dev Pod UID}"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.metadata.uid}')" = "$SFS_OBSERVER_UID"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.status.phase}')" = Running
for model in "${models[@]}"; do
  set_model "$model"
  uv run --no-project --with httpx==0.28.1 python -m evals.fleet.generation11_simple_cell validate --spec "$SPEC" --repo "$ROOT"
  test "$(kubectl -n "$NS" get job "$OLD_JOB" -o jsonpath='{.metadata.uid}')" = "$OLD_UID"
  test "$(kubectl -n "$NS" get job "$OLD_JOB" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')" = True
  test "$(kubectl -n "$NS" get pod "$OLD_POD" -o jsonpath='{.metadata.uid}')" = "$OLD_POD_UID"
  test "$(kubectl -n "$NS" get pod "$OLD_POD" -o jsonpath='{.status.containerStatuses[?(@.name=="evaluator")].state.terminated.exitCode}')" = 1
  test -z "$(kubectl -n "$NS" get job "$JOB" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$JOB" -o name)"
  test -z "$(kubectl -n "$NS" get configmap "$CM" --ignore-not-found -o name)"
  kubectl -n "$NS" exec allie-dev -- sh -c 'for p in "$@"; do test ! -e "$p" && test ! -L "$p" || exit 1; done' _ \
    "/shared/jobs/$OLD_JOB" "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$OLD_EXEC.json" \
    "/shared/jobs/$JOB" "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$NEW_EXEC.json" >/dev/null
  kubectl -n "$NS" create configmap "$CM" --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" --from-file=g11.py="$ROOT/evals/fleet/generation11_simple_cell.py" --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_generation11_simple_cell_v1.sh" --from-file=spec.json="$SPEC" --from-file=q-g10-tombstone.json="$ROOT/docs/evidence/qwen38-study/2026-09-05-qwen38-generation10-preclaim-preoutput-tombstone-v1.json" --from-file=g-g10-tombstone.json="$ROOT/docs/evidence/qwen38-study/2026-09-05-glm53-generation10-preclaim-preoutput-tombstone-v1.json" --dry-run=client -o json | python3 -c 'import json,sys; v=json.load(sys.stdin); v["immutable"]=True; json.dump(v,sys.stdout,separators=(",",":"))' >"$WORK/$model-cm.json"
  sed -e "s/GENERATION11_JOB/$JOB/g" -e "s/GENERATION11_CONFIGMAP/$CM/g" "$TEMPLATE" >"$WORK/$model-job.yaml"
  kubectl create --dry-run=server -f "$WORK/$model-cm.json" >/dev/null
  kubectl create --dry-run=server -f "$WORK/$model-job.yaml" >/dev/null
done
[[ "$MODE" == preview ]] && { echo "Generation-11 $TARGET preview valid; no objects created"; exit 0; }
for model in "${models[@]}"; do
  set_model "$model"
  test -z "$(kubectl -n "$NS" get job "$JOB" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get configmap "$CM" --ignore-not-found -o name)"
  kubectl create -f "$WORK/$model-cm.json"
  kubectl create -f "$WORK/$model-job.yaml"
done
