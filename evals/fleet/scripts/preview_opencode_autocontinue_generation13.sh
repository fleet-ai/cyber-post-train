#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# -eq 1 ]] || { echo 'usage: preview_opencode_autocontinue_generation13.sh qwen3.8-27b|glm-5.3|all' >&2; exit 2; }
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
SECRET=chris-cyber-opencode-evals-v2
SECRET_UID=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
OBSERVER_UID=73dabe56-60f8-4879-be9f-365196c502e3
WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
case "$1" in
  qwen3.8-27b) MODELS=(qwen3.8-27b) ;;
  glm-5.3) MODELS=(glm-5.3) ;;
  all) MODELS=(qwen3.8-27b glm-5.3) ;;
  *) exit 2 ;;
esac

test "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.metadata.uid}')" = "$SECRET_UID"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.metadata.uid}')" = "$OBSERVER_UID"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.status.phase}')" = Running

for model in "${MODELS[@]}"; do
  if [[ "$model" == qwen3.8-27b ]]; then
    job=chris-q38-ac-r004-a1-g13-v1
    cm=chris-q38-ac-r004-a1-g13-v1-run-v1
    execution=ecca567c5f3a76dac03610a161bc2d1c359459a06684d6ff529b170bf6f47b53
  else
    job=chris-glm53-ac-r013-a1-g13-v1
    cm=chris-glm53-ac-r013-a1-g13-v1-run-v1
    execution=41f2ddaa570451a0031a5ec1afaa6b829becf4173ea91a117d515937d5b655e7
  fi
  test -z "$(kubectl -n "$NS" get job "$job" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$job" -o name)"
  test -z "$(kubectl -n "$NS" get configmap "$cm" --ignore-not-found -o name)"
  kubectl -n "$NS" exec allie-dev -- sh -c \
    'for p in "$@"; do test ! -e "$p" && test ! -L "$p" || exit 1; done' _ \
    "/shared/jobs/$job" "/shared/jobs/$job-stages" \
    "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$execution.json" >/dev/null
  output="$WORK/${model//./_}.yaml"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.autocontinue_generation13_package render \
    --model "$model" --repo "$ROOT" --output "$output"
  kubectl create --dry-run=server -f "$output" -o name >/dev/null
done
echo 'Generation-13 preview and server dry-run valid; no objects created'
