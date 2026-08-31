#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-bf16-cast-v1
OBSERVER=chris-cyber-qwen36-sft-bf16-cast-observer-v1
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-sft-bf16-cast-job.yaml"
PLAN="$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True

configmap() {
  local cast_input=$1
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=training__init__.py="$ROOT/training/__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=training_post_sft_cast.py="$ROOT/training/post_sft_cast.py" \
    --from-file=cast-input.json="$cast_input" --dry-run=client -o json | \
    python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)'
}

require_all_absent() {
  local resource name
  while read -r resource name; do
    if kubectl -n "$NAMESPACE" get "$resource" "$name" >/dev/null 2>&1; then
      echo "Resource $NAMESPACE/$resource/$name already exists; refusing partial replacement" >&2
      exit 1
    fi
  done <<EOF
serviceaccount $OBSERVER
role.rbac.authorization.k8s.io $OBSERVER
rolebinding.rbac.authorization.k8s.io $OBSERVER
configmap $NAME
job.batch $NAME
EOF
}

validate_job() {
  PYTHONPATH="$ROOT" uv run python - "$JOB" <<'PY'
import pathlib
import sys
import yaml
from training.cluster_policy import validate_training_manifest

documents = list(yaml.safe_load_all(pathlib.Path(sys.argv[1]).read_text()))
jobs = [value for value in documents if isinstance(value, dict) and value.get("kind") == "Job"]
assert len(jobs) == 1
validate_training_manifest(jobs[0])
PY
}

case "$MODE" in
  preview)
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_cast validate-bundle \
      --plan "$PLAN" --root "$ROOT" >/dev/null
    validate_job
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    echo "create-only queued CPU cast server dry-run passed; no resources created"
    ;;
  submit)
    test "$#" = 3
    observation=$2
    raw_manifest=$3
    test "$(kubectl -n "$NAMESPACE" get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    test "$(kubectl -n "$NAMESPACE" get job chris-cyber-qwen36-sft-evidence-v2 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
    require_all_absent
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_cast validate-bundle \
      --plan "$PLAN" --root "$ROOT" >/dev/null
    cast_input=$(mktemp)
    config_map=$(mktemp)
    trap 'rm -f "$cast_input" "$config_map"' EXIT
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_cast build-input \
      --plan "$PLAN" --observation "$observation" --raw-manifest "$raw_manifest" \
      --output "$cast_input"
    configmap "$cast_input" > "$config_map"
    validate_job
    kubectl create --dry-run=server -f "$config_map" >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    require_all_absent
    kubectl create -f "$config_map"
    kubectl create -f "$JOB"
    ;;
  *) echo "usage: $0 [preview|submit OBSERVATION RAW_MANIFEST]" >&2; exit 2 ;;
esac
