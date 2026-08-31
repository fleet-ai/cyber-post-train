#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-sft-evidence-v4
OBSERVER=chris-cyber-qwen36-sft-evidence-observer-v4
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-sft-evidence-v4-job.yaml"
PLAN="$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$(kubectl -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=training__init__.py="$ROOT/evals/post_sft/runtime/training__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=training_tokenizer_equivalence.py="$ROOT/training/tokenizer_equivalence.py" \
    --from-file=base-registration.json="$ROOT/evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json" \
    --from-file=plan.json="$PLAN" --dry-run=client -o json | \
    python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)'
}

require_absent() {
  local resource name
  while read -r resource name; do
    if kubectl -n "$NAMESPACE" get "$resource" "$name" >/dev/null 2>&1; then
      echo "Resource $NAMESPACE/$resource/$name already exists; refusing replacement" >&2
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
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_artifacts \
      validate-evidence-bundle --plan "$PLAN" --root "$ROOT" >/dev/null
    validate_job
    configmap | kubectl create --dry-run=server -f - >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    echo "create-only v4 evidence server dry-run passed; no resources created"
    ;;
  submit)
    test "$(kubectl -n "$NAMESPACE" get rayjob ft-run-29f2bedf -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    require_absent
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_artifacts \
      validate-evidence-bundle --plan "$PLAN" --root "$ROOT" >/dev/null
    validate_job
    config_map=$(mktemp)
    trap 'rm -f "$config_map"' EXIT
    configmap > "$config_map"
    kubectl create --dry-run=server -f "$config_map" >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    require_absent
    kubectl create -f "$config_map"
    kubectl create -f "$JOB"
    ;;
  *) echo "usage: $0 [preview|submit]" >&2; exit 2 ;;
esac
