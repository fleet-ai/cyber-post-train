#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NAME=chris-cyber-qwen36-base-artifact-inspect-6a9e13bd-v1
NAMESPACE=inference
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
JOB="$ROOT/evals/post_sft/cluster/qwen36-base-artifact-inspect-job.yaml"
PLAN="$ROOT/configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
MODE=${1:-preview}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"

configmap() {
  kubectl -n "$NAMESPACE" create configmap "$NAME" \
    --from-file=evals__init__.py="$ROOT/evals/__init__.py" \
    --from-file=evals_webexploitbench__init__.py="$ROOT/evals/webexploitbench/__init__.py" \
    --from-file=post_sft_evidence.py="$ROOT/evals/webexploitbench/post_sft_evidence.py" \
    --from-file=training__init__.py="$ROOT/training/__init__.py" \
    --from-file=training_io.py="$ROOT/training/io.py" \
    --from-file=training_post_sft_artifacts.py="$ROOT/training/post_sft_artifacts.py" \
    --from-file=post-sft-plan.json="$PLAN" --dry-run=client -o json | \
    python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout)'
}

owned_resources() {
  printf '%s\n' \
    "configmap/$NAME" \
    "serviceaccount/chris-cyber-qwen36-base-artifact-observer-v1" \
    "role.rbac.authorization.k8s.io/chris-cyber-qwen36-base-artifact-observer-v1" \
    "rolebinding.rbac.authorization.k8s.io/chris-cyber-qwen36-base-artifact-observer-v1" \
    "job.batch/$NAME"
}

case "$MODE" in
  preview)
    configmap | kubectl create --dry-run=server -f - >/dev/null
    kubectl create --dry-run=server -f "$JOB" >/dev/null
    echo "create-only server dry-run passed; no resources created"
    ;;
  submit)
    test "$#" = 1
    while IFS= read -r resource; do
      if kubectl -n "$NAMESPACE" get "$resource" >/dev/null 2>&1; then
        echo "$resource already exists in $NAMESPACE; refusing partial reuse or replacement" >&2
        exit 1
      fi
    done < <(owned_resources)
    configmap | kubectl create -f -
    kubectl create -f "$JOB"
    ;;
  collect)
    test "$#" = 2
    OUTPUT=$2
    test ! -e "$OUTPUT" && test ! -L "$OUTPUT"
    test "$(kubectl -n "$NAMESPACE" get job "$NAME" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
    temporary=$(mktemp "${OUTPUT}.tmp.XXXXXX")
    trap 'rm -f "$temporary"' EXIT
    pod=$(kubectl -n "$NAMESPACE" get pod -l "batch.kubernetes.io/job-name=$NAME" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
    test -n "$pod" && test "$(printf '%s\n' "$pod" | wc -l | tr -d ' ')" = 1
    pod_uid=$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.metadata.uid}')
    kubectl -n "$NAMESPACE" logs "pod/$pod" --container inspect > "$temporary"
    test "$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.metadata.uid}')" = "$pod_uid"
    PYTHONPATH="$ROOT" uv run python -c 'import json,sys; from evals.webexploitbench.post_sft_evidence import _embedded_digest; value=json.load(open(sys.argv[1])); _embedded_digest(value,"base_artifact_receipt_sha256")' "$temporary"
    chmod 600 "$temporary"
    ln "$temporary" "$OUTPUT"
    ;;
  *) echo "usage: $0 [preview|submit|collect OUTPUT]" >&2; exit 2 ;;
esac
