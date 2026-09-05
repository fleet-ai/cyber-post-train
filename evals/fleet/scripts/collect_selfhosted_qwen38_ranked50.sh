#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${1:-preview}
NAMESPACE=fleet-train-jobs
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")
SOURCE_JOB=chris-cyber-qwen38-qcode-fleet-ranked50-base-v1
ACCEPT_JOB=chris-cyber-qwen38-qcode-fleet-ranked50-accept-v1
ACCEPTOR=chris-cyber-qwen38-qcode-fleet-ranked50-acceptor-v1
MANIFEST=$ROOT/evals/fleet/cluster/qwen38-ranked50-acceptance-job.yaml
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

test "$#" -le 1
test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test "$("${KUBECTL[@]}" -n "$NAMESPACE" get localqueue training-lq -o jsonpath='{.status.conditions[?(@.type=="Active")].status}')" = True

validate_source() {
  "${KUBECTL[@]}" -n "$NAMESPACE" get job "$SOURCE_JOB" -o json >"$tmp_dir/job.json"
  "${KUBECTL[@]}" -n "$NAMESPACE" get configmap "$SOURCE_JOB" -o json >"$tmp_dir/configmap.json"
  local source_uid
  source_uid=$(jq -r '.metadata.uid' "$tmp_dir/job.json")
  "${KUBECTL[@]}" -n "$NAMESPACE" get pods \
    -l "batch.kubernetes.io/controller-uid=$source_uid" -o json >"$tmp_dir/pods.json"
  python3 - "$tmp_dir/job.json" "$tmp_dir/pods.json" "$tmp_dir/configmap.json" \
    "$tmp_dir/source-binding.json" <<'PY'
import json
import os
import sys

job, pod_list, config_map = (json.load(open(path)) for path in sys.argv[1:4])
status = job.get("status") or {}
conditions = status.get("conditions") or []
complete = [row for row in conditions if row.get("type") == "Complete" and row.get("status") == "True"]
failed = [row for row in conditions if row.get("type") == "Failed" and row.get("status") == "True"]
pods = pod_list.get("items") or []
if (
    len(complete) != 1
    or failed
    or status.get("succeeded") != 1
    or int(status.get("failed") or 0)
    or int(status.get("active") or 0)
):
    raise SystemExit("source Job is not exclusively Complete")
if len(pods) != 1 or pods[0].get("status", {}).get("phase") != "Succeeded":
    raise SystemExit("source Pod terminal identity drift")
if config_map.get("immutable") is not True:
    raise SystemExit("source ConfigMap is not immutable")
binding = {
    "schema_version": "fleet-qwen38-ranked50-acceptance-source-binding-v1",
    "job_uid": job.get("metadata", {}).get("uid"),
    "pod_uid": pods[0].get("metadata", {}).get("uid"),
    "configmap_uid": config_map.get("metadata", {}).get("uid"),
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(sys.argv[4], flags, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(binding, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
}

acceptance_configmap() {
  "${KUBECTL[@]}" -n "$NAMESPACE" create configmap "$ACCEPTOR" \
    --from-file=source-binding.json="$tmp_dir/source-binding.json" \
    --dry-run=client -o json | jq '.immutable = true'
}

require_acceptance_absent() {
  local resource name
  while read -r resource name; do
    if "${KUBECTL[@]}" -n "$NAMESPACE" get "$resource" "$name" >/dev/null 2>&1; then
      echo "Resource $NAMESPACE/$resource/$name already exists; refusing to replace it" >&2
      exit 1
    fi
  done <<EOF
configmap $ACCEPTOR
serviceaccount $ACCEPTOR
role.rbac.authorization.k8s.io $ACCEPTOR
rolebinding.rbac.authorization.k8s.io $ACCEPTOR
job.batch $ACCEPT_JOB
EOF
}

case "$MODE" in
  preview)
    "${KUBECTL[@]}" create --dry-run=server -f "$MANIFEST" >/dev/null
    echo "acceptance collector server dry-run passed; source terminal state is checked only at submit"
    ;;
  submit)
    validate_source
    require_acceptance_absent
    acceptance_configmap | "${KUBECTL[@]}" create --dry-run=server -f - >/dev/null
    "${KUBECTL[@]}" create --dry-run=server -f "$MANIFEST" >/dev/null
    rm -f "$tmp_dir/source-binding.json"
    validate_source
    require_acceptance_absent
    acceptance_configmap | "${KUBECTL[@]}" create -f -
    "${KUBECTL[@]}" create -f "$MANIFEST"
    ;;
  *)
    echo "usage: $0 [preview|submit]" >&2
    exit 2
    ;;
esac
