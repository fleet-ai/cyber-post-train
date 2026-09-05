#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  printf '%s\n' \
    'usage: submit_opencode_autocontinue_generation10_glm53_v1.sh held-preview' \
    '   or: submit_opencode_autocontinue_generation10_glm53_v1.sh preview|submit RELEASE DUPLICATE LAUNCH_ROUTE' >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
MODE=$1
[[ "$MODE" == held-preview || "$MODE" == preview || "$MODE" == submit ]] || usage
SOURCE_ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
JOB=chris-glm53-ac-r013-a1-g10-v1
MODEL_CM=chris-glm53-ac-r013-a1-g10-v1-run-v1
CORE_A=chris-glm53-ac-g10-runtime-core-a-v1
CORE_B=chris-glm53-ac-g10-runtime-core-b-v1
SECRET=chris-cyber-opencode-evals-v2
SECRET_UID=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
MANIFEST=evals/fleet/cluster/opencode-autocontinue-generation10-glm53-held-v1.yaml

if [[ "$MODE" == held-preview ]]; then
  [[ $# -eq 1 ]] || usage
  uv run python -m evals.fleet.autocontinue_generation10_glm53_v1 preview --repo "$SOURCE_ROOT" >/dev/null
  uv run python -m evals.fleet.autocontinue_generation10_glm53_package_v1 preview \
    --repo "$SOURCE_ROOT"
  exit 0
fi
[[ $# -eq 4 ]] || usage
RELEASE=$(realpath "$2")
DUPLICATE=$(realpath "$3")
LAUNCH_ROUTE=$(realpath "$4")
PACKAGE_COMMIT=$(jq -er '.package_commit | select(test("^[0-9a-f]{40}$"))' "$RELEASE")
WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
SNAPSHOT=$WORK/package
RENDERED=$WORK/rendered.yaml
ALLOWLIST=$WORK/allowlist.json
PACKAGE_MANIFEST=$WORK/package-manifest.json

uv run python -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$SOURCE_ROOT" --expected-head "$PACKAGE_COMMIT"
uv run python -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$SOURCE_ROOT" --commit "$PACKAGE_COMMIT" --destination "$SNAPSHOT"
(cd "$SNAPSHOT" && uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation10_glm53_package_v1 \
  render-package-manifest --repo "$SNAPSHOT" --output "$PACKAGE_MANIFEST")

# Metadata-only binding: the Secret value is neither decoded nor printed.
test "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.metadata.uid}')" = "$SECRET_UID"
test "$(kubectl -n "$NS" get secret "$SECRET" -o go-template='{{range $k,$v := .data}}{{$k}}{{"\n"}}{{end}}')" = FLEET_API_KEY
for name in "$JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done
for name in "$CORE_A" "$CORE_B" "$MODEL_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done

: "${SFS_OBSERVER_UID:?SFS_OBSERVER_UID must bind the exact allie-dev Pod}"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.metadata.uid}')" = "$SFS_OBSERVER_UID"
test "$(kubectl -n "$NS" get pod allie-dev -o jsonpath='{.status.phase}')" = Running
kubectl -n "$NS" exec allie-dev -- sh -c \
  'for p in "$@"; do test ! -e "$p" && test ! -L "$p" || exit 1; done' _ \
  /shared/jobs/chris-glm53-ac-r013-a1-g10-v1 \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/a8b2f9b65f55c3c1f6cccc8189a5fece1023dd6cb238a02391766c28b3ba3bb8.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/ffedddf7dcf6c0df2f4b4999f3016f387b6b39eba8c8f0b27779197ecf36b3ad.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/fa6623334f96957d0879a97d3d08f1fae226719f080043e274d067a73f0fe0e2.json \
  >/dev/null

(cd "$SNAPSHOT" && uv run --no-project --with pyyaml --with httpx==0.28.1 python - \
  "$SNAPSHOT" "$RELEASE" "$DUPLICATE" "$LAUNCH_ROUTE" "$MANIFEST" \
  "$PACKAGE_COMMIT" "$PACKAGE_MANIFEST" "$RENDERED" "$ALLOWLIST") <<'PY'
import copy
import json
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation10_glm53_package_v1 as package
from evals.fleet import autocontinue_generation10_glm53_v1 as runtime
from evals.fleet import kubernetes_create_relay as relay

root, release_path, duplicate_path, route_path, manifest_path, commit, package_manifest_path, rendered, allowlist = sys.argv[1:]
root = Path(root)
release_raw = Path(release_path).read_bytes()
duplicate_raw = Path(duplicate_path).read_bytes()
release = runtime.load(Path(release_path))
duplicate = runtime.load(Path(duplicate_path))
route = runtime.load(Path(route_path))
configmaps = package.released_configmaps(
    root, release, release_raw, duplicate, duplicate_raw, commit, route
)
built_manifest = package.build_package(root)["model_manifest"]
if runtime.load(Path(package_manifest_path)) != built_manifest:
    raise ValueError("tool-rendered package manifest drifted")
job = yaml.safe_load((root / manifest_path).read_text())
job = copy.deepcopy(job)
job["metadata"]["annotations"].update({
    "cyber-post-train.fleet.ai/preview-only": "false",
    "cyber-post-train.fleet.ai/launch-authorized": "true",
    "cyber-post-train.fleet.ai/release-receipt-sha256": release["receipt_sha256"],
})
items = [*configmaps.values(), job]
Path(rendered).write_text(yaml.safe_dump({"apiVersion": "v1", "kind": "List", "items": items}, sort_keys=False))
allow = relay.build_allowlist(items)
relay.build_envelope(items, allow)
Path(allowlist).write_bytes(relay.canonical_json(allow) + b"\n")
PY

(cd "$SNAPSHOT" && uv run --no-project --with pyyaml python \
  -m evals.fleet.kubernetes_create_relay validate \
  --manifest "$RENDERED" --allowlist "$ALLOWLIST") >/dev/null

if [[ "$MODE" == preview ]]; then
  printf '%s\n' 'preview valid; no objects created'
  exit 0
fi
: "${GENERATION10_CREATE_RECEIPT:?GENERATION10_CREATE_RECEIPT must be an unused absolute path}"
[[ "$GENERATION10_CREATE_RECEIPT" == /* ]] || usage
test ! -e "$GENERATION10_CREATE_RECEIPT" && test ! -L "$GENERATION10_CREATE_RECEIPT"

# Recheck immediately before the create-only relay. A partial create is
# preserved for reconciliation and must never be repeated.
for name in "$JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done
for name in "$CORE_A" "$CORE_B" "$MODEL_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
(cd "$SNAPSHOT" && uv run --no-project --with pyyaml python \
  -m evals.fleet.kubernetes_create_relay create \
  --manifest "$RENDERED" --allowlist "$ALLOWLIST" --transport auto \
  --receipt "$GENERATION10_CREATE_RECEIPT")
