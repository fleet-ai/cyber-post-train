#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-endpoint-flock-preflight-v1.yaml"
SOURCE="$ROOT/evals/fleet/endpoint_lease_preflight.py"
RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-flock-preflight-release-v1.json"
MANIFEST_SHA=d9fa9ed9b1c144977d4132e1a4b2929a1037400b7aec3fab62e047c0d825a7ce
SOURCE_SHA=dc012e6cdaf34f8297d48d43a483c1af2b2099f637403fc93d9ea45946d18088
CONFIGMAP=chris-opencode11827-ac-flock-preflight-v1
HOLDER=chris-opencode11827-ac-flock-holder-v1
PROBER=chris-opencode11827-ac-flock-prober-v1
SFS_ROOT=/mnt/sfs/endpoint-leases/preflight-opencode11827-autocontinue-v1
A_HEAD=ft-run-98c32208-5gpb2-head-7qxwc
A_HEAD_UID=3f37ab91-4afa-4e4d-8f29-a3eeda70a774

test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA"
uv run python -m evals.fleet.autocontinue_flock_release --release "$RELEASE" >/dev/null

test "$(kubectl -n "$NS" get pod "$A_HEAD" -o jsonpath='{.metadata.uid}')" = "$A_HEAD_UID"
for name in "$HOLDER" "$PROBER"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done
test -z "$(kubectl -n "$NS" get configmap "$CONFIGMAP" --ignore-not-found -o name)"
kubectl -n "$NS" exec "$A_HEAD" -- test ! -e "$SFS_ROOT"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
CONFIG_JSON="$TMP_ROOT/configmap.json"
kubectl -n "$NS" create configmap "$CONFIGMAP" \
  --from-file=endpoint_lease_preflight.py="$SOURCE" \
  --dry-run=client -o json | jq '.immutable=true' >"$CONFIG_JSON"
kubectl -n "$NS" create --dry-run=server -f "$CONFIG_JSON" -o name >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"mode":"preview","objects_absent":true,"sfs_root_absent":true,"scored_launch_authorized":false}'
  exit 0
fi

kubectl -n "$NS" create -f "$CONFIG_JSON" -o name
awk '/^---$/{exit} {print}' "$MANIFEST" | kubectl -n "$NS" create -f - -o name
awk 'seen{print} /^---$/{seen=1}' "$MANIFEST" | kubectl -n "$NS" create -f - -o name
for name in "$HOLDER" "$PROBER"; do
  kubectl -n "$NS" get job "$name" -o jsonpath='{.metadata.name}{" "}{.metadata.uid}{"\n"}'
done
