#!/usr/bin/env bash
set -euo pipefail
umask 077

MODE=${1:-preview}
case "$MODE" in preview|prepare-release|submit-source|submit-accept) ;; *)
  printf '%s\n' 'usage: submit_exact_pass4_prebulk_reconciliation_v4.sh preview|prepare-release|submit-source|submit-accept' >&2
  exit 2
esac

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
SOURCE=chris-cyber-exact100-prebulk-reconcile-source-v4
ACCEPT=chris-cyber-exact100-prebulk-reconcile-accept-v4
INTENT=chris-cyber-exact100-prebulk-intent-v4
RELEASE_CM=chris-exact100-prebulk-release-v4
RELEASE_REL=docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-prebulk-reconciliation-release-v4.json
RELEASE=$ROOT/$RELEASE_REL
MANIFEST_REL=evals/fleet/cluster/exact-pass4-prebulk-reconciliation-held-v4.yaml
OUT=/shared/jobs/chris-cyber-exact100-prebulk-reconcile-v4
work=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$work"' EXIT

if [[ "$MODE" == preview ]]; then
  uv run --with httpx==0.28.1 python \
    -m evals.fleet.exact_pass4_prebulk_reconciliation_package_v4 preview --repo "$ROOT"
  kubectl -n "$NS" create --dry-run=server -f "$ROOT/$MANIFEST_REL" -o name >/dev/null
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

SOURCE_HEAD=$(git -C "$ROOT" rev-parse HEAD)
test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ "$MODE" == prepare-release ]]; then
  output=${PREBULK_RELEASE_OUTPUT:?}
  case "$output" in /*) ;; *) printf '%s\n' 'PREBULK_RELEASE_OUTPUT must be absolute' >&2; exit 2 ;; esac
  case "$output" in "$ROOT"/*) printf '%s\n' 'release output must be outside the repository' >&2; exit 2 ;; esac
  test ! -e "$output" && test ! -L "$output"
  uv run python -m evals.fleet.exact_pass4_prebulk_reconciliation_v4 render-release \
    --root "$ROOT" --package-commit "$SOURCE_HEAD" >"$output"
  chmod 0400 "$output"
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

test -f "$RELEASE" && test ! -L "$RELEASE"
PACKAGE_COMMIT=$(jq -er .package_commit "$RELEASE")
[[ "$PACKAGE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$ROOT" cat-file -e "$PACKAGE_COMMIT^{commit}"
git -C "$ROOT" merge-base --is-ancestor "$PACKAGE_COMMIT" "$SOURCE_HEAD"
snapshot=$work/package
uv run python -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$ROOT" --commit "$PACKAGE_COMMIT" --destination "$snapshot"
install -D -m 0400 "$RELEASE" "$snapshot/$RELEASE_REL"
(
  cd "$snapshot"
  uv run --no-project --with httpx==0.28.1 python \
    -m evals.fleet.exact_pass4_prebulk_reconciliation_v4 validate-release --root "$snapshot"
)

bundle=$work/bundle.yaml
uv run python - "$snapshot" "$PACKAGE_COMMIT" "$RELEASE_REL" "$bundle" "$MODE" <<'PY'
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

from evals.fleet import exact_pass4_prebulk_reconciliation_package_v4 as package

root, commit, release_rel, output, mode = sys.argv[1:]
root = Path(root)
built = package.build_package(root)
observer = built["configmaps"][package.OBSERVER_NAME]
observer["data"]["package_aggregate_sha256"] = built["aggregate_sha256"]
observer["data"]["package_commit"] = commit
release_raw = (root / release_rel).read_bytes()
release_cm = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": "chris-exact100-prebulk-release-v4", "namespace": "fleet-train-jobs"},
    "immutable": True,
    "data": {
        "release.json": release_raw.decode(),
        "release_file_sha256": "sha256:" + hashlib.sha256(release_raw).hexdigest(),
    },
}
manifest = yaml.safe_load((root / package.MANIFEST_PATH).read_text())
static = [item for item in manifest["items"] if item["kind"] != "Job"]
job_name = (
    "chris-cyber-exact100-prebulk-reconcile-source-v4"
    if mode == "submit-source"
    else "chris-cyber-exact100-prebulk-reconcile-accept-v4"
)
job = next(item for item in manifest["items"] if item["kind"] == "Job" and item["metadata"]["name"] == job_name)
intent = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": "chris-cyber-exact100-prebulk-intent-v4", "namespace": "fleet-train-jobs"},
    "immutable": True,
    "data": {
        "package_commit": commit,
        "package_aggregate_sha256": built["aggregate_sha256"],
        "release_file_sha256": release_cm["data"]["release_file_sha256"],
    },
}
items = (
    [*built["configmaps"].values(), release_cm, intent, *static, copy.deepcopy(job)]
    if mode == "submit-source"
    else [copy.deepcopy(job)]
)
Path(output).write_text(yaml.safe_dump({"apiVersion": "v1", "kind": "List", "items": items}, sort_keys=False))
PY

for name in "$SOURCE" "$ACCEPT"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done
while IFS=$'\t' read -r kind name; do
  test -z "$(kubectl -n "$NS" get "$kind" "$name" --ignore-not-found -o name)"
done < <(uv run python - "$bundle" <<'PY'
import sys
from pathlib import Path

import yaml

for item in yaml.safe_load(Path(sys.argv[1]).read_text())["items"]:
    print(f"{item['kind']}\t{item['metadata']['name']}")
PY
)
if [[ "$MODE" == submit-source ]]; then
  test ! -e "$OUT" && test ! -L "$OUT"
  test -z "$(kubectl -n "$NS" get configmap "$INTENT" --ignore-not-found -o name)"
else
  test -f "$OUT/OBSERVATION.json" && test ! -L "$OUT/OBSERVATION.json"
  kubectl -n "$NS" get job "$SOURCE" -o json | jq -e '
    any(.status.conditions[]?; .type=="Complete" and .status=="True") and
    (all(.status.conditions[]?; .type!="Failed" or .status!="True"))' >/dev/null
  kubectl -n "$NS" get configmap "$INTENT" -o name >/dev/null
fi
kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
test "$(git -C "$ROOT" rev-parse HEAD)" = "$SOURCE_HEAD"
test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
kubectl -n "$NS" create -f "$bundle" -o name
trap - EXIT
rm -rf "$work"
