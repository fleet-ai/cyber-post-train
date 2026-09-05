#!/usr/bin/env bash
set -euo pipefail
umask 077

MODE=${1:-preview}
[[ "$MODE" == preview || "$MODE" == prepare-release || "$MODE" == submit-source || "$MODE" == submit-accept ]] || {
  printf '%s\n' 'usage: submit_exact_pass4_prebulk_reconciliation_v3.sh preview|prepare-release|submit-source|submit-accept' >&2
  exit 2
}
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
SOURCE=chris-cyber-exact100-prebulk-reconcile-source-v3
ACCEPT=chris-cyber-exact100-prebulk-reconcile-accept-v3
OBSERVER_RBAC=chris-cyber-exact100-prebulk-observer-v3
SOURCE_INTENT=chris-cyber-exact100-prebulk-source-intent-v3
ACCEPT_INTENT=chris-cyber-exact100-prebulk-accept-intent-v3
RELEASE_CM=chris-exact100-prebulk-release-v3
RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-prebulk-reconciliation-release-v3.json"
MANIFEST_REL=evals/fleet/cluster/exact-pass4-prebulk-reconciliation-held-v3.yaml
OUT=/shared/jobs/chris-cyber-exact100-prebulk-reconcile-v3

work=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$work"' EXIT

if [[ "$MODE" == preview ]]; then
  uv run --with httpx==0.28.1 python \
    -m evals.fleet.exact_pass4_prebulk_reconciliation_package_v3 \
    preview --repo "$ROOT"
  kubectl -n "$NS" create --dry-run=server \
    -f "$ROOT/$MANIFEST_REL" -o name >/dev/null
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

RELEASE_SNAPSHOT="$work/prebulk-release.json"
SOURCE_HEAD=$(git -C "$ROOT" rev-parse HEAD)
test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ "$MODE" == prepare-release ]]; then
  PACKAGE_COMMIT=${PREBULK_PACKAGE_COMMIT:-$SOURCE_HEAD}
  test "$PACKAGE_COMMIT" = "$SOURCE_HEAD"
else
  test -f "$RELEASE"
  uv run --no-project python - "$RELEASE" "$RELEASE_SNAPSHOT" <<'PY'
import os
import stat
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
if source.is_symlink() or not source.is_file() or destination.exists():
    raise SystemExit("prebulk release source or snapshot path is unsafe")
flags = os.O_RDONLY
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
source_fd = os.open(source, flags)
try:
    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1_048_576:
        raise SystemExit("prebulk release source is not a bounded regular file")
    raw = b""
    while len(raw) <= 1_048_576:
        chunk = os.read(source_fd, 65_536)
        if not chunk:
            break
        raw += chunk
    after = os.fstat(source_fd)
    if len(raw) > 1_048_576 or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SystemExit("prebulk release source changed while snapshotting")
finally:
    os.close(source_fd)
destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
try:
    os.write(target_fd, raw)
    os.fsync(target_fd)
finally:
    os.close(target_fd)
os.chmod(destination, 0o400)
PY
  test "$(stat -f '%Lp' "$RELEASE_SNAPSHOT" 2>/dev/null || stat -c '%a' "$RELEASE_SNAPSHOT")" = 400
  PACKAGE_COMMIT=$(jq -er .package_commit "$RELEASE_SNAPSHOT")
fi
[[ "$PACKAGE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$ROOT" cat-file -e "$PACKAGE_COMMIT^{commit}"
git -C "$ROOT" merge-base --is-ancestor "$PACKAGE_COMMIT" "$SOURCE_HEAD"
snapshot="$work/package"
uv run python -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$ROOT" --commit "$PACKAGE_COMMIT" --destination "$snapshot"

observer_uid=${SFS_OBSERVER_UID:?}
assert_observer() {
  kubectl -n "$NS" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" '
    .metadata.uid==$uid and .status.phase=="Running" and
    ([.status.initContainerStatuses[]?.restartCount, .status.containerStatuses[]?.restartCount] | add // 0)==0' >/dev/null
}
assert_observer

evidence_root="$work/evidence-sfs"
mkdir -m 0700 "$evidence_root"
mirror_one() {
  logical=$1
  remote="/shared/${logical#/mnt/sfs/}"
  destination="$evidence_root/${logical#/mnt/sfs/}"
  mkdir -p "$(dirname "$destination")"
  remote_before=$(kubectl -n "$NS" exec allie-dev -- sh -c \
    'test -f "$1" && test ! -L "$1" && sha256sum "$1" | cut -d" " -f1' _ "$remote")
  test -n "$remote_before"
  temporary="$destination.partial"
  test ! -e "$destination" && test ! -L "$destination"
  kubectl -n "$NS" exec allie-dev -- sh -c 'cat "$1"' _ "$remote" >"$temporary"
  chmod 0400 "$temporary"
  mv "$temporary" "$destination"
  local_sha=$(sha256sum "$destination" | cut -d' ' -f1)
  remote_after=$(kubectl -n "$NS" exec allie-dev -- sh -c \
    'test -f "$1" && test ! -L "$1" && sha256sum "$1" | cut -d" " -f1' _ "$remote")
  test "$remote_before" = "$local_sha"
  test "$remote_after" = "$local_sha"
}

fixed_evidence_paths=(
  /mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json
  /mnt/sfs/jobs/chris-q38-ac-r004-a1-g7-v1/CANARY-TERMINAL.json
  /mnt/sfs/jobs/chris-glm53-ac-r013-a1-g7-v1/CANARY-TERMINAL.json
  /mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535.json
  /mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524.json
)
(
  cd "$snapshot"
  uv run --no-project --with httpx==0.28.1 python - "$snapshot" \
    "${fixed_evidence_paths[@]}" <<'PY'
import sys
from pathlib import Path

from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconcile

root = Path(sys.argv[1])
declared = set(sys.argv[2:])
expected = {str(path) for path in reconcile.fixed_evidence_paths(root).values()}
if declared != expected or len(declared) != 5:
    raise SystemExit("prebulk evidence mirror allowlist drifted")
PY
)
for logical in "${fixed_evidence_paths[@]}"; do
  mirror_one "$logical"
done
assert_observer

if [[ "$MODE" == prepare-release ]]; then
  release_output=${PREBULK_RELEASE_OUTPUT:?}
  case "$release_output" in
    /*) ;;
    *) printf '%s\n' 'PREBULK_RELEASE_OUTPUT must be absolute' >&2; exit 2 ;;
  esac
  case "$release_output" in
    "$ROOT"/*) printf '%s\n' 'PREBULK_RELEASE_OUTPUT must be outside the repository' >&2; exit 2 ;;
  esac
  test ! -e "$release_output" && test ! -L "$release_output"
  uv run --no-project --with httpx==0.28.1 python - \
    "$ROOT" "$PACKAGE_COMMIT" "$evidence_root" "$release_output" <<'PY'
import json
import os
import sys
from pathlib import Path

from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconcile

root, commit, evidence_root, output = sys.argv[1:]
root = Path(root)
evidence_root = Path(evidence_root)
first = reconcile.render_release_from_commit(root, commit, evidence_root=evidence_root)
second = reconcile.render_release_from_commit(root, commit, evidence_root=evidence_root)
if first != second:
    raise SystemExit("prebulk release rendering is nondeterministic")
raw = json.dumps(first, indent=2, sort_keys=True).encode() + b"\n"
output = Path(output)
output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
try:
    os.write(fd, raw)
    os.fsync(fd)
finally:
    os.close(fd)
os.chmod(output, 0o400)
reconcile.validate_release(first, root, evidence_root=evidence_root)
PY
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$SOURCE_HEAD"
  test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
  test "$(stat -f '%Lp' "$release_output" 2>/dev/null || stat -c '%a' "$release_output")" = 400
  assert_observer
  for logical in "${fixed_evidence_paths[@]}"; do
    remote="/shared/${logical#/mnt/sfs/}"
    mirrored="$evidence_root/${logical#/mnt/sfs/}"
    remote_sha=$(kubectl -n "$NS" exec allie-dev -- sh -c \
      'test -f "$1" && test ! -L "$1" && sha256sum "$1" | cut -d" " -f1' _ "$remote")
    test "$remote_sha" = "$(sha256sum "$mirrored" | cut -d' ' -f1)"
  done
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

package_json="$work/package.json"
(
  cd "$snapshot"
  uv run --no-project --with httpx==0.28.1 python - \
    "$snapshot" "$package_json" "$PACKAGE_COMMIT" "$RELEASE_SNAPSHOT" "$evidence_root" <<'PY'
import json
import sys
from pathlib import Path

from evals.fleet import exact_pass4_prebulk_reconciliation_package_v3 as package
from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconcile

root, output, commit, release_path, evidence_root = sys.argv[1:]
root = Path(root)
built = package.build_package(root)
release = reconcile._load(Path(release_path))
reconcile.validate_release(release, root, evidence_root=Path(evidence_root))
if (
    release.get("schema_version") != reconcile.RELEASE_SCHEMA
    or release.get("status") != "RELEASED"
    or release.get("launch_authorized") is not True
    or release.get("bulk_launch_authorized") is not False
    or release.get("package_commit") != commit
    or release.get("package", {}).get("aggregate_sha256") != built["aggregate_sha256"]
    or release.get("receipt_sha256") != reconcile.bulk.digest(release, "receipt_sha256")
):
    raise SystemExit("prebulk release static binding is invalid")
built["configmaps"][package.OBSERVER_NAME]["data"].update(
    {"package_aggregate_sha256": built["aggregate_sha256"], "package_commit": commit}
)
Path(output).write_text(json.dumps(built["configmaps"], sort_keys=True))
PY
)

release_sha="sha256:$(sha256sum "$RELEASE_SNAPSHOT" | awk '{print $1}')"
release_snapshot_sha=$release_sha
objects="$work/source-objects.yaml"
accept_job="$work/accept-job.yaml"
(
  cd "$snapshot"
  uv run --no-project --with pyyaml==6.0.3 python - \
    "$package_json" "$RELEASE_SNAPSHOT" "$release_sha" "$snapshot/$MANIFEST_REL" \
    "$objects" "$accept_job" <<'PY'
import json
import sys
from pathlib import Path

import yaml

package_path, release_path, release_sha, manifest_path, objects_path, accept_path = sys.argv[1:]
configmaps = list(json.loads(Path(package_path).read_text()).values())
release_cm = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": "chris-exact100-prebulk-release-v3",
        "namespace": "fleet-train-jobs",
    },
    "immutable": True,
    "data": {
        "release.json": Path(release_path).read_text(),
        "release_file_sha256": release_sha,
    },
}
items = yaml.safe_load(Path(manifest_path).read_text())["items"]
for item in items:
    item.setdefault("metadata", {})["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
    }
source = next(
    item for item in items
    if item["kind"] == "Job" and item["metadata"]["name"].endswith("source-v3")
)
accept = next(
    item for item in items
    if item["kind"] == "Job" and item["metadata"]["name"].endswith("accept-v3")
)
rbac = [item for item in items if item["kind"] != "Job"]
Path(objects_path).write_text(
    yaml.safe_dump(
        {"apiVersion": "v1", "kind": "List", "items": [*configmaps, release_cm, *rbac, source]},
        sort_keys=False,
    )
)
Path(accept_path).write_text(yaml.safe_dump(accept, sort_keys=False))
PY
)

objects_sha=$(sha256sum "$objects" | cut -d' ' -f1)
accept_job_sha=$(sha256sum "$accept_job" | cut -d' ' -f1)

final_stability_check() {
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$SOURCE_HEAD"
  test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
  test "sha256:$(sha256sum "$RELEASE_SNAPSHOT" | cut -d' ' -f1)" = "$release_snapshot_sha"
  test "$(sha256sum "$objects" | cut -d' ' -f1)" = "$objects_sha"
  test "$(sha256sum "$accept_job" | cut -d' ' -f1)" = "$accept_job_sha"
  assert_observer
  for logical in "${fixed_evidence_paths[@]}"; do
    remote="/shared/${logical#/mnt/sfs/}"
    mirrored="$evidence_root/${logical#/mnt/sfs/}"
    remote_sha=$(kubectl -n "$NS" exec allie-dev -- sh -c \
      'test -f "$1" && test ! -L "$1" && sha256sum "$1" | cut -d" " -f1' _ "$remote")
    test "$remote_sha" = "$(sha256sum "$mirrored" | cut -d' ' -f1)"
  done
}

if [[ "$MODE" == submit-source ]]; then
  for name in "$SOURCE_INTENT" "$ACCEPT_INTENT" \
    chris-exact100-prebulk-runtime-core-a-v3 \
    chris-exact100-prebulk-runtime-core-b-v3 \
    chris-exact100-prebulk-runtime-observer-v3 "$RELEASE_CM"; do
    test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
  done
  for kind in serviceaccount role rolebinding; do
    test -z "$(kubectl -n "$NS" get "$kind" "$OBSERVER_RBAC" --ignore-not-found -o name)"
  done
  for name in "$SOURCE" "$ACCEPT"; do
    test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
    test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
  done
  for prerequisite in chris-cyber-exact100-pass4-inventory-v2 \
    chris-q38-ac-r004-a1-g7-v1 chris-glm53-ac-r013-a1-g7-v1; do
    kubectl -n "$NS" get job "$prerequisite" -o json | jq -e '
      ([.status.conditions[]? | select(.type=="Complete" and .status=="True")]|length)==1 and
      ([.status.conditions[]? | select(.type=="Failed" and .status=="True")]|length)==0' >/dev/null
  done
  kubectl -n "$NS" exec allie-dev -- sh -c 'test ! -e "$1" && test ! -L "$1"' _ "$OUT"
  kubectl -n "$NS" create --dry-run=server -f "$objects" -o name >/dev/null
  final_stability_check
  kubectl -n "$NS" create cm "$SOURCE_INTENT" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-literal=release_file_sha256="$release_sha" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl create -f - >/dev/null
  final_stability_check
  kubectl -n "$NS" create -f "$objects" -o name
else
  test -n "$(kubectl -n "$NS" get configmap "$SOURCE_INTENT" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get configmap "$ACCEPT_INTENT" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get job "$ACCEPT" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$ACCEPT" -o name)"
  uv run --no-project python - "$package_json" "$RELEASE_CM" <<'PY'
import json
import subprocess
import sys

expected = json.load(open(sys.argv[1]))
for name, configmap in expected.items():
    live = json.loads(subprocess.check_output(
        ["kubectl", "-n", "fleet-train-jobs", "get", "configmap", name, "-o", "json"]
    ))
    if live.get("immutable") is not True or live.get("data") != configmap.get("data"):
        raise SystemExit("live prebulk package ConfigMap drifted")
live_release = json.loads(subprocess.check_output(
    ["kubectl", "-n", "fleet-train-jobs", "get", "configmap", sys.argv[2], "-o", "json"]
))
if live_release.get("immutable") is not True:
    raise SystemExit("live prebulk release ConfigMap is mutable")
PY
  live_release_sha=$(kubectl -n "$NS" get configmap "$RELEASE_CM" -o jsonpath='{.data.release_file_sha256}')
  live_release=$(kubectl -n "$NS" get configmap "$RELEASE_CM" -o jsonpath='{.data.release\.json}')
  test "$live_release_sha" = "$release_sha"
  test "$live_release" = "$(cat "$RELEASE_SNAPSHOT")"
  kubectl -n "$NS" get job "$SOURCE" -o json | jq -e '
    ([.status.conditions[]? | select(.type=="Complete" and .status=="True")]|length)==1 and
    ([.status.conditions[]? | select(.type=="Failed" and .status=="True")]|length)==0' >/dev/null
  source_uid=$(kubectl -n "$NS" get job "$SOURCE" -o jsonpath='{.metadata.uid}')
  kubectl -n "$NS" get pod -l job-name="$SOURCE" -o json | jq -e --arg uid "$source_uid" '
    [.items[] | select(.metadata.ownerReferences[]?.uid==$uid)] as $pods |
    ($pods|length)==1 and $pods[0].status.phase=="Succeeded" and
    ([$pods[0].status.initContainerStatuses[]?.restartCount, $pods[0].status.containerStatuses[]?.restartCount] | add // 0)==0 and
    ([$pods[0].status.initContainerStatuses[]?.state.terminated.exitCode, $pods[0].status.containerStatuses[]?.state.terminated.exitCode] | all(.==0))' >/dev/null
  kubectl -n "$NS" exec allie-dev -- sh -c \
    'test -f "$1/OBSERVATION.json" && test ! -e "$1/TERMINAL.json" && test ! -L "$1/TERMINAL.json"' _ "$OUT"
  kubectl -n "$NS" create --dry-run=server -f "$accept_job" -o name >/dev/null
  final_stability_check
  kubectl -n "$NS" create cm "$ACCEPT_INTENT" \
    --from-literal=source_job="$SOURCE" \
    --from-literal=source_job_uid="$source_uid" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl create -f - >/dev/null
  final_stability_check
  kubectl -n "$NS" create -f "$accept_job" -o name
fi

trap - EXIT
rm -rf "$work"
