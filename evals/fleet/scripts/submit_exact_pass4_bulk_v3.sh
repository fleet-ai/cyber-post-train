#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'usage: submit_exact_pass4_bulk_v3.sh prepare-release|preview|submit RELEASE INVENTORY' >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
MODE=$1
RELEASE=$2
INVENTORY=$3
[[ "$MODE" == prepare-release || "$MODE" == preview || "$MODE" == submit ]] || usage

SOURCE_ROOT=$(git rev-parse --show-toplevel)
WORK_DIR=$(mktemp -d)
trap 'rm -rf -- "$WORK_DIR"' EXIT
BINDINGS_FILE=$WORK_DIR/bindings
if [[ "$MODE" == prepare-release ]]; then
  python3 - "$RELEASE" "$INVENTORY" "$SOURCE_ROOT" >"$BINDINGS_FILE" <<'PY'
import sys
from pathlib import Path

release_path = Path(sys.argv[1]).resolve()
inventory_path = Path(sys.argv[2]).resolve(strict=True)
source_root = Path(sys.argv[3]).resolve(strict=True)
if release_path.exists() or release_path.is_relative_to(source_root):
    raise SystemExit("new append-only release must be an absent path outside the repository")
print(release_path)
print(inventory_path)
PY
  PACKAGE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
else
  python3 - "$RELEASE" "$INVENTORY" >"$BINDINGS_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

release_path = Path(sys.argv[1]).resolve(strict=True)
inventory_path = Path(sys.argv[2]).resolve(strict=True)
value = json.loads(release_path.read_text())
commit = value.get("package_commit") if isinstance(value, dict) else None
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit("bulk release package commit is invalid")
print(release_path)
print(inventory_path)
print(commit)
PY
  PACKAGE_COMMIT=$(sed -n '3p' "$BINDINGS_FILE")
fi
ABS_RELEASE=$(sed -n '1p' "$BINDINGS_FILE")
ABS_INVENTORY=$(sed -n '2p' "$BINDINGS_FILE")
FIXED_RELEASE=$WORK_DIR/release.json
FIXED_INVENTORY=$WORK_DIR/inventory.json
if [[ "$MODE" != prepare-release ]]; then
  install -m 0400 "$ABS_RELEASE" "$FIXED_RELEASE"
fi
install -m 0400 "$ABS_INVENTORY" "$FIXED_INVENTORY"
MANIFEST=$WORK_DIR/released.yaml
MANIFEST_RECHECK=$WORK_DIR/released-recheck.yaml
SNAPSHOT=$WORK_DIR/package-commit
EVIDENCE_ROOT=$WORK_DIR/evidence
COLLECTOR_JOB=$WORK_DIR/collector-job.json
COLLECTOR_PODS=$WORK_DIR/collector-pods.json
NS=fleet-train-jobs
SFS_OBSERVER=allie-dev
ACCEPT_JOB=chris-cyber-exact100-prebulk-reconcile-accept-v3

python3 -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$SOURCE_ROOT" --expected-head "$PACKAGE_COMMIT"
python3 -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$SOURCE_ROOT" --commit "$PACKAGE_COMMIT" --destination "$SNAPSHOT"

# Mirror only the fixed, schema-sanitized evidence files needed by the release
# validator.  The desktop never reads attempt directories, prompts, traces,
# flags, scores, or credentials.  The long-lived SFS observer is UID-bound on
# both sides of the copy so a replacement Pod cannot silently change authority.
observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required}
verify_observer() {
  kubectl -n "$NS" get pod "$SFS_OBSERVER" -o json | jq -e --arg uid "$observer_uid" '
    .metadata.uid==$uid and .status.phase=="Running" and
    ([.status.containerStatuses[]?.restartCount] | add // 0)==0 and
    ([.status.containerStatuses[]?.ready] | all)' >/dev/null
}
fetch_evidence() {
  logical=$1
  case "$logical" in
    /mnt/sfs/jobs/*) relative=${logical#/mnt/sfs/jobs/} ;;
    *) printf '%s\n' 'unsafe bulk evidence path' >&2; exit 1 ;;
  esac
  local_path=$EVIDENCE_ROOT/$relative
  remote_path=/shared/jobs/$relative
  mkdir -p "$(dirname "$local_path")"
  kubectl -n "$NS" exec "$SFS_OBSERVER" -- /bin/sh -c \
    'test -f "$1" && test ! -L "$1" && cat "$1"' _ "$remote_path" >"$local_path"
  chmod 0400 "$local_path"
}
mkdir -m 0700 "$EVIDENCE_ROOT"
verify_observer
for evidence in \
  /mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/OBSERVATION.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/TERMINAL.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/exact100-inventory-package.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/qwen3.8-27b-generation7-gate.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/glm-5.3-generation7-gate.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/qwen3.8-27b-generation7-claim.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/glm-5.3-generation7-claim.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/qwen3.8-27b-generation7-release.json \
  /mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v3/glm-5.3-generation7-release.json \
  /mnt/sfs/jobs/chris-q38-ac-r004-a1-g7-v1/CANARY-TERMINAL.json \
  /mnt/sfs/jobs/chris-glm53-ac-r013-a1-g7-v1/CANARY-TERMINAL.json
do
  fetch_evidence "$evidence"
done
verify_observer
kubectl -n "$NS" get job "$ACCEPT_JOB" -o json >"$COLLECTOR_JOB"
kubectl -n "$NS" get pods -l job-name="$ACCEPT_JOB" -o json >"$COLLECTOR_PODS"
chmod 0400 "$COLLECTOR_JOB" "$COLLECTOR_PODS"

if [[ "$MODE" == prepare-release ]]; then
  # Build the authorization only from a still-clean exact Git snapshot and the
  # fixed sanitized evidence mirror.  O_EXCL output preserves append-only
  # semantics; this mode performs no Kubernetes mutation.
  python3 -m evals.fleet.immutable_submission_snapshot assert-stable \
    --repo "$SOURCE_ROOT" --expected-head "$PACKAGE_COMMIT"
  released_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  (
    cd "$SOURCE_ROOT"
    python3 -m evals.fleet.exact_pass4_bulk_v3 build-release \
      --repo "$SOURCE_ROOT" --package-commit "$PACKAGE_COMMIT" \
      --released-at-utc "$released_at" --evidence-root "$EVIDENCE_ROOT" \
      --collector-job "$COLLECTOR_JOB" --collector-pods "$COLLECTOR_PODS" \
      --output "$ABS_RELEASE"
  )
  printf '%s\n' "append-only bulk release prepared at $ABS_RELEASE; no objects created"
  exit 0
fi

# Import and render only the exact archived package commit.  The authority
# repository is used solely for `git show <commit>:<path>` byte verification;
# no mutable working-tree source is read after the snapshot is materialized.
(
  cd "$SNAPSHOT"
  PYTHONPATH="$SNAPSHOT" python3 -m evals.fleet.exact_pass4_bulk_release_renderer_v3 \
    --release "$FIXED_RELEASE" --inventory "$FIXED_INVENTORY" --repo "$SNAPSHOT" \
    --package-commit-authority "$SOURCE_ROOT" --evidence-root "$EVIDENCE_ROOT" \
    --collector-job "$COLLECTOR_JOB" --collector-pods "$COLLECTOR_PODS" \
    --output "$MANIFEST"
)

kubectl create --dry-run=server -f "$MANIFEST" -o name
if [[ "$MODE" == preview ]]; then
  printf '%s\n' 'preview only; no objects created'
  exit 0
fi

# Close the dry-run/apply gap with a second live UID-bound collector check.  A
# byte-identical render is required before the already-reviewed manifest is
# created; neither evidence nor package bytes are rebuilt from the worktree.
kubectl -n "$NS" get job "$ACCEPT_JOB" -o json >"$COLLECTOR_JOB"
kubectl -n "$NS" get pods -l job-name="$ACCEPT_JOB" -o json >"$COLLECTOR_PODS"
(
  cd "$SNAPSHOT"
  PYTHONPATH="$SNAPSHOT" python3 -m evals.fleet.exact_pass4_bulk_release_renderer_v3 \
    --release "$FIXED_RELEASE" --inventory "$FIXED_INVENTORY" --repo "$SNAPSHOT" \
    --package-commit-authority "$SOURCE_ROOT" --evidence-root "$EVIDENCE_ROOT" \
    --collector-job "$COLLECTOR_JOB" --collector-pods "$COLLECTOR_PODS" \
    --output "$MANIFEST_RECHECK"
)
cmp "$MANIFEST" "$MANIFEST_RECHECK"

# The release validator has already consumed the fresh exhaustive API/Kubernetes/
# SFS/claim observer.  Create semantics are intentional: an existing immutable
# ConfigMap or Job aborts instead of updating or replacing live state.
kubectl create -f "$MANIFEST"
