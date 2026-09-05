#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview || "$MODE" == submit ]] || {
  printf '%s\n' 'usage: submit_exact_pass4_task_inventory_v1.sh preview|submit' >&2
  exit 2
}
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
JOB=chris-cyber-exact100-pass4-inventory-v1
BOOTSTRAP=chris-cyber-exact100-pass4-inventory-bootstrap-v1
INTENT=chris-cyber-exact100-pass4-inventory-intent-v1
SECRET=chris-cyber-opencode-evals-v2
SECRET_UID=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
MANIFEST_REL=evals/fleet/cluster/exact-pass4-task-inventory-observer-v1.yaml
PACKAGE_REL=evals/fleet/exact_pass4_task_inventory_package.py
OUT_ROOT=/shared/jobs/chris-cyber-exact100-pass4-inventory-v1
PACKAGE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)

stable_checkout() {
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$PACKAGE_COMMIT"
  test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
}
if [[ "$MODE" == submit ]]; then stable_checkout; fi

test "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.metadata.uid}')" = "$SECRET_UID"
for name in "$BOOTSTRAP" "$INTENT"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
test -z "$(kubectl -n "$NS" get job "$JOB" --ignore-not-found -o name)"
test -z "$(kubectl -n "$NS" get pod -l job-name="$JOB" -o name)"

work=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$work"' EXIT
package="$work/package.json"
intent="$work/intent.json"
manifest="$work/job.yaml"
builder="$work/package.py"
git -C "$ROOT" show "$PACKAGE_COMMIT:$PACKAGE_REL" >"$builder"
uv run --no-project python "$builder" \
  render-configmap --repo "$ROOT" --package-commit "$PACKAGE_COMMIT" >"$package"
uv run --no-project python "$builder" \
  render-intent --configmap "$package" >"$intent"
uv run --no-project python "$builder" \
  render-job-manifest --repo "$ROOT" --package-commit "$PACKAGE_COMMIT" >"$manifest"
test "$(jq -r '.data["package.json"]|fromjson|.job_manifest.sha256' "$package")" = \
  "sha256:$(sha256sum "$manifest" | awk '{print $1}')"
chmod 0400 "$builder" "$package" "$intent" "$manifest"

kubectl -n "$NS" create --dry-run=server -f "$package" -o name >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$intent" -o name >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$manifest" -o name >/dev/null

if [[ "$MODE" == preview ]]; then
  printf '%s\n' 'exact pass@4 task inventory observer preview passed; no objects created'
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

stable_checkout
observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required for the create-once SFS gate}
kubectl -n "$NS" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and
   ([.status.containerStatuses[].restartCount]|add)==0 and
   ([.status.containerStatuses[].ready]|all)' >/dev/null
kubectl -n "$NS" exec allie-dev -- sh -c \
  'test ! -e "$1" && test ! -L "$1"' _ "$OUT_ROOT"
stable_checkout

# The immutable intent is the atomic create-once boundary.  A partial create is
# never retried under these names; it is reconciled and succeeded by a new
# reviewed generation if necessary.
kubectl -n "$NS" create -f "$intent" -o name >/dev/null
kubectl -n "$NS" create -f "$package" -o name >/dev/null
kubectl -n "$NS" create -f "$manifest" -o name
trap - EXIT
rm -rf "$work"
