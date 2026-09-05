#!/usr/bin/env bash
set -euo pipefail

mode=${1:-preview}
[[ "$mode" == preview || "$mode" == submit ]] || {
  printf '%s\n' 'usage: submit_hosted_concurrency4_qualification_release_v1.sh preview|submit' >&2
  exit 2
}

root=$(git rev-parse --show-toplevel)
namespace=fleet-train-jobs
release_rel=docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-launch-release-v1.json
release="$root/$release_rel"
job=chris-cyber-hosted-c4-qualification-v1
bootstrap=chris-cyber-hosted-c4-qualification-bootstrap-v1
held_intent=chris-cyber-hosted-c4-qualification-intent-v1
release_intent=chris-cyber-hosted-c4-qualification-release-v1
secret=chris-cyber-opencode-evals-v2
secret_uid=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
q_job=chris-q38-ac-r004-a1-g5-v1
g_job=chris-glm53-ac-r013-a1-g5-v1
q_execution=0e359500ab08e29efc95c8cb3ee6c2bc4d719e31cb8c9aadf53fedda03e1c3f1
g_execution=54008eb3de475a49d737cf8f7a4c344bf1815b4f2e8818a49d6ec236a3d8a83b

work=$(mktemp -d /private/tmp/hosted-c4-release.XXXXXX)
snapshot="$work/snapshot"
cleanup() {
  git -C "$root" worktree remove --force "$snapshot" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

test -f "$release"
implementation_commit=$(jq -er '.implementation.commit' "$release")
package_commit=$(jq -er '.package.commit' "$release")
test "$package_commit" = cbe8b2c7654cbda1f5392c8466b953e335be2641
git -C "$root" cat-file -e "$implementation_commit^{commit}"
git -C "$root" merge-base --is-ancestor "$package_commit" "$implementation_commit"
if git -C "$root" cat-file -e "$implementation_commit:$release_rel" 2>/dev/null; then
  printf '%s\n' 'release receipt must be append-only after the implementation commit' >&2
  exit 1
fi

git -C "$root" worktree add --detach "$snapshot" "$implementation_commit" >/dev/null
install -m 0600 "$release" "$snapshot/$release_rel"
(
  cd "$snapshot"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v1 \
    validate-release --release "$snapshot/$release_rel" --repo "$snapshot"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v1 \
    render-bundle --release "$snapshot/$release_rel" --repo "$snapshot" \
    --output "$work/bundle.yaml"
)

bundle_sha=$(sha256sum "$work/bundle.yaml" | awk '{print $1}')
uv run --project "$root" python - "$work/bundle.yaml" <<'PY'
import sys
import yaml

bundle = yaml.safe_load(open(sys.argv[1]))
assert bundle["kind"] == "List" and len(bundle["items"]) == 3
bootstrap, intent, job = bundle["items"]
assert bootstrap["kind"] == intent["kind"] == "ConfigMap"
assert job["kind"] == "Job"
assert intent["data"]["launch_authorized"] == "true"
assert intent["data"]["scored_bulk_launch_authorized"] == "false"
pod = job["spec"]["template"]["spec"]
assert pod["priorityClassName"] == "fleet-serve-low"
assert pod["preemptionPolicy"] == "Never"
assert job["spec"]["backoffLimit"] == 0
assert not any(
    "gpu" in key.lower()
    for container in pod["containers"]
    for bucket in container.get("resources", {}).values()
    for key in bucket
)
PY

if command -v kubectl >/dev/null && kubectl config current-context >/dev/null 2>&1; then
  kubectl -n "$namespace" create --dry-run=server -f "$work/bundle.yaml" -o name >/dev/null
fi

if [[ "$mode" == preview ]]; then
  jq -n \
    --arg package_commit "$package_commit" \
    --arg implementation_commit "$implementation_commit" \
    --arg receipt "$(jq -er .receipt_sha256 "$release")" \
    '{ok:true,status:"RELEASED_PREVIEW",package_commit:$package_commit,implementation_commit:$implementation_commit,release_receipt_sha256:$receipt,objects_created:false,chat_completion_requests:24,task_instance_session_scoring_or_verifier_calls:0}'
  exit 0
fi

for name in "$bootstrap" "$held_intent" "$release_intent"; do
  test -z "$(kubectl -n "$namespace" get configmap "$name" --ignore-not-found -o name)"
done
test -z "$(kubectl -n "$namespace" get job "$job" --ignore-not-found -o name)"
test -z "$(kubectl -n "$namespace" get pod -l job-name="$job" -o name)"
test "$(kubectl -n "$namespace" get secret "$secret" -o jsonpath='{.metadata.uid}')" = "$secret_uid"

for pair in "qwen3.8-27b:$q_job:$q_execution" "glm-5.3:$g_job:$g_execution"; do
  IFS=: read -r model source_job execution <<<"$pair"
  kubectl -n "$namespace" get job "$source_job" -o json >"$work/$model-job.json"
  kubectl -n "$namespace" get pods -l job-name="$source_job" -o json >"$work/$model-pods.json"
  (
    cd "$snapshot"
    uv run --no-project --with pyyaml --with httpx==0.28.1 python \
      -m evals.fleet.hosted_concurrency4_qualification_release_v1 \
      validate-terminal-job --job-json "$work/$model-job.json" \
      --pods-json "$work/$model-pods.json" --job-name "$source_job"
  )
  live_job_uid=$(jq -er '.metadata.uid' "$work/$model-job.json")
  live_pod_uid=$(jq -er '.items[0].metadata.uid' "$work/$model-pods.json")
  printf '%s\n' "$live_job_uid" >"$work/$model-job-uid"
  printf '%s\n' "$live_pod_uid" >"$work/$model-pod-uid"
done

observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required for fixed-path evidence checks}
kubectl -n "$namespace" get pod allie-dev -o json >"$work/observer.json"
jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and ([.status.containerStatuses[].restartCount]|add)==0' \
  "$work/observer.json" >/dev/null
kubectl -n "$namespace" exec allie-dev -- sh -ceu \
  'test ! -e "$1" && test ! -L "$1"' _ \
  /shared/jobs/chris-cyber-hosted-c4-qualification-v1

for pair in "qwen3.8-27b:$q_job:$q_execution" "glm-5.3:$g_job:$g_execution"; do
  IFS=: read -r model source_job execution <<<"$pair"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/jobs/$source_job/CANARY-TERMINAL.json" >"$work/$model-terminal.json"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$execution.json" \
    >"$work/$model-claim.json"
  (
    cd "$snapshot"
    uv run --no-project --with pyyaml --with httpx==0.28.1 python \
      -m evals.fleet.hosted_concurrency4_qualification_release_v1 \
      validate-generation5-evidence --repo "$snapshot" --model "$model" \
      --terminal "$work/$model-terminal.json" --claim "$work/$model-claim.json" \
      --live-job-uid "$(cat "$work/$model-job-uid")" \
      --live-pod-uid "$(cat "$work/$model-pod-uid")"
  )
done

for active_job in \
  chris-q38-ac-exact100-bulk-a199-v1 \
  chris-q38-ac-exact100-bulk-b200-v1 \
  chris-glm53-ac-exact100-bulk-a199-v1 \
  chris-glm53-ac-exact100-bulk-b200-v1; do
  state=$(kubectl -n "$namespace" get job "$active_job" --ignore-not-found -o json)
  if [[ -n "$state" ]]; then
    jq -e '(.status.active // 0)==0' <<<"$state" >/dev/null
  fi
done

api_key=$(kubectl -n "$namespace" get secret "$secret" -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
(
  cd "$snapshot"
  FLEET_API_KEY="$api_key" uv run --no-project --with pyyaml --with httpx==0.28.1 \
    python - <<'PY'
from evals.fleet.hosted_concurrency4_qualification_v1 import validate_identity
import os

validate_identity(os.environ["FLEET_API_KEY"])
PY
)
unset api_key

# Recheck the complete create-once boundary immediately before the only create.
for name in "$bootstrap" "$held_intent" "$release_intent"; do
  test -z "$(kubectl -n "$namespace" get configmap "$name" --ignore-not-found -o name)"
done
test -z "$(kubectl -n "$namespace" get job "$job" --ignore-not-found -o name)"
test -z "$(kubectl -n "$namespace" get pod -l job-name="$job" -o name)"
test "$bundle_sha" = "$(sha256sum "$work/bundle.yaml" | awk '{print $1}')"
kubectl -n "$namespace" create -f "$work/bundle.yaml" -o name
