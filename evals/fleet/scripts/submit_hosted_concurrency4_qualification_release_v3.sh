#!/usr/bin/env bash
set -euo pipefail

mode=${1:-preview}
[[ "$mode" == preview || "$mode" == submit ]] || {
  printf '%s\n' 'usage: submit_hosted_concurrency4_qualification_release_v3.sh preview|submit' >&2
  exit 2
}

root=$(git rev-parse --show-toplevel)
namespace=fleet-train-jobs
held_rel=docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-g7-gated-held-v3.json
release_rel=docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-launch-release-v3.json
q_release_rel=docs/evidence/qwen38-study/2026-09-05-qwen38-autocontinue-generation7-scoring-release-v1.json
g_release_rel=docs/evidence/qwen38-study/2026-09-05-glm53-autocontinue-generation7-scoring-release-v1.json
held="$root/$held_rel"
release="$root/$release_rel"
q_release="$root/$q_release_rel"
g_release="$root/$g_release_rel"
job=chris-cyber-hosted-c4-qualification-v1
bootstrap=chris-cyber-hosted-c4-qualification-bootstrap-v1
held_intent=chris-cyber-hosted-c4-qualification-intent-v1
release_intents=(
  chris-cyber-hosted-c4-qualification-release-v1
  chris-cyber-hosted-c4-qualification-release-v2
  chris-cyber-hosted-c4-qualification-release-v3
)
secret=chris-cyber-opencode-evals-v2
secret_uid=e0febd8e-94a2-46b0-a0bf-dd6b3154187b
probe_package_commit=cbe8b2c7654cbda1f5392c8466b953e335be2641
g7_package_commit=e549ed9588159af9aaf6186e5b458a9eb070c114
q_job=chris-q38-ac-r004-a1-g7-v1
g_job=chris-glm53-ac-r013-a1-g7-v1
q_execution=77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535
g_execution=b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524
bulk_jobs=(
  chris-q38-ac-exact100-bulk-a199-v1
  chris-q38-ac-exact100-bulk-b200-v1
  chris-glm53-ac-exact100-bulk-a199-v1
  chris-glm53-ac-exact100-bulk-b200-v1
)
bulk_configmaps=(
  chris-q38-ac-exact100-bulk-a199-run-v1
  chris-q38-ac-exact100-bulk-b200-run-v1
  chris-glm53-ac-exact100-bulk-a199-run-v1
  chris-glm53-ac-exact100-bulk-b200-run-v1
)

test -f "$held"
uv run --project "$root" python \
  -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
  validate-held --held "$held"

if [[ ! -f "$release" ]]; then
  if [[ "$mode" == submit ]]; then
    printf '%s\n' 'HELD: append-only G7-gated v3 launch release has not been rendered' >&2
    exit 2
  fi
  uv run --project "$root" python \
    -m evals.fleet.hosted_concurrency4_qualification_package_v1 \
    render-configmap --repo "$root" --package-commit "$probe_package_commit" >/dev/null
  jq -n \
    --arg held_receipt "$(jq -er .receipt_sha256 "$held")" \
    --arg g7_package_commit "$g7_package_commit" \
    '{ok:true,status:"HELD_PREVIEW",held_receipt_sha256:$held_receipt,generation7_package_commit:$g7_package_commit,release_present:false,objects_created:false,chat_completion_requests:0,task_instance_session_scoring_or_verifier_calls:0}'
  exit 0
fi

test -f "$q_release" -a -f "$g_release"
work=$(mktemp -d /private/tmp/hosted-c4-release-v3.XXXXXX)
snapshot="$work/snapshot"
cleanup() {
  git -C "$root" worktree remove --force "$snapshot" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

implementation_commit=$(jq -er '.implementation.commit' "$release")
test "$(jq -er '.probe_package.commit' "$release")" = "$probe_package_commit"
test "$(jq -er '.generation7_package.commit' "$release")" = "$g7_package_commit"
git -C "$root" cat-file -e "$implementation_commit^{commit}"
git -C "$root" merge-base --is-ancestor "$probe_package_commit" "$implementation_commit"
git -C "$root" merge-base --is-ancestor "$g7_package_commit" "$implementation_commit"
if git -C "$root" cat-file -e "$implementation_commit:$release_rel" 2>/dev/null; then
  printf '%s\n' 'release receipt must be append-only after the implementation commit' >&2
  exit 1
fi

git -C "$root" worktree add --detach "$snapshot" "$implementation_commit" >/dev/null
install -m 0600 "$release" "$snapshot/$release_rel"
install -m 0600 "$q_release" "$snapshot/$q_release_rel"
install -m 0600 "$g_release" "$snapshot/$g_release_rel"
(
  cd "$snapshot"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
    validate-release --release "$snapshot/$release_rel" --repo "$snapshot"
  for pair in \
    "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation7-v1.json:$q_release_rel" \
    "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation7-v1.json:$g_release_rel"; do
    uv run --no-project --with httpx==0.28.1 python \
      -m evals.fleet.autocontinue_generation7_authority_v1 validate-release \
      --spec "${pair%%:*}" --release "${pair#*:}" \
      --authority docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation7-root-authorization-v1.json \
      --repo "$snapshot" --package-commit "$g7_package_commit"
  done
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
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
assert intent["data"]["generation7_evidence_required"] == "true"
pod = job["spec"]["template"]["spec"]
assert pod["priorityClassName"] == "fleet-serve-low"
assert pod["preemptionPolicy"] == "Never"
assert pod["restartPolicy"] == "Never"
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
    --arg probe_package_commit "$probe_package_commit" \
    --arg generation7_package_commit "$g7_package_commit" \
    --arg implementation_commit "$implementation_commit" \
    --arg receipt "$(jq -er .receipt_sha256 "$release")" \
    '{ok:true,status:"RELEASED_PREVIEW",probe_package_commit:$probe_package_commit,generation7_package_commit:$generation7_package_commit,implementation_commit:$implementation_commit,release_receipt_sha256:$receipt,objects_created:false,chat_completion_requests:24,task_instance_session_scoring_or_verifier_calls:0}'
  exit 0
fi

for name in "$bootstrap" "$held_intent" "${release_intents[@]}"; do
  test -z "$(kubectl -n "$namespace" get configmap "$name" --ignore-not-found -o name)"
done
test -z "$(kubectl -n "$namespace" get job "$job" --ignore-not-found -o name)"
test -z "$(kubectl -n "$namespace" get pod -l job-name="$job" -o name)"
test "$(kubectl -n "$namespace" get secret "$secret" -o jsonpath='{.metadata.uid}')" = "$secret_uid"

observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required for fixed-path evidence checks}
kubectl -n "$namespace" get pod allie-dev -o json >"$work/observer.json"
jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and ([.status.containerStatuses[].restartCount]|add)==0' \
  "$work/observer.json" >/dev/null
kubectl -n "$namespace" exec allie-dev -- sh -ceu \
  'for path in "$@"; do test ! -e "$path" && test ! -L "$path" || exit 1; done' _ \
  /shared/jobs/chris-cyber-hosted-c4-qualification-v1 \
  /shared/jobs/chris-q38-ac-exact100-bulk-a199-v1 \
  /shared/jobs/chris-q38-ac-exact100-bulk-b200-v1 \
  /shared/jobs/chris-glm53-ac-exact100-bulk-a199-v1 \
  /shared/jobs/chris-glm53-ac-exact100-bulk-b200-v1

for bulk_job in "${bulk_jobs[@]}"; do
  test -z "$(kubectl -n "$namespace" get job "$bulk_job" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$namespace" get pod -l job-name="$bulk_job" -o name)"
done
for bulk_configmap in "${bulk_configmaps[@]}"; do
  test -z "$(kubectl -n "$namespace" get configmap "$bulk_configmap" --ignore-not-found -o name)"
done

for pair in "qwen3.8-27b:$q_job:$q_execution" "glm-5.3:$g_job:$g_execution"; do
  IFS=: read -r model source_job execution <<<"$pair"
  kubectl -n "$namespace" get job "$source_job" -o json >"$work/$model-job.json"
  kubectl -n "$namespace" get pods -l job-name="$source_job" -o json >"$work/$model-pods.json"
  (
    cd "$snapshot"
    uv run --no-project --with pyyaml --with httpx==0.28.1 python \
      -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
      validate-terminal-job --job-json "$work/$model-job.json" \
      --pods-json "$work/$model-pods.json" --job-name "$source_job"
  )
  jq -er '.metadata.uid' "$work/$model-job.json" >"$work/$model-job-uid"
  jq -er '.items[0].metadata.uid' "$work/$model-pods.json" >"$work/$model-pod-uid"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/jobs/$source_job/CANARY-TERMINAL.json" >"$work/$model-terminal.json"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$execution.json" \
    >"$work/$model-claim.json"
  (
    cd "$snapshot"
    uv run --no-project --with pyyaml --with httpx==0.28.1 python \
      -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
      validate-generation7-evidence --repo "$snapshot" --model "$model" \
      --terminal "$work/$model-terminal.json" --claim "$work/$model-claim.json" \
      --live-job-uid "$(cat "$work/$model-job-uid")" \
      --live-pod-uid "$(cat "$work/$model-pod-uid")"
  )
done

(
  cd "$snapshot"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
    validate-live-binding --repo "$snapshot" --release "$snapshot/$release_rel" \
    --evidence-dir "$work"
)

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

# Recheck every create-once and bulk-absence boundary immediately before create.
for name in "$bootstrap" "$held_intent" "${release_intents[@]}"; do
  test -z "$(kubectl -n "$namespace" get configmap "$name" --ignore-not-found -o name)"
done
test -z "$(kubectl -n "$namespace" get job "$job" --ignore-not-found -o name)"
test -z "$(kubectl -n "$namespace" get pod -l job-name="$job" -o name)"
for bulk_job in "${bulk_jobs[@]}"; do
  test -z "$(kubectl -n "$namespace" get job "$bulk_job" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$namespace" get pod -l job-name="$bulk_job" -o name)"
done
for bulk_configmap in "${bulk_configmaps[@]}"; do
  test -z "$(kubectl -n "$namespace" get configmap "$bulk_configmap" --ignore-not-found -o name)"
done
test "$(kubectl -n "$namespace" get secret "$secret" -o jsonpath='{.metadata.uid}')" = "$secret_uid"
kubectl -n "$namespace" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and ([.status.containerStatuses[].restartCount]|add)==0' >/dev/null
kubectl -n "$namespace" exec allie-dev -- sh -ceu \
  'for path in "$@"; do test ! -e "$path" && test ! -L "$path" || exit 1; done' _ \
  /shared/jobs/chris-cyber-hosted-c4-qualification-v1 \
  /shared/jobs/chris-q38-ac-exact100-bulk-a199-v1 \
  /shared/jobs/chris-q38-ac-exact100-bulk-b200-v1 \
  /shared/jobs/chris-glm53-ac-exact100-bulk-a199-v1 \
  /shared/jobs/chris-glm53-ac-exact100-bulk-b200-v1
for pair in "qwen3.8-27b:$q_job:$q_execution" "glm-5.3:$g_job:$g_execution"; do
  IFS=: read -r model source_job execution <<<"$pair"
  kubectl -n "$namespace" get job "$source_job" -o json >"$work/$model-job.json"
  kubectl -n "$namespace" get pods -l job-name="$source_job" -o json >"$work/$model-pods.json"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/jobs/$source_job/CANARY-TERMINAL.json" >"$work/$model-terminal.json"
  kubectl -n "$namespace" exec allie-dev -- sh -ceu 'cat "$1"' _ \
    "/shared/cell-execution-claims/opencode11827-autocontinue-v1/$execution.json" \
    >"$work/$model-claim.json"
done
(
  cd "$snapshot"
  uv run --no-project --with pyyaml --with httpx==0.28.1 python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v3 \
    validate-live-binding --repo "$snapshot" --release "$snapshot/$release_rel" \
    --evidence-dir "$work"
)
test "$bundle_sha" = "$(sha256sum "$work/bundle.yaml" | awk '{print $1}')"
kubectl -n "$namespace" create -f "$work/bundle.yaml" -o name
