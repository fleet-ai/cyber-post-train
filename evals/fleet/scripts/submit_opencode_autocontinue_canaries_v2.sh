#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
SECRET=chris-cyber-opencode-evals-v2
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
PRE_MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
HOSTED_RELEASE="$ROOT/evals/fleet/autocontinue_canary_hosted_release.py"
HOSTED_RUNTIME="$ROOT/evals/fleet/autocontinue_canary_hosted_runtime.py"
HOSTED_HEALTH="$ROOT/evals/fleet/autocontinue_hosted_health.py"
MANIFEST_AUTH="$ROOT/evals/fleet/autocontinue_successor_manifest_authorization.py"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
SCORED_V1_INCIDENT="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-successor-hosted-scoring-release-v4.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-successor-hosted-scoring-release-v4.json"
Q_PREAUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v3.json"
G_PREAUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v3.json"
Q_PREFLIGHT="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v3-pass.json"
G_PREFLIGHT="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v3-pass.json"
Q_POST="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v3-post-exit.json"
G_POST="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v3-post-exit.json"
Q_DUP="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-duplicate-inventory-v3.json"
G_DUP="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-duplicate-inventory-v3.json"
Q_CM=chris-q38-ac-canary1-run-v3
G_CM=chris-glm53-ac-canary1-run-v3
Q_JOB=chris-q38-ac-canary1-v2
G_JOB=chris-glm53-ac-canary1-v2
INTENT=chris-ac-canary1-hosted-scored-submit-v2
SFS_OBSERVER=allie-dev
SFS_OBSERVER_UID=73dabe56-60f8-4879-be9f-365196c502e3

validate_phase_c() {
  uv run python - "$ROOT" "$1" <<'PY'
import sys
from pathlib import Path
from evals.fleet import autocontinue_canary_controller as controller
from evals.fleet import autocontinue_canary_hosted_release as release

root = Path(sys.argv[1])
plan = controller.load_object(Path(sys.argv[2]))
release.validate_preflight_bundle(plan, root)
PY
}

validate_phase_c "$Q_PLAN"
validate_phase_c "$G_PLAN"
uv run python -m evals.fleet.autocontinue_successor_manifest_authorization \
  --manifest "$MANIFEST" \
  --expected-job "$Q_JOB" \
  --expected-job "$G_JOB" >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null

assert_k8s_absent() {
  local name
  for name in "$Q_CM" "$G_CM" "$INTENT"; do
    test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
  done
  for name in "$Q_JOB" "$G_JOB"; do
    test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
    test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
  done
}

assert_k8s_absent
if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"mode":"preview","hosted_only":true,"scored_launch_authorized":false,"bulk_release_authorized":false,"objects_created":false}'
  exit 0
fi

test -f "$Q_RELEASE" && test -f "$G_RELEASE"
Q_PACKAGE=$(jq -er '.implementation.package_commit' "$Q_RELEASE")
G_PACKAGE=$(jq -er '.implementation.package_commit' "$G_RELEASE")
test "$Q_PACKAGE" = "$G_PACKAGE"
PACKAGE_COMMIT=$Q_PACKAGE
git cat-file -e "$PACKAGE_COMMIT^{commit}"

for pair in "$Q_PLAN:$Q_RELEASE" "$G_PLAN:$G_RELEASE"; do
  plan=${pair%%:*}
  release=${pair#*:}
  uv run python -m evals.fleet.autocontinue_canary_hosted_runtime validate-release \
    --plan "$plan" --release "$release" --repo "$ROOT" \
    --package-commit "$PACKAGE_COMMIT" >/dev/null
done

for path in \
  evals/fleet/autocontinue_canary_controller.py \
  evals/fleet/autocontinue_canary_hosted_release.py \
  evals/fleet/autocontinue_canary_hosted_runtime.py \
  evals/fleet/autocontinue_successor_manifest_authorization.py \
  evals/fleet/autocontinue_hosted_health.py \
  evals/fleet/hosted_sweep_controller.py \
  evals/fleet/self_hosted.py \
  evals/fleet/opencode_train_sweep_runner.py \
  evals/fleet/endpoint_lease.py \
  evals/fleet/fixed_proxy.py \
  evals/fleet/Dockerfile.opencode \
  evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json \
  evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json \
  evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml \
  evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml \
  evals/fleet/scripts/run_opencode_autocontinue_canary.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json \
  docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v3.json \
  docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v3.json \
  docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v3-pass.json \
  docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v3-pass.json \
  docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v3-post-exit.json \
  docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v3-post-exit.json \
  docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-duplicate-inventory-v3.json \
  docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-duplicate-inventory-v3.json; do
  test "$(git show "$PACKAGE_COMMIT:$path" | sha256sum | awk '{print $1}')" = \
    "$(sha256sum "$ROOT/$path" | awk '{print $1}')"
done

route_dir=$(mktemp -d)
trap 'rm -rf "$route_dir"' EXIT
route_file="$route_dir/launch-route.json"
api_key=$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python -m evals.fleet.autocontinue_canary_hosted_runtime \
  observe-route --out "$route_file" \
  --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
uv run python -m evals.fleet.autocontinue_canary_hosted_runtime \
  validate-route --receipt "$route_file" --maximum-age-seconds 120 \
  --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null

FLEET_API_KEY="$api_key" uv run python - "$Q_PLAN" "$G_PLAN" <<'PY'
import os
import tempfile
from pathlib import Path
from evals.fleet import autocontinue_canary_controller as controller

for path in map(Path, __import__("sys").argv[1:]):
    plan = controller.load_object(path)
    with tempfile.TemporaryDirectory() as directory:
        empty_root = Path(directory) / "output-root"
        controller.hosted._validate_inventory_for_task(
            plan, empty_root, plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
PY
unset api_key

kubectl -n "$NS" get pod "$SFS_OBSERVER" -o json | jq -e \
  --arg uid "$SFS_OBSERVER_UID" \
  '.metadata.uid == $uid and .status.phase == "Running" and
   ([.status.conditions[] | select(.type == "Ready" and .status == "True")] | length) == 1 and
   ([.status.containerStatuses[].restartCount] | add) == 0' >/dev/null
claim_paths=$(uv run python - "$Q_PLAN" "$G_PLAN" <<'PY'
import sys
from pathlib import Path
from evals.fleet import autocontinue_canary_controller as controller
from evals.fleet import self_hosted

for path in map(Path, sys.argv[1:]):
    plan = controller.load_object(path)
    identity = {
        "context_management": controller.CONTEXT,
        "served_id": plan["model"]["served_id"],
        "session_model": plan["model"]["session_model"],
        "task_version_id": plan["tasks"][0]["task"]["version_id"],
        "attempt": int(plan["attempts"][0]["attempt"]),
    }
    digest = self_hosted.sha256(self_hosted.canonical_json(identity)).removeprefix("sha256:")
    print(f"/shared/cell-claims/opencode11827-autocontinue-primary-v1/{digest}.json")
PY
)
claim_1=$(printf '%s\n' "$claim_paths" | sed -n '1p')
claim_2=$(printf '%s\n' "$claim_paths" | sed -n '2p')
test -n "$claim_1" && test -n "$claim_2"
test "$(printf '%s\n' "$claim_paths" | wc -l | tr -d ' ')" = 2
kubectl -n "$NS" exec "$SFS_OBSERVER" -- sh -ceu -- '
  test -d /shared/jobs
  test ! -L /shared/jobs
  test -z "$(find /shared/jobs -type l -print -quit)"
  test ! -e /shared/jobs/chris-q38-ac-canary1-v2
  test ! -L /shared/jobs/chris-q38-ac-canary1-v2
  test ! -e /shared/jobs/chris-glm53-ac-canary1-v2
  test ! -L /shared/jobs/chris-glm53-ac-canary1-v2
  test ! -e "$1"
  test ! -L "$1"
  test ! -e "$2"
  test ! -L "$2"
  test -z "$(find /shared/jobs -path "*/claims/chris-q38-ac-canary1-v2-sr004-a1-02dd4e3f.json" -print -quit)"
  test -z "$(find /shared/jobs -path "*/attempts/chris-q38-ac-canary1-v2-sr004-a1-02dd4e3f" -print -quit)"
  test -z "$(find /shared/jobs -path "*/claims/chris-glm53-ac-canary1-v2-sr013-a1-9375a9b9.json" -print -quit)"
  test -z "$(find /shared/jobs -path "*/attempts/chris-glm53-ac-canary1-v2-sr013-a1-9375a9b9" -print -quit)"
' _ "$claim_1" "$claim_2" >/dev/null
assert_k8s_absent

Q_RELEASE_SHA=$(jq -er '.receipt_sha256' "$Q_RELEASE")
G_RELEASE_SHA=$(jq -er '.receipt_sha256' "$G_RELEASE")
ROUTE_SHA=$(jq -er '.receipt_sha256' "$route_file")
kubectl -n "$NS" create configmap "$INTENT" \
  --from-literal=package_commit="$PACKAGE_COMMIT" \
  --from-literal=qwen_release_sha256="$Q_RELEASE_SHA" \
  --from-literal=glm_release_sha256="$G_RELEASE_SHA" \
  --from-literal=launch_route_sha256="$ROUTE_SHA" \
  --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null

create_scored_cm() {
  local cm=$1 plan=$2 release=$3 preauth=$4 preflight=$5 post_exit=$6 duplicate=$7
  kubectl -n "$NS" create configmap "$cm" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-file=plan.json="$plan" \
    --from-file=release.json="$release" \
    --from-file=launch-route.json="$route_file" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=hosted_release.py="$HOSTED_RELEASE" \
    --from-file=hosted_runtime.py="$HOSTED_RUNTIME" \
    --from-file=hosted_health.py="$HOSTED_HEALTH" \
    --from-file=manifest_authorization.py="$MANIFEST_AUTH" \
    --from-file=endpoint_lease.py="$ROOT/evals/fleet/endpoint_lease.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_canary.sh" \
    --from-file=submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh" \
    --from-file=campaign.json="$CAMPAIGN" \
    --from-file=preflight-manifest.yaml="$PRE_MANIFEST" \
    --from-file=scored-manifest.yaml="$MANIFEST" \
    --from-file=compatibility.json="$COMPATIBILITY" \
    --from-file=scored-v1-incident.json="$SCORED_V1_INCIDENT" \
    --from-file=preauth.json="$preauth" \
    --from-file=preflight.json="$preflight" \
    --from-file=post-exit.json="$post_exit" \
    --from-file=duplicate.json="$duplicate" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null
}

create_scored_cm "$Q_CM" "$Q_PLAN" "$Q_RELEASE" "$Q_PREAUTH" "$Q_PREFLIGHT" "$Q_POST" "$Q_DUP"
create_scored_cm "$G_CM" "$G_PLAN" "$G_RELEASE" "$G_PREAUTH" "$G_PREFLIGHT" "$G_POST" "$G_DUP"
uv run python -m evals.fleet.autocontinue_canary_hosted_runtime \
  validate-route --receipt "$route_file" --maximum-age-seconds 120 \
  --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
kubectl -n "$NS" create -f "$MANIFEST" -o name
