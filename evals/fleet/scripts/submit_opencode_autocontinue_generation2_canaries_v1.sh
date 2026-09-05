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
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json"
Q_JOB=chris-q38-ac-r004-a1-g2-v1
G_JOB=chris-glm53-ac-r013-a1-g2-v1
Q_CM=chris-q38-ac-r004-a1-g2-run-v1
G_CM=chris-glm53-ac-r013-a1-g2-run-v1
INTENT=chris-ac-g2-canary-submit-v1
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-generation2-scoring-release-v1.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-generation2-scoring-release-v1.json"

uv run python -m evals.fleet.autocontinue_generation2_canary preview \
  --plan "$Q_PLAN" --plan "$G_PLAN" --release "$HELD" --repo "$ROOT" >/dev/null
uv run python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
docs = list(yaml.safe_load_all(path.read_text()))
expected = {
    "chris-q38-ac-r004-a1-g2-v1": "chris-q38-ac-r004-a1-g2-run-v1",
    "chris-glm53-ac-r013-a1-g2-v1": "chris-glm53-ac-r013-a1-g2-run-v1",
}
if len(docs) != 2 or {doc["metadata"]["name"] for doc in docs} != set(expected):
    raise SystemExit("generation-2 manifest roster drifted")
for doc in docs:
    name = doc["metadata"]["name"]
    annotations = doc["metadata"]["annotations"]
    pod = doc["spec"]["template"]["spec"]
    if (
        set(doc) != {"apiVersion", "kind", "metadata", "spec"}
        or doc["kind"] != "Job"
        or doc["metadata"]["namespace"] != "fleet-train-jobs"
        or annotations != {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        or doc["spec"]["backoffLimit"] != 0
        or pod["restartPolicy"] != "Never"
        or pod["priorityClassName"] != "fleet-train-high"
        or pod["preemptionPolicy"] != "Never"
        or pod["volumes"][0]["configMap"]["name"] != expected[name]
        or len(pod["containers"]) != 1
        or len(pod["initContainers"]) != 1
    ):
        raise SystemExit("generation-2 manifest safety policy drifted")
PY

kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null
for name in "$INTENT" "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l "job-name=$name" -o name)"
done

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"status":"HELD","execution_generation":2,"statistical_cells":2,"launch_authorized":false,"bulk_release_authorized":false,"objects_created":false}'
  exit 0
fi

# A held manifest fails here before a secret read or cluster mutation. A later
# executable commit may flip only the two exact annotation pairs after adding
# append-only model-specific RELEASED receipts.
uv run python - "$MANIFEST" <<'PY'
import sys
import yaml

docs = list(yaml.safe_load_all(open(sys.argv[1])))
for doc in docs:
    if doc["metadata"]["annotations"] != {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
    }:
        raise SystemExit("generation-2 package is HELD; no launch is authorized")
PY
test -f "$Q_RELEASE" && test -f "$G_RELEASE"
Q_PACKAGE=$(jq -er '.package_commit' "$Q_RELEASE")
G_PACKAGE=$(jq -er '.package_commit' "$G_RELEASE")
test "$Q_PACKAGE" = "$G_PACKAGE"
PACKAGE_COMMIT=$Q_PACKAGE
git cat-file -e "$PACKAGE_COMMIT^{commit}"

for pair in "$Q_PLAN:$Q_RELEASE" "$G_PLAN:$G_RELEASE"; do
  plan=${pair%%:*}
  release=${pair#*:}
  uv run python -m evals.fleet.autocontinue_generation2_canary validate-release \
    --plan "$plan" --release "$release" --repo "$ROOT" \
    --package-commit "$PACKAGE_COMMIT" >/dev/null
done

for path in \
  evals/fleet/autocontinue_generation2_canary.py \
  evals/fleet/exact_pass4_universe.py \
  evals/fleet/autocontinue_canary_controller.py \
  evals/fleet/autocontinue_canary_hosted_release.py \
  evals/fleet/autocontinue_canary_hosted_runtime.py \
  evals/fleet/autocontinue_hosted_health.py \
  evals/fleet/hosted_sweep_controller.py \
  evals/fleet/self_hosted.py \
  evals/fleet/opencode_train_sweep_runner.py \
  evals/fleet/endpoint_lease.py \
  evals/fleet/fixed_proxy.py \
  evals/fleet/Dockerfile.opencode \
  evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json \
  evals/fleet/configs/opencode-easiest-train100-selection-v2.json \
  evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json \
  evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json \
  evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml \
  evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json; do
  test "$(git show "$PACKAGE_COMMIT:$path" | sha256sum | awk '{print $1}')" = \
    "$(sha256sum "$ROOT/$path" | awk '{print $1}')"
done

route_dir=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$route_dir"' EXIT
route_file="$route_dir/launch-route.json"
api_key=$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python \
  -m evals.fleet.autocontinue_canary_hosted_runtime observe-route \
  --out "$route_file" --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
FLEET_API_KEY="$api_key" uv run python - "$Q_PLAN" "$G_PLAN" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

from evals.fleet import autocontinue_generation2_canary as generation2
from evals.fleet import hosted_sweep_controller as hosted

for path in map(Path, sys.argv[1:]):
    plan = generation2.validate_plan(generation2.load(path), Path.cwd())
    with tempfile.TemporaryDirectory() as directory:
        hosted._validate_inventory_for_task(
            plan, Path(directory) / "absent", plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
PY
unset api_key

observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required for authoritative shared-SFS checks}
kubectl -n "$NS" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" \
  '.metadata.uid == $uid and .status.phase == "Running" and
   ([.status.conditions[] | select(.type == "Ready" and .status == "True")] | length) == 1 and
   ([.status.containerStatuses[].restartCount] | add) == 0' >/dev/null
kubectl -n "$NS" exec allie-dev -- sh -ceu -- '
  for path in "$@"; do test ! -e "$path" && test ! -L "$path"; done
' _ \
  "/shared/jobs/$Q_JOB" "/shared/jobs/$G_JOB" \
  "/shared/cell-execution-claims/opencode11827-autocontinue-v1/71236ff990db210eb4989ee9e2b4c7afc4b7153b008178c2ae31d5149c3dc8ce.json" \
  "/shared/cell-execution-claims/opencode11827-autocontinue-v1/db5c16dfac428436d4b70dd68ac65b5d1f835937cd4a620839ec9c9a0f4bac5f.json" >/dev/null
assert_old_claim() {
  local path=$1 expected=$2
  test "$(kubectl -n "$NS" exec allie-dev -- sha256sum "$path" | awk '{print $1}')" = "$expected"
}
assert_old_claim /shared/jobs/chris-q38-ac-canary1-v2/claims/chris-q38-ac-canary1-v2-sr004-a1-02dd4e3f.json d0e4eb7b47505cbabcca0c77f0a42aef7d244cd249b0752fc60abfbf1375ed51
assert_old_claim /shared/cell-claims/opencode11827-autocontinue-primary-v1/d9a8b7af84dde45ed8ef0a4ad6b744827608427c5ced6528c04544dd7f8c813d.json 8e3ddf02d6de80f9df1d36234ff8449488585c89ce72fcb2e84079b840e68d13
assert_old_claim /shared/jobs/chris-glm53-ac-canary1-v2/claims/chris-glm53-ac-canary1-v2-sr013-a1-9375a9b9.json 8315b08568cd1fdfbbf1a4bf63d1f3408acd2d7c82ec4d75b8bc034518ada834
assert_old_claim /shared/cell-claims/opencode11827-autocontinue-primary-v1/29b5a7875caa52f4d90544a6ede30dbefd5f723659674816a704cf994153872c.json 3b8e6a6ee66e432ad0f51febd58542a454c1940c6e46fc075b50d1288b8cbdc1

for name in "$INTENT" "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l "job-name=$name" -o name)"
done

Q_RELEASE_SHA=$(jq -er '.receipt_sha256' "$Q_RELEASE")
G_RELEASE_SHA=$(jq -er '.receipt_sha256' "$G_RELEASE")
ROUTE_SHA=$(jq -er '.receipt_sha256' "$route_file")
kubectl -n "$NS" create configmap "$INTENT" \
  --from-literal=package_commit="$PACKAGE_COMMIT" \
  --from-literal=qwen_release_sha256="$Q_RELEASE_SHA" \
  --from-literal=glm_release_sha256="$G_RELEASE_SHA" \
  --from-literal=launch_route_sha256="$ROUTE_SHA" \
  --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null

create_runtime_configmap() {
  local cm=$1 plan=$2 predecessor=$3 release=$4
  kubectl -n "$NS" create configmap "$cm" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-file=plan.json="$plan" \
    --from-file=predecessor-plan.json="$predecessor" \
    --from-file=release.json="$release" \
    --from-file=launch-route.json="$route_file" \
    --from-file=generation2.py="$ROOT/evals/fleet/autocontinue_generation2_canary.py" \
    --from-file=exact_universe.py="$ROOT/evals/fleet/exact_pass4_universe.py" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=canary_controller.py="$ROOT/evals/fleet/autocontinue_canary_controller.py" \
    --from-file=hosted_release.py="$ROOT/evals/fleet/autocontinue_canary_hosted_release.py" \
    --from-file=hosted_runtime.py="$ROOT/evals/fleet/autocontinue_canary_hosted_runtime.py" \
    --from-file=hosted_health.py="$ROOT/evals/fleet/autocontinue_hosted_health.py" \
    --from-file=endpoint_lease.py="$ROOT/evals/fleet/endpoint_lease.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=campaign.json="$ROOT/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json" \
    --from-file=selection.json="$ROOT/evals/fleet/configs/opencode-easiest-train100-selection-v2.json" \
    --from-file=incident.json="$INCIDENT_PATH" \
    --from-file=tombstones.json="$TOMBSTONE_PATH" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null
}

INCIDENT_PATH="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json"
TOMBSTONE_PATH="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json"
create_runtime_configmap "$Q_CM" "$Q_PLAN" \
  "$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json" "$Q_RELEASE"
create_runtime_configmap "$G_CM" "$G_PLAN" \
  "$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json" "$G_RELEASE"
kubectl -n "$NS" create -f "$MANIFEST" -o name
trap - EXIT
rm -rf "$route_dir"
