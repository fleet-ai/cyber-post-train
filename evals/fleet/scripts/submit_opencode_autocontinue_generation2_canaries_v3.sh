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
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml"
Q_SPEC="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json"
G_SPEC="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-generation2-scoring-release-v3.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-generation2-scoring-release-v3.json"
Q_JOB=chris-q38-ac-r004-a1-g2-v1
G_JOB=chris-glm53-ac-r013-a1-g2-v1
Q_CM=chris-q38-ac-r004-a1-g2-run-v1
G_CM=chris-glm53-ac-r013-a1-g2-run-v1
INTENT=chris-ac-g2-canary-submit-v1

uv run python -m evals.fleet.autocontinue_generation2_runtime_v3 preview \
  --spec "$Q_SPEC" --spec "$G_SPEC" --held "$HELD" --repo "$ROOT" >/dev/null
uv run python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
items = manifest.get("items", [])
expected = {
    "chris-q38-ac-r004-a1-g2-v1": "chris-q38-ac-r004-a1-g2-run-v1",
    "chris-glm53-ac-r013-a1-g2-v1": "chris-glm53-ac-r013-a1-g2-run-v1",
}
if set(manifest) != {"apiVersion", "kind", "items"} or manifest.get("kind") != "List":
    raise SystemExit("generation-2 v3 manifest envelope drifted")
if len(items) != 2 or {item["metadata"]["name"] for item in items} != set(expected):
    raise SystemExit("generation-2 v3 manifest roster drifted")
for item in items:
    name = item["metadata"]["name"]
    pod = item["spec"]["template"]["spec"]
    if (
        item["metadata"]["namespace"] != "fleet-train-jobs"
        or item["metadata"]["annotations"]
        != {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        or item["spec"]["backoffLimit"] != 0
        or pod["restartPolicy"] != "Never"
        or pod["priorityClassName"] != "fleet-train-high"
        or pod["preemptionPolicy"] != "Never"
        or pod["volumes"][0]["configMap"]["name"] != expected[name]
    ):
        raise SystemExit("generation-2 v3 manifest safety policy drifted")
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
  printf '%s\n' '{"ok":true,"status":"HELD","execution_generation":2,"statistical_cells":2,"launch_authorized":false,"objects_created":false,"v3_executable_package":true}'
  exit 0
fi

# Held manifests fail before secret reads or mutations. A reviewed append-only
# executable successor may flip only these annotations after adding both exact
# model-specific RELEASED receipts.
uv run python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path

import yaml

for item in yaml.safe_load(Path(sys.argv[1]).read_text())["items"]:
    if item["metadata"]["annotations"] != {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
    }:
        raise SystemExit("generation-2 v3 executable package is HELD")
PY

test -f "$Q_RELEASE" && test -f "$G_RELEASE"
Q_PACKAGE=$(jq -er '.package_commit' "$Q_RELEASE")
G_PACKAGE=$(jq -er '.package_commit' "$G_RELEASE")
test "$Q_PACKAGE" = "$G_PACKAGE"
PACKAGE_COMMIT=$Q_PACKAGE
git cat-file -e "$PACKAGE_COMMIT^{commit}"

validate_release() {
  local spec=$1 release=$2
  local at statement
  at=$(jq -er '.authorization.authorized_at_utc' "$release")
  statement=$(jq -er '.authorization.statement' "$release")
  uv run python -m evals.fleet.autocontinue_generation2_canary_v3 validate-release \
    --spec "$spec" --release "$release" --repo "$ROOT" \
    --package-commit "$PACKAGE_COMMIT" --authorized-at-utc "$at" \
    --authorization-statement "$statement" >/dev/null
}
validate_release "$Q_SPEC" "$Q_RELEASE"
validate_release "$G_SPEC" "$G_RELEASE"

for path in \
  evals/fleet/autocontinue_generation2_canary.py \
  evals/fleet/autocontinue_generation2_canary_v2.py \
  evals/fleet/autocontinue_generation2_canary_v3.py \
  evals/fleet/autocontinue_generation2_runtime_v3.py \
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
  evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json \
  evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml \
  evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml \
  evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml \
  evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh \
  evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh \
  evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json \
  docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json; do
  test "$(git show "$PACKAGE_COMMIT:$path" | sha256sum | awk '{print $1}')" = \
    "$(sha256sum "$ROOT/$path" | awk '{print $1}')"
done

route_dir=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for reconciliation" >&2; rm -rf "$route_dir"' EXIT
route_file="$route_dir/launch-route.json"
api_key=$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python \
  -m evals.fleet.autocontinue_canary_hosted_runtime observe-route \
  --out "$route_file" --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
FLEET_API_KEY="$api_key" uv run python - "$Q_SPEC" "$G_SPEC" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import hosted_sweep_controller as hosted

for path in map(Path, sys.argv[1:]):
    plan = generation2.validate_spec(generation2.load(path), Path.cwd())
    with tempfile.TemporaryDirectory() as directory:
        hosted._validate_inventory_for_task(
            plan, Path(directory) / "absent", plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
PY
unset api_key

observer_uid=${SFS_OBSERVER_UID:?SFS_OBSERVER_UID is required for shared-SFS checks}
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
  local cm=$1 spec=$2 release=$3
  local at statement
  at=$(jq -er '.authorization.authorized_at_utc' "$release")
  statement=$(jq -er '.authorization.statement' "$release")
  kubectl -n "$NS" create configmap "$cm" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-literal=authorized_at_utc="$at" \
    --from-literal=authorization_statement="$statement" \
    --from-file=spec.json="$spec" \
    --from-file=q-spec.json="$Q_SPEC" --from-file=g-spec.json="$G_SPEC" \
    --from-file=q-v1-spec.json="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json" \
    --from-file=g-v1-spec.json="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json" \
    --from-file=q-predecessor.json="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json" \
    --from-file=g-predecessor.json="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json" \
    --from-file=release.json="$release" --from-file=launch-route.json="$route_file" \
    --from-file=generation2_v1.py="$ROOT/evals/fleet/autocontinue_generation2_canary.py" \
    --from-file=generation2_v2.py="$ROOT/evals/fleet/autocontinue_generation2_canary_v2.py" \
    --from-file=generation2_v3.py="$ROOT/evals/fleet/autocontinue_generation2_canary_v3.py" \
    --from-file=runtime_v3.py="$ROOT/evals/fleet/autocontinue_generation2_runtime_v3.py" \
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
    --from-file=incident.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json" \
    --from-file=tombstones.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json" \
    --from-file=held-v1.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json" \
    --from-file=held-v2.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json" \
    --from-file=held-v3.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json" \
    --from-file=executable-held.json="$HELD" \
    --from-file=v1-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml" \
    --from-file=v1-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh" \
    --from-file=v1-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh" \
    --from-file=v2-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml" \
    --from-file=v2-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh" \
    --from-file=v2-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh" \
    --from-file=v3-manifest.yaml="$MANIFEST" \
    --from-file=v3-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh" \
    --from-file=v3-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null
}

create_runtime_configmap "$Q_CM" "$Q_SPEC" "$Q_RELEASE"
create_runtime_configmap "$G_CM" "$G_SPEC" "$G_RELEASE"
kubectl -n "$NS" create -f "$MANIFEST" -o name
trap - EXIT
rm -rf "$route_dir"
