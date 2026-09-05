#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview || "$MODE" == submit ]] || exit 2
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v5.yaml"
AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-root-authorization-v1.json"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v2.json"
Q_SPEC="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json"
G_SPEC="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-generation2-scoring-release-v5.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-generation2-scoring-release-v5.json"
Q_JOB=chris-q38-ac-r004-a1-g2-v1
G_JOB=chris-glm53-ac-r013-a1-g2-v1
Q_CM=chris-q38-ac-r004-a1-g2-run-v1
G_CM=chris-glm53-ac-r013-a1-g2-run-v1
INTENT=chris-ac-g2-canary-submit-v1

uv run python -m evals.fleet.autocontinue_generation2_authority_v5 preview \
  --authority "$AUTH" --held "$HELD" --repo "$ROOT" >/dev/null
uv run python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
expected = {
    "chris-q38-ac-r004-a1-g2-v1": "chris-q38-ac-r004-a1-g2-run-v1",
    "chris-glm53-ac-r013-a1-g2-v1": "chris-glm53-ac-r013-a1-g2-run-v1",
}
if set(manifest) != {"apiVersion", "kind", "items"} or manifest["kind"] != "List":
    raise SystemExit("v5 manifest envelope drifted")
if len(manifest["items"]) != 2:
    raise SystemExit("v5 manifest item count drifted")
for item in manifest["items"]:
    name = item["metadata"]["name"]
    pod = item["spec"]["template"]["spec"]
    if (
        name not in expected
        or item["metadata"]["annotations"]
        != {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        or item["spec"]["backoffLimit"] != 0
        or pod["priorityClassName"] != "fleet-train-high"
        or pod["preemptionPolicy"] != "Never"
        or pod["restartPolicy"] != "Never"
        or pod["volumes"][0]["configMap"]["name"] != expected[name]
    ):
        raise SystemExit("v5 manifest policy drifted")
PY
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null
for name in "$INTENT" "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get cm "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done
if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"status":"HELD","authority_bound":true,"launch_authorized":false,"objects_created":false}'
  exit 0
fi

# The committed source manifest remains held forever. A separately reviewed,
# valid pair of model releases is the sole authority to render its in-memory
# launch annotations after every package/hash gate succeeds.
uv run python - "$MANIFEST" <<'PY'
import sys
import yaml

for item in yaml.safe_load(open(sys.argv[1]))["items"]:
    if item["metadata"]["annotations"] != {
        "cyber-post-train.fleet.ai/preview-only": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
    }:
        raise SystemExit("v5 immutable source manifest is not held")
PY
test -f "$Q_RELEASE" && test -f "$G_RELEASE"
Q_PACKAGE=$(jq -er .package_commit "$Q_RELEASE")
G_PACKAGE=$(jq -er .package_commit "$G_RELEASE")
test "$Q_PACKAGE" = "$G_PACKAGE"
PACKAGE_COMMIT=$Q_PACKAGE
git cat-file -e "$PACKAGE_COMMIT^{commit}"

uv run python -m evals.fleet.autocontinue_generation2_authority_v4 validate-authority \
  --spec "$Q_SPEC" --spec "$G_SPEC" --authority "$AUTH" --repo "$ROOT" >/dev/null
for pair in "$Q_SPEC:$Q_RELEASE" "$G_SPEC:$G_RELEASE"; do
  uv run python -m evals.fleet.autocontinue_generation2_authority_v5 validate-release \
    --spec "${pair%%:*}" --release "${pair#*:}" --authority "$AUTH" \
    --held "$HELD" --repo "$ROOT" --package-commit "$PACKAGE_COMMIT" >/dev/null
done

paths=(
 evals/fleet/autocontinue_generation2_canary.py
 evals/fleet/autocontinue_generation2_canary_v2.py
 evals/fleet/autocontinue_generation2_canary_v3.py
 evals/fleet/autocontinue_generation2_runtime_v3.py
 evals/fleet/autocontinue_generation2_authority_v4.py
 evals/fleet/autocontinue_generation2_authority_v5.py
 evals/fleet/exact_pass4_universe.py
 evals/fleet/autocontinue_canary_controller.py
 evals/fleet/autocontinue_canary_hosted_release.py
 evals/fleet/autocontinue_canary_hosted_runtime.py
 evals/fleet/autocontinue_hosted_health.py
 evals/fleet/hosted_sweep_controller.py
 evals/fleet/self_hosted.py
 evals/fleet/opencode_train_sweep_runner.py
 evals/fleet/endpoint_lease.py
 evals/fleet/fixed_proxy.py
 evals/fleet/Dockerfile.opencode
 evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json
 evals/fleet/configs/opencode-easiest-train100-selection-v2.json
 evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json
 evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json
 evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json
 evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json
 evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json
 evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json
 evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml
 evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh
 evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh
 evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml
 evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh
 evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh
 evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml
 evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh
 evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh
 evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v4.yaml
 evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v4.sh
 evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v4.sh
 evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v5.yaml
 evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v5.sh
 evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v5.sh
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-root-authorization-v1.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v1.json
 docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v2.json
)
for package_path in "${paths[@]}"; do
  test "$(git show "$PACKAGE_COMMIT:$package_path" | sha256sum | awk '{print $1}')" = \
    "$(sha256sum "$ROOT/$package_path" | awk '{print $1}')"
done

route_dir=$(mktemp -d)
trap 'echo "partial create preserved for reconciliation" >&2; rm -rf "$route_dir"' EXIT
released_manifest="$route_dir/released.yaml"
uv run python - "$MANIFEST" "$released_manifest" <<'PY'
import sys
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
for item in manifest["items"]:
    item["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
    }
Path(sys.argv[2]).write_text(yaml.safe_dump(manifest, sort_keys=False))
PY
kubectl -n "$NS" create --dry-run=server -f "$released_manifest" -o name >/dev/null
route="$route_dir/route.json"
api_key=$(kubectl -n "$NS" get secret chris-cyber-opencode-evals-v2 -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python -m evals.fleet.autocontinue_canary_hosted_runtime observe-route \
  --out "$route" --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
FLEET_API_KEY="$api_key" uv run python - "$Q_SPEC" "$G_SPEC" <<'PY'
import os
import sys
import tempfile
from pathlib import Path
from evals.fleet import autocontinue_generation2_canary_v3 as contract
from evals.fleet import hosted_sweep_controller as hosted

for path in map(Path, sys.argv[1:]):
    plan = contract.validate_spec(contract.load(path), Path.cwd())
    with tempfile.TemporaryDirectory() as directory:
        hosted._validate_inventory_for_task(
            plan, Path(directory) / "absent", plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
PY
unset api_key
observer_uid=${SFS_OBSERVER_UID:?}
kubectl -n "$NS" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and ([.status.containerStatuses[].restartCount]|add)==0' >/dev/null
kubectl -n "$NS" exec allie-dev -- sh -c \
  'for p in "$@"; do test ! -e "$p" && test ! -L "$p" || exit 1; done' _ \
  "/shared/jobs/$Q_JOB" "/shared/jobs/$G_JOB" \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/71236ff990db210eb4989ee9e2b4c7afc4b7153b008178c2ae31d5149c3dc8ce.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/db5c16dfac428436d4b70dd68ac65b5d1f835937cd4a620839ec9c9a0f4bac5f.json >/dev/null

auth_file_sha=$(sha256sum "$AUTH" | awk '{print $1}')
auth_self=$(jq -er .receipt_sha256 "$AUTH")
q_release_file_sha=$(sha256sum "$Q_RELEASE" | awk '{print $1}')
g_release_file_sha=$(sha256sum "$G_RELEASE" | awk '{print $1}')
q_release_self=$(jq -er .receipt_sha256 "$Q_RELEASE")
g_release_self=$(jq -er .receipt_sha256 "$G_RELEASE")
kubectl -n "$NS" create cm "$INTENT" \
  --from-literal=package_commit="$PACKAGE_COMMIT" \
  --from-literal=root_authorization_file_sha256="sha256:$auth_file_sha" \
  --from-literal=root_authorization_receipt_sha256="$auth_self" \
  --from-literal=qwen_release_file_sha256="sha256:$q_release_file_sha" \
  --from-literal=qwen_release_receipt_sha256="$q_release_self" \
  --from-literal=glm_release_file_sha256="sha256:$g_release_file_sha" \
  --from-literal=glm_release_receipt_sha256="$g_release_self" \
  --dry-run=client -o json | jq '.immutable=true' | kubectl create -f - >/dev/null

create_cm() {
  local cm=$1 spec=$2 release=$3
  kubectl -n "$NS" create cm "$cm" --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-file=spec.json="$spec" --from-file=q-spec.json="$Q_SPEC" --from-file=g-spec.json="$G_SPEC" \
    --from-file=q-v1-spec.json="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json" \
    --from-file=g-v1-spec.json="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json" \
    --from-file=q-predecessor.json="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json" \
    --from-file=g-predecessor.json="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json" \
    --from-file=campaign.json="$ROOT/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json" \
    --from-file=selection.json="$ROOT/evals/fleet/configs/opencode-easiest-train100-selection-v2.json" \
    --from-file=incident.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json" \
    --from-file=tombstones.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json" \
    --from-file=held-v1.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json" \
    --from-file=held-v2.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json" \
    --from-file=held-v3.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json" \
    --from-file=executable-held-v3.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json" \
    --from-file=authority.json="$AUTH" \
    --from-file=held-v4.json="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v1.json" \
    --from-file=held-v5.json="$HELD" --from-file=release.json="$release" \
    --from-file=launch-route.json="$route" \
    --from-file=generation2_v1.py="$ROOT/evals/fleet/autocontinue_generation2_canary.py" \
    --from-file=generation2_v2.py="$ROOT/evals/fleet/autocontinue_generation2_canary_v2.py" \
    --from-file=generation2_v3.py="$ROOT/evals/fleet/autocontinue_generation2_canary_v3.py" \
    --from-file=runtime_v3.py="$ROOT/evals/fleet/autocontinue_generation2_runtime_v3.py" \
    --from-file=authority_v4.py="$ROOT/evals/fleet/autocontinue_generation2_authority_v4.py" \
    --from-file=authority_v5.py="$ROOT/evals/fleet/autocontinue_generation2_authority_v5.py" \
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
    --from-file=v1-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml" \
    --from-file=v1-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh" \
    --from-file=v1-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh" \
    --from-file=v2-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml" \
    --from-file=v2-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh" \
    --from-file=v2-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh" \
    --from-file=v3-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml" \
    --from-file=v3-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh" \
    --from-file=v3-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh" \
    --from-file=v4-manifest.yaml="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v4.yaml" \
    --from-file=v4-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v4.sh" \
    --from-file=v4-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v4.sh" \
    --from-file=v5-manifest.yaml="$MANIFEST" \
    --from-file=v5-run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v5.sh" \
    --from-file=v5-submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v5.sh" \
    --from-file=run-v5.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v5.sh" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl create -f - >/dev/null
}
create_cm "$Q_CM" "$Q_SPEC" "$Q_RELEASE"
create_cm "$G_CM" "$G_SPEC" "$G_RELEASE"
kubectl -n "$NS" create -f "$released_manifest" -o name
trap - EXIT
rm -rf "$route_dir"
