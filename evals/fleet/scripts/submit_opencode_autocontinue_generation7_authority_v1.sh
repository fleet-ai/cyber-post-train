#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview || "$MODE" == submit ]] || {
  printf '%s\n' 'usage: submit_opencode_autocontinue_generation7_authority_v1.sh preview|submit' >&2
  exit 2
}
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation7-root-authorization-v1.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation7-authority-held-v1.yaml"
Q_SPEC="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation7-v1.json"
G_SPEC="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation7-v1.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-05-qwen38-autocontinue-generation7-scoring-release-v1.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-05-glm53-autocontinue-generation7-scoring-release-v1.json"
INTENT=chris-ac-g7-canary-submit-v1
Q_JOB=chris-q38-ac-r004-a1-g7-v1
G_JOB=chris-glm53-ac-r013-a1-g7-v1
Q_CM=chris-q38-ac-r004-a1-g7-run-v1
G_CM=chris-glm53-ac-r013-a1-g7-run-v1

work=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$work"' EXIT
bundle="$work/bundle.yaml"
jobs="$work/jobs.yaml"
route="$work/route.json"

for name in "$INTENT" chris-ac-g7-runtime-core-a-v1 chris-ac-g7-runtime-core-b-v1 "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done

if [[ "$MODE" == preview ]]; then
  uv run python -m evals.fleet.autocontinue_generation7_authority_v1 \
    validate-authority --authority "$AUTH" --repo "$ROOT"
  uv run python -m evals.fleet.autocontinue_generation7_authority_package_v1 \
    preview --repo "$ROOT" >/dev/null
  uv run python - "$ROOT" "$MANIFEST" "$bundle" <<'PY'
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation7_authority_package_v1 as package
root, manifest_path, output_path = map(Path, sys.argv[1:])
built = package.build_package(root)
manifest = yaml.safe_load(manifest_path.read_text())
output_path.write_text(yaml.safe_dump({
    "apiVersion": "v1", "kind": "List",
    "items": list(built["configmaps"].values()) + manifest["items"],
}, sort_keys=False))
PY
  kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
  uv run python -m evals.fleet.autocontinue_generation7_authority_package_v1 \
    preview --repo "$ROOT"
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

test -f "$Q_RELEASE" -a -f "$G_RELEASE"
SOURCE_HEAD=$(git -C "$ROOT" rev-parse HEAD)
uv run python -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$ROOT" --expected-head "$SOURCE_HEAD"
PACKAGE_COMMIT=$(jq -er .package_commit "$Q_RELEASE")
test "$PACKAGE_COMMIT" = "$(jq -er .package_commit "$G_RELEASE")"
git cat-file -e "$PACKAGE_COMMIT^{commit}"
git merge-base --is-ancestor "$PACKAGE_COMMIT" "$SOURCE_HEAD"
Q_RELEASE_REL=${Q_RELEASE#"$ROOT/"}
G_RELEASE_REL=${G_RELEASE#"$ROOT/"}
for release_path in "$Q_RELEASE_REL" "$G_RELEASE_REL"; do
  ! git -C "$ROOT" cat-file -e "$PACKAGE_COMMIT:$release_path" 2>/dev/null
done
SNAPSHOT="$work/package"
uv run python -m evals.fleet.immutable_submission_snapshot materialize \
  --repo "$ROOT" --commit "$PACKAGE_COMMIT" --destination "$SNAPSHOT"
install -m 0644 "$Q_RELEASE" "$SNAPSHOT/$Q_RELEASE_REL"
install -m 0644 "$G_RELEASE" "$SNAPSHOT/$G_RELEASE_REL"
AUTH="$SNAPSHOT/${AUTH#"$ROOT/"}"
MANIFEST="$SNAPSHOT/${MANIFEST#"$ROOT/"}"
Q_SPEC="$SNAPSHOT/${Q_SPEC#"$ROOT/"}"
G_SPEC="$SNAPSHOT/${G_SPEC#"$ROOT/"}"
Q_RELEASE="$SNAPSHOT/$Q_RELEASE_REL"
G_RELEASE="$SNAPSHOT/$G_RELEASE_REL"
for pair in "$Q_SPEC:$Q_RELEASE" "$G_SPEC:$G_RELEASE"; do
  (cd "$SNAPSHOT" && uv run --no-project --with httpx==0.28.1 python \
    -m evals.fleet.autocontinue_generation7_authority_v1 validate-release \
    --spec "${pair%%:*}" --release "${pair#*:}" --authority "$AUTH" \
    --repo "$SNAPSHOT" --package-commit "$PACKAGE_COMMIT")
done

api_key=$(kubectl -n "$NS" get secret chris-cyber-opencode-evals-v2 \
  -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
(cd "$SNAPSHOT" && FLEET_API_KEY="$api_key" \
  uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_canary_hosted_runtime observe-route \
  --out "$route" --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null)
(cd "$SNAPSHOT" && FLEET_API_KEY="$api_key" uv run --no-project --with httpx==0.28.1 \
  python - "$Q_SPEC" "$G_SPEC") <<'PY'
import os
import sys
import tempfile
from pathlib import Path
from evals.fleet import autocontinue_generation6_canary as generation6
from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import hosted_sweep_controller as hosted
for path in map(Path, sys.argv[1:]):
    spec = generation7.load(path)
    g7_plan = generation7.validate_spec(spec, Path.cwd())
    g6_spec = generation6.load(Path(generation6.G6_SPEC_PATHS[spec["model"]]))
    g6_plan = generation6.validate_spec(g6_spec, Path.cwd())
    with tempfile.TemporaryDirectory() as directory:
        absent = Path(directory) / "absent"
        hosted._validate_inventory_for_task(
            g6_plan, absent, g6_plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
        hosted._validate_inventory_for_task(
            g7_plan, absent, g7_plan["tasks"][0], os.environ["FLEET_API_KEY"]
        )
PY
unset api_key

for row in \
  'chris-q38-ac-r004-a1-g6-v1 674ac8ec-0d20-49e0-a260-b77d96e9cd81 2a2e94ed-c86d-4159-80d5-45ef94f89bf8' \
  'chris-glm53-ac-r013-a1-g6-v1 859378a9-b2ce-4edc-9b25-18d701201e5f ea288ab8-aea8-45b4-84b8-7d011a2a5e8f'; do
  set -- $row
  kubectl -n "$NS" get job "$1" -o json | jq -e --arg uid "$2" \
    '.metadata.uid==$uid and .status.failed==1 and .status.succeeded==null and
     ([.status.conditions[]|select(.type=="Failed" and .status=="True" and .reason=="BackoffLimitExceeded")]|length)==1' >/dev/null
  kubectl -n "$NS" get pod -l job-name="$1" -o json | jq -e \
    --arg uid "$3" --arg job_uid "$2" --arg job_name "$1" \
    '(.items|length)==1 and .items[0].metadata.uid==$uid and
     ([.items[0].metadata.ownerReferences[]|select(.controller==true and
       .apiVersion=="batch/v1" and .kind=="Job" and .name==$job_name and .uid==$job_uid)]|length)==1 and
     .items[0].status.phase=="Failed" and
     .items[0].status.reason=="Evicted" and
     .items[0].status.message=="Usage of EmptyDir volume \"docker-cli\" exceeds the limit \"100Mi\". " and
     ([.items[0].status.initContainerStatuses[],.items[0].status.containerStatuses[]]
       |map(.restartCount)|add)==2 and
     ([.items[0].status.initContainerStatuses[]|select(.name=="docker-cli" and
       .restartCount==0 and .state.terminated.exitCode==0 and
       .state.terminated.reason=="Completed")]|length)==1 and
     ([.items[0].status.containerStatuses[]|select(.name=="evaluator" and
       .restartCount==1 and .state.terminated.exitCode==137 and
       .state.terminated.reason=="ContainerStatusUnknown")]|length)==1 and
     ([.items[0].status.initContainerStatuses[]|select(.name=="dind" and
       .restartCount==1 and .state.terminated.exitCode==137 and
       .state.terminated.reason=="ContainerStatusUnknown")]|length)==1' >/dev/null
done

observer_uid=${SFS_OBSERVER_UID:?}
kubectl -n "$NS" get pod allie-dev -o json | jq -e --arg uid "$observer_uid" \
  '.metadata.uid==$uid and .status.phase=="Running" and ([.status.containerStatuses[].restartCount]|add)==0' >/dev/null
kubectl -n "$NS" exec allie-dev -- sh -c \
  'for p in "$@"; do test ! -e "$p" && test ! -L "$p" || exit 1; done' _ \
  /shared/jobs/chris-q38-ac-r004-a1-g6-v1 \
  /shared/jobs/chris-glm53-ac-r013-a1-g6-v1 \
  "/shared/jobs/$Q_JOB" "/shared/jobs/$G_JOB" \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/d839f3c2ade74ff72b5034fc6e988f3acb93863e0beaf5eaaf97bdf92ba1f8f3.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/8d828cab823f0d2c551e77bc80f8d8eff79a1f55f3ef3193def475b0fc80dd70.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/71236ff990db210eb4989ee9e2b4c7afc4b7153b008178c2ae31d5149c3dc8ce.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/db5c16dfac428436d4b70dd68ac65b5d1f835937cd4a620839ec9c9a0f4bac5f.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/f4d798d1419e7e470447975162948ef4c30013cc2e684eb7b7de1a2fa7822a74.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/163b867bd0ae54c706a22f8f7ba007d9e5a1d858418b16994c118f68fa52c07d.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/232a0409f90327711144283182611d0c28e18e388578f3b062fd29f0f83e6fff.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/736c28d3fb9ce42dd7aa2b1a9865f218616171767cd19a56b6028dd006af9d71.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/0e359500ab08e29efc95c8cb3ee6c2bc4d719e31cb8c9aadf53fedda03e1c3f1.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/54008eb3de475a49d737cf8f7a4c344bf1815b4f2e8818a49d6ec236a3d8a83b.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/fa0c9b8529643267ad12a3ac4e817ea62a2bc78b83b9161d0f5b2183344288c1.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/a63dbb284779822f4beef416581c99be91fdfb3406ec77b7095a6334b034d07d.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524.json >/dev/null

(cd "$SNAPSHOT" && uv run --no-project --with pyyaml --with httpx==0.28.1 python - \
  "$SNAPSHOT" "$ROOT" "$MANIFEST" "$Q_RELEASE" "$G_RELEASE" \
  "$PACKAGE_COMMIT" "$route" "$bundle" "$jobs") <<'PY'
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation7_authority_package_v1 as package
from evals.fleet import autocontinue_generation7_authority_v1 as authority
from evals.fleet import immutable_submission_snapshot as immutable
root, source_root, manifest_path, q_path, g_path, commit, route_path, output_path, jobs_path = sys.argv[1:]
root = Path(root)
releases = {
    "qwen3.8-27b": (authority.load(Path(q_path)), authority.file_sha256(Path(q_path))),
    "glm-5.3": (authority.load(Path(g_path)), authority.file_sha256(Path(g_path))),
}
configmaps = package.released_configmaps(root, releases, commit, authority.load(Path(route_path)))
paths = {
    entry["source_path"]
    for obj in package.build_package(root)["object_manifests"].values()
    for entry in obj["entries"]
}
immutable.verify_paths(Path(source_root), commit, root, paths)
manifest = yaml.safe_load(Path(manifest_path).read_text())
for item in manifest["items"]:
    item["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/package-layout": "generation7-authority-split-v1",
    }
Path(jobs_path).write_text(yaml.safe_dump_all(manifest["items"], sort_keys=False))
Path(output_path).write_text(yaml.safe_dump({
    "apiVersion": "v1", "kind": "List",
    "items": list(configmaps.values()) + manifest["items"],
}, sort_keys=False))
PY
(cd "$SNAPSHOT" && uv run --no-project --with pyyaml --with httpx==0.28.1 python \
  -m evals.fleet.scored_manifest_authorization \
  --manifest "$jobs" --expected-job "$Q_JOB" --expected-job "$G_JOB")
bundle_sha=$(sha256sum "$bundle" | awk '{print $1}')
kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
test "$bundle_sha" = "$(sha256sum "$bundle" | awk '{print $1}')"
uv run python -m evals.fleet.immutable_submission_snapshot assert-stable \
  --repo "$ROOT" --expected-head "$SOURCE_HEAD"

kubectl -n "$NS" create cm "$INTENT" \
  --from-literal=package_commit="$PACKAGE_COMMIT" \
  --from-literal=root_authorization_receipt_sha256="$(jq -er .receipt_sha256 "$AUTH")" \
  --from-literal=qwen_release_receipt_sha256="$(jq -er .receipt_sha256 "$Q_RELEASE")" \
  --from-literal=glm_release_receipt_sha256="$(jq -er .receipt_sha256 "$G_RELEASE")" \
  --dry-run=client -o json | jq '.immutable=true' | kubectl create -f - >/dev/null
kubectl -n "$NS" create -f "$bundle" -o name
trap - EXIT
rm -rf "$work"
