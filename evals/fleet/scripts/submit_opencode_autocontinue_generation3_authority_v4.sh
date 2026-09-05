#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview || "$MODE" == submit ]] || { printf '%s\n' 'usage: $0 preview|submit' >&2; exit 2; }
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation3-root-authorization-v1.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation3-authority-held-v4.yaml"
Q_SPEC="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation3-v1.json"
G_SPEC="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation3-v1.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-generation3-scoring-release-v1.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-generation3-scoring-release-v1.json"
INTENT=chris-ac-g3-canary-submit-v1
Q_JOB=chris-q38-ac-r004-a1-g3-v1
G_JOB=chris-glm53-ac-r013-a1-g3-v1
Q_CM=chris-q38-ac-r004-a1-g3-run-v1
G_CM=chris-glm53-ac-r013-a1-g3-run-v1

uv run python -m evals.fleet.autocontinue_generation3_authority_v1 \
  validate-authority --authority "$AUTH" --repo "$ROOT"
uv run python -m evals.fleet.autocontinue_generation3_authority_package_v1 \
  preview --repo "$ROOT" >/dev/null

work=$(mktemp -d)
trap 'printf "%s\n" "partial create is preserved for manual reconciliation" >&2; rm -rf "$work"' EXIT
bundle="$work/bundle.yaml"
route="$work/route.json"

for name in "$INTENT" chris-ac-g3-runtime-core-a-v4 chris-ac-g3-runtime-core-b-v4 "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done

if [[ "$MODE" == preview ]]; then
  uv run python - "$ROOT" "$MANIFEST" "$bundle" <<'PY'
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation3_authority_package_v1 as package

root, manifest_path, output_path = map(Path, sys.argv[1:])
built = package.build_package(root)
manifest = yaml.safe_load(manifest_path.read_text())
Path(output_path).write_text(yaml.safe_dump({
    "apiVersion": "v1", "kind": "List",
    "items": list(built["configmaps"].values()) + manifest["items"],
}, sort_keys=False))
PY
  kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
  uv run python -m evals.fleet.autocontinue_generation3_authority_package_v1 preview --repo "$ROOT"
  trap - EXIT
  rm -rf "$work"
  exit 0
fi

test -f "$Q_RELEASE" -a -f "$G_RELEASE"
PACKAGE_COMMIT=$(jq -er .package_commit "$Q_RELEASE")
test "$PACKAGE_COMMIT" = "$(jq -er .package_commit "$G_RELEASE")"
git cat-file -e "$PACKAGE_COMMIT^{commit}"
for pair in "$Q_SPEC:$Q_RELEASE" "$G_SPEC:$G_RELEASE"; do
  uv run python -m evals.fleet.autocontinue_generation3_authority_v1 \
    validate-release --spec "${pair%%:*}" --release "${pair#*:}" \
    --authority "$AUTH" --repo "$ROOT" --package-commit "$PACKAGE_COMMIT"
done

uv run python - "$ROOT" "$PACKAGE_COMMIT" <<'PY'
import subprocess
import sys
from pathlib import Path
from evals.fleet import autocontinue_generation3_authority_package_v1 as package

root = Path(sys.argv[1])
commit = sys.argv[2]
built = package.build_package(root)
for obj in built["object_manifests"].values():
    for entry in obj["entries"]:
        expected = subprocess.run(
            ["git", "show", f"{commit}:{entry['source_path']}"],
            cwd=root, check=True, stdout=subprocess.PIPE,
        ).stdout
        if expected != (root / entry["source_path"]).read_bytes():
            raise SystemExit("package source differs from release commit")
PY

api_key=$(kubectl -n "$NS" get secret chris-cyber-opencode-evals-v2 -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python -m evals.fleet.autocontinue_canary_hosted_runtime observe-route \
  --out "$route" --expected-job "$Q_JOB" --expected-job "$G_JOB" >/dev/null
FLEET_API_KEY="$api_key" uv run python - "$Q_SPEC" "$G_SPEC" <<'PY'
import os
import sys
import tempfile
from pathlib import Path
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import hosted_sweep_controller as hosted

for path in map(Path, sys.argv[1:]):
    plan = generation3.validate_spec(generation3.load(path), Path.cwd())
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
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/db5c16dfac428436d4b70dd68ac65b5d1f835937cd4a620839ec9c9a0f4bac5f.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/f4d798d1419e7e470447975162948ef4c30013cc2e684eb7b7de1a2fa7822a74.json \
  /shared/cell-execution-claims/opencode11827-autocontinue-v1/163b867bd0ae54c706a22f8f7ba007d9e5a1d858418b16994c118f68fa52c07d.json >/dev/null
for name in ft-run-98c32208 ft-run-9e92209d; do
  test -z "$(kubectl -n "$NS" get rayjob "$name" --ignore-not-found -o name)"
done
for name in ft-run-98c32208-5gpb2-head-svc ft-run-9e92209d-pppzg-head-svc; do
  test -z "$(kubectl -n "$NS" get service "$name" --ignore-not-found -o name)"
done

uv run python - "$ROOT" "$MANIFEST" "$Q_RELEASE" "$G_RELEASE" "$PACKAGE_COMMIT" "$route" "$bundle" <<'PY'
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation3_authority_package_v1 as package
from evals.fleet import autocontinue_generation3_authority_v1 as authority

root, manifest_path, q_path, g_path, commit, route_path, output_path = sys.argv[1:]
root = Path(root)
releases = {
    "qwen3.8-27b": (authority.load(Path(q_path)), authority.file_sha256(Path(q_path))),
    "glm-5.3": (authority.load(Path(g_path)), authority.file_sha256(Path(g_path))),
}
configmaps = package.released_configmaps(
    root, releases, commit, authority.load(Path(route_path))
)
manifest = yaml.safe_load(Path(manifest_path).read_text())
for item in manifest["items"]:
    item["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/package-layout": "authority-split-configmap-v4",
    }
Path(output_path).write_text(yaml.safe_dump({
    "apiVersion": "v1", "kind": "List",
    "items": list(configmaps.values()) + manifest["items"],
}, sort_keys=False))
PY
kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null

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
kubectl -n "$NS" create -f "$bundle" -o name
trap - EXIT
rm -rf "$work"
