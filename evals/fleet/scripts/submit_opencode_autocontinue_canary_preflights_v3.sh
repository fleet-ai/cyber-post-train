#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
PACKAGE_COMMIT=f2c32e4571fddfa43770249f323e1843fc205f95
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
FROZEN="$ROOT/evals/fleet/hosted_sweep_controller.py"
SELF_HOSTED="$ROOT/evals/fleet/self_hosted.py"
RUNNER="$ROOT/evals/fleet/opencode_train_sweep_runner.py"
ENDPOINT_LEASE="$ROOT/evals/fleet/endpoint_lease.py"
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
INCIDENT="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
Q_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v3.json"
G_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v3.json"
Q_CM=chris-q38-ac-canary1-pre-v3
G_CM=chris-glm53-ac-canary1-pre-v3
Q_PRE=chris-q38-ac-canary1-v3-preflight
G_PRE=chris-glm53-ac-canary1-v3-preflight
Q_SCORED_CM=chris-q38-ac-canary1-run-v3
G_SCORED_CM=chris-glm53-ac-canary1-run-v3
Q_SCORED_JOB=chris-q38-ac-canary1-v2
G_SCORED_JOB=chris-glm53-ac-canary1-v2
INTENT_CM=chris-ac-canary1-preflight-submit-v3
AUTHORIZED_AT_UTC=2026-09-04T22:26:36Z
Q_STATEMENT='Authorize exactly one create-once read-only Qwen corrected-canary v3 bootstrap preflight, superseding only the terminal scored-v1 bootstrap failure; it may create the exact preflight ConfigMap, Job, and sanitized SFS evidence, but no model, verifier, scoring, session, scored ConfigMap, or scored Job call is authorized.'
G_STATEMENT='Authorize exactly one create-once read-only GLM corrected-canary v3 bootstrap preflight, superseding only the terminal scored-v1 bootstrap failure; it may create the exact preflight ConfigMap, Job, and sanitized SFS evidence, but no model, verifier, scoring, session, scored ConfigMap, or scored Job call is authorized.'
Q_PLAN_RAW=fa0f5f0f7b762d3050fab6b1c9335c6af4737a56b77c3bc96ea00c77f84d754c
G_PLAN_RAW=7096622e3b8dcaa04b8b8f286822db6564d89866a0df7e06c30ec3998b3d28bf
Q_AUTH_RAW=1059c755a352b5f2320b540094ba7f4e5cc575b7b6ff5af4ca68a601b0f4f80c
G_AUTH_RAW=de0d7aec69f8471c5e3b35797de5f7dda723a70b90c97407c8aeb9c21c31f69e
Q_AUTH_SELF=sha256:2ccba714a6885ff8de4cd13f7a7857df2ccdf75bd8a258be04969e502d1ccf10
G_AUTH_SELF=sha256:f5766496049bfe8e649e95b2ef450263b461c03b803a7b39b0da8bca63b3ea5c

test "$(git rev-parse "$PACKAGE_COMMIT^{commit}")" = "$PACKAGE_COMMIT"

verify_git_object() {
  local file=$1 relative=${1#"$ROOT/"}
  git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"
}

for file in "$CONTROLLER" "$FROZEN" "$SELF_HOSTED" "$RUNNER" "$ENDPOINT_LEASE" \
  "$CAMPAIGN" "$COMPATIBILITY" "$INCIDENT" "$MANIFEST" "$Q_PLAN" "$G_PLAN"; do
  verify_git_object "$file"
done

test "$(sha256sum "$CONTROLLER" | awk '{print $1}')" = 605a695e818f8598274d9d64111868ecc1b092514c7ea725437de0cff7585260
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a
test "$(sha256sum "$SELF_HOSTED" | awk '{print $1}')" = 16df432b5fde55112924d6106e6c03f09817846f1344ba0fe100dcf785c33d8b
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = b1f9c5028f65b0d7772538e3ce075310dc6c7a46b3de58d0196bc474e74e9e9d
test "$(sha256sum "$ENDPOINT_LEASE" | awk '{print $1}')" = 1df60ee13be8c6057113dbebadf9020343649e175b5de38aea41706748987019
test "$(sha256sum "$CAMPAIGN" | awk '{print $1}')" = 1dfcc6d9ddbd19ef5ff46b5d6308d62526db4d1a0ccf05c5fa3b10b176910f34
test "$(sha256sum "$COMPATIBILITY" | awk '{print $1}')" = f1e8ac4119a833e2c076b621f089d0500ff3b7d8e8838be349ba2a7c8ecd7bb3
test "$(sha256sum "$INCIDENT" | awk '{print $1}')" = 0de1186800caeccdefa8c9730bf00ef3208e44ba0e699cede0bbf598d86fa8d7
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = 44262e7dad2329219480f9e52b19064fab3c59d67f9c174fd11e7e058beb4f12
test "$(sha256sum "$Q_PLAN" | awk '{print $1}')" = "$Q_PLAN_RAW"
test "$(sha256sum "$G_PLAN" | awk '{print $1}')" = "$G_PLAN_RAW"
test "$(sha256sum "$Q_AUTH" | awk '{print $1}')" = "$Q_AUTH_RAW"
test "$(sha256sum "$G_AUTH" | awk '{print $1}')" = "$G_AUTH_RAW"

for row in \
  "$Q_PLAN|$Q_AUTH|$Q_PLAN_RAW|$Q_AUTH_RAW|$Q_AUTH_SELF|$Q_STATEMENT" \
  "$G_PLAN|$G_AUTH|$G_PLAN_RAW|$G_AUTH_RAW|$G_AUTH_SELF|$G_STATEMENT"; do
  IFS='|' read -r plan authorization plan_raw expected_raw expected_self statement <<<"$row"
  uv run python - "$plan" "$authorization" "$ROOT" "$PACKAGE_COMMIT" \
    "sha256:$plan_raw" "$AUTHORIZED_AT_UTC" "$statement" "$expected_raw" "$expected_self" <<'PY'
import hashlib
import sys
from pathlib import Path
from evals.fleet.autocontinue_canary_controller import (
    load_object,
    validate_preflight_authorization,
)

plan_path = Path(sys.argv[1])
authorization_path = Path(sys.argv[2])
authorization = load_object(authorization_path)
if hashlib.sha256(authorization_path.read_bytes()).hexdigest() != sys.argv[8]:
    raise SystemExit("preflight authorization immutable file digest drifted")
if authorization.get("receipt_sha256") != sys.argv[9]:
    raise SystemExit("preflight authorization immutable receipt digest drifted")
validate_preflight_authorization(
    authorization,
    load_object(plan_path),
    Path(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    sys.argv[7],
)
PY
done

for name in "$INTENT_CM" "$Q_CM" "$G_CM" "$Q_SCORED_CM" "$G_SCORED_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_PRE" "$G_PRE" "$Q_SCORED_JOB" "$G_SCORED_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done

make_immutable() {
  python3 -c 'import json,sys; value=json.load(sys.stdin); value["immutable"]=True; json.dump(value,sys.stdout,separators=(",",":"))'
}

make_configmap() {
  local name=$1 plan=$2 authorization=$3 plan_raw=$4 statement=$5 mode=$6
  local payload
  payload=$(kubectl -n "$NS" create configmap "$name" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-literal=plan_file_sha256="sha256:$plan_raw" \
    --from-literal=authorized_at_utc="$AUTHORIZED_AT_UTC" \
    --from-literal=authorization_statement="$statement" \
    --from-file=plan.json="$plan" \
    --from-file=release.json="$authorization" \
    --from-file=campaign.json="$CAMPAIGN" \
    --from-file=controller.py="$FROZEN" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=compatibility.json="$COMPATIBILITY" \
    --from-file=scored-v1-incident.json="$INCIDENT" \
    --from-file=endpoint_lease.py="$ENDPOINT_LEASE" \
    --from-file=self_hosted.py="$SELF_HOSTED" \
    --from-file=runner.py="$RUNNER" \
    --from-file=preflight-manifest.yaml="$MANIFEST" \
    --dry-run=client -o json | make_immutable)
  if [[ "$mode" == preview ]]; then
    printf '%s' "$payload" | kubectl -n "$NS" create --dry-run=server -f - -o name
  else
    printf '%s' "$payload" | kubectl -n "$NS" create -f - -o name
  fi
}

make_configmap "$Q_CM" "$Q_PLAN" "$Q_AUTH" "$Q_PLAN_RAW" "$Q_STATEMENT" preview >/dev/null
make_configmap "$G_CM" "$G_PLAN" "$G_AUTH" "$G_PLAN_RAW" "$G_STATEMENT" preview >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null

if [[ "$MODE" == submit ]]; then
  stage=create-intent
  trap 'rc=$?; printf "{\"ok\":false,\"mode\":\"submit\",\"failed_stage\":\"%s\",\"manual_reconciliation_required\":true}\n" "$stage" >&2; exit "$rc"' ERR
  kubectl -n "$NS" create configmap "$INTENT_CM" \
    --from-literal=status=CREATE_ONCE_INTENT \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-literal=qwen_preflight_authorization_file_sha256="sha256:$Q_AUTH_RAW" \
    --from-literal=qwen_preflight_authorization_receipt_sha256="$Q_AUTH_SELF" \
    --from-literal=glm_preflight_authorization_file_sha256="sha256:$G_AUTH_RAW" \
    --from-literal=glm_preflight_authorization_receipt_sha256="$G_AUTH_SELF" \
    --from-literal=scored_v1_incident_receipt_sha256=sha256:b30930aeb12cbe4d8ab5ef91a1608ec672e2536576168b50639318b866e30423 \
    --from-literal=controller_compatibility_receipt_sha256=sha256:b3920547030db509de25f3a04559d9f4173866b81b02efa0957812b35cf06ba2 \
    --from-literal=preflight_manifest_sha256=sha256:44262e7dad2329219480f9e52b19064fab3c59d67f9c174fd11e7e058beb4f12 \
    --dry-run=client -o json | make_immutable | kubectl -n "$NS" create -f - -o name >/dev/null
  stage=create-qwen-configmap
  make_configmap "$Q_CM" "$Q_PLAN" "$Q_AUTH" "$Q_PLAN_RAW" "$Q_STATEMENT" submit >/dev/null
  stage=create-glm-configmap
  make_configmap "$G_CM" "$G_PLAN" "$G_AUTH" "$G_PLAN_RAW" "$G_STATEMENT" submit >/dev/null
  stage=create-preflight-jobs
  kubectl -n "$NS" create -f "$MANIFEST" -o name
  trap - ERR
  printf '%s\n' '{"ok":true,"mode":"submit","preflights_created":2,"scored_objects_created":0}'
else
  printf '%s\n' '{"ok":true,"mode":"preview","preflight_authorized":true,"scored_launch_authorized":false,"objects_created":false}'
fi
