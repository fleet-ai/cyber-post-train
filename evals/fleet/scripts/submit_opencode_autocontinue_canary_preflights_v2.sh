#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
PACKAGE_COMMIT=9c93095f60d627c239fe2d01f35efd918f328fe6
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
FROZEN="$ROOT/evals/fleet/hosted_sweep_controller.py"
SELF_HOSTED="$ROOT/evals/fleet/self_hosted.py"
RUNNER="$ROOT/evals/fleet/opencode_train_sweep_runner.py"
ENDPOINT_LEASE="$ROOT/evals/fleet/endpoint_lease.py"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
INCIDENT="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v2.json"
G_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v2.json"
Q_CM=chris-q38-ac-canary1-pre-v2
G_CM=chris-glm53-ac-canary1-pre-v2
Q_PRE=chris-q38-ac-canary1-v2-preflight
G_PRE=chris-glm53-ac-canary1-v2-preflight
INTENT_CM=chris-ac-canary1-preflight-submit-v2
AUTHORIZED_AT_UTC=2026-09-04T21:31:11Z
Q_STATEMENT='I authorize the create-once read-only Qwen corrected-treatment canary v2 preflight only; it supersedes the terminal infrastructure-failed v1 preflight and no scored Job or scored ConfigMap may be created.'
G_STATEMENT='I authorize the create-once read-only GLM corrected-treatment canary v2 preflight only; it supersedes the terminal infrastructure-failed v1 preflight and no scored Job or scored ConfigMap may be created.'
Q_PLAN_RAW=28d3e45ae8ef317506d5414af89f4d3f7f38e3ed3762eccbe57cca3a45728ebc
G_PLAN_RAW=2eada8438613465f8d9422836e9a6cce76f5a6eefbdc31635939b6ff3fd958c8
Q_AUTH_RAW=ad4171eec459d4925f5f12b97a3040ac3fa30e65b434bf2592a8ee43c9070249
G_AUTH_RAW=7e037f0d8436ad04e66056718d7dc94d1bb5f00f57b477aebedec3c193ae06cf
Q_AUTH_SELF=sha256:d1b764cd8f96bbe5d36a62956ea32931066740c8c77b8ac60c5474c7209f3720
G_AUTH_SELF=sha256:a7cef96859a38ace334d5edbef22ffa1d87ed74d5c1d216857945035bf07f530

test "$(git rev-parse "$PACKAGE_COMMIT^{commit}")" = "$PACKAGE_COMMIT"
test "$(sha256sum "$CONTROLLER" | awk '{print $1}')" = 412bf8c0dd33d23e50a56c4597e5e0dfc90b122a8b06af0987059afdca53f7ef
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a
test "$(sha256sum "$SELF_HOSTED" | awk '{print $1}')" = 16df432b5fde55112924d6106e6c03f09817846f1344ba0fe100dcf785c33d8b
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = b1f9c5028f65b0d7772538e3ce075310dc6c7a46b3de58d0196bc474e74e9e9d
test "$(sha256sum "$ENDPOINT_LEASE" | awk '{print $1}')" = 1df60ee13be8c6057113dbebadf9020343649e175b5de38aea41706748987019
test "$(sha256sum "$COMPATIBILITY" | awk '{print $1}')" = ddc57a78fe4ef8e85352cfb9541773e0cb6b96aa29f6890903a266cf095e62fa
test "$(sha256sum "$INCIDENT" | awk '{print $1}')" = 43c7875c08a416967ac78ece28b9bb0bbf5f1c7a9bd14a414618adf732c166c9
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = 979071556343cfa23373a94742fefa684a9d9e0e39ecbe1f2bea05c58dce998a
test "$(sha256sum "$Q_PLAN" | awk '{print $1}')" = "$Q_PLAN_RAW"
test "$(sha256sum "$G_PLAN" | awk '{print $1}')" = "$G_PLAN_RAW"
test "$(sha256sum "$Q_AUTH" | awk '{print $1}')" = "$Q_AUTH_RAW"
test "$(sha256sum "$G_AUTH" | awk '{print $1}')" = "$G_AUTH_RAW"

verify_git_object() {
  local file=$1 relative=${1#"$ROOT/"}
  git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"
}

for file in "$CONTROLLER" "$FROZEN" "$SELF_HOSTED" "$RUNNER" "$ENDPOINT_LEASE" \
  "$COMPATIBILITY" "$INCIDENT" "$MANIFEST" "$Q_PLAN" "$G_PLAN"; do
  verify_git_object "$file"
done

for row in \
  "$Q_PLAN|$Q_AUTH|$Q_PLAN_RAW|$Q_AUTH_RAW|$Q_AUTH_SELF|$Q_STATEMENT" \
  "$G_PLAN|$G_AUTH|$G_PLAN_RAW|$G_AUTH_RAW|$G_AUTH_SELF|$G_STATEMENT"; do
  IFS='|' read -r plan authorization plan_raw expected_raw expected_self statement <<<"$row"
  test "$(sha256sum "$plan" | awk '{print $1}')" = "$plan_raw"
  test "$(sha256sum "$authorization" | awk '{print $1}')" = "$expected_raw"
  uv run python - "$plan" "$authorization" "$ROOT" "$PACKAGE_COMMIT" \
    "sha256:$plan_raw" "$AUTHORIZED_AT_UTC" "$statement" "$expected_self" <<'PY'
import sys
from pathlib import Path
from evals.fleet.autocontinue_canary_controller import (
    load_object,
    validate_preflight_authorization,
)

plan = load_object(Path(sys.argv[1]))
authorization = load_object(Path(sys.argv[2]))
if authorization.get("receipt_sha256") != sys.argv[8]:
    raise SystemExit("preflight authorization immutable intent digest drifted")
validate_preflight_authorization(
    authorization,
    plan,
    Path(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    sys.argv[7],
)
PY
done

for name in "$INTENT_CM" "$Q_CM" "$G_CM" chris-q38-ac-canary1-run-v2 chris-glm53-ac-canary1-run-v2; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_PRE" "$G_PRE" chris-q38-ac-canary1-v1 chris-glm53-ac-canary1-v1; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done

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
    --from-file=controller.py="$FROZEN" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=compatibility.json="$COMPATIBILITY" \
    --from-file=v1-incident.json="$INCIDENT" \
    --from-file=endpoint_lease.py="$ENDPOINT_LEASE" \
    --from-file=self_hosted.py="$SELF_HOSTED" \
    --from-file=runner.py="$RUNNER" \
    --from-file=preflight-manifest.yaml="$MANIFEST" \
    --dry-run=client -o json | jq -c '.immutable=true')
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
    --from-literal=v1_incident_receipt_sha256=sha256:c07804a36065a68cc821b0c57e7989f7e9caa077142969046b4bff913a592ee4 \
    --from-literal=controller_compatibility_receipt_sha256=sha256:7851d18a8176abb08aebcb857960380f7f8a59bb854c51437e4010f169c82b6a \
    --from-literal=preflight_manifest_sha256=sha256:979071556343cfa23373a94742fefa684a9d9e0e39ecbe1f2bea05c58dce998a \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - -o name >/dev/null
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
