#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
PACKAGE_COMMIT=fdad6c80bcbfa999cd98e6f3194040bd1d85d679
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
FROZEN="$ROOT/evals/fleet/hosted_sweep_controller.py"
SELF_HOSTED="$ROOT/evals/fleet/self_hosted.py"
RUNNER="$ROOT/evals/fleet/opencode_train_sweep_runner.py"
ENDPOINT_LEASE="$ROOT/evals/fleet/endpoint_lease.py"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml"
SCORED_MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v1.json"
G_AUTH="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v1.json"
Q_CM=chris-q38-ac-canary1-pre-v1
G_CM=chris-glm53-ac-canary1-pre-v1
Q_PRE=chris-q38-ac-canary1-v1-preflight
G_PRE=chris-glm53-ac-canary1-v1-preflight
INTENT_CM=chris-ac-canary1-preflight-submit-v1
Q_AUTH_RAW=45acad27d6389d0da90491090d9adcae5455952473aac27d8fc302f7096e0f3d
G_AUTH_RAW=8ecbfeb079a3ed0241e81835e8e1180674ba132a14645a279f602722427cb4bd
Q_AUTH_SELF=sha256:9f8eae90f2d59a1c1a39f94d38adbd345eaa5006cd73d2eec850701a32e59b59
G_AUTH_SELF=sha256:8870a2ff27c6b22a70a28e2c58add13e459754dbe12480ba9786feb7f4654ab4

test "$(git rev-parse "$PACKAGE_COMMIT^{commit}")" = "$PACKAGE_COMMIT"
test "$(sha256sum "$CONTROLLER" | awk '{print $1}')" = f023dffd60f4594ff60a6388b6ca653c24e56ebb760dc8ecdf8afcc6d7ee04f9
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a
test "$(sha256sum "$SELF_HOSTED" | awk '{print $1}')" = 16df432b5fde55112924d6106e6c03f09817846f1344ba0fe100dcf785c33d8b
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = b1f9c5028f65b0d7772538e3ce075310dc6c7a46b3de58d0196bc474e74e9e9d
test "$(sha256sum "$ENDPOINT_LEASE" | awk '{print $1}')" = 1df60ee13be8c6057113dbebadf9020343649e175b5de38aea41706748987019
test "$(sha256sum "$COMPATIBILITY" | awk '{print $1}')" = 68f4afe5c2b89d610b99b95f4c1d8454f2723e31cf34f8f5546494e80c4ab0c3
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = 45f4d0942e6d4485b35be4d0e3ebccdaa825bef5ceec377ba5f983cdd7e499d6
test "$(sha256sum "$SCORED_MANIFEST" | awk '{print $1}')" = 1217666a0bfe5ab4d9aec0191213a3f1af8967f053b046922de656ed4a953be2
test "$(sha256sum "$Q_PLAN" | awk '{print $1}')" = 28d3e45ae8ef317506d5414af89f4d3f7f38e3ed3762eccbe57cca3a45728ebc
test "$(sha256sum "$G_PLAN" | awk '{print $1}')" = 2eada8438613465f8d9422836e9a6cce76f5a6eefbdc31635939b6ff3fd958c8
test "$(sha256sum "$Q_AUTH" | awk '{print $1}')" = "$Q_AUTH_RAW"
test "$(sha256sum "$G_AUTH" | awk '{print $1}')" = "$G_AUTH_RAW"

verify_git_object() {
  local file=$1 relative=${1#"$ROOT/"}
  git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"
}

for file in "$CONTROLLER" "$FROZEN" "$SELF_HOSTED" "$RUNNER" "$ENDPOINT_LEASE" \
  "$COMPATIBILITY" "$MANIFEST" "$SCORED_MANIFEST" "$Q_PLAN" "$G_PLAN"; do
  verify_git_object "$file"
done

for row in "$Q_PLAN|$Q_AUTH|$Q_AUTH_RAW|$Q_AUTH_SELF" \
  "$G_PLAN|$G_AUTH|$G_AUTH_RAW|$G_AUTH_SELF"; do
  IFS='|' read -r plan authorization expected_raw expected_self <<<"$row"
  test "$(sha256sum "$authorization" | awk '{print $1}')" = "$expected_raw"
  uv run python - "$plan" "$authorization" "$ROOT" "$PACKAGE_COMMIT" "$expected_self" <<'PY'
import sys
from pathlib import Path
from evals.fleet.autocontinue_canary_controller import load_object, validate_preflight_authorization

plan = load_object(Path(sys.argv[1]))
authorization = load_object(Path(sys.argv[2]))
if authorization.get("receipt_sha256") != sys.argv[5]:
    raise SystemExit("preflight authorization immutable intent digest drifted")
if authorization.get("package_commit") != sys.argv[4]:
    raise SystemExit("preflight authorization package commit drifted")
if authorization.get("implementation", {}).get("package_commit") != sys.argv[4]:
    raise SystemExit("preflight authorization implementation commit drifted")
validate_preflight_authorization(authorization, plan, Path(sys.argv[3]), sys.argv[4])
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
  local name=$1 plan=$2 authorization=$3 mode=$4
  local payload
  payload=$(kubectl -n "$NS" create configmap "$name" \
    --from-literal=package_commit="$PACKAGE_COMMIT" \
    --from-file=plan.json="$plan" \
    --from-file=release.json="$authorization" \
    --from-file=controller.py="$FROZEN" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=compatibility.json="$COMPATIBILITY" \
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

make_configmap "$Q_CM" "$Q_PLAN" "$Q_AUTH" preview >/dev/null
make_configmap "$G_CM" "$G_PLAN" "$G_AUTH" preview >/dev/null
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
    --from-literal=preflight_manifest_sha256=sha256:45f4d0942e6d4485b35be4d0e3ebccdaa825bef5ceec377ba5f983cdd7e499d6 \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - -o name >/dev/null
  stage=create-qwen-configmap
  make_configmap "$Q_CM" "$Q_PLAN" "$Q_AUTH" submit >/dev/null
  stage=create-glm-configmap
  make_configmap "$G_CM" "$G_PLAN" "$G_AUTH" submit >/dev/null
  stage=create-preflight-jobs
  kubectl -n "$NS" create -f "$MANIFEST" -o name
  trap - ERR
  printf '%s\n' '{"ok":true,"mode":"submit","preflights_created":2,"scored_objects_created":0}'
else
  printf '%s\n' '{"ok":true,"mode":"preview","preflight_authorized":true,"scored_launch_authorized":false,"objects_created":false}'
fi
