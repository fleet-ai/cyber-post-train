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
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
PRE_MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml"
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
HOSTED_RELEASE="$ROOT/evals/fleet/autocontinue_canary_hosted_release.py"
HOSTED_RUNTIME="$ROOT/evals/fleet/autocontinue_canary_hosted_runtime.py"
HOSTED_HEALTH="$ROOT/evals/fleet/autocontinue_hosted_health.py"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
V1_INCIDENT="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
PHASE_C_ROUTE="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-hosted-only-route-v1.json"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-hosted-scoring-release-v3.json"
G_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-hosted-scoring-release-v3.json"
Q_CM=chris-q38-ac-canary1-run-v2
G_CM=chris-glm53-ac-canary1-run-v2
Q_JOB=chris-q38-ac-canary1-v1
G_JOB=chris-glm53-ac-canary1-v1
INTENT=chris-ac-canary1-hosted-scored-submit-v1

phase_c() {
  uv run python - "$ROOT" "$1" <<'PY'
import sys
from pathlib import Path
from evals.fleet import autocontinue_canary_controller as c
from evals.fleet import autocontinue_canary_hosted_release as h
root = Path(sys.argv[1])
h.validate_preflight_bundle(c.load_object(root / sys.argv[2]), root)
PY
}

phase_c evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json
phase_c evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null
for name in "$Q_CM" "$G_CM" "$INTENT"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"mode":"preview","hosted_only":true,"scored_launch_authorized":false,"objects_created":false}'
  exit 0
fi

test -f "$Q_RELEASE" && test -f "$G_RELEASE"
for pair in "$Q_PLAN:$Q_RELEASE" "$G_PLAN:$G_RELEASE"; do
  plan=${pair%%:*}
  release=${pair#*:}
  uv run python -m evals.fleet.autocontinue_canary_hosted_runtime validate-release \
    --plan "$plan" --release "$release" --repo "$ROOT" >/dev/null
done
test -z "$(yq -r 'select(.kind == "Job") | select(.metadata.annotations."cyber-post-train.fleet.ai/launch-authorized" != "true") | .metadata.name' "$MANIFEST")"

PACKAGE_COMMIT=$(jq -er '.implementation.package_commit' "$Q_RELEASE")
test "$PACKAGE_COMMIT" = "$(jq -er '.implementation.package_commit' "$G_RELEASE")"
for path in \
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
  evals/fleet/scripts/run_opencode_autocontinue_canary.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh \
  evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml; do
  test "$(git show "$PACKAGE_COMMIT:$path" | sha256sum | awk '{print $1}')" = "$(sha256sum "$ROOT/$path" | awk '{print $1}')"
done

route_dir=$(mktemp -d)
trap 'rm -rf "$route_dir"' EXIT
route_file="$route_dir/launch-route.json"
api_key=$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.FLEET_API_KEY}' | base64 --decode)
FLEET_API_KEY="$api_key" uv run python -m evals.fleet.autocontinue_canary_hosted_runtime \
  observe-route --out "$route_file" >/dev/null
unset api_key
uv run python -m evals.fleet.autocontinue_canary_hosted_runtime validate-route \
  --receipt "$route_file" --maximum-age-seconds 120 >/dev/null

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
    --from-file=plan.json="$plan" \
    --from-file=release.json="$release" \
    --from-file=launch-route.json="$route_file" \
    --from-file=controller.py="$ROOT/evals/fleet/hosted_sweep_controller.py" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=hosted_release.py="$HOSTED_RELEASE" \
    --from-file=hosted_runtime.py="$HOSTED_RUNTIME" \
    --from-file=hosted_health.py="$HOSTED_HEALTH" \
    --from-file=endpoint_lease.py="$ROOT/evals/fleet/endpoint_lease.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=fixed_proxy.py="$ROOT/evals/fleet/fixed_proxy.py" \
    --from-file=Dockerfile.opencode="$ROOT/evals/fleet/Dockerfile.opencode" \
    --from-file=run.sh="$ROOT/evals/fleet/scripts/run_opencode_autocontinue_canary.sh" \
    --from-file=submit.sh="$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh" \
    --from-file=campaign.json="$CAMPAIGN" \
    --from-file=preflight-manifest.yaml="$PRE_MANIFEST" \
    --from-file=scored-manifest.yaml="$MANIFEST" \
    --from-file=compatibility.json="$COMPATIBILITY" \
    --from-file=v1-incident.json="$V1_INCIDENT" \
    --from-file=phase-c-route.json="$PHASE_C_ROUTE" \
    --from-file=preauth.json="$preauth" \
    --from-file=preflight.json="$preflight" \
    --from-file=post-exit.json="$post_exit" \
    --from-file=duplicate.json="$duplicate" \
    --dry-run=client -o json | jq '.immutable=true' | kubectl -n "$NS" create -f - >/dev/null
}

create_scored_cm "$Q_CM" "$Q_PLAN" "$Q_RELEASE" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v2.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v2-pass.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-preflight-v2-post-exit.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-duplicate-inventory-v2.json"
create_scored_cm "$G_CM" "$G_PLAN" "$G_RELEASE" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-authorization-v2.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v2-pass.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v2-post-exit.json" \
  "$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-duplicate-inventory-v2.json"

uv run python -m evals.fleet.autocontinue_canary_hosted_runtime validate-route \
  --receipt "$route_file" --maximum-age-seconds 120 >/dev/null
kubectl -n "$NS" create -f "$MANIFEST" -o name
