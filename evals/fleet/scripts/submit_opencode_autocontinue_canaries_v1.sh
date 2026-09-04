#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
CAMPAIGN_RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-primary-campaign-release-preview-v1.json"
PRE_MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml"
SCORED_MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
CONTROLLER="$ROOT/evals/fleet/autocontinue_canary_controller.py"
COMPATIBILITY="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
FROZEN="$ROOT/evals/fleet/hosted_sweep_controller.py"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-held-release-v1.json"
G_HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-held-release-v1.json"
Q_FINAL="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-scoring-release-v2.json"
G_FINAL="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-scoring-release-v2.json"
Q_PRE_CM=chris-q38-ac-canary1-pre-v1
G_PRE_CM=chris-glm53-ac-canary1-pre-v1
Q_RUN_CM=chris-q38-ac-canary1-run-v2
G_RUN_CM=chris-glm53-ac-canary1-run-v2
Q_PRE=chris-q38-ac-canary1-v1-preflight
G_PRE=chris-glm53-ac-canary1-v1-preflight
Q_JOB=chris-q38-ac-canary1-v1
G_JOB=chris-glm53-ac-canary1-v1

uv run python -m evals.fleet.autocontinue_campaign \
  --campaign "$CAMPAIGN" --release "$CAMPAIGN_RELEASE" >/dev/null
for pair in "$Q_PLAN:$Q_HELD" "$G_PLAN:$G_HELD"; do
  plan=${pair%%:*}
  release=${pair#*:}
  uv run python -m evals.fleet.autocontinue_canary_controller validate-held \
    --plan "$plan" --release "$release" --repo "$ROOT" >/dev/null
done
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a

for name in "$Q_PRE_CM" "$G_PRE_CM" "$Q_RUN_CM" "$G_RUN_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_PRE" "$G_PRE" "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done
if [[ "$MODE" == submit ]]; then
  test -f "$Q_FINAL" && test -f "$G_FINAL"
  for pair in "$Q_PLAN:$Q_FINAL" "$G_PLAN:$G_FINAL"; do
    plan=${pair%%:*}
    release=${pair#*:}
    uv run python -m evals.fleet.autocontinue_canary_controller validate-release \
      --plan "$plan" --release "$release" --repo "$ROOT" >/dev/null
  done
  test -z "$(yq -r 'select(.kind == "Job") | select(.metadata.annotations."cyber-post-train.fleet.ai/launch-authorized" != "true") | .metadata.name' "$SCORED_MANIFEST")"
  echo "final scored submission requires a separately reviewed v2 manifest and binding" >&2
  exit 3
fi

for spec in "$Q_PRE_CM:$Q_PLAN:$Q_HELD" "$G_PRE_CM:$G_PLAN:$G_HELD"; do
  cm=${spec%%:*}; rest=${spec#*:}; plan=${rest%%:*}; release=${rest#*:}
  kubectl -n "$NS" create configmap "$cm" \
    --from-file=plan.json="$plan" \
    --from-file=release.json="$release" \
    --from-file=controller.py="$FROZEN" \
    --from-file=canary_controller.py="$CONTROLLER" \
    --from-file=compatibility.json="$COMPATIBILITY" \
    --from-file=endpoint_lease.py="$ROOT/evals/fleet/endpoint_lease.py" \
    --from-file=self_hosted.py="$ROOT/evals/fleet/self_hosted.py" \
    --from-file=runner.py="$ROOT/evals/fleet/opencode_train_sweep_runner.py" \
    --from-file=preflight-manifest.yaml="$PRE_MANIFEST" \
    --dry-run=client -o json | jq '.immutable=true' | \
    kubectl -n "$NS" create --dry-run=server -f - -o name >/dev/null
done
kubectl -n "$NS" create --dry-run=server -f "$PRE_MANIFEST" -o name >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$SCORED_MANIFEST" -o name >/dev/null
printf '%s\n' '{"ok":true,"mode":"preview","scored_launch_authorized":false,"objects_created":false}'
