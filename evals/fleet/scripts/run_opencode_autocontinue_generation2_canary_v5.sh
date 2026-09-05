#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
mkdir -p "$ROOT/evals/fleet/configs" "$ROOT/evals/fleet/scripts" \
  "$ROOT/evals/fleet/cluster" "$ROOT/docs/evidence/qwen38-study"
touch "$ROOT/evals/__init__.py" "$ROOT/evals/fleet/__init__.py"

install_module() { install -m 0644 "/bootstrap/$1" "$ROOT/evals/fleet/$2"; }
install_module self_hosted.py self_hosted.py
install_module runner.py opencode_train_sweep_runner.py
install_module controller.py hosted_sweep_controller.py
install_module canary_controller.py autocontinue_canary_controller.py
install_module hosted_release.py autocontinue_canary_hosted_release.py
install_module hosted_runtime.py autocontinue_canary_hosted_runtime.py
install_module hosted_health.py autocontinue_hosted_health.py
install_module endpoint_lease.py endpoint_lease.py
install_module fixed_proxy.py fixed_proxy.py
install_module exact_universe.py exact_pass4_universe.py
install_module generation2_v1.py autocontinue_generation2_canary.py
install_module generation2_v2.py autocontinue_generation2_canary_v2.py
install_module generation2_v3.py autocontinue_generation2_canary_v3.py
install_module runtime_v3.py autocontinue_generation2_runtime_v3.py
install_module authority_v4.py autocontinue_generation2_authority_v4.py
install_module authority_v5.py autocontinue_generation2_authority_v5.py
install_module Dockerfile.opencode Dockerfile.opencode

install -m 0644 /bootstrap/spec.json "$ROOT/evals/fleet/configs/generation2-spec.json"
install -m 0644 /bootstrap/q-spec.json "$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json"
install -m 0644 /bootstrap/g-spec.json "$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json"
install -m 0644 /bootstrap/q-v1-spec.json "$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json"
install -m 0644 /bootstrap/g-v1-spec.json "$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json"
install -m 0644 /bootstrap/q-predecessor.json "$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
install -m 0644 /bootstrap/g-predecessor.json "$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
install -m 0644 /bootstrap/campaign.json "$ROOT/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$ROOT/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/incident.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json"
install -m 0644 /bootstrap/tombstones.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json"
install -m 0644 /bootstrap/held-v1.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json"
install -m 0644 /bootstrap/held-v2.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json"
install -m 0644 /bootstrap/held-v3.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json"
install -m 0644 /bootstrap/executable-held-v3.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json"
install -m 0644 /bootstrap/authority.json "$ROOT/${GENERATION2_AUTHORITY_PATH:?}"
install -m 0644 /bootstrap/held-v4.json "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v1.json"
install -m 0644 /bootstrap/held-v5.json "$ROOT/${GENERATION2_HELD_PATH:?}"
install -m 0644 /bootstrap/release.json "$ROOT/evals/fleet/configs/generation2-release.json"
install -m 0644 /bootstrap/launch-route.json "$ROOT/evals/fleet/configs/generation2-launch-route.json"
for version in 1 2 3 4 5; do
  install -m 0644 "/bootstrap/v${version}-manifest.yaml" "$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v${version}.yaml"
done
install -m 0755 /bootstrap/v1-run.sh "$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh"
install -m 0755 /bootstrap/v1-submit.sh "$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh"
for version in 2 3 4 5; do
  install -m 0755 "/bootstrap/v${version}-run.sh" "$ROOT/evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v${version}.sh"
  install -m 0755 "/bootstrap/v${version}-submit.sh" "$ROOT/evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v${version}.sh"
done

uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation2_authority_v5 validate-held \
  --authority "$ROOT/$GENERATION2_AUTHORITY_PATH" \
  --held "$ROOT/$GENERATION2_HELD_PATH" --repo "$ROOT"
if [[ "${GENERATION2_RENDERER_GATE_ONLY:-0}" == 1 ]]; then exit 0; fi

: "${PACKAGE_COMMIT:?}" "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
OUT_ROOT=${FLEET_EVAL_OUT_ROOT:-/mnt/sfs/jobs/${GENERATION2_JOB_NAME:?}}
test ! -e "$OUT_ROOT"
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$ROOT/evals/fleet/Dockerfile.opencode" "$ROOT/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
exec uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation2_authority_v5 run \
  --spec "$ROOT/evals/fleet/configs/generation2-spec.json" \
  --authority "$ROOT/$GENERATION2_AUTHORITY_PATH" \
  --held "$ROOT/$GENERATION2_HELD_PATH" \
  --release "$ROOT/evals/fleet/configs/generation2-release.json" \
  --launch-route "$ROOT/evals/fleet/configs/generation2-launch-route.json" \
  --out-dir "$OUT_ROOT" --proxy "$ROOT/evals/fleet/fixed_proxy.py" --repo "$ROOT" \
  --package-commit "$PACKAGE_COMMIT"
