#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CAMPAIGN="$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
RELEASE="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-primary-campaign-release-preview-v1.json"

if [[ ${1:-preview} != preview ]]; then
  echo "This held artifact supports preview only; no workload submission is authorized." >&2
  exit 2
fi

test "$(jq -r '.status' "$CAMPAIGN")" = HELD
test "$(jq -r '.launch_authorized' "$CAMPAIGN")" = false
test "$(jq -r '.canary_launch_authorized' "$RELEASE")" = false
test "$(jq -r '.bulk_launch_authorized' "$RELEASE")" = false
test "$(jq -r '.cluster_objects_created' "$RELEASE")" = false

cd "$ROOT"
exec uv run python -m evals.fleet.autocontinue_campaign \
  --campaign "$CAMPAIGN" \
  --release "$RELEASE"
