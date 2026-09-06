#!/usr/bin/env bash
set -euo pipefail

test -n "${V23_SERVER_BINDING:-}"
test -n "${V23_WATCHDOG_TERMINAL:-}"
test -n "${V23_RELEASE_CONFIRMATION_OUT:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
exec uv run python -m evals.fleet.glm53_dedicated_v23_scorefree_gpu_observer_v1 \
  confirm-release \
  --binding "$V23_SERVER_BINDING" \
  --terminal "$V23_WATCHDOG_TERMINAL" \
  --out "$V23_RELEASE_CONFIRMATION_OUT"
