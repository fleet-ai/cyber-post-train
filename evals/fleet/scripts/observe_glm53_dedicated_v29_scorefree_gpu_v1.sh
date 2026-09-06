#!/usr/bin/env bash
set -euo pipefail

test -n "${V29_QUALIFICATION_AUTHORIZATION:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
exec uv run python -m evals.fleet.glm53_dedicated_v29_scorefree_gpu_observer_v1 \
  --authorization "$V29_QUALIFICATION_AUTHORIZATION"
