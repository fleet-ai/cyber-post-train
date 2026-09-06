#!/usr/bin/env bash
set -euo pipefail

test -n "${V23_QUALIFICATION_AUTHORIZATION:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
exec uv run python -m evals.fleet.glm53_dedicated_v23_scorefree_gpu_observer_v1 \
  --authorization "$V23_QUALIFICATION_AUTHORIZATION"
