#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
exec uv run python -m evals.fleet.glm53_dedicated_v22_concurrency_gpu_observer_v1 \
  --root "$repo_root"
