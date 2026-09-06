#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${JOB_UID:?}" "${POD_UID:?}"
export REPO_ROOT=/workspace/cyber-post-train
python /bootstrap/bootstrap.py
cd "$REPO_ROOT"
exec uv run --no-project --with httpx==0.28.1 \
  python -m evals.fleet.hosted_glm_rank30_preclaim_phase_observer_v1
