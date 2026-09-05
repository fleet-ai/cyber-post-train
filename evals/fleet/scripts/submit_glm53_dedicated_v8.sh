#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 || ! $1 =~ ^(preview|submit)$ ]]; then
  printf '%s\n' 'usage: submit_glm53_dedicated_v8.sh preview|submit RECEIPT' >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
exec uv run python -m evals.fleet.glm53_dedicated_v8_live "$1" --output "$2"
