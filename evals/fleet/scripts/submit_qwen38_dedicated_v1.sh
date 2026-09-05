#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 || ! $1 =~ ^(preview|submit)$ ]]; then
  printf '%s\n' 'usage: submit_qwen38_dedicated_v1.sh preview|submit RECEIPT' >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
exec uv run python -m evals.fleet.qwen38_dedicated_v1_live "$1" --output "$2"
