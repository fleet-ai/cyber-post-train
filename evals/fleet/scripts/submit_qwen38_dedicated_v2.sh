#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  printf '%s\n' "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
OUT=${QWEN38_V2_RECEIPT:?QWEN38_V2_RECEIPT is required}
cd "$ROOT"
python3 -m evals.fleet.qwen38_dedicated_v2_live "$MODE" --output "$OUT"
