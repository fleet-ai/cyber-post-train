#!/usr/bin/env bash
set -euo pipefail

ROOT=${CYBER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PACKAGE=${BULK_PACKAGE:-$ROOT/evals/fleet/configs/q38-glm53-opencode-autocontinue-bulk-held-v1.json}
HELD_AUTH=${BULK_HELD_AUTHORIZATION:-$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-bulk-held-authorization-v1.json}
MODE=${1:-preview}

case "$MODE" in
  preview)
    test "$#" -eq 1
    exec uv run --no-project --with httpx==0.28.1 python \
      -m evals.fleet.autocontinue_bulk_controller preview \
      --package "$PACKAGE" \
      --held-authorization "$HELD_AUTH" \
      --repo "$ROOT"
    ;;
  materialize)
    test "$#" -eq 5
    AUTHORIZATION=$2
    QWEN_OBSERVER=$3
    GLM_OBSERVER=$4
    OUT_DIR=$5
    exec uv run --no-project --with httpx==0.28.1 python \
      -m evals.fleet.autocontinue_bulk_controller materialize \
      --package "$PACKAGE" \
      --authorization "$AUTHORIZATION" \
      --qwen-observer "$QWEN_OBSERVER" \
      --glm-observer "$GLM_OBSERVER" \
      --out-dir "$OUT_DIR" \
      --repo "$ROOT"
    ;;
  authorize-shard)
    test "$#" -eq 8
    PLAN=$2
    AUTHORIZATION=$3
    QWEN_OBSERVER=$4
    GLM_OBSERVER=$5
    PREFLIGHT=$6
    PREFLIGHT_OBSERVER=$7
    OUT=$8
    exec uv run --no-project --with httpx==0.28.1 python \
      -m evals.fleet.autocontinue_bulk_controller authorize-shard \
      --plan "$PLAN" \
      --package "$PACKAGE" \
      --authorization "$AUTHORIZATION" \
      --qwen-observer "$QWEN_OBSERVER" \
      --glm-observer "$GLM_OBSERVER" \
      --preflight "$PREFLIGHT" \
      --preflight-observer "$PREFLIGHT_OBSERVER" \
      --out "$OUT" \
      --repo "$ROOT"
    ;;
  *)
    echo "usage: $0 preview | materialize AUTH Q_OBS G_OBS OUT_DIR | authorize-shard PLAN AUTH Q_OBS G_OBS PREFLIGHT PREFLIGHT_OBSERVER OUT" >&2
    exit 64
    ;;
esac
