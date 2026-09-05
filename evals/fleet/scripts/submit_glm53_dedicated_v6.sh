#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' 'usage: submit_glm53_dedicated_v6.sh held-preview|submit' >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
MODE=$1
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

python3 -m evals.fleet.glm53_dedicated_v6 validate >/dev/null

case "$MODE" in
  held-preview)
    python3 -m evals.fleet.glm53_dedicated_v6 held-preview
    ;;
  submit)
    printf '%s\n' \
      'HELD: G5, prebulk reconciliation, authenticated Jobs API image-pull preview, and UID-bound canary gates are not attached' >&2
    exit 78
    ;;
  *) usage ;;
esac
