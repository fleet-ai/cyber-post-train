#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  preview)
    shift
    exec uv run python -m evals.fleet.exact_pass4_dedicated_bulk_v4 preview "$@"
    ;;
  validate)
    shift
    exec uv run python -m evals.fleet.exact_pass4_dedicated_bulk_v4 validate "$@"
    ;;
  prepare-prebulk-release)
    shift
    exec "$(git rev-parse --show-toplevel)/evals/fleet/scripts/submit_exact_pass4_prebulk_reconciliation_v4.sh" prepare-release "$@"
    ;;
  submit-prebulk-source)
    shift
    exec "$(git rev-parse --show-toplevel)/evals/fleet/scripts/submit_exact_pass4_prebulk_reconciliation_v4.sh" submit-source "$@"
    ;;
  submit-prebulk-accept)
    shift
    exec "$(git rev-parse --show-toplevel)/evals/fleet/scripts/submit_exact_pass4_prebulk_reconciliation_v4.sh" submit-accept "$@"
    ;;
  *)
    printf '%s\n' 'usage: submit_exact_pass4_dedicated_bulk_v4.sh preview|validate|prepare-prebulk-release|submit-prebulk-source|submit-prebulk-accept' >&2
    exit 2
    ;;
esac
