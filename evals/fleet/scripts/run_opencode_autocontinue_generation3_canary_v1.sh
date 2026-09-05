#!/usr/bin/env bash
set -euo pipefail

# Generation 3 remains held. A later executable release must package the exact
# repository tree and replace this stop only in an append-only successor.
ROOT=${CYBER_ROOT:-/workspace/cyber-post-train}
test -f "$ROOT/evals/fleet/autocontinue_generation3_canary.py"
uv run --no-project --with httpx==0.28.1 python \
  -m evals.fleet.autocontinue_generation3_canary validate-held \
  --held "$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation3-canaries-held-v1.json" \
  --repo "$ROOT"
printf '%s\n' 'generation-3 canary package is held; no release is installed' >&2
exit 3
