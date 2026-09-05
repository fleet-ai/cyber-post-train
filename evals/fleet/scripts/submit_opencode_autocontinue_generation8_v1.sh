#!/usr/bin/env bash
set -euo pipefail

mode=${1:-preview}
case "$mode" in
  preview) ;;
  submit)
    printf '%s\n' \
      'generation-8 is HELD: require terminal G7 preclaim tombstone and append-only releases' >&2
    exit 78
    ;;
  *) printf '%s\n' 'usage: submit_opencode_autocontinue_generation8_v1.sh preview|submit' >&2; exit 2 ;;
esac

root=$(git rev-parse --show-toplevel)
test -z "$(git -C "$root" status --porcelain=v1 --untracked-files=all -- \
  evals/fleet/autocontinue_generation8_optimized_v1.py \
  evals/fleet/autocontinue_generation8_package_v1.py \
  evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-v1.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-v1.json \
  evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-plan-v1.json \
  evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-plan-v1.json \
  evals/fleet/cluster/opencode-autocontinue-generation8-held-v1.yaml \
  evals/fleet/scripts/run_opencode_autocontinue_generation8_v1.sh \
  evals/fleet/scripts/submit_opencode_autocontinue_generation8_v1.sh \
  docs/GENERATION8_OPTIMIZED_FALLBACK.md \
  docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation8-optimized-held-v1.json \
  tests/test_autocontinue_generation8_optimized_v1.py)"

uv run python -m evals.fleet.autocontinue_generation8_optimized_v1 preview --repo "$root"
uv run python -m evals.fleet.autocontinue_generation8_package_v1 preview --repo "$root"
uv run python - "$root/evals/fleet/cluster/opencode-autocontinue-generation8-held-v1.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

value = yaml.safe_load(Path(sys.argv[1]).read_text())
items = value.get("items", [])
if len(items) != 2:
    raise SystemExit("expected exactly two held Generation-8 Jobs")
for job in items:
    pod = job["spec"]["template"]["spec"]
    if (
        job["metadata"]["annotations"].get("cyber-post-train.fleet.ai/launch-authorized") != "false"
        or pod.get("priorityClassName") != "fleet-serve-low"
        or pod.get("preemptionPolicy") != "Never"
        or job["spec"].get("backoffLimit") != 0
    ):
        raise SystemExit("Generation-8 held manifest safety contract drifted")
PY

printf '%s\n' 'Generation-8 optimized fallback validated HELD; no objects created'
