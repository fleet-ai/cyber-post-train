#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview ]] || {
  printf '%s\n' 'generation-4 successor is held; only preview is available' >&2
  exit 3
}
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation4-held-v1.yaml"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation4-canaries-held-v1.json"

uv run python -m evals.fleet.autocontinue_generation4_canary validate-held \
  --held "$HELD" --repo "$ROOT"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
bundle="$work/generation4-held.yaml"
uv run python - "$ROOT" "$MANIFEST" "$bundle" <<'PY'
import sys
from pathlib import Path
import yaml
from evals.fleet import autocontinue_generation4_executable_package_v1 as package

root, manifest_path, output_path = map(Path, sys.argv[1:])
built = package.build_package(root)
manifest = yaml.safe_load(manifest_path.read_text())
Path(output_path).write_text(yaml.safe_dump({
    "apiVersion": "v1",
    "kind": "List",
    "items": list(built["configmaps"].values()) + manifest["items"],
}, sort_keys=False))
PY
kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
for name in chris-ac-g4-canary-submit-v1 chris-ac-g4-runtime-core-a-v1 \
  chris-ac-g4-runtime-core-b-v1 chris-q38-ac-r004-a1-g4-run-v1 \
  chris-glm53-ac-r013-a1-g4-run-v1; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in chris-q38-ac-r004-a1-g4-v1 chris-glm53-ac-r013-a1-g4-v1; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done
uv run python -m evals.fleet.autocontinue_generation4_executable_package_v1 \
  preview --repo "$ROOT"
