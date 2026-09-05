#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview ]] || {
  printf '%s\n' 'generation-3 split package is held; only preview is available' >&2
  exit 3
}
ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation3-executable-held-v2.yaml"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation3-executable-package-held-v2.json"

uv run python -m evals.fleet.autocontinue_generation3_executable_package \
  validate-held --held "$HELD" --repo "$ROOT" >/dev/null

preview_dir=$(mktemp -d)
trap 'rm -rf "$preview_dir"' EXIT
bundle="$preview_dir/split-package-held.yaml"
uv run python - "$ROOT" "$MANIFEST" "$bundle" <<'PY'
import sys
from pathlib import Path
import yaml

from evals.fleet import autocontinue_generation3_executable_package as package

root, manifest_path, output_path = map(Path, sys.argv[1:])
built = package.build_package(root)
manifest = yaml.safe_load(manifest_path.read_text())
if manifest.get("kind") != "List" or len(manifest.get("items", [])) != 2:
    raise SystemExit("generation-3 executable manifest envelope drifted")
expected_sources = {
    "chris-q38-ac-r004-a1-g3-v1": [
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.MODEL_NAMES["qwen3.8-27b"],
    ],
    "chris-glm53-ac-r013-a1-g3-v1": [
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.MODEL_NAMES["glm-5.3"],
    ],
}
for item in manifest["items"]:
    name = item["metadata"]["name"]
    pod = item["spec"]["template"]["spec"]
    sources = [
        source["configMap"]["name"]
        for source in pod["volumes"][0]["projected"]["sources"]
    ]
    if (
        item["metadata"]["annotations"].get("cyber-post-train.fleet.ai/launch-authorized") != "false"
        or item["metadata"]["annotations"].get("cyber-post-train.fleet.ai/preview-only") != "true"
        or pod["priorityClassName"] != "fleet-serve-low"
        or pod["preemptionPolicy"] != "Never"
        or item["spec"]["backoffLimit"] != 0
        or sources != expected_sources.get(name)
    ):
        raise SystemExit("generation-3 executable manifest policy drifted")
objects = list(built["configmaps"].values()) + manifest["items"]
output_path.write_text(yaml.safe_dump({"apiVersion": "v1", "kind": "List", "items": objects}, sort_keys=False))
PY

kubectl -n "$NS" create --dry-run=server -f "$bundle" -o name >/dev/null
for name in chris-ac-g3-runtime-core-a-v2 chris-ac-g3-runtime-core-b-v2 \
  chris-q38-ac-r004-a1-g3-run-v1 chris-glm53-ac-r013-a1-g3-run-v1; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in chris-q38-ac-r004-a1-g3-v1 chris-glm53-ac-r013-a1-g3-v1; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l job-name="$name" -o name)"
done

uv run python -m evals.fleet.autocontinue_generation3_executable_package \
  preview --repo "$ROOT"
