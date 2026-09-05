#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
[[ "$MODE" == preview ]] || {
  printf '%s\n' 'generation-3 canaries are held; only preview is available' >&2
  exit 3
}
ROOT=$(git rev-parse --show-toplevel)
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation3-canaries-held-v1.json"
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation3-canaries-v1.yaml"
uv run python -m evals.fleet.autocontinue_generation3_canary preview \
  --held "$HELD" --repo "$ROOT" >/dev/null
uv run python - "$MANIFEST" <<'PY'
from pathlib import Path
import sys
import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
expected = {
    "chris-q38-ac-r004-a1-g3-v1": "chris-q38-ac-r004-a1-g3-run-v1",
    "chris-glm53-ac-r013-a1-g3-v1": "chris-glm53-ac-r013-a1-g3-run-v1",
}
if set(manifest) != {"apiVersion", "kind", "items"} or manifest["kind"] != "List":
    raise SystemExit("generation-3 manifest envelope drifted")
if len(manifest["items"]) != 2:
    raise SystemExit("generation-3 manifest item count drifted")
for item in manifest["items"]:
    name = item["metadata"]["name"]
    pod = item["spec"]["template"]["spec"]
    if (
        name not in expected
        or item["metadata"]["annotations"]
        != {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        or item["spec"]["backoffLimit"] != 0
        or pod["priorityClassName"] != "fleet-serve-low"
        or pod["preemptionPolicy"] != "Never"
        or pod["restartPolicy"] != "Never"
        or pod["volumes"][0]["configMap"]["name"] != expected[name]
    ):
        raise SystemExit("generation-3 manifest policy drifted")
PY
kubectl -n fleet-train-jobs create --dry-run=server -f "$MANIFEST" -o name >/dev/null
printf '%s\n' '{"ok":true,"status":"HELD","execution_generation":3,"priority_class":"fleet-serve-low","preemption_policy":"Never","launch_authorized":false,"objects_created":false}'
