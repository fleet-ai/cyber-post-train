#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview && "$MODE" != submit ]]; then
  echo "usage: $0 preview|submit" >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml"
Q_SPEC="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json"
G_SPEC="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json"
HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json"

uv run python -m evals.fleet.autocontinue_generation2_canary_v2 preview \
  --spec "$Q_SPEC" --spec "$G_SPEC" --release "$HELD" --repo "$ROOT" >/dev/null
uv run python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
items = manifest.get("items", [])
expected = {
    "chris-q38-ac-r004-a1-g2-v1": "chris-q38-ac-r004-a1-g2-run-v1",
    "chris-glm53-ac-r013-a1-g2-v1": "chris-glm53-ac-r013-a1-g2-run-v1",
}
if set(manifest) != {"apiVersion", "kind", "items"} or manifest.get("kind") != "List":
    raise SystemExit("generation-2 successor manifest envelope drifted")
if len(items) != 2 or {item["metadata"]["name"] for item in items} != set(expected):
    raise SystemExit("generation-2 successor manifest roster drifted")
for item in items:
    name = item["metadata"]["name"]
    pod = item["spec"]["template"]["spec"]
    if (
        item["metadata"]["namespace"] != "fleet-train-jobs"
        or item["metadata"]["annotations"]
        != {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        or item["spec"]["backoffLimit"] != 0
        or pod["restartPolicy"] != "Never"
        or pod["priorityClassName"] != "fleet-train-high"
        or pod["preemptionPolicy"] != "Never"
        or pod["volumes"][0]["configMap"]["name"] != expected[name]
    ):
        raise SystemExit("generation-2 successor manifest safety policy drifted")
PY

kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null
for name in chris-ac-g2-canary-submit-v1 \
  chris-q38-ac-r004-a1-g2-run-v1 chris-glm53-ac-r013-a1-g2-run-v1; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in chris-q38-ac-r004-a1-g2-v1 chris-glm53-ac-r013-a1-g2-v1; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pod -l "job-name=$name" -o name)"
done

if [[ "$MODE" == preview ]]; then
  printf '%s\n' '{"ok":true,"status":"HELD","execution_generation":2,"statistical_cells":2,"runtime_plans_self_digesting":true,"launch_authorized":false,"objects_created":false}'
  exit 0
fi

# This append-only successor remains held. Release requires a new reviewed
# executable package and model-specific scoring receipts; this file never
# mutates the historical v1 or held-v2 package in place.
echo "generation-2 successor package is HELD; no launch is authorized" >&2
exit 1
