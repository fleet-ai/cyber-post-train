#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preview}
if [[ "$MODE" != preview ]]; then
  echo "successor canaries remain held; only preview is authorized" >&2
  exit 3
fi

ROOT=$(git rev-parse --show-toplevel)
NS=fleet-train-jobs
MANIFEST="$ROOT/evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
Q_PLAN="$ROOT/evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_PLAN="$ROOT/evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
Q_HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-successor-held-release-v3.json"
G_HELD="$ROOT/docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-successor-held-release-v3.json"
Q_CM=chris-q38-ac-canary1-run-v3
G_CM=chris-glm53-ac-canary1-run-v3
Q_JOB=chris-q38-ac-canary1-v2
G_JOB=chris-glm53-ac-canary1-v2

uv run python - "$ROOT" "$Q_PLAN" "$Q_HELD" "$G_PLAN" "$G_HELD" <<'PY'
import sys
from pathlib import Path
from evals.fleet import autocontinue_canary_controller as controller
from evals.fleet import autocontinue_canary_hosted_release as release

root = Path(sys.argv[1])
for plan_path, release_path in ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])):
    plan = controller.load_object(Path(plan_path))
    held = controller.load_object(Path(release_path))
    release.validate_successor_held_release(held, plan, root)
PY

uv run python -m evals.fleet.autocontinue_successor_manifest_authorization \
  --held \
  --manifest "$MANIFEST" \
  --expected-job "$Q_JOB" \
  --expected-job "$G_JOB" >/dev/null
kubectl -n "$NS" create --dry-run=server -f "$MANIFEST" -o name >/dev/null

for name in "$Q_CM" "$G_CM"; do
  test -z "$(kubectl -n "$NS" get configmap "$name" --ignore-not-found -o name)"
done
for name in "$Q_JOB" "$G_JOB"; do
  test -z "$(kubectl -n "$NS" get job "$name" --ignore-not-found -o name)"
  test -z "$(kubectl -n "$NS" get pods -l "job-name=$name" -o name)"
done

printf '%s\n' '{"ok":true,"mode":"preview","hosted_only":true,"scored_launch_authorized":false,"bulk_release_authorized":false,"objects_created":false}'
