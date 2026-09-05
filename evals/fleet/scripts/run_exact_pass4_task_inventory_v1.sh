#!/usr/bin/env bash
set -euo pipefail

: "${PACKAGE_COMMIT:?}" "${JOB_UID:?}" "${POD_UID:?}" "${FLEET_API_KEY:?}"
RUNTIME_ROOT=/workspace/exact-pass4-task-inventory-v1
OUT_ROOT=/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v1

python /bootstrap/package.py materialize \
  --projected /bootstrap --destination "$RUNTIME_ROOT" \
  --package-commit "$PACKAGE_COMMIT" >/dev/null

cd "$RUNTIME_ROOT"
python \
  -m evals.fleet.exact_pass4_task_inventory \
  --root "$RUNTIME_ROOT" --prepare "$RUNTIME_ROOT/EXPECTED.json"
exec python \
  -m evals.fleet.exact_pass4_task_inventory \
  --root "$RUNTIME_ROOT" --out-dir "$OUT_ROOT"
