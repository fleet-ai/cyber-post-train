#!/usr/bin/env bash
set -euo pipefail

: "${PACKAGE_COMMIT:?}" "${PACKAGE_SHA256:?}" "${JOB_UID:?}" "${POD_UID:?}"
runtime=/runtime/hosted-q38-c4-addon-v1
python /bootstrap/package.py materialize \
  --projected /bootstrap \
  --destination "$runtime" \
  --package-commit "$PACKAGE_COMMIT" >/dev/null
test "$(cat /bootstrap/package_sha256)" = "$PACKAGE_SHA256"
cd "$runtime"
exec python -m evals.fleet.hosted_c4_addon_qualifier_v1 run \
  --out-dir /mnt/sfs/jobs/chris-cyber-hosted-q38-c4-addon-v2
