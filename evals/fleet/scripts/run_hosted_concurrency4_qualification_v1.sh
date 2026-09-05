#!/usr/bin/env bash
set -euo pipefail

runtime=/runtime/hosted-concurrency4-qualification-v1
python /bootstrap/package.py materialize \
  --projected /bootstrap \
  --destination "$runtime" \
  --package-commit "$PACKAGE_COMMIT" >/dev/null
test "$(cat /bootstrap/package_sha256)" = "$PACKAGE_SHA256"
exec python "$runtime/evals/fleet/hosted_concurrency4_qualification_v1.py" run \
  --out-dir /mnt/sfs/jobs/chris-cyber-hosted-c4-qualification-v1
