#!/usr/bin/env bash
set -euo pipefail

: "${DUPLICATE_MODE:?DUPLICATE_MODE is required}"
: "${DUPLICATE_GROUP:?DUPLICATE_GROUP is required}"
: "${PACKAGE_AGGREGATE:?PACKAGE_AGGREGATE is required}"
: "${PACKAGE_COMMIT:?PACKAGE_COMMIT is required}"
: "${RELEASE_FILE_SHA256:?RELEASE_FILE_SHA256 is required}"
: "${JOB_UID:?JOB_UID is required}"
: "${POD_UID:?POD_UID is required}"
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"

case "${DUPLICATE_MODE}" in
  source|accept) ;;
  *) echo "invalid duplicate mode" >&2; exit 64 ;;
esac

repo=/workspace/cyber-post-train
release="${repo}/duplicate-release.json"
python3 /bootstrap/package.py verify-mounted \
  --manifest /bootstrap/package-manifest.json \
  --bootstrap /bootstrap \
  --destination "${repo}" \
  --expected "${PACKAGE_AGGREGATE}"
cp /bootstrap/duplicate-release.json "${release}"
actual="sha256:$(sha256sum "${release}" | awk '{print $1}')"
test "${actual}" = "${RELEASE_FILE_SHA256}"

exec python3 -m evals.fleet.exact_pass4_final_duplicate_observer_v5 \
  "${DUPLICATE_MODE}" \
  --root "${repo}" \
  --release "${release}" \
  --group "${DUPLICATE_GROUP}" \
  --output-root "/mnt/sfs/jobs/chris-cyber-final-v5-dup-${DUPLICATE_GROUP}"
