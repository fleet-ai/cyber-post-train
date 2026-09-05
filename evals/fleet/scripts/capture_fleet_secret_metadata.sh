#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:?usage: capture_fleet_secret_metadata.sh /restricted/new-metadata.json}
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6

test ! -e "$OUT"
test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
umask 077
set -C
cd "$ROOT"
uv run python - "$OUT" "$EXPECTED_CONTEXT" <<'PY'
import json
import sys
from pathlib import Path

from evals.credential_rotation import read_secret_metadata

Path(sys.argv[1]).write_text(
    json.dumps(read_secret_metadata(context=sys.argv[2]), sort_keys=True) + "\n"
)
PY
