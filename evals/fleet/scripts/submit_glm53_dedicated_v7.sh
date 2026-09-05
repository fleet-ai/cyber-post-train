#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  printf '%s\n' 'usage: submit_glm53_dedicated_v7.sh held-preview | server-preview A|B | submit A|B' >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
MODE=$1
REPLICA=${2:-}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
uv run python -m evals.fleet.glm53_dedicated_v7 validate >/dev/null

case "$MODE" in
  held-preview)
    [[ $# -eq 1 ]] || usage
    exec uv run python -m evals.fleet.glm53_dedicated_v7 held-preview
    ;;
  server-preview)
    [[ $# -eq 2 && "$REPLICA" =~ ^[AB]$ ]] || usage
    exec uv run python -m evals.fleet.glm53_dedicated_v7 server-preview --replica "$REPLICA"
    ;;
  submit)
    [[ $# -eq 2 && "$REPLICA" =~ ^[AB]$ ]] || usage
    ;;
  *) usage ;;
esac

: "${GLM53_PREVIEW_RECEIPT:?}" "${GLM53_DUPLICATE_RECEIPT:?}"
: "${GLM53_PREBULK_TERMINAL:?}" "${GLM53_SUBMISSION_RECEIPT_OUTPUT:?}"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -f "$GLM53_PREVIEW_RECEIPT" && test ! -L "$GLM53_PREVIEW_RECEIPT"
test -f "$GLM53_DUPLICATE_RECEIPT" && test ! -L "$GLM53_DUPLICATE_RECEIPT"
test -f "$GLM53_PREBULK_TERMINAL" && test ! -L "$GLM53_PREBULK_TERMINAL"
test ! -e "$GLM53_SUBMISSION_RECEIPT_OUTPUT" && test ! -L "$GLM53_SUBMISSION_RECEIPT_OUTPUT"

uv run python -m evals.fleet.glm53_dedicated_v7 validate-preview \
  --replica "$REPLICA" --receipt "$GLM53_PREVIEW_RECEIPT" >/dev/null
uv run python -m evals.fleet.glm53_dedicated_v7 validate-duplicate \
  --replica "$REPLICA" --receipt "$GLM53_DUPLICATE_RECEIPT" >/dev/null
uv run python - "$GLM53_PREBULK_TERMINAL" <<'PY'
import sys
from pathlib import Path
from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as prior
from evals.fleet import exact_pass4_prebulk_reconciliation_v4 as reconcile

reconcile.validate_terminal(prior._load(Path(sys.argv[1])), Path.cwd())  # noqa: SLF001
PY

if [[ "$REPLICA" == B ]]; then
  : "${GLM53_REPLICA_A_PARITY_RECEIPT:?}" "${GLM53_REPLICA_A_CANARY_RECEIPT:?}"
  : "${GLM53_REPLICA_A_RUNTIME_RECEIPT:?}"
  uv run python -m evals.fleet.glm53_dedicated_v7 validate-runtime --replica A \
    --receipt "$GLM53_REPLICA_A_RUNTIME_RECEIPT" \
    --parity "$GLM53_REPLICA_A_PARITY_RECEIPT" \
    --canary "$GLM53_REPLICA_A_CANARY_RECEIPT" >/dev/null
fi

title=$(uv run python - <<PY
from pathlib import Path
from evals.fleet import glm53_dedicated_v7 as v7
print(v7.spec(Path.cwd())["replicas"]["$REPLICA"]["title"])
PY
)
run_dir=$(uv run python - <<PY
from pathlib import Path
from evals.fleet import glm53_dedicated_v7 as v7
print(v7.spec(Path.cwd())["replicas"]["$REPLICA"]["run_dir"])
PY
)
test -z "$(kubectl -n fleet-train-jobs get rayjobs.ray.io -o name 2>/dev/null | grep -F -- "$title" || true)"
test -z "$(kubectl -n fleet-train-jobs get jobs,pods,services -o name | grep -F -- "$title" || true)"
test ! -e "$run_dir" && test ! -L "$run_dir"

payload=$(mktemp)
trap 'rm -f -- "$payload"' EXIT
uv run python -m evals.fleet.glm53_dedicated_v7 server-preview --replica "$REPLICA" >"$payload"
REPLICA="$REPLICA" PAYLOAD="$payload" OUTPUT="$GLM53_SUBMISSION_RECEIPT_OUTPUT" \
  uv run --with httpx==0.28.1 python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

import httpx

from evals.fleet import self_hosted

payload = json.loads(Path(os.environ["PAYLOAD"]).read_text())
token = subprocess.run(
    ["gh", "auth", "token", "--hostname", "github.com"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
with httpx.Client(base_url="https://api.ft.flt.build", headers=headers, timeout=60) as client:
    offset = 0
    while True:
        page_response = client.get("/v1/runs", params={"limit": 200, "offset": offset})
        page_response.raise_for_status()
        page = page_response.json()
        if any(item.get("run_dir") == payload["run_dir"] for item in page["items"]):
            raise SystemExit("Jobs API create-once run directory is not absent")
        if not page["has_more"]:
            break
        offset += page["limit"]
    response = client.post("/v1/runs", json=payload)
    response.raise_for_status()
    if response.status_code != 202:
        raise SystemExit("Jobs API submit did not return 202")
    value = response.json()
run_id = value.get("name")
if not isinstance(run_id, str) or not run_id.startswith("ft-run-"):
    raise SystemExit("Jobs API response omitted exact run id")
receipt = {
    "schema_version": "fleet-glm53-dedicated-serving-v7-submission-v1",
    "status": "SUBMITTED",
    "replica": os.environ["REPLICA"],
    "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
    "api_run_id": run_id,
    "api_run_uid_bound_by_parity": True,
    "title": payload["title"],
    "run_dir": payload["run_dir"],
    "route": "POST /v1/runs",
    "http_status": 202,
    "credentials_included": False,
}
receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
output = Path(os.environ["OUTPUT"])
output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
output.chmod(0o400)
print(json.dumps({"api_run_id": run_id, "status": "SUBMITTED"}, sort_keys=True))
PY
