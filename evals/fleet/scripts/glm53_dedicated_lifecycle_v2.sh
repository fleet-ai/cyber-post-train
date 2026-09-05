#!/usr/bin/env bash
set -euo pipefail

# Future-only GLM TP8 lifecycle. It is not bound to or authorized by v8.
: "${GLM53_RUN_DIR:?GLM53_RUN_DIR is required}"
: "${GLM53_REPLICA:?GLM53_REPLICA is required}"

LIFECYCLE_DIR=$GLM53_RUN_DIR/lifecycle
INVENTORY_TMP=$LIFECYCLE_DIR/.gpu-inventory.csv
PREFLIGHT_FILE=$LIFECYCLE_DIR/HARDWARE-PREFLIGHT.json
EXIT_FILE=$LIFECYCLE_DIR/SERVER-EXIT.json
READY_FILE=$LIFECYCLE_DIR/READY
IDLE_FILE=$LIFECYCLE_DIR/IDLE-TIMEOUT
DRAIN_FILE=$LIFECYCLE_DIR/DRAIN
HEARTBEAT_A=$LIFECYCLE_DIR/traffic-stream-1
HEARTBEAT_B=$LIFECYCLE_DIR/traffic-stream-2
IDLE_SECONDS=600
POLL_SECONDS=5

mkdir -p "$LIFECYCLE_DIR"
chmod 0700 "$LIFECYCLE_DIR"
umask 077

nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
  --format=csv,noheader,nounits >"$INVENTORY_TMP"
python3 - "$INVENTORY_TMP" "$PREFLIGHT_FILE" <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
rows = []
with source.open(newline="") as handle:
    for raw in csv.reader(handle):
        if len(raw) != 4:
            raise SystemExit("GPU inventory row shape drifted")
        index, name, memory_mib, compute_capability = (item.strip() for item in raw)
        rows.append({
            "index": int(index),
            "name": name,
            "memory_mib": int(memory_mib),
            "compute_capability": compute_capability,
        })
valid = (
    len(rows) == 8
    and [row["index"] for row in rows] == list(range(8))
    and {row["name"] for row in rows} == {"NVIDIA B300 SXM6 PC"}
    and {row["memory_mib"] for row in rows} == {275040}
    and {row["compute_capability"] for row in rows} == {"10.3"}
)
body = {
    "schema_version": "fleet-glm53-dedicated-hardware-preflight-v1",
    "status": "PASSED" if valid else "FAILED",
    "gpu_count": len(rows),
    "products": sorted({row["name"] for row in rows}),
    "memory_mib": [row["memory_mib"] for row in rows],
    "compute_capabilities": sorted({row["compute_capability"] for row in rows}),
}
body["receipt_sha256"] = "sha256:" + hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
os.replace(temporary, destination)
source.unlink(missing_ok=True)
if not valid:
    raise SystemExit(78)
PY

server_pid=
stop_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    wait "$server_pid" || true
  fi
}
trap stop_server EXIT INT TERM

"$@" &
server_pid=$!

while kill -0 "$server_pid" 2>/dev/null; do
  if python3 - <<'PY'
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except (OSError, urllib.error.URLError):
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep "$POLL_SECONDS"
done

if ! kill -0 "$server_pid" 2>/dev/null; then
  set +e
  wait "$server_pid"
  status=$?
  set -e
  SERVER_EXIT_CODE=$status python3 - "$EXIT_FILE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

destination = Path(sys.argv[1])
body = {
    "schema_version": "fleet-glm53-dedicated-server-exit-v1",
    "status": "FAILED_BEFORE_READINESS",
    "server_exit_code": int(os.environ["SERVER_EXIT_CODE"]),
    "ready": False,
}
body["receipt_sha256"] = "sha256:" + hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
os.replace(temporary, destination)
PY
  server_pid=
  exit "$status"
fi

ready_at=$(date +%s)
printf '%s\n' "$ready_at" >"$READY_FILE"

while kill -0 "$server_pid" 2>/dev/null; do
  if [[ -f "$DRAIN_FILE" && ! -L "$DRAIN_FILE" ]]; then
    stop_server
    server_pid=
    exit 0
  fi
  now=$(date +%s)
  newest=$ready_at
  for heartbeat in "$HEARTBEAT_A" "$HEARTBEAT_B"; do
    if [[ -f "$heartbeat" && ! -L "$heartbeat" ]]; then
      modified=$(stat -c %Y "$heartbeat")
      if (( modified > now )); then
        modified=$now
      fi
      if (( modified > newest )); then
        newest=$modified
      fi
    fi
  done
  age=$((now - newest))
  if (( age >= IDLE_SECONDS )); then
    printf '%s\n' "$now" >"$IDLE_FILE"
    stop_server
    server_pid=
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

set +e
wait "$server_pid"
status=$?
set -e
server_pid=
exit "$status"
