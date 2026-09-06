#!/usr/bin/env bash
set -euo pipefail

: "${QWEN38_RUN_DIR:?QWEN38_RUN_DIR is required}"
: "${QWEN38_OBSERVER_SCRIPT:?QWEN38_OBSERVER_SCRIPT is required}"

LIFECYCLE_DIR=$QWEN38_RUN_DIR/lifecycle
PREFLIGHT_FILE=$LIFECYCLE_DIR/HARDWARE-PREFLIGHT.json
EXIT_FILE=$LIFECYCLE_DIR/SERVER-EXIT.json
READY_FILE=$LIFECYCLE_DIR/READY
IDLE_FILE=$LIFECYCLE_DIR/IDLE-TIMEOUT
DRAIN_FILE=$LIFECYCLE_DIR/DRAIN
TRAFFIC_FILE=$LIFECYCLE_DIR/traffic
COUNTER_STATE=$LIFECYCLE_DIR/.request-counters.json
TRAFFIC_OBSERVATION=$LIFECYCLE_DIR/REAL-TRAFFIC.json
IDLE_SECONDS=600
POLL_SECONDS=5

mkdir -p "$LIFECYCLE_DIR"
chmod 0700 "$LIFECYCLE_DIR"
umask 077

nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
  --format=csv,noheader,nounits >"$LIFECYCLE_DIR/.gpu-inventory.csv"
python3 - "$LIFECYCLE_DIR/.gpu-inventory.csv" "$PREFLIGHT_FILE" <<'PY'
import csv, hashlib, json, os, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
rows=[]
with source.open(newline="") as handle:
    for raw in csv.reader(handle):
        if len(raw) != 4: raise SystemExit("GPU inventory row shape drifted")
        index,name,memory,cap=(item.strip() for item in raw)
        rows.append({"index":int(index),"name":name,"memory_mib":int(memory),"compute_capability":cap})
valid=(len(rows)==8 and [row["index"] for row in rows]==list(range(8)) and all(row["name"]=="NVIDIA B300 SXM6 PC" for row in rows) and all(row["memory_mib"]==275040 for row in rows) and all(row["compute_capability"]=="10.3" for row in rows))
body={"schema_version":"fleet-qwen38-dedicated-dp8-hardware-preflight-v2","status":"PASSED" if valid else "FAILED","gpu_count":len(rows),"gpu_indices":[row["index"] for row in rows],"products":sorted({row["name"] for row in rows}),"memory_mib":[row["memory_mib"] for row in rows],"compute_capabilities":sorted({row["compute_capability"] for row in rows})}
body["receipt_sha256"]="sha256:"+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
tmp=destination.with_suffix(".tmp"); tmp.write_text(json.dumps(body,sort_keys=True,separators=(",",":"))+"\n"); os.replace(tmp,destination); source.unlink(missing_ok=True)
if not valid: raise SystemExit(78)
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
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=5) as response:
        raise SystemExit(0 if response.status==200 else 1)
except OSError: raise SystemExit(1)
PY
  then break; fi
  sleep "$POLL_SECONDS"
done

if ! kill -0 "$server_pid" 2>/dev/null; then
  set +e; wait "$server_pid"; status=$?; set -e
  SERVER_EXIT_CODE=$status python3 - "$EXIT_FILE" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); body={"schema_version":"fleet-qwen38-dedicated-dp8-server-exit-v2","status":"FAILED_BEFORE_READINESS","server_exit_code":int(os.environ["SERVER_EXIT_CODE"]),"ready":False}; body["receipt_sha256"]="sha256:"+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest(); p.write_text(json.dumps(body,sort_keys=True,separators=(",",":"))+"\n")
PY
  server_pid=; exit "$status"
fi

ready_at=$(date +%s)
printf '%s\n' "$ready_at" >"$READY_FILE"

while kill -0 "$server_pid" 2>/dev/null; do
  if [[ -f "$DRAIN_FILE" && ! -L "$DRAIN_FILE" ]]; then
    stop_server; server_pid=; exit 0
  fi
  python3 "$QWEN38_OBSERVER_SCRIPT" \
    --state-path "$COUNTER_STATE" \
    --receipt-path "$TRAFFIC_OBSERVATION" \
    --traffic-path "$TRAFFIC_FILE" \
    --server-run-dir "$QWEN38_RUN_DIR" || true
  now=$(date +%s)
  newest=$ready_at
  if [[ -f "$TRAFFIC_FILE" && ! -L "$TRAFFIC_FILE" ]]; then
    modified=$(stat -c %Y "$TRAFFIC_FILE")
    if (( modified > now )); then modified=$now; fi
    if (( modified > newest )); then newest=$modified; fi
  fi
  if (( now - newest >= IDLE_SECONDS )); then
    printf '%s\n' "$now" >"$IDLE_FILE"
    stop_server; server_pid=; exit 0
  fi
  sleep "$POLL_SECONDS"
done

set +e; wait "$server_pid"; status=$?; set -e
server_pid=; exit "$status"
