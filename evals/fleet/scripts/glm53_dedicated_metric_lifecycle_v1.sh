#!/usr/bin/env bash
set -euo pipefail

# This script is embedded byte-for-byte in each Jobs API request. The runtime
# image does not depend on a mutable checkout or an SFS bootstrap file.
: "${GLM53_RUN_DIR:?GLM53_RUN_DIR is required}"
: "${GLM53_REPLICA:?GLM53_REPLICA is required}"

LIFECYCLE_DIR=$GLM53_RUN_DIR/lifecycle
READY_FILE=$LIFECYCLE_DIR/READY
EXIT_FILE=$LIFECYCLE_DIR/SERVER-EXIT
IDLE_FILE=$LIFECYCLE_DIR/IDLE-TIMEOUT
DRAIN_FILE=$LIFECYCLE_DIR/DRAIN
TRAFFIC_FILE=$LIFECYCLE_DIR/MODEL-TRAFFIC.json
IDLE_SECONDS=${GLM53_IDLE_SECONDS:-600}
POLL_SECONDS=${GLM53_POLL_SECONDS:-5}

[[ $IDLE_SECONDS =~ ^[1-9][0-9]*$ ]] && (( IDLE_SECONDS <= 600 ))
[[ $POLL_SECONDS =~ ^[1-9][0-9]*$ ]] && (( POLL_SECONDS <= 5 ))

mkdir -p "$LIFECYCLE_DIR"
chmod 0700 "$LIFECYCLE_DIR"
umask 077

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

# Model loading is productive startup and is deliberately outside the idle
# budget. The ten-minute clock begins only when /health first returns 200.
while kill -0 "$server_pid" 2>/dev/null; do
  if python3 - <<'PY'
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
  then
    break
  fi
  sleep "$POLL_SECONDS"
done

if ! kill -0 "$server_pid" 2>/dev/null; then
  wait "$server_pid"
  exit $?
fi

ready_at=$(date +%s)
printf '%s\n' "$ready_at" >"$READY_FILE"

# The watchdog observes the inference server from inside the exact serving Pod.
# Health probes and controller liveness are deliberately excluded: only changes
# to SGLang's inference request/token counters extend the lease.
metrics_digest() {
  python3 - <<'PY'
import hashlib
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5) as response:
    if response.status != 200:
        raise SystemExit(1)
    text = response.read(8_000_000).decode(errors="ignore")

allowed = {
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:prompt_tokens_total",
}
rows = []
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name = line.split("{", 1)[0].split(None, 1)[0]
    if name in allowed:
        rows.append(line)
if not rows:
    raise SystemExit(1)
print("sha256:" + hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest())
PY
}

last_metrics=$(metrics_digest 2>/dev/null || true)
last_productive=$ready_at

while kill -0 "$server_pid" 2>/dev/null; do
  if [[ -f "$DRAIN_FILE" && ! -L "$DRAIN_FILE" ]]; then
    stop_server
    server_pid=
    exit 0
  fi

  now=$(date +%s)
  current_metrics=$(metrics_digest 2>/dev/null || true)
  if [[ -n $current_metrics && -n $last_metrics && $current_metrics != "$last_metrics" ]]; then
    last_productive=$now
    MODEL_TRAFFIC_DIGEST=$current_metrics MODEL_TRAFFIC_AT=$now python3 - "$TRAFFIC_FILE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

destination = Path(sys.argv[1])
if destination.is_symlink():
    raise SystemExit("traffic marker must not be a symlink")
body = {
    "schema_version": "fleet-glm53-server-local-model-traffic-v1",
    "status": "MODEL_COUNTER_ADVANCED",
    "metrics_sha256": os.environ["MODEL_TRAFFIC_DIGEST"],
    "observed_at_unix": int(os.environ["MODEL_TRAFFIC_AT"]),
    "source": "local_sglang_inference_metrics",
}
body["receipt_sha256"] = "sha256:" + hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
temporary = destination.with_name(destination.name + ".tmp")
temporary.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
os.replace(temporary, destination)
PY
  fi
  if [[ -n $current_metrics ]]; then
    last_metrics=$current_metrics
  fi
  age=$((now - last_productive))
  if (( age >= IDLE_SECONDS )); then
    printf '%s\n' "$now" >"$IDLE_FILE"
    stop_server
    server_pid=
    exit 0
  fi
  remaining=$((IDLE_SECONDS - age))
  if (( remaining < POLL_SECONDS )); then
    sleep "$remaining"
  else
    sleep "$POLL_SECONDS"
  fi
done

set +e
wait "$server_pid"
status=$?
set -e
printf '%s\n' "$status" >"$EXIT_FILE"
server_pid=
exit "$status"

