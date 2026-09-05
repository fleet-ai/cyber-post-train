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
HEARTBEAT_A=$LIFECYCLE_DIR/traffic-stream-1
HEARTBEAT_B=$LIFECYCLE_DIR/traffic-stream-2
IDLE_SECONDS=600
POLL_SECONDS=5

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
      # A controller clock ahead of the serving node must not extend the
      # ten-minute idle lease beyond ten minutes on the serving node.
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
