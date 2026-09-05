#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'usage: glm53_dedicated_v6_controller_heartbeat.sh once REPLICA STREAM | watch REPLICA STREAM PID' >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
MODE=$1
REPLICA=$2
STREAM=$3
case "$REPLICA" in
  A) RUN_DIR=/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v6 ;;
  B) RUN_DIR=/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-b-v6 ;;
  *) usage ;;
esac
case "$STREAM" in 1|2) ;; *) usage ;; esac

LIFECYCLE_DIR=$RUN_DIR/lifecycle
READY_FILE=$LIFECYCLE_DIR/READY
HEARTBEAT=$LIFECYCLE_DIR/traffic-stream-$STREAM
[[ -d "$LIFECYCLE_DIR" && ! -L "$LIFECYCLE_DIR" ]]
[[ -f "$READY_FILE" && ! -L "$READY_FILE" ]]

write_once() {
  temporary=$LIFECYCLE_DIR/.traffic-stream-$STREAM.$$
  trap 'rm -f -- "${temporary:-}"' RETURN
  umask 077
  printf '%s\n' "$(date +%s)" >"$temporary"
  mv -T "$temporary" "$HEARTBEAT"
  temporary=
  trap - RETURN
}

case "$MODE" in
  once)
    [[ $# -eq 3 ]] || usage
    write_once
    ;;
  watch)
    [[ $# -eq 4 ]] || usage
    PID=$4
    [[ "$PID" =~ ^[1-9][0-9]*$ ]]
    cleanup() { rm -f -- "$HEARTBEAT"; }
    trap cleanup EXIT INT TERM
    while kill -0 "$PID" 2>/dev/null; do
      write_once
      sleep 60
    done
    ;;
  *) usage ;;
esac
