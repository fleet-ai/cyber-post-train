#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
EXPECTED_CONTEXT=nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6
NAMESPACE=fleet-train-jobs
MODE=${1:?usage: $0 runtime-core SNAPSHOT_DIR EXPORT_REQUEST | runtime-submitter SNAPSHOT_DIR EXPORT_REQUEST | terminal-capture SNAPSHOT_DIR EXPORT_REQUEST | terminal SNAPSHOT_DIR EXPORT_REQUEST OUTPUT}
SNAPSHOT_DIR=${2:?snapshot directory is required}
EXPORT_REQUEST=${3:?export request receipt is required}

test "$(kubectl config current-context)" = "$EXPECTED_CONTEXT"
test -s "$EXPORT_REQUEST"
umask 077

IFS=$'\t' read -r RUN_NAME RAYJOB_UID < <(
  python3 - "$EXPORT_REQUEST" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
assert value["schema"] == "cyber_sft_zero_step_export_request_v1"
print(value["export_run"]["name"], value["export_run"]["rayjob_uid"], sep="\t")
PY
)
test -n "$RUN_NAME" && test -n "$RAYJOB_UID"

case "$MODE" in
  runtime-core)
    test "$#" = 3
    if test -e "$SNAPSHOT_DIR" || test -L "$SNAPSHOT_DIR"; then
      echo "refusing pre-existing runtime snapshot path: $SNAPSHOT_DIR" >&2
      exit 1
    fi
    mkdir -m 0700 -- "$SNAPSHOT_DIR"
    kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o json > "$SNAPSHOT_DIR/rayjob-runtime.json"
    test "$(kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o jsonpath='{.metadata.uid}')" = "$RAYJOB_UID"
    CLUSTER=$(kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o jsonpath='{.status.rayClusterName}')
    test -n "$CLUSTER"
    kubectl -n "$NAMESPACE" get raycluster "$CLUSTER" -o json \
      > "$SNAPSHOT_DIR/raycluster-runtime.json"
    kubectl -n "$NAMESPACE" get pods -l "ray.io/cluster=$CLUSTER,ray.io/node-type=head" \
      -o json > "$SNAPSHOT_DIR/head-pods.json"
    python3 - \
      "$SNAPSHOT_DIR/head-pods.json" "$SNAPSHOT_DIR/runtime-pod.json" <<'PY'
import json
import os
import sys

value = json.load(open(sys.argv[1]))
items = value.get("items")
assert isinstance(items, list) and len(items) == 1, "expected exactly one Ray head Pod"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
fd = os.open(sys.argv[2], flags, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump(items[0], stream, sort_keys=True)
    stream.write("\n")
PY
    rm -- "$SNAPSHOT_DIR/head-pods.json"
    echo "captured read-only RayCluster and head Pod identity at $SNAPSHOT_DIR"
    ;;
  runtime-submitter)
    test "$#" = 3
    test -s "$SNAPSHOT_DIR/rayjob-runtime.json"
    test -s "$SNAPSHOT_DIR/raycluster-runtime.json"
    test -s "$SNAPSHOT_DIR/runtime-pod.json"
    for path in "$SNAPSHOT_DIR/submitter-job.json" "$SNAPSHOT_DIR/submitter-pod.json"; do
      if test -e "$path" || test -L "$path"; then
        echo "refusing pre-existing runtime snapshot file: $path" >&2
        exit 1
      fi
    done
    kubectl -n "$NAMESPACE" get job "$RUN_NAME" -o json \
      > "$SNAPSHOT_DIR/submitter-job.json"
    kubectl -n "$NAMESPACE" get pods -l "batch.kubernetes.io/job-name=$RUN_NAME" \
      -o json > "$SNAPSHOT_DIR/submitter-pods.json"
    python3 - "$SNAPSHOT_DIR/submitter-pods.json" "$SNAPSHOT_DIR/submitter-pod.json" <<'PY'
import json
import os
import sys

value = json.load(open(sys.argv[1]))
items = value.get("items")
assert isinstance(items, list) and len(items) == 1, "expected exactly one submitter Pod"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
fd = os.open(sys.argv[2], flags, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump(items[0], stream, sort_keys=True)
    stream.write("\n")
PY
    rm -- "$SNAPSHOT_DIR/submitter-pods.json"
    echo "captured read-only submitter identity at $SNAPSHOT_DIR"
    ;;
  terminal-capture)
    test "$#" = 3
    test -s "$SNAPSHOT_DIR/runtime-pod.json"
    test -s "$SNAPSHOT_DIR/raycluster-runtime.json"
    test -s "$SNAPSHOT_DIR/submitter-job.json"
    test -s "$SNAPSHOT_DIR/submitter-pod.json"
    if test -e "$SNAPSHOT_DIR/terminal" || test -L "$SNAPSHOT_DIR/terminal"; then
      echo "refusing pre-existing terminal snapshot path: $SNAPSHOT_DIR/terminal" >&2
      exit 1
    fi
    : "${FLEET_TRAINING_API_TOKEN:?FLEET_TRAINING_API_TOKEN must be set}"
    TEMP_DIR=$(mktemp -d "$SNAPSHOT_DIR/.terminal.XXXXXX")
    trap 'rm -rf -- "$TEMP_DIR"' EXIT
    kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o json > "$TEMP_DIR/rayjob-terminal.json"
    test "$(kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o jsonpath='{.metadata.uid}')" = "$RAYJOB_UID"
    test "$(kubectl -n "$NAMESPACE" get rayjob "$RUN_NAME" -o jsonpath='{.status.jobStatus}')" = SUCCEEDED
    CURL_CONFIG="$TEMP_DIR/curl-auth.cfg"
    case "$FLEET_TRAINING_API_TOKEN" in *$'\n'*|*'"'*) echo "invalid API token encoding" >&2; exit 1;; esac
    printf 'header = "Authorization: Bearer %s"\n' "$FLEET_TRAINING_API_TOKEN" > "$CURL_CONFIG"
    chmod 0600 "$CURL_CONFIG"
    curl --fail --silent --show-error --config "$CURL_CONFIG" \
      "https://api.ft.flt.build/v1/rl/run/$RUN_NAME" > "$TEMP_DIR/api-run.json"
    unset FLEET_TRAINING_API_TOKEN
    IFS=$'\t' read -r SUBMITTER_POD SUBMITTER_POD_UID < <(
      python3 - "$SNAPSHOT_DIR/submitter-pod.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
print(value["metadata"]["name"], value["metadata"]["uid"], sep="\t")
PY
    )
    test "$(kubectl -n "$NAMESPACE" get pod "$SUBMITTER_POD" -o jsonpath='{.metadata.uid}')" = \
      "$SUBMITTER_POD_UID"
    kubectl -n "$NAMESPACE" logs "pod/$SUBMITTER_POD" -c ray-job-submitter \
      > "$TEMP_DIR/driver.log"
    test "$(kubectl -n "$NAMESPACE" get pod "$SUBMITTER_POD" -o jsonpath='{.metadata.uid}')" = \
      "$SUBMITTER_POD_UID"
    chmod 0600 "$TEMP_DIR/rayjob-terminal.json" "$TEMP_DIR/api-run.json" "$TEMP_DIR/driver.log"
    mkdir -m 0700 -- "$SNAPSHOT_DIR/terminal"
    for name in rayjob-terminal.json api-run.json driver.log; do
      ln -- "$TEMP_DIR/$name" "$SNAPSHOT_DIR/terminal/$name"
    done
    echo "captured immutable terminal inputs at $SNAPSHOT_DIR/terminal"
    ;;
  terminal)
    test "$#" = 4
    OUTPUT=$4
    test -s "$SNAPSHOT_DIR/runtime-pod.json"
    test -s "$SNAPSHOT_DIR/raycluster-runtime.json"
    test -s "$SNAPSHOT_DIR/submitter-job.json"
    test -s "$SNAPSHOT_DIR/submitter-pod.json"
    if test -e "$OUTPUT" || test -L "$OUTPUT"; then
      echo "refusing pre-existing terminal observation path: $OUTPUT" >&2
      exit 1
    fi
    test -s "$SNAPSHOT_DIR/terminal/rayjob-terminal.json"
    test -s "$SNAPSHOT_DIR/terminal/api-run.json"
    test -s "$SNAPSHOT_DIR/terminal/driver.log"
    PYTHONPATH="$ROOT" uv run python -m training.post_sft_cli collect-export-run \
      --export-request "$EXPORT_REQUEST" \
      --terminal-rayjob "$SNAPSHOT_DIR/terminal/rayjob-terminal.json" \
      --runtime-raycluster "$SNAPSHOT_DIR/raycluster-runtime.json" \
      --runtime-pod "$SNAPSHOT_DIR/runtime-pod.json" \
      --submitter-job "$SNAPSHOT_DIR/submitter-job.json" \
      --submitter-pod "$SNAPSHOT_DIR/submitter-pod.json" \
      --api-run "$SNAPSHOT_DIR/terminal/api-run.json" \
      --driver-log "$SNAPSHOT_DIR/terminal/driver.log" \
      --output "$OUTPUT"
    ;;
  *)
    echo "usage: $0 runtime-core SNAPSHOT_DIR EXPORT_REQUEST | runtime-submitter SNAPSHOT_DIR EXPORT_REQUEST | terminal-capture SNAPSHOT_DIR EXPORT_REQUEST | terminal SNAPSHOT_DIR EXPORT_REQUEST OUTPUT" >&2
    exit 2
    ;;
esac
