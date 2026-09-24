#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
: "${ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}"
: "${DOCKER_BIND_ROOT:?DOCKER_BIND_ROOT is required}"
: "${EVAL_CONFIG_NAME:?EVAL_CONFIG_NAME is required}"
: "${EVAL_TASK_SET_NAME:?EVAL_TASK_SET_NAME is required}"
: "${EVAL_OUTPUT:?EVAL_OUTPUT is required}"
: "${EVAL_DATABASE:?EVAL_DATABASE is required}"

root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet" "$root/configs/evaluation"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"

install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in \
  cluster_entry.py evaluate.py exact_pass4_crypto.py exact_pass4_universe.py \
  fixed_proxy.py model_artifact.py model_artifact_v2.py model_artifact_v3.py \
  opencode_self_hosted.py rollout_campaign.py rollout_ledger.py \
  rollout_postgres.py rollout_worker.py; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
install -m 0644 /bootstrap/config.json "$root/configs/evaluation/$EVAL_CONFIG_NAME"
install -m 0644 /bootstrap/task-set.json \
  "$root/configs/evaluation/$EVAL_TASK_SET_NAME"
test -r "$root/configs/evaluation/$EVAL_TASK_SET_NAME"

for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null

cd "$root"
cmp --silent /bootstrap/config.json "/bootstrap/$EVAL_CONFIG_NAME"
artifact_args=()
if [[ -e /bootstrap/model-artifact.json || -e /bootstrap/model-artifact-acceptance.json ]]; then
  [[ -f /bootstrap/model-artifact.json && -f /bootstrap/model-artifact-acceptance.json ]]
  artifact_packet="$root/configs/evaluation/model-artifact.json"
  artifact_acceptance="$root/configs/evaluation/model-artifact-acceptance.json"
  install -m 0600 /bootstrap/model-artifact.json "$artifact_packet"
  install -m 0600 /bootstrap/model-artifact-acceptance.json "$artifact_acceptance"
  artifact_args=(
    --model-artifact-binding "$artifact_packet"
    --model-artifact-acceptance "$artifact_acceptance"
  )
fi
exec uv run --no-project \
  --with httpx==0.28.1 \
  --with pyyaml==6.0.3 \
  --with 'psycopg[binary]==3.3.5' \
  python -m evals.fleet.cluster_entry \
  --config "configs/evaluation/$EVAL_CONFIG_NAME" \
  --output "$EVAL_OUTPUT" \
  --database "$EVAL_DATABASE" \
  --harness-tar \
    /mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/opencode-1.18.27-linux-amd64.tar \
  --harness-receipt /mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/BUILD.json \
  --harness-receipt-sha256 \
    sha256:4781514a8431d6d87c671c49120639062f5a8e3afb016a04360752c896181745 \
  "${artifact_args[@]}"
