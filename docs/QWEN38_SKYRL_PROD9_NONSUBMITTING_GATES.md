# Prod9: non-submitting live gates

This is the exact preparation sequence for the fresh prod9 SkyRL canary after
the successor change has merged. It intentionally contains **no command that
creates a Kubernetes object**. `kubectl create --dry-run=server` asks the
cluster to validate an object and immediately discards it; it does not queue a
Job or allocate a GPU.

The final GPU create is deliberately absent. The two CPU-only Jobs that stage
the private package and run the exact-image preflight are also deliberately
absent: each needs a separate, explicit decision to create it after these
checks pass.

All paths below are local operator paths. The command never prints a private
row, prompt, response, credential, or W&B key.

## 1. Start from the merged source and make a private local package

Set `PROD9_SOURCE` to the restricted local directory containing the earlier
**input** package only. It must not point at any prod8 output or checkpoint.
The destination must be new and empty.

```sh
export PROD9_SOURCE_REPO="$(git rev-parse --show-toplevel)"
git -C "$PROD9_SOURCE_REPO" fetch origin main
export PROD9_REPO="$(mktemp -d /private/tmp/cpt-prod9-main.XXXXXX)"
rmdir "$PROD9_REPO"
git -C "$PROD9_SOURCE_REPO" worktree add --detach "$PROD9_REPO" origin/main
cd "$PROD9_REPO"
test "$(jq -r .sha256 configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json)" \
  = "sha256:c666bbe82dfc1f084afc8233865edfd4bf1f34fbdf5f8f5f10d69a4f91fad311"

umask 077
export PROD9_GATE_DIR="$(mktemp -d /private/tmp/q38-prod9-gates.XXXXXX)"
chmod 700 "$PROD9_GATE_DIR"
export PROD9_SOURCE="/restricted/local/path/to/rlreward-inputs-prod8-v1/data"
export PROD9_REBOUND="$PROD9_GATE_DIR/rebound-data"
test -d "$PROD9_SOURCE"
test ! -e "$PROD9_REBOUND"

uv run --locked python - "$PROD9_SOURCE" "$PROD9_REBOUND" <<'PY'
import sys
from pathlib import Path
from training import skyrl_reward_rayjob as direct

root = Path.cwd()
identity = direct.load_identity(
    root / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
)
direct.rebind_private_source_for_identity(
    Path(sys.argv[1]), Path(sys.argv[2]), identity
)
PY

uv run --locked python scripts/prepare_qwen38_skyrl_prod9_successor.py \
  --manifest "$PROD9_REBOUND/manifest.json" \
  > "$PROD9_GATE_DIR/OFFLINE_PREPARATION.json"
```

This changes only the embedded run identity and its per-row checksums in a
local successor package. The offline receipt must say
`prepared_not_authorized`. It does not contact Fleet, W&B, Kubernetes, or SFS.

## 2. Generate the stage/preflight objects, inspect absence, and server-preview

This one command does only local file creation, read-only inventory calls, Jobs
API preview calls, and server dry-runs. It does **not** call the Jobs API
creation endpoint or `kubectl create` without `--dry-run=server`.

`FLEET_API_KEY` must already be available in the operator environment. Do not
put it in this document, a shell history file, or an artifact.

```sh
test -n "${FLEET_API_KEY:-}"

uv run --locked python - "$PROD9_GATE_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

from cyber_post_train.jobs import API_URLS, Jobs, digest
from scripts import prepare_qwen38_skyrl_prod9_successor as prepared
from training import skyrl_reward_rayjob as direct

gate = Path(sys.argv[1])
root = Path.cwd()
identity = direct.load_identity(
    root / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
)
manifest = json.loads((gate / "rebound-data" / "manifest.json").read_bytes())
run = prepared._load(prepared.RUN)
plan, request = prepared._compile(run, manifest)
direct._identity_for_plan(plan, identity)

def write(name, value):
    path = gate / name
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o600)
    return path

# This archive remains private and local. It is not uploaded in this command.
stage = direct.stage_plan(plan, gate / "rebound-data", gate / "stage-data.tar.gz", identity=identity)
stage_job = direct.stage_job_manifest(stage, identity=identity)
preflight_job = direct.preflight_job_manifest(plan, identity=identity)

with Jobs(os.environ["FLEET_API_KEY"], base_url=API_URLS["prod"]) as jobs:
    source_preview = jobs.raw_preview(request)
expected = direct.manifest(plan, request, source_preview, identity=identity)

proofs = {}
for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT):
    rendered_stage = direct.server_dry_run(stage_job, context=context)
    rendered_preflight = direct.server_dry_run(preflight_job, context=context)
    rendered_rayjob = direct.server_dry_run(expected, context=context)
    proofs[context] = {
        "data_stage": direct.validate_cpu_preview(
            stage_job, rendered_stage, context=context, purpose="data_stage"
        ),
        "preflight": direct.validate_cpu_preview(
            preflight_job, rendered_preflight, context=context, purpose="preflight"
        ),
        "rayjob": direct.validate_preview(
            plan, request, source_preview, expected, rendered_rayjob,
            context=context, identity=identity,
        ),
    }

# Checks every relevant resource kind in both clusters and both Jobs API
# histories. It fails if the prod9 run name or output root is already owned.
absence = direct.duplicate_checks(
    plan, token=os.environ["FLEET_API_KEY"], identity=identity
)

# The GPU identity is covered above. CPU identities are separate create-once
# names, so prove they do not already exist in either cluster too.
import subprocess
for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT):
    for resource, name in (("job", identity.stage_name), ("job", identity.preflight_name)):
        result = subprocess.run(
            ["kubectl", "--context", context, "--namespace", direct.NAMESPACE,
             "get", resource, name, "--ignore-not-found", "--output", "json"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode or result.stdout.strip():
            raise SystemExit(f"existing or unreadable {resource}/{name} in {context}")

write("PLAN.json", plan)
write("REQUEST.json", request)
write("JOBS_SOURCE_PREVIEW.json", source_preview)
write("RAYJOB.json", expected)
write("DATA_STAGE_JOB.json", stage_job)
write("PREFLIGHT_JOB.json", preflight_job)
write("STAGE_PLAN.json", stage)
write("NON_SUBMITTING_LIVE_GATE_RECEIPT.json", {
    "schema": "cyber_qwen38_skyrl_prod9_non_submitting_live_gate_v1",
    "status": "passed_not_authorized",
    "external_workloads_created": 0,
    "identity_sha256": identity.sealed_mapping()["sha256"],
    "plan_sha256": "sha256:" + digest(plan),
    "request_sha256": "sha256:" + digest(request),
    "rayjob_manifest_sha256": "sha256:" + digest(expected),
    "data_stage_manifest_sha256": "sha256:" + digest(stage_job),
    "preflight_manifest_sha256": "sha256:" + digest(preflight_job),
    "stage_plan_sha256": stage["sha256"],
    "duplicate_checks": absence,
    "server_previews": proofs,
})
PY
```

The command fails closed unless all six server previews pass: data-stage,
preflight, and final root RayJob in both dev and prod. Each proof verifies the
root `fleet.ai/failure-alerts: "off"` annotation. The RayJob proof also
verifies the one-node/eight-GPU shape, `c1`/`q1`, fixed image, and the exact
262K compaction contract.

The duplicate check covers both Kubernetes clusters and both Jobs API histories
for the fresh prod9 run/output identity; the command also checks the two fresh
CPU Job names directly in both clusters. It cannot prove an SFS directory is
absent without mounting SFS; that proof is supplied only by the controlled
zero-GPU stage/preflight sequence below. It also deliberately does not use a
W&B read as an absence gate: W&B may collapse missing-run, subscription, and
transient failures into the same error. The fresh W&B ID is instead bound to
`resume="never"` and accepted or rejected by W&B at training startup.

## 3. Controlled later steps — intentionally not executed here

Only after the receipt above passes may a separately authorized operator create
the two zero-GPU Jobs. Do not substitute a manual `kubectl create` for the
existing stage/preflight rails.

Immediately before each CPU create, start its exact observer in a separate
terminal. The observer has a 120-second creation allowance, so do **not** arm
it early. It binds the first Job UID it sees, records its terminal result and
release, and can release only that exact UID.

```sh
# Stage observer. Start immediately before the separately authorized stage create.
uv run --locked python -m training.dev_cleanup_observer \
  --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 \
  --namespace fleet-train-jobs --kind job --name chris-q38-prod9-data-v1 \
  --maximum-seconds 1800 --expected-gpus 0 \
  --plan-sha256 "$(jq -r .plan_sha256 "$PROD9_GATE_DIR/NON_SUBMITTING_LIVE_GATE_RECEIPT.json")" \
  --manifest-sha256 "$(jq -r .data_stage_manifest_sha256 "$PROD9_GATE_DIR/NON_SUBMITTING_LIVE_GATE_RECEIPT.json")" \
  --profile production-cpu \
  --armed "$PROD9_GATE_DIR/STAGE_OBSERVER_ARMED.json" \
  --result "$PROD9_GATE_DIR/STAGE_OBSERVER_RESULT.json"

# Preflight observer. Start immediately before the separately authorized preflight create.
uv run --locked python -m training.dev_cleanup_observer \
  --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 \
  --namespace fleet-train-jobs --kind job --name chris-q38-prod9-preflight-v1 \
  --maximum-seconds 1800 --expected-gpus 0 \
  --plan-sha256 "$(jq -r .plan_sha256 "$PROD9_GATE_DIR/NON_SUBMITTING_LIVE_GATE_RECEIPT.json")" \
  --manifest-sha256 "$(jq -r .preflight_manifest_sha256 "$PROD9_GATE_DIR/NON_SUBMITTING_LIVE_GATE_RECEIPT.json")" \
  --profile production-cpu \
  --armed "$PROD9_GATE_DIR/PREFLIGHT_OBSERVER_ARMED.json" \
  --result "$PROD9_GATE_DIR/PREFLIGHT_OBSERVER_RESULT.json"
```

The data-stage Job is the create-once SFS absence check: it refuses an existing
prod9 data directory before publishing the newly rebound package. The
exact-image preflight then refuses an existing prod9 output directory, reopens
the staged data, checks the parser and 262K compaction behavior, and proves it
used zero GPUs. The GPU observer is armed only after both CPU receipts are
accepted and immediately before the separate one-time RayJob create; its command
is intentionally withheld with the GPU creation path.

If any check fails, preserve the sanitized receipt, release only an exact owned
CPU object if one exists, and make a new successor identity rather than
replaying prod9.
