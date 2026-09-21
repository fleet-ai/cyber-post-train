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

### SFS-only rebind alternative (design only)

The operator machine may not have `/mnt/sfs` mounted. In that case, the local
rebind above cannot be improvised by copying private inputs through a laptop.
The safe alternative is a **new, separate, zero-GPU rebind Job**. It is not
implemented or authorized by this document; no existing data-stage Job may be
repurposed for it.

The implementation must use a fresh create-once name such as
`chris-q38-prod9-rebind-v1`, not the prod9 data-stage or preflight name. Its
sealed plan must bind all of the following before any create:

1. the sealed prod9 identity digest, the exact prod8 **input** directory, and
   the empty prod9 destination directory;
2. the exact pinned runtime image and source bundle; a fixed five-file input
   inventory; and the rule that only `run_id` plus its row checksum may change;
3. a root Kubernetes `Job` with
   `metadata.annotations["fleet.ai/failure-alerts"] == "off"`, `c1` priority,
   zero GPU requests and limits, the SFS PVC, and the same non-root runtime
   user as the training reader; and
4. a server preview in every intended cluster plus a UID-bound zero-GPU cleanup
   observer before the single create.

Inside that Job, the rebind code must reject a present destination or temporary
sibling, read only the predecessor input package on SFS, write into a private
temporary sibling, rehash every result, and atomically rename it into the new
destination. It must never touch a prod8 output/checkpoint, a task instance,
or a GPU. Its terminal receipt may contain only file counts, byte counts,
digests, paths, the source/destination identity bindings, and release state;
it must not contain rows, prompts, responses, flags, credentials, or scores.

One small supporting change is still required before this design can be used:
the rebind Job must emit a sanitized public-manifest receipt sufficient for the
following preflight to compile and verify prod9 without the operator reading
the private SFS rows. That receipt must be a new schema and a new testable
input to the plan compiler. Until it exists, fail closed rather than staging an
archive or a hand-written manifest from an unmounted filesystem.

## 2. What source is ready now, and what is intentionally not yet runnable

The current source has three fresh prod9-only pieces:

1. `training.skyrl_prod9_training` builds the exact GPU bundle and the matching
   CPU-preflight bundle. Both use the fresh rollout module.
2. `training.skyrl_prod9_rollout.Generator` constructs
   `training.skyrl_prod9_hardening.Recorder` directly. The finished batch
   receipt names that recorder, so a later acceptance check can prove it ran.
3. `training.skyrl_prod9_direct` can render (but cannot create) the exact
   one-node/eight-GPU root RayJob and the zero-GPU CPU-preflight Job. Both have
   the required root `fleet.ai/failure-alerts: "off"` annotation before any
   server preview.

The old `training.skyrl_reward_rayjob` direct rail is deliberately rejected for
this plan. It was built around the historical runtime bundle and must never be
used to render, preview, or create prod9. The fresh renderer also deliberately
has no `create` or `submit` function: its `live_create_is_available()` result is
currently `False`.

That is a safety boundary, not a pause in the scientific design. It prevents a
new bundle from accidentally being launched through old job code while the
remaining SFS-only rebind and create-once evidence gates are made explicit.

## 3. Exact next gates before any workload can be created

The next implementation must stay on the fresh prod9 rail and produce these
separate, sanitized proofs in order:

1. a fresh zero-GPU SFS rebind Job, as described above, which publishes the
   exact prod9 input package without exposing task rows;
2. a fresh zero-GPU CPU-preflight Job produced by
   `skyrl_prod9_direct.preflight_job_manifest`, server-previewed with the root
   alert annotation present and then observed to release its exact UID;
3. fresh Jobs-API and Kubernetes absence checks for the prod9 name, output
   directory, and the CPU Job names;
4. fresh server previews of the prod9 root RayJob in every intended cluster.
   `skyrl_prod9_direct.validate_preview` must show the exact image, one node,
   eight GPUs, `c1`/`q1`, the fresh bundle entrypoint, and the root annotation;
5. immediately before a future one-time GPU create, a new all-namespace
   project-capacity census. `skyrl_prod9_hardening.capacity_gate` includes the
   planned one node/eight GPUs and rejects a stale, incomplete, or over-budget
   result; and
6. a separately reviewed create-once function that consumes only those exact
   proofs, arms an exact-UID cleanup observer, records a no-retry create intent,
   and never falls back to the prod8 rail.

Until all six exist and are reviewed, no GPU workload is authorized. If a gate
fails, preserve its sanitized receipt, release only a known owned zero-GPU
object if one exists, and make a new successor identity rather than replaying
prod8 or partially reusing prod9.

## 4. A finished training process is not an accepted model

`NATIVE_TRAINING_COMPLETE.json` only says that the native training process
returned. It is **not** permission to call the resulting checkpoint valid,
evaluate it, serve it, or start a successor from it.

The only terminal acceptance marker is
`/mnt/sfs/jobs/chris-q38-rlreward-prod9/ACCEPTED.json`. It may be written once
by `training.skyrl_prod9_hardening.accept_terminal`; no shell command or
hand-written receipt is a substitute. The historical `skyrl_posttrain` module
can seal/export a checkpoint, but it cannot by itself accept prod9. The fresh
prod9 function accepts only these fixed paths for the exact planned final step:

| Evidence | Required fixed path | What it proves |
| --- | --- | --- |
| Checkpoint seal | `.../checkpoint-seals-v1/step-<final-step>.json` | The final checkpoint is complete, changed model parameters, and has not changed since it was sealed. |
| BF16 export | `.../hf-export-step<final-step>-v1/EXPORT.json` | Every model tensor and required sidecar was rebuilt and re-opened as BF16 from that seal. |
| One-GPU reload check | `...-p<final-step>-reload-v1/GPU_CHECK.json` | The exact export loads with the model configuration and tokenizer, produces finite output, and performs no optimizer update. |
| Cleanup observer | `...-p<final-step>-reload-v1/OBSERVER_RESULT.json` | The exact one-GPU reload RayJob succeeded without a restart and its RayJob, Ray cluster, workload, Pod, and GPU allocation were all released. |

The reload Job is a separate, future one-GPU operation. Its final server
preview must prove the root `RayJob` annotation
`fleet.ai/failure-alerts: "off"`, the fixed `c1`/`q1` priority, and exactly one
GPU before it may be created. Its observer must be armed against the exact
RayJob name and manifest before creation. The acceptance function rejects a
missing or changed seal, a non-BF16 export, a reload that did not use exactly
one GPU, a failed/restarted reload, or any unreleased resource. It also rejects
all alternate file paths, so an old receipt cannot be substituted for prod9.

## 5. Reasoning and long-context safety

This RL run does not ingest teacher reasoning. It trains only on actions and
working-memory summaries generated by the student during its own rollout. A
summary is a normal, separately recorded model action: it is used to continue
the same task and is not presented as an external answer key.

If a future SFT data lane uses reasoning, it may contain only reasoning the
model was explicitly allowed to see as an assistant message. It must never
copy a teacher's hidden reasoning, hidden analysis, private prompt, trace,
flag, credential, answer, or sealed score into a training target, W&B record,
or public receipt. Those inputs remain private even when the corresponding
task outcome is useful for training.

The fresh prod9 bundle already binds
`training.skyrl_prod9_rollout.Generator`, which constructs
`training.skyrl_prod9_hardening.Recorder` directly rather than using the frozen
prod8 recorder. It treats
`tool_result_chars` only as an input-size limit. It does **not** assume that
one character equals one model token. Before a tool result becomes part of the
next model prompt, the recorder tokenizes it with the exact Qwen tokenizer and
renders the full next prompt with the exact chat template. If the next action
would need a summary, it also proves that the summary prompt and its reserved
output fit in the configured context. If either does not fit, the episode ends
with a declared context-limit result; it does not silently cut off or discard
text.
