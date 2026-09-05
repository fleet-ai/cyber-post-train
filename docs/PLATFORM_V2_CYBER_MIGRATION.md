# Platform v2 migration for the frozen Fleet cyber cohort

This runbook migrates the exact 160-task cohort selected by source Job
`a62dd51f-a52b-4941-8207-4679e4b25b51` into Registry alpha. It is an
artifact-migration procedure, not authorization to publish tasks or launch
scored evaluations.

## Frozen identities

- Selection: `configs/data/fleet-a62-task-split-v1.json`
- File SHA-256:
  `ab444cb664ffb4da351e4d33298cc052f848f2833487807bf984a753fa7c3b47`
- Manifest digest:
  `sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a`
- Membership: 160 unique task versions: 130 train, 10 dev, and 20 test
- Import Repository: `gentle-ember-ledger/fleet-cyber-a62-frozen`
- Final composition identities:
  `all160-fb09668f-v1`, `train130-fb09668f-v1`,
  `dev10-fb09668f-v1`, and `test20-fb09668f-v1`

The complete content-free composition contract is
`evals/platform_v2/fleet_a62_compositions_v1.json`. Do not substitute a
mutable `latest` tag or a task's current catalog pointer.

## Current hold and pending exact protocol

There are two deliberately different planning modes:

- Discovery protocol `fleet.legacy-task-import.v2` is diagnostics-only. It
  selects by task key and therefore follows a mutable current pointer. The most
  recent reconciliation found 14 frozen versions that no longer equal current,
  and its TaskDump topology omits one environment shape used by two rows. The
  controller reports those facts but refuses every write.
- Pending discovery protocol `fleet.legacy-task-import.v3` accepts a
  `source_selection` with schema `fleet.taskdump.selection.v1`. It selects the
  frozen source row directly and supports the previously omitted environment
  topology. Only this protocol can produce write-admissible rows.

Do not publish while discovery still reports v2. Do not publish only the
apparently eligible rows and later call the mixed result the frozen cohort.
After v3 is deployed, regenerate the plan from the exact manifest; do not
upgrade or hand-edit a v2 plan.

Every v3 `source_selection` binds all nine fields below:

```text
schema, task_key, task_version_id, team_id, environment_version_id,
env_key, env_version, data_key, data_version
```

Registry returns both that complete selection and its digest. The controller
requires both to match the request in the create receipt and every later status
read. The digest is the SHA-256 of the compact JSON object in the server's
declared field order; it is distinct from the request, plan, and published
TaskSet digests.

## Credentials and private output

Authenticate to Registry alpha through the normal Fleet Registry login. When
discovery reports v2, make `FLEET_API_KEY` available as an environment secret
for the read-only current-pointer diagnostic. V3 planning does not read the
current pointer and does not need that credential. Never put either credential
in a command argument, plan, journal, commit, or chat transcript.

Keep the plan and append-only receipt journal in a private operator directory:

```sh
PLATFORM_V2_RUN_DIR=$(mktemp -d)
chmod 700 "$PLATFORM_V2_RUN_DIR"
```

The controller emits identities, states, counts, and digests only. It must not
copy task content, prompts, traces, flags, scores, or API error bodies into
operator artifacts.

## 1. Generate the read-only plan

Run from the repository root:

```sh
uv run python -m evals.platform_v2.legacy_import \
  --namespace gentle-ember-ledger \
  --repository fleet-cyber-a62-frozen \
  --plan-output "$PLATFORM_V2_RUN_DIR/import-plan.json"
```

This does not create a Repository or an import. Review the create-once plan and
record its digest.

If discovery reports v2, the output is a diagnostic snapshot with
`eligible_current_equals_frozen`, `blocked_current_differs_from_frozen`, and
`blocked_deployed_topology_omits_environment` counts. It is never valid input
to submission, even if all mutable pointers happened to agree.

If discovery reports v3, submission remains held unless the plan has exactly:

```text
total = 160
eligible_exact_frozen_version = 160
registry_protocol = fleet.legacy-task-import.v3
deployed_exact_task_version_selector = true
```

Every v3 row must have a request digest and source-selection digest derived
from the frozen manifest. Its `current_task_version_id` is intentionally null:
current-pointer equality is not an authority in v3.

## 2. Import one create-once canary

Only after the full plan passes and publication is explicitly approved, submit
one row from that reviewed plan:

```sh
uv run python -m evals.platform_v2.legacy_import \
  --namespace gentle-ember-ledger \
  --repository fleet-cyber-a62-frozen \
  --plan-input "$PLATFORM_V2_RUN_DIR/import-plan.json" \
  --journal "$PLATFORM_V2_RUN_DIR/import-receipts.jsonl" \
  --limit 1 \
  --submit
```

Monitor the same durable journal:

```sh
uv run python -m evals.platform_v2.legacy_import \
  --journal "$PLATFORM_V2_RUN_DIR/import-receipts.jsonl" \
  --monitor
```

Proceed only when the canary is exclusively `published` and verification binds
the returned source selection and digest, requested tag, one exact task digest,
prepared Environment, detached bare Environment, and immutable image lineage.
`failed`, `needs_input`, identity drift, or conflicting evidence is a stop, not
a reason to mint another import.

## 3. Import and reconcile all 160 rows

Reuse the same reviewed plan and journal without `--limit`. Idempotency is
defined by the frozen task-version identity, and the journal prevents a
completed row from being posted twice.

```sh
uv run python -m evals.platform_v2.legacy_import \
  --namespace gentle-ember-ledger \
  --repository fleet-cyber-a62-frozen \
  --plan-input "$PLATFORM_V2_RUN_DIR/import-plan.json" \
  --journal "$PLATFORM_V2_RUN_DIR/import-receipts.jsonl" \
  --submit
```

Do not compose until all 160 imports are exclusively `published`, every
published tag resolves to the root in its receipt, and every root contains
exactly its expected task-version digest. Unknown or conflicting rows block
composition.

## 4. Preview and publish the four final compositions

Build each Registry composition from the published one-task tags according to
`evals/platform_v2/fleet_a62_compositions_v1.json`. First call the Registry's
`/api/v1/taskset-compositions/preview` route. The preview must fill and bind each
source's expected TaskSet root, tag revision, and task digest. Submit only the
byte-equivalent pinned `request` returned by that preview to
`/api/v1/taskset-compositions/publish`.

Publish in this order: `train130`, `dev10`, `test20`, then `all160`. Require
the exact task count and membership digest from the composition spec after
every publication. Keep `advance_latest=false`; the four immutable tags are
the experiment inputs.

## 5. Handoff to evaluation

An evaluation may reference a final composition only after its immutable root
digest and Registry reference are recorded in the experiment plan. Preserve
train/dev/test boundaries: training consumes `train130`; development may use
`dev10`; `test20` stays sealed evaluation-only. Platform publication alone
does not authorize a model run or make a benchmark scientifically valid.
