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
- Reviewed task-ID roster:
  `configs/data/fleet-a62-task-identity-roster-v1.json`
- Roster file SHA-256:
  `d58f146e1c5cdc2ba372f13b079a13eab997ba7addd47d943f2ab9eceeec6bb9`
- Membership: 160 unique task versions: 130 train, 10 dev, and 20 test
- Import Repository: `gentle-ember-ledger/fleet-cyber-a62-frozen`
- Final composition identities:
  `all160-fb09668f-v1`, `train130-fb09668f-v1`,
  `dev10-fb09668f-v1`, and `test20-fb09668f-v1`

The complete content-free composition contract is
`evals/platform_v2/fleet_a62_compositions_v1.json`. Do not substitute a
mutable `latest` tag or a task's current catalog pointer.

## Current hold and deployed v3 contract

There are two deliberately different planning modes:

- Discovery protocol `fleet.legacy-task-import.v2` is diagnostics-only. It
  selects by task key and therefore follows a mutable current pointer. The most
  recent reconciliation found 14 frozen versions that no longer equal current,
  and its TaskDump topology omits one environment shape used by two rows. The
  controller reports those facts but refuses every write.
- Discovery protocol `fleet.legacy-task-import.v3`, as merged in Platform PR
  #905, accepts two top-level pins: `expected_eval_task_id` and
  `expected_current_task_version_id`. The latter must still equal
  `eval_tasks.current_version_id` when extraction begins. V3 prevents a later
  pointer move from silently changing the import; it does not select a
  historical non-current task version.

Do not publish while discovery still reports v2. Do not publish only the
apparently eligible rows and later call the mixed result the frozen cohort.
Regenerate a v3 plan from the exact manifest and reviewed identity roster; do
not upgrade or hand-edit a v2 plan.

The frozen split manifest carries each task key and frozen task-version UUID,
but not its stable `eval_tasks.id`. Supply the separately reviewed, content-free
160-row identity roster. The controller pins its exact raw SHA-256:

```text
sha256:d58f146e1c5cdc2ba372f13b079a13eab997ba7addd47d943f2ab9eceeec6bb9
```

It must have schema `chris.cyber.v2.source-roster.v1`, the exact source Job ID,
and a one-to-one ordered mapping of all 160 task keys to canonical
`eval_task_id` values. The roster's captured current-version values are
diagnostic only. The request always takes
`expected_current_task_version_id` from the frozen split manifest.

The create response and every status response must echo the v3 protocol and
both pins. The append-only receipt binds those values, the full request digest,
a source-identity digest, and the plan digest.

## Credentials and private output

Authenticate to Registry alpha through the normal Fleet Registry login. Make
`FLEET_API_KEY` available as an environment secret for the GET-only task-ID and
current-pointer preflight in both modes. Never put either credential in a
command argument, plan, journal, commit, or chat transcript.

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

If discovery reports v3, each row sends the reviewed stable task ID plus the
frozen task-version UUID. Submission remains held unless the plan has exactly:

```text
total = 160
eligible_exact_current_frozen = 160
registry_protocol = fleet.legacy-task-import.v3
deployed_exact_current_source_pins = true
```

The latest reviewed snapshot has only 146
`eligible_exact_current_frozen` rows and 14
`blocked_frozen_version_not_current` rows. Therefore the 160-task campaign is
still held. Importing those 14 exact historical versions requires a further
Platform contract extension; v3 from PR #905 cannot do it.

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
the returned protocol and two source pins, requested tag, one exact task digest,
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
