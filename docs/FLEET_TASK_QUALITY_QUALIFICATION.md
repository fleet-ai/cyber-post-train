# Fleet task-quality qualification waves

`evals.fleet.task_quality_qualification` turns unproven production blackbox
task versions into runtime-qualified roster candidates without asking a model
to solve them and without creating training examples.

The source inventory currently contains 1,093 production blackbox versions.
Eighty have an exact execution receipt. Of the remaining 1,013, 47 are already
known broken and are excluded, 17 are the first priority wave (7 QA-clean and
10 prior agent failures), and 949 are unanalysed. The expansion schedule is the
priority wave followed by at most fourteen 64-version waves and one 53-version
tail. These are supply bounds, not automatic admission counts.

## What one qualification proves

For every selected exact task/version, the controller:

1. binds the exact environment version, complete starting-data selection,
   verifier version/hash, reviewed Registry atom/task-graph lineage, and safe
   task-content digests; exact source locators retain their versions while the
   grouping family is the version-independent Registry artifact key;
2. rejects any candidate sharing an atom artifact key with the immutable
   17-task dev or eight-task final held-out split;
3. proves the durable create-request claim GET/DELETE routes are deployed,
   exclusively records a cell intent, provisions through the exact
   task-version authority once, and verifies the response, Fleet-team runtime,
   and durable create-request claim all bind the same environment;
4. initializes MCP, proves each advertised input schema admits the exact probe,
   then calls `bash` and `submit_report` with the runtime's `no_flag` verdict
   plus a nonempty qualification explanation;
5. invokes the v3 direct verifier authority and creates a new zero-message,
   trace-free evidence session (an already-existing session fails closed); and
6. terminates the exact environment and requires termination evidence.

A finite authoritative verifier failure is valid task-health evidence. The
qualification does not select on success, failure, or score. A task is excluded
only when environment startup, tool reachability, verifier execution, evidence
ingestion, or cleanup is infrastructure-invalid. An ambiguous mutation is
quarantined and is never replayed automatically.

No prompt, tool response, answer, flag, trace, credential, or numeric score is
written. Exact task/session/runtime identities remain in the private operation
root. `AGGREGATE_RECEIPT.json` contains counts and digests only.

## Exactly-once priority wave

Prepare is read-only against Fleet. Always regenerate it from the exact merged
source commit. The controller requires clean HEAD to be an ancestor of the
freshly fetched `origin/main`. `--allow-unmerged-preview` produces a diagnostic
plan with external mutations cryptographically disabled; `run` refuses it.

```sh
uv run --locked python -m evals.fleet.task_quality_qualification prepare \
  --inventory configs/data/fleet-blackbox-current-production-20260921-v1.json \
  --coverage configs/data/fleet-blackbox-training-coverage-20260921-v1.json \
  --protected-split configs/data/fleet-blackbox-current-study-split-20260914-v2.json \
  --wave-id priority17-v1 \
  --qa-status clean --qa-status agent_failure \
  --limit 64 --concurrency 8 \
  --private-base /private/qualification-root
```

Review the content-free prepare response and private plan. Then invoke `run`
once. `RUN_INTENT.json` is exclusive: any second invocation fails before a
Fleet mutation.

```sh
uv run --locked python -m evals.fleet.task_quality_qualification run \
  /private/qualification-root/task-quality-priority17-v1-<plan-prefix>

uv run --locked python -m evals.fleet.task_quality_qualification cleanup \
  /private/qualification-root/task-quality-priority17-v1-<plan-prefix>

uv run --locked python -m evals.fleet.task_quality_qualification status \
  /private/qualification-root/task-quality-priority17-v1-<plan-prefix>
```

The private output includes a source-supply catalog and qualified catalog in
the exact schemas consumed by `training.fleet_collection_roster`. The public
aggregate receipt is not itself permission to train: the qualified catalog
must still be combined with the trusted rooted family anchor, and any newly
allocated dev/final roles remain held out.

Cleanup is independently resumable and never reruns provisioning, tools,
verification, or ingestion. If a provisioning response is lost, it reads the
durable create-request claim by the exact request ID, cancels an accepted claim
or discovers the exact materialized instance, and then deletes only that
instance. If an exact delete response is lost, a later `cleanup` invocation
reuses the same sealed intent, reads the same instance ID, and safely reissues
only that delete. The wave is not resource-terminal until
`CLEANUP_AGGREGATE.json` records zero unresolved task versions.

### Claim-absent historical recovery

If a terminal wave predates a deployed create-claim integration and the normal
cleanup cannot resolve a provision intent, do not replay the qualification or
guess an instance ID. The separate, read-only recovery issuer can seal only a
strict independent-absence case:

```sh
git fetch origin main
uv run --locked python -m evals.fleet.task_quality_cleanup_recovery \
  /private/qualification-root/task-quality-<wave>-<plan-prefix>
```

The issuer requires the recovery code to equal freshly fetched `origin/main`,
reopens the original merged controller bytes from Git, validates the sealed
plan/run/cell/provision/terminal chain and deterministic request/run IDs,
rechecks Fleet-team authentication and the deployed claim API, and then
requires two successive observations of both a `404` create claim and an empty
authoritative instance listing for the exact run ID. It performs no external
mutation and fails on any claim, instance, binding change, nonempty list, or
partial evidence. Each successful private resolution binds its independent
absence-evidence digest. Finally, rerun the **original frozen controller's**
normal `cleanup` command so it can consume those exact resolutions and seal
`CLEANUP_AGGREGATE.json`. This issuer is an evidence-closure tool; it does not
authorize another qualification wave. Repair and re-review the live ownership
contract before any later wave.

## Continuous expansion

For each later `not_analyzed` wave, pass the preceding private
`ATTEMPTED_CATALOG.private.json` with `--exclude-catalog`, use a new immutable
wave ID, and cap the selected versions at 64. No qualified-only catalog schema
is accepted for these waves. The attempted catalog is cumulative and
excludes qualified, infrastructure-invalid, and ambiguous cells from automatic
replay. Merge the qualified catalogs only through a digest-checked private
adapter; never infer validity from inventory metadata or copy a task because a
related version passed. After each sealed anchored roster, render a new
base-Qwen/OpenCode visible-action pass@4 campaign over all admitted train
families. Student-visible reasoning and stronger-teacher visible-action data
remain separate corpus arms and never inherit qualification authority from this
runtime probe.
