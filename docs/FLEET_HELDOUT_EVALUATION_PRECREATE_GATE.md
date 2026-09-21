# One new Fleet held-out evaluation: launch gate

This note describes the smallest safe path to launch **one new** Qwen
held-out Fleet evaluation after its model is ready. The guarded command is
implemented as `cyber-post-train eval heldout-create`; this note does not make
an old campaign, an old Job YAML, or a partially prepared packet launchable.

## Reuse the existing evaluator, not an older campaign

The worker runtime already exists in
[`evals/fleet/cluster_entry.py`](../evals/fleet/cluster_entry.py), with its
bootstrap script in
[`evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh`](../evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh).
The next campaign should make a **new** immutable config, ConfigMap, Job name,
output directory, and database name for the selected checkpoint. Do not copy
or apply an old campaign's Job YAML: its names and evidence belong to that
campaign.

The current checked-in final template is deliberately unresolved, and the
prepared seed-44 comparison is explicitly not launchable. Existing seed-43
campaigns are historical records, not reusable inputs. Therefore no new Fleet
held-out evaluation is launch-ready today.

## Guarded launch packet and command

Once a candidate has an accepted checkpoint and serving-route proof, mint one
new launch packet. The packet is a JSON file whose sibling files contain the
exact config, task selection, representative split manifest, immutable
ConfigMap, Job, checkpoint provenance, serving-route proof, and shared matched
comparison protocol. Each of those sealed files has a SHA-256 digest in the
packet. The output directory, database, matched-comparison identity, sampling,
harness, and selected arm are bound into the same identity. The mutable
evaluation ledger is included for the duplicate census but is deliberately not
treated as a sealed input.

The launcher is intentionally not a generic training launcher and not a clone
of an earlier campaign. It accepts one arm of a predeclared base-versus-
checkpoint comparison at a time; the packet must name all comparison arms and
the current arm. The shared protocol is sealed in every arm's packet. It fixes
the task selection and split, complete harness and sampling settings, evaluator
images, retry rule, and intended model revision for each arm. This is what
makes separately launched baseline and checkpoint arms a single matched
comparison rather than two merely similar runs.

## Preserve the held-out task set as it grows

The historical 75-task split is a frozen record, not a ceiling on future
evaluation. A larger representative split must be made as a **new** sealed
manifest from a separately reviewed, digest-bound task-quality inventory. That
inventory is metadata-only: it says which exact task versions are eligible and
their reviewed family labels. It must not guess that an unknown task is broken,
and it must not contain prompts, rollout content, answers, or scores. A split
must keep every exact version of the same reviewed application/task-family
together and record the intended partitions (for example, training,
development, and final test). This prevents an updated version of one task
family leaking across partitions while allowing the task population to grow.

For each new evaluation arm, the packet must include both:

- the complete sealed split manifest, including its file digest and its
  self-digest; and
- a sealed task-selection file naming one non-training role from that manifest
  and the exact task key/version pairs in that role.

The launcher's route must contain exactly those selected task versions—no more,
no fewer, and no duplicates. The same selected role and manifest must be used
for the baseline and every checkpoint being compared. A changed population,
partition policy, task family assignment, or selected role is a new experiment
identity, never an in-place update to an old evaluation.

Run it only with an explicit Kubernetes context and a **new** local journal
path:

```sh
cyber-post-train eval heldout-create \
  /safe/new-campaign/LAUNCH_PACKET.json \
  --context <fleet-cluster-context> \
  --journal /safe/new-campaign/CREATE_INTENT.jsonl
```

The command does not initialize the evaluation database, write a score, read a
rollout trace, or decide whether the model is better. Those belong to the
existing evaluator and later score-blind reconciliation.

Immediately before its one `kubectl create`, the command does these live safety
checks:

1. **Fresh duplicate check.** Read the exact new Job, ConfigMap, Kueue
   Workload, Pod, output directory, database, and evaluation ledger. Stop if
   any exists, or if the ledger already contains the same complete evaluation
   identity. This protects both against name reuse and against an accidental
   scientific duplicate.
2. **Server-rendered alert proof.** Send the exact Job to Kubernetes with a
   server dry-run, then inspect the returned root `batch/v1 Job`. Require:

   ```yaml
   metadata:
     annotations:
       fleet.ai/failure-alerts: "off"
   ```

   A source YAML value, request flag, or Pod-template annotation alone does
   not count. Stop if the returned root object lacks that exact value.

The server preview happens twice. The command removes only Kubernetes-generated
UID projections before hashing each reply and requires the two normalized
previews to agree. It then repeats the complete duplicate census, so an object
or output created during preview still blocks creation.

After both checks, write a durable create-intent record and make exactly one
create request. If its response is uncertain, re-read the exact names; never
repeat the create request blindly.

When Kubernetes reports the Job terminal, collect a separate read-only record:

```sh
cyber-post-train eval heldout-terminal-collect \
  /safe/new-campaign/LAUNCH_PACKET.json \
  --context <fleet-cluster-context> \
  --receipt /safe/new-campaign/TERMINAL_OBSERVATION.json
```

The terminal record contains only object identities, matched-protocol identity,
terminal states, output-presence, and aggregate database state counts. It never
reads prompts, responses, traces, flags, rewards, or score values; it never
retries a rollout or scores an outcome. A later evaluator-specific review must
compare the same protocol digest across arms and interpret valid capability
results.

## Required regression tests for this command

- reject an existing object or matching evaluation ledger;
- reject a server-rendered Job missing the root annotation even when its Pod
  template has it; and
- reject a task selection or evaluator route that does not exactly match its
  sealed representative split; and
- prove that a successful double dry-run is followed by one create at most; and
- prove terminal collection is score-blind and cannot retry or score.

This is deliberately narrow. It preserves the existing evaluator runtime and
adds only the two safeguards required before a new Fleet object can exist.
