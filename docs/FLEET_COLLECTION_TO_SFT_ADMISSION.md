# Fleet collection admission for action SFT

## Purpose

`data-fleet-admit` is the small boundary between a reviewed Fleet rollout
campaign and later offline SFT corpus construction.  It decides which **metadata
records** may be considered for an action-only SFT corpus.  It is deliberately
not a corpus builder.

The command is local and CPU-only.  It does not create a rollout, evaluation,
training job, cluster object, or model artifact.  It does not open a session
transcript or any other source text.

This separation matters.  A rollout can be a genuine task success without
being safe or scientifically valid training material.  Admission first proves
that the session belongs to the intended collection, remains on the training
side of a family-safe split, and has the right evidence.  A later, private
corpus step must still match the admitted identifiers to normalized records and
apply the existing dense-SFT message and window rules.

## What it does not produce

The adapter never produces a training corpus.  In particular, it does **not**
write:

- `train.parquet`, JSONL training rows, packed windows, tokens, or loss masks;
- prompts, assistant text, tool input or output, task answers, flags,
  credentials, or reasoning text;
- an evaluation result, a checkpoint, or a launch request.

Its output must not be passed directly to a trainer.  A name such as
`selection.private.json` is evidence of selection, not evidence that the
selected material has passed the downstream dense-corpus checks.

## Required sealed inputs

The request has schema `cyber_fleet_collection_admission_request_v1`.  Each
file reference names a regular file and supplies its exact `sha256:<hex>` file
digest.  The adapter checks every reference before reading it and hashes every
input again immediately before publishing output.  A changed file therefore
cannot be mistaken for the reviewed admission result.

| Input | What it binds | Why it is required |
| --- | --- | --- |
| Fleet campaign plan | A self-digested `cyber_fleet_eval_v1` plan, selected source model, task/version bindings, action-tool treatment, and planned attempts | Prevents a session from another campaign, model route, or tool contract being silently mixed in.  The plan must set `training_data_eligible: true`. |
| Sanitized task catalog inventory | A self-digested inventory of reviewed task-version and lineage metadata | Supplies the complete task universe used to make the split.  It contains no session text. |
| Family-safe split | A sealed anchored family split for that exact inventory | Keeps every version of a reviewed application/task family in one role: `train`, `dev`, or `final_test`. Generic v1 re-splits are not accepted on the broad collection path. |
| Reviewed root role anchor | The one source-derived historical role map that the anchored split names | The adapter requires the exact root derived from the locked September study split, not merely a self-sealed file. It compares every inherited role and task-family mapping to it, so a child split or a replacement root cannot make a rewritten role map authoritative. |
| Protected-family lock | A self-digested list of held-out family digests tied to the exact split digest | Adds an independent fail-closed check: a family named as protected cannot appear on the training side. |
| Private attempt-metadata JSONL | One sealed metadata record per observed attempt | Provides outcome, provenance, ingestion, and content-policy evidence without exposing a transcript. |

The request also declares whether the source is `teacher` or `self`, the exact
model alias and chat-template digest, an explicit maximum number of sessions
per task version, and a create-once output directory.

### Campaign qualification

`training_data_eligible: true` is necessary but not sufficient.  The campaign
must also be self-digested, route the selected model to unambiguous task
versions, and bind the reviewed action surface exactly to `bash` and
`submit_report`.  Each expected task/model/attempt combination is represented
by a deterministic cell digest.  The adapter records a missing expected cell;
it never silently treats an incomplete collection as complete.

### Family-safe split and catalog binding

The inventory file has its own self-digest, the request records the inventory
file digest, and the family-safe split is validated against that exact
inventory.  This prevents a catalog refresh from changing which task versions
or families a previously reviewed split represents. When the catalog grows,
the split inherits every historical role and allocates only genuinely new
families. The separate root-anchor file is required and must be the checked-in,
source-derived root; it must exactly match the split's recorded historical
roles and task-family mapping. Resealing a child split or creating a fresh
generic split cannot change that authority. See [broad collection
expansion](FLEET_BROAD_COLLECTION_EXPANSION.md).

Only `train` assignments can be selected.  `dev` and `final_test` are never
training candidates, even when a rollout there succeeded.  The protected-family
lock adds a second check before selection and again before publication.  This
is task-family-held-out, not application-held-out: different task families can
share an application, but no version of one reviewed family may cross roles.

## Per-attempt admission gates

Attempt records are strict metadata objects.  Unknown fields are rejected, so
an attempt cannot smuggle raw messages, prompts, transcript content, private
reasoning, or credentials through this interface.  Each record is self-digested
and must bind to the exact campaign cell, model identity, tool treatment,
template, and tool-catalog digest.

An attempt is admitted only when all applicable gates below pass.

| Gate | Required evidence | Result when it does not pass |
| --- | --- | --- |
| Planned identity | Exact campaign/cell/task/model/attempt binding | Rejected as unplanned, ambiguous, or wrongly bound. |
| Authoritative task success | `completed` status, successful verifier process, and score at least one | Rejected as an invalid authoritative success. |
| Complete ingestion | A complete normalized-record, normalized-trajectory, and transcript digest record | Rejected as incomplete; no missing source is guessed. |
| Visible-action-only targets | `target_mode: visible_actions_only` | Rejected. |
| No private or unknown reasoning | `reasoning_visibility: absent` | Rejected.  Provider or tool fields named `thinking`, `reasoning`, or similar are not assumed trainable. |
| Reconstructable context | `compaction: none` | Opaque compaction is rejected explicitly.  Any other unapproved compaction mode is also rejected rather than replayed as if no summary occurred. |
| Training split membership | Exact catalog/split assignment is `train` and not protected | Rejected as held out or non-training. |
| Exact deduplication | One record per campaign cell, one cell per session, and one retained normalized-trajectory digest | Duplicates are rejected or deterministically reduced before the cap is applied. |
| Per-task balance | Explicit `max_sessions_per_task_version` | Extra otherwise-eligible sessions are deterministically excluded. |

The reasoning and compaction rules are intentionally conservative.  The
action-only corpus may teach visible decisions and tool calls.  It must not
claim to teach private chain-of-thought, nor train later actions against a
different context from the one that produced them.  See
[the reasoning and compaction policy](QWEN38_REASONING_AND_COMPACTION_POLICY.md)
for the separate contract required before a student-visible reasoning lane can
exist.

## Output and confidentiality

The output directory is create-once and restricted.  It contains two JSON
files, neither of which is a corpus.

| File | Audience | Contents |
| --- | --- | --- |
| `selection.private.json` | Private corpus operator only | Selected session and task identifiers plus the campaign, split, model, harness, template, and normalized-record digests needed for an exact later lookup.  It contains no source text or packed examples. |
| `ADMISSION.json` | Aggregate-only audit receipt | Input file digests, campaign/split/lock bindings, aggregate counts and fixed rejection reasons, policy facts, and the selection digest.  It contains no per-session identifiers or source content. |

Both documents are self-digested.  Treat the selection file as private even
though it contains only metadata: identifiers and source-record digests are
not public research output.  Do not commit either file or copy it into W&B,
dashboards, or a public report.

The command's standard output is likewise an aggregate handoff summary.  It
reports that no external action was submitted, the two output digests, admitted
counts, and rejection counts.  It is not a shortcut for reading selected data.

## Local invocation and preflight

Run it from a checkout with the request file and all referenced private inputs
already staged:

```sh
uv run --locked cyber-post-train data-fleet-admit \
  /private/collection-wave/admission.request.json
```

This command needs no Fleet, Kubernetes, TensorLake, model-provider, or W&B
credential.  It must fail closed before publishing when any input is malformed,
unsealed, changed, outside the reviewed campaign, or inconsistent with the
catalog/split/lock.  It is not a submission command and does not authorize the
subsequent corpus build or an SFT run.

Before invoking it, the operator should verify all of the following:

1. The campaign was designed for data collection and explicitly set
   `training_data_eligible: true`; an evaluation-only campaign cannot be
   retroactively repurposed.
2. The campaign's selected model, template, harness treatment, tool catalog,
   and task versions are the intended frozen values.
3. The sanitized inventory, family split, and protected-family lock were made
   together and their digests agree.
4. The private attempt metadata was generated by the reviewed collection
   exporter and contains only the defined metadata schema.
5. The output directory does not already exist.  Preserve an earlier receipt;
   do not overwrite it to retry or revise a selection.

## Downstream handoff

Admission is the first half of the data path.  The next component is the
[private corpus materializer](FLEET_PRIVATE_CORPUS_MATERIALIZATION.md).  It
must:

1. reads `selection.private.json` without printing its identifiers;
2. resolves each selected normalized record only by its recorded exact digest;
3. fails if any session, campaign, split, model, template, or normalized-record
   binding differs from the selection;
4. applies the existing dense-SFT visibility, tool-contract, target uniqueness,
   and no-opaque-compaction rules while creating windows; and
5. publishes a new corpus manifest and aggregate receipt that bind back to the
   admission selection digest.

The existing `training/fleet_teacher_corpus.py` remains the frozen 75-task
teacher-corpus path and must not be misrepresented as a generic consumer of
this family-safe split.  The generic materializer writes an immutable corpus
manifest, but it still cannot feed SFT until it reaches its packet's 20M unique
visible-action-token gate; below that threshold its manifest deliberately fails
the existing SFT compiler.  Admission therefore remains auditable selection
evidence, not trainable examples by itself.
