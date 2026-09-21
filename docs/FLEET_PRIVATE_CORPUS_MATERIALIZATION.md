# Private Fleet collection corpus materialization

## Purpose

`data-fleet-materialize` is the second, private half of the reusable Fleet
collection path.  It turns a **previously admitted** set of successful
trajectory records into dense SFT windows.  It does not contact Fleet, start a
job, call a model, or print trace content.  It is intentionally separate from
the metadata-only `data-fleet-admit` command.

The separation is important:

1. Collection admission decides whether a successful rollout is real, belongs
   to the intended campaign, and is allowed on the training side of a
   family-safe split.
2. This command reads the private normalized records only after checking those
   admission decisions against the immutable collection packet and task roster.
3. Existing dense-SFT code then makes packed windows with visible assistant
   actions as targets.  It never treats a private chain of thought as a target.

This command is a local CPU operation.  It is not a rollout, an evaluation, a
training launch, or permission to launch any of those.

## Inputs and exact bindings

The request schema is `cyber_fleet_private_corpus_materialization_request_v1`.
Every file input is a regular-file reference with its exact SHA-256.  The
materializer checks each file before reading it and again immediately before
writing the result.

| Input | What it proves |
| --- | --- |
| Private selection and aggregate admission receipt | Exact successful sessions, source model, template, tool treatment, campaign, and metadata-only admission policy from `data-fleet-admit`. |
| Collection packet, task selection, and evaluation configuration | The approved source type, target of at least 20 million unique visible-action tokens, **digest of the source-authorization receipt**, OpenCode tool surface, and complete train roster. |
| Catalog inventory, family split, protected-family lock, and runtime bindings | The complete current task universe, one immutable role per task family, every held-out family, and the exact runnable task/environment/data binding. |
| Private normalized records | The admitted records only.  Their self-digests, session identities, campaign cells, source settings, lineage, and trajectory digests must match the selection exactly. |
| Model lock, local tokenizer, and native masking helper | Exact local tokenizer bytes and the approved dense-window implementation. |

An inventory refresh therefore cannot quietly change an old split.  All task
versions of a family stay in one role; `dev` and `final_test` families are
rejected even if they have a successful rollout.

The packet binds the exact digest of the authorization evidence and, for a
teacher, the strength-evidence digest.  This local materializer can prove that
those identities have not changed; it cannot independently prove who issued an
opaque external receipt.  A collection launcher must therefore verify that
receipt against its source-of-truth registry before collection begins.  This is
an intentional boundary: this command has no credentials or network access.

## What records are accepted

Only the current `cyber_fleet_visible_action_record_v1` envelope is accepted.
It has a fixed, small set of fields.  Unknown fields are rejected rather than
silently ignored.  In particular, it must prove all of the following:

- the record is one of the sealed admitted sessions and has the exact same
  normalized-record and trajectory digests;
- its policy is exactly `visible_actions_only`, reasoning is `absent`, and
  compaction is `none`;
- its source model, template, action-tool treatment, task version, and family
  agree with both admission and the collection packet;
- assistant/tool messages have only the reviewed fields and only `bash` plus
  `submit_report` calls.  Tool IDs are bounded opaque identifiers, a report
  call cannot share an assistant turn with other tool work, and
  `submit_report.explanation` is empty in this action-only contract because it
  would otherwise become a supervised free-text target; and
- each target can be reconstructed without an opaque summary.

Private or unknown reasoning fields, `<think>`-style assistant content,
unknown message fields, any assistant prose without a reviewed tool action,
nonopaque tool identifiers, mixed report/tool turns, nonempty report
explanations, opaque compaction, missing tool results, unsupported tools,
duplicate trajectories, duplicate packed windows, and held-out families all
fail closed.
No substitute example is chosen after a failure.

The current implementation accepts **teacher visible actions** and **self
visible actions** as separate collection packets.  It deliberately does not
accept student-visible reasoning.  That future lane needs a new packet schema,
explicit permission, and a reviewed exporter; no flag can relax this rule.

OpenCode can compact context during a long live rollout, but that does not make
the resulting summary safe training context.  In this v1 corpus path, a record
that compacted must be labeled and rejected.  The collection exporter is a
separate required gate: it must not label a compacted session as `none`.  A
future path may accept compaction only after it records and validates the exact
before-summary, summary, and after-summary continuation; it is not implemented
here.

The materializer validates the message envelope and rejects known unsafe fields,
but it intentionally does not try to infer whether an arbitrary shell command
contains hidden reasoning.  That distinction has to be made by the signed
normalization/export path that labels the record `visible_actions_only` before
this command runs.  The local checks bind that label to the sealed admission
selection; they do not turn untrusted free-form command text into provenance.

## Outputs and the 20M gate

The create-once output directory is private (`0700`; output files are `0600`):

| File | Contents |
| --- | --- |
| `train.parquet` | Private packed visible-action windows and masks. |
| `source-selection.private.jsonl` | Private per-session provenance and aggregate coverage facts, without raw text. |
| `coverage.private.json` | Aggregate target/window/token coverage. |
| `manifest.json` | Immutable corpus identity and every upstream binding. |
| `MATERIALIZATION.json` | Small receipt that binds the manifest and coverage files. |

The command itself prints only an aggregate summary: counts, receipt digests,
and whether the corpus has reached its target.  It never prints a training row,
prompt, trace, tool result, token IDs, flag, or private reasoning.

The collection packet currently requires at least **20,000,000 unique visible
assistant target tokens**.  A smaller materialization is preserved as evidence
with `validation_mode: collection_pending_target`, which the existing SFT
compiler rejects.  Once the threshold is reached, the manifest instead uses
`validation_mode: task_outcomes_only`; an SFT plan can then use it while keeping
held-out evaluation separate from teacher-token loss.

At the 20M gate, the materializer also rejects a corpus if one task family
provides more than 25% of its supervised action tokens.  It records the
aggregate concentration result privately.  This prevents a handful of long
trajectories from satisfying the token goal while leaving most of the roster
underrepresented.

## Practical campaign sequence

For a scalable, family-safe teacher or self-action collection campaign:

1. Build a fresh sanitized inventory of reviewed runnable task versions and
   validate the tasks independently.
2. Build one parameterized family split and seal its protected non-training
   family lock.  Do not reuse a tiny frozen task list as the whole campaign.
3. Render a source-only packet for either teacher visible actions or self
   visible actions.  It must carry the exact source authorization, template,
   task runtime bindings, and at least the 20M visible-action-token target.
4. After normal operational launch approval, collect the planned attempts and
   export only the sealed attempt metadata needed by `data-fleet-admit`.
5. Run `data-fleet-admit`; it selects only genuine verifier-confirmed,
   family-safe successes and applies exact deduplication plus the per-task cap.
6. Run `data-fleet-materialize` using the selected private records and all
   exact packet/split bindings.  Check `MATERIALIZATION.json` and confirm
   `sft_ready: true`.
7. Only then compile the SFT plan.  Evaluate a resulting checkpoint on the
   protected Fleet families and an independently configured external benchmark;
   neither is folded back into the training corpus.

The first four steps are planning and operational gates, not actions performed
by this command.  It will not launch them on its own.
