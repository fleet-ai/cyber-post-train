# Qwen3.8 Teacher3K step-1000 regression audit

Date: 2026-09-24  
Status: root cause established; corrective corpus and confirmation run pending

## Conclusion

The step-1000 checkpoint did not fail because training silently skipped
updates, exported the wrong model, or served corrupt weights. It successfully
optimized the objective it was given. The dominant problem was that the
objective and most of its input windows did not represent a valid OpenCode
black-box-exploitation interaction.

Two independently observed defects are sufficient to explain falling training
loss alongside worse task performance:

1. **Most 32K training windows started in the middle of an earlier message or
   tool result.** The rechunker copied a raw token suffix and did not restore
   the original system prompt, task, tool definitions, or message boundary.
   The exact step-1000 sampler consumed 6,451 such rows among its first 8,000
   examples (80.64%).
2. **The training and evaluation tool interfaces differed.** Training targets
   used bare `bash` and `submit_report` names and were rendered without the
   formal tool catalog. The pinned OpenCode evaluator exposed those same
   underlying MCP tools to the model as `fleet_bash` and
   `fleet_submit_report`. The evaluator preserved those exact model-generated
   names in storage; it did not add the `fleet_` prefix during ingestion.

These are real training-contract defects, not merely speculative differences.
They were measured in the exact corpus consumed by the checkpoint and in its
actual development-evaluation traces. Other recipe choices likely made the
regression worse, but they are secondary to these defects.

The old development comparison is useful for this structural diagnosis, but it
is not a valid capability result. It incorrectly treated output-limit and
nonzero-process attempts as ordinary failures, and four of its 17 development
task families were exposed in training through aliases. No pass@4 value,
paired effect, or confidence interval can be claimed from those attempts.

## Exact artifact under investigation

- API run: `34148b91-efe7-45e3-861f-1edd1798935b`
- RayJob UID: `e4d1376c-5d3f-49e5-9f0c-6469cd32c753`
- Base: `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Frozen plan SHA-256:
  `8488f03a65e3d158a46697eefda2751736681e3a503c26ee9c39c63fb130679c`
- Corpus: 14,693 rows and 57,384,881 supervised-token occurrences
- Recipe: full-weight FSDP, batch 8, 32,768-token ceiling, constant LR `3e-6`,
  no warmup, gradient-norm ceiling 1.0
- Evaluated checkpoint: `global_step_1000`, after 31,035,607 supervised-token
  occurrences, or 54.08% of the planned epoch
- Accepted BF16 export revision:
  `sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5`

The complete run later reached all 1,837 planned optimizer steps. Step 1000 was
an intermediate checkpoint, not the one-epoch final model.

## What the attempted evaluation showed

The Fleet development attempt used the same 17 task versions, four seeds per
task, OpenCode 1.18.27, sampling settings, context and time limits, environment
bindings, verifier bindings, and serving settings.

| Model | Planned attempts | Normally completed | Output-limit holds | Process errors |
|---|---:|---:|---:|---:|
| Exact base | 68 | 52 | 15 | 1 |
| Teacher3K step 1000 | 68 | 18 | 49 | 1 |

Only exit zero plus a normal OpenCode completion is a valid model outcome.
Output-limit and process-error attempts must be held outside the denominator,
not converted to zero. The two arms therefore contain only 52 and 18 valid
attempts, with different missing cells. The earlier 7/17 versus 0/17 statement,
its 41.18-point difference, and its bootstrap interval are superseded and must
not be cited as a valid pass@4 or capability comparison. Missing paired cells
must be reacquired under a score-blind, predeclared replacement policy before
any effect estimate is calculated.

All 18 normally completed candidate attempts scored zero and none attempted the
report tool. That remains a useful diagnostic observation about these retained
traces, not a complete task-level capability estimate.

The candidate also behaved differently from base:

- 49/68 candidate attempts reached their output limit;
- 18/68 ended normally without solving;
- 1/68 had a process error;
- candidate tool-call volume was much lower than base;
- the candidate generated 32 invalid bare-`bash` call attempts across 24
  attempts; OpenCode rejected every one before MCP execution;
- candidate emitted zero report-submission calls.

The comparison also was not training-lineage held out. A later exact-lineage
audit established that four of these 17 development families occurred in the
training corpus through composite or hinted aliases. This is separate from the
incomplete-attempt defect above and independently prevents a held-out capability
claim.

## Primary defect 1: malformed, anchorless windows

The rechunker in `training/dense_rechunk.py` selected each later window with:

```text
start = target_start - context_tokens
window = input_ids[start:end]
```

For this corpus, `context_tokens` was 8,192. The transform preserved target
tokens and their loss masks, but it did not align `start` to a message boundary
and did not prepend the system prompt, task request, or tool definitions.

Sanitized inspection of the exact sealed Parquet established:

- 11,794/14,693 rows (80.27%) place their first supervised target exactly at
  offset 8,192, proving that earlier raw tokens were clipped;
- among a fixed sample of 256 clipped rows, 254 began with ordinary vocabulary
  tokens rather than a Qwen chat delimiter;
- all 256 control rows that were not clipped began at the expected chat
  delimiter;
- the seed-20260920 sampler used for step 1000 consumed 6,451 clipped rows
  among its 8,000 examples.

The runtime correctly checked target hashes, loss masks, token ranges and
once-only target coverage. It did not check that a row began with a valid
system/task anchor or complete message boundary. Training therefore received
valid labels attached to mostly invalid conversational prefixes.

This explains how token-level cross entropy could improve without learning the
causal task: the model was rewarded for predicting actions from fragments that
often omitted the problem statement and the observations that justified those
actions.

## Primary defect 2: tool-interface mismatch

The historical training materializer at preparation commit
`daa86d54e38bcd347c1e79e9dd7a0c081d47e63f` rendered with `tools=[]`, admitted
only the bare tool names `bash` and `submit_report`, and retained those names in
assistant targets. Its exact `training/dense.py` SHA-256 is
`712038f1abf4f461cf471afc50f7fbbfe90c9d2d7b9e0be1a81583378c933012`.

The pinned evaluator's MCP server advertises the underlying executor names
`bash` and `submit_report`. OpenCode registers that server under the client name
`fleet`, constructs each model-facing name as `<client>_<tool>`, and sends
`fleet_bash` and `fleet_submit_report` with their full schemas to the provider.
When a valid prefixed call executes, OpenCode maps it back to the underlying
bare MCP name. Bare `bash` and bare `submit_report` are not valid model-facing
names in this harness. Evaluation also uses OpenCode's coding-agent system
prompt, reasoning mode, long context and native compaction. Training used
heterogeneous teacher prompts, no formal tool catalog, visible actions only,
32K windows and raw truncation.

This path is fixed by the pinned OpenCode 1.18.27 source. Its MCP catalog builds
the prefixed key, the session request passes that exact key to the provider, and
its only repair for an unknown tool is lowercasing before routing it to an
`invalid` handler. Lowercasing cannot map `bash` to `fleet_bash`. The evaluator's
normalizer copies raw OpenCode `part.tool` into stored `function.name` without
renaming it.

The evaluator pins OpenCode release asset SHA-256
`4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702`;
the v1.18.27 tag resolves to source commit
`4b7e19e315cca414121ba1d61523fef74bb3ae8b`. The fixed request proxy retained
request byte counts and status, not request bodies or body hashes, so the
provider-facing catalog is established by this exact pinned construction and
the matching raw OpenCode events rather than by a retained provider payload.

All 136 retained raw OpenCode traces were read from their exact bound paths and
verified against their recorded file SHA-256. The raw and stored aggregate names
agree exactly:

| Raw OpenCode tool event | Base | Step 1000 |
|---|---:|---:|
| valid `fleet_bash`, completed | 9,929 | 4,929 |
| valid `fleet_bash`, tool returned an error | 677 | 94 |
| invalid bare `bash`, rejected before execution | 0 | 32 |
| attempts with an invalid bare `bash` event | 0 / 68 | 24 / 68 |
| valid `fleet_submit_report`, completed | 20 | 0 |

The base and candidate saw the same evaluation interface. The candidate's 32
bare `bash` values were model-generated invalid call attempts, not executed
shell actions. They match the names used in training and provide direct evidence
of the interface mismatch. No retained attempt generated bare `submit_report`,
but the same deterministic mapping makes it invalid; the valid model-facing
name is `fleet_submit_report`. Zero report actions from the candidate is
mechanically important because Fleet grading requires a submission.

## Additional material problems

### Successful-session fragments were not necessarily successful fragments

The broad corpus salvaged valid lifecycle prefixes from 149 malformed source
sessions while retaining the original full session's success eligibility. The
salvage removed 5,878 assistant turns. Sanitized inspection found that 148/149
salvaged sessions no longer contained a retained report, yet contributed about
1.65M supervised tokens. A source session's eventual success therefore did not
guarantee that the retained training fragment included the successful outcome.

### The objective mostly imitated visible actions

Private reasoning was intentionally omitted. Approximately 94.21% of assistant
targets were non-submit shell/tool actions; only about 1.65% were plain visible
decision or reasoning turns. Final-report tokens were only 2.30% of supervised
tokens, so “the model only learned the final submission” is not supported.
The problem is closer to action imitation without the planning state and exact
tool interface required at evaluation.

### Data weighting was highly uneven

Loss was correctly normalized over supervised tokens, but there was no
task/session/teacher balancing. The largest task supplied 7.04% of supervised
tokens; the top five supplied 23.22%; one teacher family supplied roughly half
of the supervised tokens. Long, prolific trajectories therefore dominated the
gradient.

### The full-weight recipe was aggressive

The optimizer executed correctly, but every reported pre-clipping gradient norm
exceeded the 1.0 clipping threshold. The run used full-weight LR `3e-6`, no
warmup and no held-out capability selection. Those choices may have amplified
forgetting. They cannot, however, explain away the proven malformed-context and
tool-interface defects.

### Five claimed held-out families were exposed through aliases

The historical builder excluded 25 exact task-key strings, but exact-string
exclusion was scientifically insufficient. A complete read-only lineage audit
resolved all 496 training keys and 1,176 exact training versions to their
reviewed vulnerability lineages. It found zero exact task-key or version
overlap, yet five of the 25 claimed held-out families occurred through six
composite or hinted training aliases: four development families and one final
family.

The historical manifest claim
`held_out_task_families_excluded_across_all_versions: 25` is therefore false
for the bound source selection. The checkpoint and immutable manifests remain
valid provenance, but that field must not be used as leakage evidence. The
largest proven training-lineage-clean subset is 20 families: 13 development and
7 final. “Lineage-clean” here means absent from this exact teacher3k corpus, not
globally untouched.

## What was ruled out

The following explanations have low likelihood given the evidence:

- **No optimizer update.** The run recorded all 1,837 contiguous optimizer
  steps, finite positive gradient norms and a 27.7% reduction in mean loss from
  the first to last 100 steps.
- **Repeated rows before step 1000.** The seed-bound sampler consumed 8,000
  distinct rows before the first epoch boundary.
- **Bad loss normalization.** The exact SkyRL source computes global
  supervised-token-mean cross entropy and correctly compensates for FSDP
  gradient averaging.
- **Wrong or base checkpoint served.** Checkpoint, export, stage and live route
  receipts bind the expected trained artifact. Candidate logits differed from
  base.
- **Missing tensors or export dtype corruption.** The export contains the exact
  1,199-tensor BF16 layout: 1,184 trained tensors plus 15 exact-base MTP tensors.
  CPU and GPU reloads produced finite logits.
- **Base/candidate treatment drift.** The attempted comparison used matching
  tasks, seeds, harness, tools, budgets and non-weight serving settings. This
  does not repair the incomplete-attempt or training-lineage defects.
- **Final-answer-only learning.** Submit/report output was a small fraction of
  the objective.

## Required correction

The existing 32K/64K/96K derivative corpora must not be treated as a clean
training recipe merely because target and mask checks pass. A replacement
corpus generation must:

1. rebuild rows from structured messages, never raw-slice token arrays;
2. include the exact system prompt, task request and formal OpenCode tool
   catalog in every row;
3. use the exact model-facing tool names used at evaluation;
4. retain only complete tool-call/result pairs and complete message boundaries;
5. prove every row begins with the pinned chat-template system anchor;
6. require success evidence within the retained fragment, or label the fragment
   independently rather than inheriting full-session success;
7. exclude held-out task families by reviewed vulnerability lineage across
   every alias, hint, composite and version before materialization;
8. bind the parent builder revision, selection roster and source cutoffs;
9. publish per-task/session/teacher concentration and checkpoint-exposure
   statistics;
10. define actual compaction explicitly. An 8,192-token suffix is truncation,
    not compaction.

## Confirmation plan

The smallest decisive confirmation is:

1. materialize a corrected create-once corpus with the requirements above;
2. run a bounded corrected SFT canary, preferably with a lower LR and warmup;
3. demonstrate finite loss/gradients, changed weights and a reloadable
   checkpoint;
4. run a matched pass@4 base-versus-corrected-checkpoint comparison on the exact
   20-family lineage-clean Fleet set, reacquiring every output-limit or process
   failure as a paired replacement rather than scoring it as zero;
5. separately compare earlier and final checkpoints from the flawed run to
   localize when capability declined; and
6. ablate tool-contract repair versus context repair if attribution between the
   two primary defects is scientifically important.

The flawed checkpoint remains useful as a negative control. It should not be
promoted as evidence that stronger-teacher SFT intrinsically hurts cyber
capability.
