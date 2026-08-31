# Fleet Agent Runtime versus native SkyRL harness parity

Status: **local correction and pre-submission design only**. No job was submitted, changed,
reprioritized, unsuspended, or cancelled. The paired canary remains intentionally blocked.

## Result

The running native RL experiment is not an interpretable capability comparison with the original
Fleet web-app evaluations. The exact 129-task treatment set was a productive middle-difficulty set
for the frontier controls (508/1,021 verified successes, 49.8%), but the native Qwen run was given a
much shorter effective interaction horizon. Its configured 16,384-token prompt budget plus 2,048
post-prompt tokens yielded only an 18,432-token flat trajectory and the launcher forced 12 model
turns. The source Fleet job allowed 600 agent steps; successful source sessions used a median 49
assistant turns and failed sessions a median 104 tool calls in the earlier trace analysis. Zero
native reward is therefore confounded by the harness budget and must not be read as evidence that
Qwen cannot solve the tasks.

The local Theseus correction makes the following controls explicit while retaining every old
default for existing jobs:

- configurable `max_turns` (1–600; historical native default 12);
- a per-turn token cap separate from the total flat-trajectory token budget;
- a configurable native adapter tool-result clip (historical default 4,000 characters);
- a fail-closed task contract that requires every exact task version to declare the canonical
  ordered list `[bash, submit_report]` before any parquet is written;
- canonical tool-schema ordering matching the requested order rather than MCP discovery order.

For a successor diagnostic, the proposed native shape uses the model's full 65,536-token context:
16,384 prompt tokens plus 49,152 post-prompt trajectory tokens, at most 2,048 sampled tokens per
assistant turn, with a 600-turn ceiling. The context, not the step counter, will normally become the
binding limit. This is 3.56 times the current 18,432-token trajectory and can cover the median
successful source trace far more plausibly, while never pretending Qwen can accept more than its
declared context window.

Unbounded tool output is not a safe substitute for compaction. Of 56,808 parseable historical
`bash` observations, 15,051 (26.5%) exceeded 4,000 bytes, 5,393 (9.5%) exceeded 16,000, 1,826
(3.2%) exceeded 65,536, and the maximum was 524,396 bytes (the sandbox independently caps stdout
and stderr). One such response can consume Qwen's entire context. The proposed canary therefore
uses a 16,000-character observation cap in **both** arms: it preserves about 90.5% of historical
results without truncation while bounding one result to roughly a few thousand tokens. If Agent
Runtime cannot apply the same cap, the comparison must be labelled diagnostic rather than matched.

This 65,536-token shape is a **model/configuration ceiling, not a measured B300-feasible bound**.
The existing B300 jobs prove the 27B model and shorter rollout shape run, but no observed job proves
the longer vLLM KV-cache, rollout memory, and SkyRL batch shape fit together. Preview/config parsing
can be checked now; GPU feasibility requires one explicitly reviewed, minimal canary after all
identity and protocol gates pass.

## What the source export proves

The private export manifest binds 1,265 sessions from source job
`a62dd51f-a52b-4941-8207-4679e4b25b51` at
`sha256:160607dc4f4c53a4822b1dafccfbd8283ed773c113d8e61d3dfc8d3437ca59a9`.
The checked-in evidence contains no prompt bodies, model text, tool arguments, or tool output.

- All 1,265 records advertise the ordered task-facing tools `bash`, `submit_report` and
  `max_steps=600` in tool-use mode.
- In all 1,265 records, the first user message is byte-equal to the task-version prompt after
  normalizing Fleet's one-element text-content envelope. A paired run must hydrate the same exact
  `eval_task_version_id`; it must not copy or reconstruct prompt prose.
- The GPT arm used one 1,036-character (1,038 UTF-8 byte) system prompt in all 627 sessions, hash
  `sha256:8512560b0706f8859774bfe8e93c2c86ecf6a6a4603fe59db004fd04fcc738d1`.
  Grok used two prompt byte variants. The proposed Qwen diagnostic freezes the GPT byte variant;
  that is an explicit new-Qwen choice, not evidence that the original Qwen used it.
- The source job scored 587/1,265 (46.4%). GPT-5.6-sol scored 290/627, Grok-4.6 263/584, and
  Grok-4.5 34/54.
- Source sessions had a median 76 assistant turns, p90 154, and maximum 436. Their recorded Fleet
  step counts had median 79 and maximum 482. Durations had median 1,396.7 seconds and maximum
  7,033.1 seconds.
- The exact 129 treatment task keys had 1,021 source attempts and 508 passes: 41 always solved,
  43 never solved, and 45 mixed.

## What the export does not prove

The export records the two tool names, but not their JSON schemas. Latest source describes current
`bash` as a challenge-sandboxed script tool with its own timeout/output bounds and current
`submit_report` as the flag/report submission tool. That is useful implementation context, not proof
of the schemas served to the historical job. A launch gate therefore runtime-discovers both arms,
requires exactly those two names, canonicalizes their complete OpenAI tool objects, and requires
equal whole-surface and per-tool SHA-256 hashes.

The historical deployed Agent Runtime revision and configured whole-session timeout are also absent.
The observed maximum duration is evidence of what happened, not a timeout setting. The original
verifier execution is version-bound and binary; current native scoring already calls Fleet's exact
task-version/environment-version authority and preserves infrastructure failures as failures rather
than silently converting them to reward zero.

The source GPT transcript uses standard assistant `tool_calls` followed by `role=tool` observations.
Some Grok traces contain `search_tool`/`use_tool` wrapper calls even though the harness surface still
advertises only `bash` and `submit_report`. Current native Qwen parses Qwen3.6's attribute-XML
function syntax (and the legacy Hermes JSON form), then represents consecutive tool results as one
`role=user` message containing `<tool_response>` blocks. That representation exists because SkyRL's
loss-mask helper accepts only user/assistant roles; Qwen's pinned chat template currently renders it
like tool-role messages. This is source-level evidence, not sufficient parity proof. The paired gate
must render a fixed one-call exchange through both paths and compare the exact Qwen prompt token IDs
through the first observation. Equal tool names or equal JSON schemas alone cannot prove equal
model-visible protocol.

## Compaction and policy-gradient correctness

GPT produced 111 automatic `context_compaction` events in 97 sessions. The median recorded event
reduced roughly 244,705 pre-compaction tokens to 4,381. The compaction payload says it is opaque;
neither the summary algorithm nor its hidden state is present in the export. Grok recorded no such
events, and `job_launch_params.context_management` was null throughout.

Native SkyRL returns one flat causal trajectory with sampled token IDs and the log probability of
each policy token under the exact prefix used at sampling time. If the adapter compacts or deletes
old tokens while sampling later turns, then flattens those later tokens behind the un-compacted
prefix for training, the stored log probabilities no longer correspond to the trajectory the loss
replays. That is a wrong-gradient bug, not a cosmetic transcript difference. Re-tokenizing an
opaque summary cannot repair it.

Therefore this correction does **not** implement fake compaction. Safe options are:

1. use one uncompacted trajectory up to Qwen's exact 65,536-token ceiling (the present proposal);
2. add a trainer-native segmented-trajectory objective that records each compacted sampling prefix,
   action span, and log probabilities, then validate the loss against those segments; or
3. train a policy-visible memory/summarization action as a separately specified intervention.

Options 2 and 3 change the trainer/objective and require their own review and controlled experiment.
They cannot be inferred from the opaque Codex compaction events.

## Safest paired canary

The plan selects three exact dev versions across Current, Fentry, and Fubspot with historical
frontier rates 5/8, 2/8, and 6/8. Four repetitions per task per arm give 12 episodes per arm. Both
arms must bind the frozen Qwen checkpoint and tokenizer revision, exact task/environment versions,
system-prompt bytes, two runtime-equal tool schemas, binary authoritative verifier, temperature,
top-p, per-turn cap, seed behavior, and parallel-tool policy.

For a truly paired comparison, Agent Runtime must disable compaction and accept an equivalent
65,536-token whole-trajectory ceiling. If its API cannot expose those controls, the run may still be
a useful diagnostic, but it is not matched causal evidence and must report each arm's censoring
separately. The 600-step value remains a ceiling in both arms; Qwen's model context is the justified
effective horizon until a correct segmented objective exists.

No paid launch should occur until every blocker in
`configs/evaluation/qwen36-27b-fleet-harness-paired-canary-v1.json` is proven and reviewed.

## Reproduction

`training/harness_parity.py` streams the private JSONL export and emits only aggregate, non-content
evidence. Its checked-in result is
`docs/evidence/harness-parity/2026-08-31-fleet-export-v1.json`. The trace export itself remains
private and ignored.
