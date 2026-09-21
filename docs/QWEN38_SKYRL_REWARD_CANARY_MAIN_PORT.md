# Qwen3.8 SkyRL reward canary

Status: **prod8 was created once on 2026-09-21, but its terminal result and
GPU release are now unknown. It is not accepted and must never be resumed.**

The historical launch record says prod8 was created, but it contains neither a
terminal receipt nor verified GPU release. The current Jobs API and both
Kubernetes clusters no longer contain the RayJob, which cannot prove whether it
finished, failed, or released its GPUs. The authoritative current conclusion is
the sanitized [prod8 reconciliation receipt](evidence/qwen38-study/2026-09-21-skyrl-prod8-reconciliation-v1.json):
`unknown`. Do not use any later statement in this historical prod8 document as
proof that it is running, terminal, released, or reusable. A fresh-from-base
successor is prepared separately in
[`QWEN38_SKYRL_REWARD_CANARY_PROD9.md`](QWEN38_SKYRL_REWARD_CANARY_PROD9.md).

The exact, non-authorizing packet is
[`qwen38-rl-reward-canary-prod8-launch-packet-v1.json`](../configs/qualification/qwen38-rl-reward-canary-prod8-launch-packet-v1.json).
It is frozen historical evidence. It is never refreshed to match newer runtime
code, and it cannot authorize or recreate prod8. The fresh prod9 preparation
receipt is the only place that may bind a current runtime, plan, request, and
CPU-preflight digest. The independent broader audit is
[`2026-09-20-skyrl-launch-readiness-audit-v1.json`](evidence/qwen38-study/2026-09-20-skyrl-launch-readiness-audit-v1.json).
Neither file creates or authorizes a workload. The separate, sanitized live
launch record is
[`2026-09-21-skyrl-prod8-launch-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod8-launch-v1.json).

## Exact prod8 run

The immutable inputs are:

- data config:
  [`qwen38-rl-reward-canary-data-prod-v8.json`](../configs/qualification/qwen38-rl-reward-canary-data-prod-v8.json);
- run config:
  [`qwen38-rl-reward-canary-prod-v8.json`](../configs/qualification/qwen38-rl-reward-canary-prod-v8.json);
- sanitized data manifest:
  [`qwen38-rl-reward-canary-manifest-prod-v8.json`](../configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json);
- source and submission contract:
  [`qwen38-rl-reward-canary-port-v8.json`](../configs/qualification/qwen38-rl-reward-canary-port-v8.json).

The sealed identities are:

- plan SHA-256 `09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de`;
- request SHA-256 `7c2df31feceb5c741cb16463b554b91d00b11c6bf50ec85b3203706823c2fb44`;
- runtime SHA-256 `acb1d1a1ff5aec9d8c153dfc57e52a13bb0b081fc666a4a0d0db139e8cb6831b`;
- sanitized data-manifest SHA-256
  `5561f1a349abbe1a580dd5763368c1e6c1524861c39d744f7dad4f25d9950dd3`.

Prod8 uses one B300 node with eight GPUs at `c1`/`q1`. It asks the exact
Qwen3.8-27B revision to attempt one train task eight times, then make exactly
one optimizer update at learning rate `1e-6`. It evaluates once before the
update and once after it. The horizon is 262,144 context tokens, up to 32,768
tokens in one assistant turn, 4,096-token generation chunks, context
compaction at 163,840 tokens with an 8,192-token policy summary, up to 1,200
turns, and up to four hours per episode. The total episode generation allowance
is 4,194,304 tokens so long tasks can continue across many turns and
compactions.

In this historical specification, semantic compaction means the student emits
a separately recorded working-memory summary. Before it replaces earlier
history, the exact Qwen tokenizer and full chat template must prove that the
summary plus the next action fit in 262,144 tokens. Otherwise an episode ends
cleanly; it never silently truncates older history. This describes the intended
contract, not evidence that prod8 completed it.

The ordered tools are `bash` and `submit_report`. Only Fleet's authoritative
grader may produce reward. A response that merely reaches its 32,768-token
turn limit is still graded, but an unfinished tool call is never executed.
Other context, timeout, transport, parser, or infrastructure failures remain
hard failures rather than invented zero rewards.

## Checks already completed

On 2026-09-21, read-only Jobs preview and Kubernetes server dry-run checks
passed in both the development and production clusters. They proved the final
root RayJob shape is one node/eight GPUs at `c1`/`q1`, and that the final root
RayJob carries `fleet.ai/failure-alerts: "off"`. They also proved the exact
CPU-preflight Job requests zero GPUs, uses `c1`, and carries the same root
annotation. No workload was created and no private task payload was read.

The sanitized receipts are
[`2026-09-21-skyrl-prod8-read-only-previews-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod8-read-only-previews-v1.json).
They are useful evidence about the current object shape, but they are not
launch authority. Server previews and absence checks must be repeated
immediately before the sole GPU create.

The old `v17` topology packet remains historical evidence. It is not a prod8
gate. Prod6 and prod7 already exercised the one-node topology live. At the
time, prod8's historical direct path validated its recorded object shape; that
path is frozen evidence and must never render, preview, or create prod9.

## Historical launch record; not acceptance evidence

The launch record reports that the operator completed these steps in order on
2026-09-21. That record is useful historical context only; the reconciliation
above supersedes its former live-status claim:

1. Rehash the exact private one-train/one-development package and prove its new
   SFS destination, staging Job name, and upload archive are absent.
2. Server-preview the zero-GPU staging Job in development and production.
3. Create that CPU-only stage once. Its root Job must carry the alert opt-out;
   the operator must monitor it to terminal release and independently verify
   ownership, inventory, and every digest at the destination.
4. Server-preview and create the exact-image, zero-GPU CPU preflight once. It
   must reopen the staged data and return literal `true` for both the
   gradeable-output-limit and partial-tool-blocked checks.
5. Immediately before a GPU create, repeat the Jobs history, Kubernetes, SFS,
   W&B ID, and output-path absence checks; repeat both direct RayJob server
   previews; and arm the exact cleanup observer.
6. Authorize one RayJob create. An ambiguous create response is reconciled and
   never retried.

The exact RayJob was created once at 2026-09-21T10:44:48Z. Its UIDs, launch
receipt, capacity check, staged-data receipt, CPU-preflight receipt, and cleanup
observer receipt are recorded in the live launch record linked above. Do not
repeat the create.

Acceptance still requires eight genuine rollouts, one
authoritative verifier execution ID per rollout, non-constant rewards, one
finite nonzero optimizer update, a sealed step-1 checkpoint, and complete
release of the eight GPUs. All-zero genuine rewards are a valid experiment
result but do not prove a learning update. Missing or truncated evidence is an
infrastructure failure, not a model score.

Review the historical packet without external access. This confirms that its
bytes are still the recorded prod8 evidence and reports whether current source
code has changed; it never turns prod8 into a runnable job:

```sh
uv run --locked python scripts/prepare_qwen38_skyrl_prod8_launch_packet.py --check
uv run --locked python scripts/prepare_qwen38_skyrl_production_queue.py --check
uv run --locked python scripts/audit_qwen38_skyrl_launch_readiness.py --check
```

## Retired hypothetical checkpoint and resume plan (do not execute)

Prod8 saves every optimizer step and keeps the latest two checkpoints. Because
it contains only one step, the expected checkpoint is
`/mnt/sfs/jobs/chris-q38-rlreward-prod8/checkpoints/global_step_1`. A complete
native checkpoint must contain trainer state, the data-sampler cursor, the FSDP
configuration, eight model shards, eight optimizer shards, eight rank-local
extra-state shards, and Hugging Face configuration/tokenizer files. The
checkpoint sealer rejects missing, empty, extra, or linked files and checks
that parameters really changed.

That is not yet the same as proven recovery. The current configuration sets
`trainer.resume_mode` to `none`; therefore no existing receipt can truthfully
claim that a stopped job will continue from this checkpoint. Checkpoint size
and write duration are also still unobserved.

The former smallest resume qualification was a separate create-once successor named
`chris-q38-rlreward-resume-step2-v1`. It must reopen the sealed step-1
checkpoint without changing it; restore all model, optimizer, scheduler,
rank-local random, trainer-step, and sampler state; resume at step 1; collect
one new group of eight without replaying step-1 episodes or verifier IDs; make
one finite update to step 2; seal the complete step-2 checkpoint; then pass an
independent zero-update GPU reload and finite inference check. Only that result
can qualify native resume for a broader run.

The non-authorizing offline packet for that retired chain is
[`qwen38-rl-reward-prod8-resume-qualification-v1.json`](../configs/qualification/qwen38-rl-reward-prod8-resume-qualification-v1.json).
It fixes the three create-once identities, resources, and acceptance checks for
the step-1 reload, step-1-to-step-2 continuation, and step-2 reload. Its live
step-1 receipt and checkpoint-manifest digests are deliberately `null`, and its
implementation and launch gates are deliberately false. Do not edit those
fields in place. No prod8 acceptance or resume is possible. Any future resume
or broad arm must begin from a terminally accepted fresh prod9-or-later run,
with new identities and current source.

The first intended broad-arm document remains historical context only:
[`qwen38-skyrl-production-a1-v1.json`](../configs/runs/qwen38-skyrl-production-a1-v1.json),
but it is not launch-safe yet. Its old 98,304-token non-compacting horizon must
be replaced by a current long-context and output-limit contract. It cannot be
launched from prod8 or any of its historical artifacts.

## A successor data package must change its embedded run identity

The prod7 V2 data-stage Job completed correctly, but its exact-image CPU
preflight rejected the package before any GPU allocation. The public manifest
named prod7 while the private train and development rows still named prod6
inside `cyber_config_json.run_id`. Changing only the manifest does not create a
valid successor: the embedded episode identity is what the environment,
recorder, and checkpoint lineage use during training.

The V2 destination is immutable and retired. A replacement uses a new V3 data
destination and `rebind_private_source_run_id`. That helper permits exactly two
changes per row: the run ID and the row's checksum after that run-ID change. It
canonicalizes the row again, recomputes the train/development payload hashes,
and recomputes the manifest checksum. Any prompt, task, environment, reward,
model, split, tool, or horizon change fails closed. The exact-image CPU
preflight must then reopen the staged package and pass before a GPU RayJob may
be authorized.

Sanitized rejection and release evidence is
[`2026-09-21-skyrl-prod7-stale-episode-run-id-preflight-rejection-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod7-stale-episode-run-id-preflight-rejection-v1.json).
It records zero collected episodes, zero rewards, zero optimizer updates, zero
checkpoints, and complete release. It is not a capability result.

## Cleanup observers must prove a cleanup reason

Prod6 established a separate operational failure class. The RL process was
still generating its pre-update development episode, the Pod was Ready with
zero restarts, and GPUs were actively computing. The local cleanup observer
then suffered five consecutive observation errors and its old `finally` block
deleted the exact RayJob. SFS contained only the run and batch `STARTED`
markers: no collection, reward, update, checkpoint, runtime failure, rejection,
or completion marker existed. This was an observer-induced infrastructure
failure, not a model result.

An observer exists to release a workload after there is evidence that cleanup
is required. Failure of the observer's own Kubernetes read is not that
evidence. The observer may now delete only when at least one positive condition
is true:

1. the exact bound workload reports a terminal state;
2. the exact bound workload violates its reviewed GPU-resource contract; or
3. the plan-bound maximum runtime is reached.

Transient or repeated read failures are recorded using a short sanitized code
and the observer keeps watching. They never authorize deletion of a
nonterminal run. The result receipt records the total and maximum consecutive
observation failures, the last sanitized failure code, and the positive reason
that eventually authorized cleanup. The regression deliberately exhausts the
old five-cycle threshold while an eight-GPU run remains active, proves no
delete occurs, then proves normal exact-UID cleanup after a genuine terminal
status.

Sanitized incident and release evidence is recorded in
[`2026-09-21-skyrl-prod6-observer-induced-release-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod6-observer-induced-release-v1.json).
It records zero collected training batches, zero optimizer updates, zero
checkpoints, and complete release of the eight GPUs. It makes no capability
claim.

## A model output limit is graded, but partial tools never run

Prod7 reached the pre-training development episode and generated continuously
for about one hour and fifty minutes. The recorder correctly continued each
4,096-token generation chunk from the exact preceding token IDs. The episode
ended only when one assistant response consumed its declared 32,768-token
allowance without an end marker. This is the same model-visible `output_limit`
outcome reported by the matched OpenCode protocol. It is not a context-window
overflow, compaction failure, cluster failure, reward, optimizer update, or
capability result.

Rejecting this outcome before scoring stops the whole grouped RL batch and
throws away a useful failed policy attempt. Assigning zero locally would be
worse because that would invent a reward. The repaired boundary therefore does
four exact things:

1. retain the sampled token IDs, masks, and log probabilities as the final
   policy step;
2. never parse or execute a tool call from the unfinished response;
3. ask Fleet's authoritative grader for the reward and release the instance;
4. report `length` to native SkyRL so the output limit remains visible.

Only `turn_response_budget_exhausted` gets this treatment. Context exhaustion,
the total episode-response limit, the 1,200-turn limit, episode timeout,
transport failure, parser failure, and infrastructure failure remain
fail-closed. The 262,144-token context, 4,096-token chunk continuation,
compaction policy, 32,768-token assistant-response limit, and four-hour episode
limit do not change.

The sanitized prod7 evidence is
[`2026-09-21-skyrl-prod7-output-limit-rejection-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod7-output-limit-rejection-v1.json).
It records zero collected episodes, zero authoritative rewards, zero optimizer
updates, no checkpoint, and complete release of all eight GPUs. It contains no
task prompt, model response, tool result, flag, credential, or sealed score.
