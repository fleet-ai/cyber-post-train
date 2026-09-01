# Qwen3.8-27B training and serving qualification

This is the launch plan for repeating the Qwen3.6 cyber post-training study with
`Qwen/Qwen3.8-27B`. It deliberately stops before a paid SFT or RL launch. The
machine-readable source of truth is
[`configs/qualification/qwen38-27b-v1.json`](../configs/qualification/qwen38-27b-v1.json).

## Current operational state

**Proven:** the exact train-only corpus is locally verified, the Training API
catalog has no Qwen3.8 row, serving and training storage are distinct, and the
Ready trainer candidate has not demonstrated Qwen3.8 forward/backward,
optimizer, checkpoint, or export compatibility. **Pending:** the 55.6 GB model
rematerialization, corpus staging, catalog registration, positive reward gate,
and one-step capability run. Draft PR
[#78](https://github.com/fleet-ai/cyber-post-train/pull/78) remains an unmerged,
dry-run-only staging bridge and requires explicit transfer approval. See the
[`sanitized overnight handoff`](OVERNIGHT_OPERATIONAL_STATE_2026-09-01.md).

## Current decision

Qwen3.8-27B is an appropriate second student because it is the same dense
`qwen3_5` architecture and exact 27,781,427,952-parameter scale as Qwen3.6, but
has materially stronger served capability. It keeps the topology and most of
the hard-earned Qwen trainer work relevant while directly testing whether the
Qwen3.6 reward floor was model-specific.

It is not a drop-in replacement:

- the immutable checkpoint is a different 18-shard BF16 artifact;
- the tokenizer configuration and chat template have different hashes;
- it was released after WebExploitBench, so external results support a matched
  before/after intervention claim, not a temporally clean absolute benchmark
  claim; and
- the Fleet Training API does not currently know or stage this model.

The exact model and serving locks are supplied by the external-evaluation track
in PR #65. The training qualification is stacked on that shared identity rather
than copying it.

## Exact model and serving identity

The frozen checkpoint is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0. Its 18 BF16
safetensor files contain 1,199 tensors and 27,781,427,952 parameters. The shard
files total 55,563,006,776 bytes; tensor payload is 55,562,855,904 bytes. The
canonical LFS manifest is
`sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352`.

The live base route was observed as `qwen3.8-27b`, BF16, SGLang, TP1, 262,144
context, FP8-E4M3 KV cache, `trtllm_mha`, reasoning parser `qwen3`, and tool
parser `qwen3_coder`. Its exact image is
`lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`.
A forced structured-tool probe passed. A post-training route must clone this
registration and change only checkpoint identity, paths, and the new served id.

## Live trainability blocker

At 2026-09-01 06:39 UTC, `GET https://api.ft.flt.build/v1/models` listed only
`qwen2.5-1.5b-instruct` and `qwen3.6-27b` as staged. It contained no Qwen3.8
catalog row. A read-only SFT preview with `staged_model=qwen3.8-27b` returned
HTTP 422:

> Unknown base model 'qwen3.8-27b' ... A model must be staged to
> /mnt/sfs/models by the model-stage chart and registered in the models table
> before a run can use it.

The prepared correction mirrors the exact Qwen3.6 staging recipe without
submitting it. `chris-cyber-qwen38-stage-1d4bf0f2` is a CPU-only, Kueue-managed
job in `training-lq`. It downloads into a partial directory, verifies all 18
LFS shard digests and every runtime sidecar, writes an immutable checkpoint
lock, then atomically promotes the exact revision directory. A separate
create-only `chris-cyber-qwen38-canonical-link` job exposes that exact tree as
`/mnt/sfs/models/qwen3.8-27b`; it refuses to replace any existing nonmatching
path. Both checked-in manifests remain suspended plans and were not submitted.

The end-to-end correction is:

1. add `Qwen3.8-27B` as a base-model catalog row with provider `Qwen`, exact
   repository `Qwen/Qwen3.8-27B`, and context 262,144;
2. run the reviewed immutable stage job with the exact 40-character revision,
   never `main`;
3. verify its promoted SFS tree and lock against the committed model lock;
4. create the canonical link only after that verification;
5. register local name `qwen3.8-27b` in the Training API catalog; and
6. repeat the server preview and bind the returned model path and staged bytes.

This is a shared Training API/catalog change and is intentionally not hidden in
this repository. No manual alternate model path or mutable alias is accepted.

## What is reusable and what must be rebuilt

The task split, verified-success selection, objective, learning rate, one-epoch
rule, exact task-version bindings, deterministic verifiers, external holdouts,
and evaluation harnesses remain unchanged. SFT and RL configs are separate from
the model adapter, so another dense Qwen checkpoint changes only model-specific
identity and compatibility fields.

The Qwen3.6 token windows are not reusable. Qwen3.8's exact tokenizer config
and chat template differ, and window boundaries and target coverage are
tokenizer-dependent. Rebuild the same five assistant-ending windows per
verified success under job id `chris-cyber-qwen38-windowed-v2`, then prove:

- 130/10/20 train/dev/test lineage identity is unchanged;
- all verified-success final turns remain selected;
- every window is at most 14,336 Qwen3.8 tokens;
- the staged corpus manifest binds the new tokenizer revision and hashes; and
- no WebExploitBench, ExploitGym, Fleet test20, grader, or verifier source is
  present.

The `v2` corpus name supersedes an unstaged local `v1` draft whose output
manifest did not bind the tokenizer files. The builder now admits only the
`train` split before eligibility or tokenization and verifies every tokenizer
file against the exact model lock before writing any corpus output.

The ignored local `v2` artifact is now built and verified: 508 successful train
sessions from 86 train lineages produced 2,540 windows, with no oversized
targets and a maximum of 14,335 tokens. Its manifest is
`sha256:95a4f1293fe3d964e0b046754feb05fec268a15aa1623f87dfc30ea1282bbb80`.
It has not been copied to SFS or registered with the Training API.

## SFT rail

The candidate trainer is the exact image that completed Qwen3.6 SFT:
`fleet/skyrl-train:q36-torchgdn-6db8d0c9@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`.
Both models use `qwen3_5`, so this is the best starting point, but it is not
evidence that Qwen3.8 works. The config therefore leaves `trainer_version_id`
unset.

The first paid training job should be
`chris-cyber-qwen38-27b-sft-one-step-capability-gate-v1`, only after every
blocker is closed. It uses one 8×B300 worker, FSDP, BF16, microbatch one per GPU,
the exact rebuilt corpus, a before-train development evaluation, one optimizer
step, one checkpoint, and a post-step evaluation. Acceptance requires finite
loss and gradients, an actual backward and optimizer event, a complete durable
checkpoint, exact runtime image identity, and no model or data substitution.

Only that green gate releases the one-epoch config. The full run is never the
compatibility test.

### Resource arithmetic

For 27,781,427,952 parameters, the static model-state lower bound is:

| State | Bytes |
|---|---:|
| BF16 weights | 55,562,855,904 |
| BF16 gradients | 55,562,855,904 |
| two FP32 Adam moments | 222,251,423,616 |
| optional FP32 master weights | 111,125,711,808 |
| total with master weights | 444,502,847,232 |
| even FSDP share over eight ranks | 55,562,855,904 per rank |

This makes the successful Qwen3.6 one-node topology plausible. It does not
prove activation, temporary all-gather, kernel, or fragmentation headroom; the
one-step gate measures those rather than inferring them.

## RL rail

RL is gated twice. First, the independent 20-task Qwen3.8 Fleet calibration
must return at least one positive authoritative reward with valid cleanup. A
served model success only establishes capability; it does not release training
unless the exact task version, verifier execution, and reward path are retained.

Second, the native RL canary must use the two metadata-only exact-tool
successors already created for Qwen3.6. Those task versions differ from their
parents only by ordered `metadata.tools=[bash, submit_report]`; they were not
promoted to mutable current. The canary requires:

- an exact Ready trainer image containing the long-horizon and execution
  allowlist controls;
- 600 turns, a 65,536-token trajectory, 16,384 prompt tokens, up to 49,152
  generated tokens, and 16,000-character tool-result clipping;
- two TP4 inference engines on one 8×B300 worker;
- one true optimizer step through explicit
  `trainer.max_training_steps=1`;
- eight planned rollouts, exact per-episode authoritative verifier execution
  IDs, at least one nonzero reward, and one durable checkpoint; and
- no context truncation before `submit_report`.

All-zero valid rewards remain a measured model result. Missing verifier IDs,
truncation, parser failure, or environment failure is infrastructure-invalid.
The full 129-train/10-dev as-treated RL run is released only after this canary
is reward-bearing and its checkpoint is verified.

## Checkpoint, export and serving handoff

The Qwen3.6 handoff is reused as an algorithm, not as a set of artifact values:

1. select exactly one final promoted training checkpoint;
2. resume it at its final step in a separate queued export and prove zero new
   optimizer steps;
3. inspect actual safetensor dtypes and never relabel FP32 as BF16;
4. if necessary, deterministically cast FP32 to BF16 and reopen every tensor to
   prove bit equality with a direct cast;
5. compare the export to the exact Qwen3.8 base. Do not assume Qwen3.8 omits the
   same 15 MTP tensors as Qwen3.6; derive and review the exact difference;
6. compose trained weights with byte-identical Qwen3.8 runtime sidecars;
7. atomically stage to a new inference path with no replacement and complete
   source/destination manifests;
8. clone the exact base SGLang registration and change only checkpoint identity
   and paths; and
9. require Ready/live identity, tokenizer, structured-tool, fixed-logit, and
   no-speculative-decoding parity receipts before any post-training evaluation.

Every stage binds request, controller and Pod UIDs, requested and resolved image
digests, queue/resources, logs, optimizer count, and complete file manifests.

## Current blockers and next action

No cluster job has been submitted by this qualification track. The next useful
external actions are model catalog registration/staging and completion of the
Qwen3.8 Fleet reward calibration. After both are green, rebuild and stage the
Qwen3.8-tokenized SFT corpus, bind an exact Ready trainer version, run an
unpaid server preview, and submit exactly the one-step SFT capability gate.

Full SFT and RL remain prohibited until their respective gates pass.
