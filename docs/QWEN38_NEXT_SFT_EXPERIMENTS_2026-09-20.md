# Qwen3.8 next SFT experiments — 2026-09-20

## Decision

The next paid SFT comparisons are **not another learning-rate sweep**. The
current 32K full-weight study already covers batch 8 versus 16 and learning
rate `3e-6` versus `1e-6`; the LoRA lane separately covers adapter learning
rate. The next two arms instead ask one new question:

> Does preserving more of a successful cyber trajectory before each supervised
> action improve held-out task outcomes?

The 64K and 96K arms below use exactly the same 57,384,881 supervised target
tokens as the 32K anchor. They change only how those targets and their preceding,
loss-masked history are grouped into windows. Both exact treatments already
passed a one-step forward, backward, optimizer, checkpoint, and resource-release
gate on one eight-GPU B300 node.

The corpus is deliberately visible-action SFT rather than teacher
chain-of-thought distillation.  See
[`QWEN38_REASONING_AND_COMPACTION_POLICY.md`](QWEN38_REASONING_AND_COMPACTION_POLICY.md)
for the private-reasoning boundary, current compaction treatment, and the
separate admission gate for a future student-visible reasoning corpus.

## Shared scientific contract

- Model: `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Data: 2,886 verified stronger-teacher success sessions, 496 training task
  keys, 176,654 supervised assistant responses, and 57,384,881 unique
  supervised target tokens.
- Split: `sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`.
- Held-out protection: all 25 frozen Fleet development/final task families are
  excluded across every task version. WebExploitBench is absent from training.
- Optimization: full-weight SFT, one epoch, global batch 8, microbatch one per
  GPU, learning rate `3e-6`, seed `20260920`, one eight-GPU node, c1 priority.
- Selection: fresh outcomes on the protected Fleet development tasks. Training
  loss is a health signal, not the checkpoint-selection metric.
- Confirmation: the untouched Fleet final split and WebExploitBench are used
  only after development-set selection.

The source identities and exclusions are identical across 32K, 64K, and 96K.
The number of optimizer steps differs because longer windows pack the same
targets into fewer rows; this is an inseparable, declared part of the context
treatment rather than an accidental change in target-token exposure.

## Prioritized arms

### 1. 64K history treatment — deterministic successor qualification

**Motivation.** A 32K window may omit earlier reconnaissance and tool outcomes
that explain why a later expert action was taken. Doubling the window to 65,536
tokens, with up to 16,384 preceding tokens copied as loss-masked context, should
make intermediate actions more interpretable without approaching the model's
262K context ceiling.

- Current config: `configs/runs/qwen38-teacher3k-64k-full-b8-lr3e6-v5.json`
- Data: 8,953 windows; all 57,384,881 targets appear exactly once.
- Planned work: 1,120 optimizer steps on one eight-GPU node.
- W&B: group `qwen38-teacher3k-64k-v1`, create-once successor run
  `chris-q38-t3k64-b8-v5`.
- Principal contrast: 65,536-token windows and 16,384-token masked history
  instead of 32,768/8,192. Model, targets, split, batch, learning rate, epoch,
  seed, and topology are fixed.
- Stop rule: stop only for a non-finite update, restart, deterministic input
  mismatch, or confirmed lack of progress under the operator idle bound.
  Otherwise finish the predeclared single epoch. Do not stop or select from a
  favorable training-loss fluctuation.

The active `v5` identity replaces the earlier prepared identities only to make
checkpoint recovery practical: it saves every 15 steps, retains the latest two
checkpoints, and carries a bounded first-checkpoint horizon.  Its model, data,
context treatment, optimizer, batch, seed, epoch count, and topology remain the
same scientific comparison.  The checkpoint policy is recorded separately in
[`qwen38-broad-sft-checkpoint-retention-policy-20260921.json`](evidence/qwen38-broad-sft-checkpoint-retention-policy-20260921.json).

### 2. 96K history treatment — deterministic successor qualification

**Motivation.** If 64K still loses necessary early trajectory state, 98,304
tokens with up to 24,576 masked history may preserve it. Comparing 96K with the
matched 64K and 32K arms tells us whether the benefit continues, saturates, or
reverses as sequences become longer and each step becomes more expensive.

- Fresh successor config: `configs/runs/qwen38-teacher3k-96k-full-b8-lr3e6-v3.json`
- Data: 6,847 windows; all 57,384,881 targets appear exactly once.
- Planned work: 856 optimizer steps on one eight-GPU node.
- W&B: group `qwen38-teacher3k-96k-v1`, create-once successor run
  `chris-q38-t3k96-b8-v3`.
- Principal contrast: 98,304-token windows and 24,576-token masked history.
  Every other scientific input is fixed.
- Stop rule: the same operational stop rule as the 64K arm. A successful
  one-step canary is necessary but does not by itself prove full-epoch lift.

The old `v2` name and output root are never reused.  The `v3` successor changes
only create-once identities and recovery settings: it saves every 50 updates
and keeps two checkpoints.  A full checkpoint is large, so this is deliberately
less frequent than the 64K arm's 15-update cadence but much safer than the
old 210-update interval.  It does not set a numerical recovery-time horizon:
we have no accepted 96K step-rate measurement from which to honestly derive
one.  The zero-GPU preflight must confirm the exact loader and checkpoint path;
the launch watchdog must use observed progress rather than inventing a timing
claim.

## Runtime incident and exact repair

The first full-run identities for both arms reached model setup but failed
before optimizer step 1. This was one shared infrastructure defect, not a model
or data result:

- `chris-q38-t3k64-b8-v1` ended at `2026-09-20T23:39:48Z`.
- `chris-q38-t3k96-b8-v1` ended at `2026-09-20T23:41:59Z`.
- Both public receipts stopped at runtime stage `device_ready`, reported
  `ModuleNotFoundError`, recorded optimizer step zero, and released their Pods,
  RayClusters, and GPU nodes.

The exact full-weight image is built from SkyRL `f5bc3b78`. Its SFT trainer
accepts the prepared `list[dict]` rows directly and does not contain
`skyrl.train.dataset.sft_dataset`. A later repair for the separately pinned
Megatron-LoRA image had accidentally imported that LoRA-only `TextDataset`
wrapper for every SFT plan. The current repair routes by the already-frozen plan
family: full-weight FSDP returns the prepared rows unchanged; Qwen3.8
Megatron-LoRA retains its native wrapper. The CPU preflight now calls the exact
staged training loader inside the pinned image so this class of mismatch fails
before a GPU is requested.

The `v2` identities change only the create-once output/W&B names and the runtime
repair tag. Model, data, context treatment, optimizer, batch, seed, epoch count,
and topology are byte-for-byte the same scientific comparison as `v1`.

### 3. Conditional second epoch of the best context — held

**Motivation.** Closely related cyber SFT work, including CTF-Dojo, has used two
epochs over successful trajectories. A second epoch is worth testing only after
we know which context treatment transfers best; running it now would confound
context with training duration and spend compute before the first-epoch result
exists.

- Principal contrast: exactly one additional pass over the winning first-epoch
  corpus, resumed from its exact terminal checkpoint. No data, context, batch,
  learning-rate, or seed change.
- Admission rule: launch only if a 32K/64K/96K first-epoch checkpoint improves
  the predeclared Fleet development outcome metric without a material regression
  in reliability. WebExploitBench is confirmation, not the tuning target.
- Compute: one more epoch on one eight-GPU node; the exact step count is the
  winning arm's row count divided by batch 8.
- Stop rule: do not launch when no first-epoch arm beats the base model on the
  protected development set. If launched, stop on the same operational defects
  above and compare epoch one versus epoch two on identical evaluation inputs.

## Explicitly deferred

- **Tiny self-success SFT:** the currently prepared self-success corpus has only
  about 1.3M supervised tokens from 49 sessions. It is too small and narrow to
  be the next production comparison against the 57.38M-token teacher corpus.
- **Teacher/self mixture:** repeating the small self corpus until it has visible
  weight would change both source identity and repetition rate. Build a broader,
  held-out-safe self-success corpus first, preferably at least 20M unique
  supervised tokens.
- **More LoRA rank/LR knobs:** the LoRA lane is already testing optimizer scale,
  and its memory-accumulation defect must be understood before another adapter
  arm is scientifically useful.
- **Task-balanced resampling:** promising, but it requires a new deterministic
  corpus manifest that proves retained target coverage and the same held-out
  exclusions. It should not be improvised inside a launch config.

## Literature transfer limits

- The official Qwen3.8 release documents a 262,144-token inference context,
  making 64K and 96K conservative relative to the model limit. It does not
  publish a directly transferable cyber SFT recipe:
  <https://github.com/QwenLM/Qwen3.8/blob/main/README.md>.
- CTF-Dojo trains on successful cyber trajectories and reports global batch 16,
  learning rate `5e-6`, and two epochs. It supports testing successful-trajectory
  SFT and a conditional second epoch, but its tasks and models differ from ours:
  <https://openreview.net/pdf?id=FAPEir5GyY>.
- The separate LoRA study follows the Tinker guidance that adapter learning
  rates are often much higher than full-weight rates and that broad supervised
  data can require substantial rank. That guidance does not justify changing
  the full-weight context study:
  <https://thinkingmachines.ai/blog/lora/>.

These are priors, not proof. Held-out Fleet outcomes and matched
WebExploitBench confirmation remain the decision evidence.
