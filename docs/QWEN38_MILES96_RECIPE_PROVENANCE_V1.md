# Qwen3.8 Miles96 recipe provenance

No completed exact-adapter update receipt exists in the reviewed sources for
the frozen adapter at `978df19a1f6b344e2f88d9502060700a59294681`.
Neeraj's Dataminer v004 is
strong evidence for the broad one-node Qwen3.8 TP4/CP2 96K mechanics, but the
maintained FTI 0.10.27 adapter changes the training-memory contract from 8,192
tokens/GPU with optimizer CPU offload to 49,152 tokens/GPU with whole-trainer
phase offload and no optimizer CPU offload. Those changes require their own
qualification. The frozen implementation states that its exact update/reload
schemas have never admitted an external run
(`training/miles96_mechanics_canary.py:37-45`), and the one-update artifact is
still a non-launchable template
(`configs/qualification/qwen38-miles96-mechanics-canary-v1.template.json:2-35`).

The machine-readable authority is
[`2026-09-24-miles96-recipe-provenance-v1.json`](evidence/qwen38-study/2026-09-24-miles96-recipe-provenance-v1.json).
It contains no task prompt, task/version UUID, verifier UUID, credential,
trajectory, or private cyber reward.

## Evidence boundary

| Source | What it proves | What it does not prove |
| --- | --- | --- |
| `fleet-ai/dataminer_v2@10afa8d064bb3dd1c11c50768590e432dfa69097` evidence snapshot, executed code `8452eda94667567e9357dac0d92a05fc48e2d727`; `experiments/rl-transfer-v004/protocol.md:41-53`, `run_v004.py:21-27,47-67,91-138,151-208`, job manifest `:1-70`, `state.md:2889-2963` | One 8×B300 node, TP4/CP2, 96K, synchronous rollout/update/save; checkpoint-75 resume through steps 76-79 and a completed iteration-79 checkpoint | Fleet cyber distribution, the maintained 0.10.27 source closure, 49,152 tokens/GPU, or the new offload contract |
| `fleet-ai/theseus@d23116f018cb9213f0a3ee6c228d213abcd85d80`, `services/fti/src/fti/trainers/miles/run_fleet.py:143-265,1080-1196,1213-1315` | Exact maintained recipe source: one node, TP4/CP2, 98,304 context, 49,152 tokens/GPU, whole-trainer CPU phase offload, no optimizer CPU offload, Adam LR `1e-6` | A completed update or reloadable checkpoint on that exact stack |
| `fleet-ai/theseus@68da13aa9c226d1bfed0ad59b990b99321d44239`, `services/fti/CHANGELOG.md:29-30,104` | Primary operational reports for reported maintained-Miles optimizer steps, zero-spread filtering and radix behavior | A sanitized UID-bound terminal receipt chain or proof of the frozen 0.10.27 closure |
| `cyber-post-train@978df19a…`, `docs/evidence/qwen36-study/2026-08-31-miles-canary03-terminal-v1.json` | A sealed different-model operational precedent for a one-node/8-GPU optimizer and checkpoint path on Qwen3.6 | Qwen3.8 identity, learning signal, or exact verifier-ID retention; rewards and gradients were zero |
| Maintained 256K route: `run_fleet.py:520-541`, `payloads/tool-use-qwen38-256k-v1.json:1-14`, `CHANGELOG.md:104` | Four-node TP8/CP4 source shape and a historical save/restore attempt | A valid resumed update: the restored trainer met base-weight rollout engines and failed equality |

## Field-level decision

| Field | Neeraj v004 | Frozen adapter / smallest cyber choice |
| --- | --- | --- |
| Runner | Copied custom `run_v004.py` | Call `python -m fti.trainers.miles.run_fleet` directly |
| Shape | 1 node, 8 GPUs, TP4/CP2, 96K | Keep the same broad shape |
| Train packing | 8,192 tokens/GPU | 49,152 tokens/GPU; first qualification |
| Offload | Optimizer CPU offload | Whole-trainer CPU phase offload; no optimizer CPU offload |
| Wave | 8 prompts × 4 samples, concurrency 32 | Qualification: one task × 8 fixed slots, concurrency 2 |
| Sampling | `T=.7`, `top-p=.8`, `top-k=20` | Keep maintained `T=1`; do not import study knobs |
| Optimizer | Adam, LR `2e-6`, 80 steps, save every 4 | Phase one has zero updates; successor has one update at `1e-6`, save 1 |
| Reward | Custom bridge over a non-cyber distribution | Session-open task key/version, verifier-version and raw/projected tool-hash checks; non-zero-spread filter for training |
| Compaction | None | None; bounded 96K qualification only |
| Evidence | Completed broad-mechanics run | Exact source binding; no exact-recipe terminal receipt in the reviewed scope |

## Planned qualification and minimum successor

The score-blind review at
`cyber-post-train@7858ead1f2c498850816168fded1be9d3927fa32` binds four
independent one-node lanes. Each has eight fixed sample slots, concurrency two,
32 turns, 8,192 tokens/turn, a 2,400-second episode limit, zero outer
replacements, zero optimizer steps, and no checkpoint. The job contract is
`c1`/`q1`, backoff zero, no requeue, and root
`fleet.ai/failure-alerts: "off"`. The review file is
explicitly `launchable: false`; it is qualification, not training. No launch or
terminal receipt for it is present in the reviewed sources.

After selecting exactly one lane whose receipt proves at least two normally completed gradeable
episodes, two distinct finite rewards, unique verifier executions, and complete
instance release, the intended training-recipe delta is deliberately narrow:

1. `--mode eval` to `--mode normal`.
2. Zero optimizer steps to one at LR `1e-6`.
3. Enable `check_no_aborted_and_nonzero_std` and checkpoint interval one.
4. Seal the step-0 checkpoint and complete HF export, then independently reload
   on one GPU for one ordinary zero-update episode.

Acceptance requires the optimizer update to complete, at least one trained
tensor to change, and every trained tensor to remain finite; a generic
"finite non-zero update" claim is not the validator contract.

Everything else remains pinned: adapter/source closure, model/image, TP4/CP2
shape, eight slots, concurrency two, the exact session-open checks above, no
outer replacement, evidence separation, and complete resource release. The
successor additionally needs fresh task/runtime preflights and a fresh prepared-
model inventory over both HF and Megatron trees; it must not reuse the phase-one
HF-only model-binding digest as phase-two authority.

Do not create a new runner, copy v004's reward bridge or retry loop, import its
LR/offload/sampling/fan-out, make the four signal lanes training arms, or add
compaction to Miles96. Long-horizon compaction belongs to a separately qualified
rail.
