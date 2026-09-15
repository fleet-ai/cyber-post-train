# Qwen3.8 fresh-catalog teacher SFT

This arm trains only on successful stronger-teacher sessions from the frozen
50-task training split in
`configs/data/fleet-blackbox-current-study-split-20260914-v2.json`. The 17
development tasks and 8 final-test tasks never enter the corpus.

## Data receipt

- 424 stronger-teacher successes were found in the session catalog.
- 274 match an exact current task version in the training split.
- 115 sessions pass the Qwen tool-interface and task-richness checks, covering
  37 exact task versions.
- Source mix: 113 `gpt-5.6-sol`, 1 `grok-4.5`, and 1 `kimi-k3` session.
- The corpus contains 916 windows, 1,036,061 supervised tokens, and 4,725
  assistant responses.
- 106 sessions using an unproven `search_tool`/`use_tool` interface, 11 with
  opaque context compaction, and 42 with extra user/system transitions are
  excluded. No exact trajectory or window duplicate was found.
- The final wrapper is removed only when it is an exact two-message
  `submit_final_answer` suffix after a completed `submit_report`. The original
  transcript digest remains bound in every row.

The public aggregate manifest and receipt are
`configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json` and
`configs/data/qwen38-fresh75-teacher-sft-train-v1.receipt.json`. Session IDs,
transcripts, scores, and token IDs remain private.

`cyber-post-train data-fleet-teachers` reproduces the private Parquet file from
digest-bound normalized records and success evidence without network or GPU
access. Two independent create-once builds produced the same manifest and
Parquet digests.

## Training recipe

The run configuration is
`configs/runs/qwen38-fresh75-teacher-sft-largest-v1.json`:

- Qwen3.8-27B at exact revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- two epochs, global batch 8, microbatch 1 per GPU
- learning rate `1e-5`
- 230 optimizer steps: `ceil(916 / 8) * 2`
- checkpoint every 20 steps, retaining the newest three
- one 8-GPU node at `c1` priority
- W&B run `thefleet/cyber-post-train/chris-q38-f75-teacher-max-v1`

There is no held-out teacher-token loss. W&B records training loss and runtime
health. Checkpoint choice must use task outcomes on the frozen 17-task Fleet
development split.

## Evaluation handoff

After a checkpoint is sealed and exported:

1. Evaluate it on the 17 Fleet development tasks with the same OpenCode-facing
   tool contract used for the training traces.
2. Run matched base-versus-checkpoint WebExploitBench with OpenCode only.
3. Choose a checkpoint without consulting the 8 final-test tasks.
4. Evaluate the chosen checkpoint once on the 8 final-test tasks.

Every evaluation must bind the exact checkpoint receipt, task selection,
harness image/config, model-serving identity, and sampling settings.
