# Qwen3.8 32K dense-versus-LoRA matched study V1

Status: **source-only successors prepared; no API preview, Kubernetes object,
GPU allocation, optimizer step, checkpoint, or evaluation was created.**

## Question and hypothesis

The primary question is whether a high-capacity, all-linear rank-64 LoRA can
preserve protected Fleet cyber-task outcome quality closely enough to justify
its smaller trainable/checkpoint surface relative to full-weight SFT after the
same one-epoch exposure to the same successful-action targets.

The directional hypothesis is that LoRA at `3e-5` can approach the dense
`3e-6` arm because its rank is relatively high, it targets every linear layer,
and each method uses its own credible learning rate.  The counter-hypothesis is
important: cyber tool-use adaptation may require higher-rank weight changes
than rank 64 can express, and published code/math studies have found that LoRA
can learn less even when it forgets less.  LoRA is worth keeping only if the
predeclared protected-development task metric and invalid-run rate are within
the study's frozen acceptance margin **and** measured checkpoint, transfer, and
serving benefits survive its merge/export workflow.  Smaller adapters alone do
not establish lower wall time: 32K activations and TP8 communication can
dominate.

These are the best-grounded **starting** hyperparameters, not a claim of
optimality.  “Optimal” means the smallest stable method-specific LR/rank/batch/
epoch setting that maximizes the predeclared protected Fleet-development task
metric under the same data-exposure contract; the conditional grid below is
the only authorized way to refine that answer.  Training loss alone cannot
select the winner.

The numeric non-inferiority/reliability margin is intentionally still unknown.
It must be chosen with the matched evaluation protocol before either arm is
launched, not after seeing its score.

## Exact source pair

| Binding | Dense successor | LoRA successor |
|---|---|---|
| Config | [`qwen38-27b-sft-32k-dense-matched-v1.json`](../configs/runs/qwen38-27b-sft-32k-dense-matched-v1.json) | [`qwen38-27b-sft-32k-lora-r64-a32-matched-v1.json`](../configs/runs/qwen38-27b-sft-32k-lora-r64-a32-matched-v1.json) |
| Create-once run | `chris-q38-sft32-dense-m1-v1` | `chris-q38-sft32-lora-m1-v1` |
| Method | Full-weight | Rank 64, alpha 32, zero dropout, Kaiming initialization, all linear layers |
| Learning rate | `3e-6` | `3e-5` |
| Trainer | Pinned SkyRL FSDP image | Pinned `skyrl-fleet-v2` Megatron-LoRA image at `7e9356c8e02e7382e84b8484638baccdd1bbf680` |

Both configs select the versioned `qwen38_sft_32k_matched_v1` wrapper.  The
wrapper binds the unchanged historical runtime bytes separately, admits only
these two identities, and renders the same explicit optimizer fields into both
native runtimes.  Historical dense and LoRA create-once configs are not edited.

## A2 terminal incident and successor repair

The terminal A2 LoRA RayJob was admitted on one B300 node with the exact e04 GPU
cluster selector; placement was not its terminal cause.  It failed before an
optimizer step because its versioned wrapper plan reached the inherited
production qualification helper without first restoring its already-validated
delegated base identity.  The delegated identity itself matched with an empty
diff; the original plan's `runtime_variant` key and wrapper runtime digest made
the later historical helper raise `ValueError: Qwen3.8 LoRA recovery root
differs from its reviewed broad plan`.

The matched wrapper fixes that exact boundary: it validates the complete fresh
identity first, then passes a delegated plan with `runtime_variant` and
`optimizer` removed and the historical runtime digest restored into the
unchanged qualification helper.  The regression first reproduces the A2 error
on the undecorated wrapper plan, then proves the delegated successor reopens the
same digest-bound export receipt reference and reaches the unchanged receipt
validator.  The exact terminal identities, admission timing, error, and
zero-resource cleanup boundary are sealed in the
[sanitized A2 incident receipt](evidence/qwen38-lora-anchor-a2-runtime-root-terminal-20260921.json)
(receipt SHA-256
`92911231923c38ca5d45bfcdfc4dff16d37398640d7f1bc6b0c2d3d818ebe075`).
The receipt also records the one successful Jobs API release
(empty HTTP 204) and the subsequent absence of the exact RayJob UID, Workload
UID, RayCluster, and run-owned pods.  Neither A2 identity is reused or patched.

## What is actually matched

- Model: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Corpus and train split: manifest
  `sha256:a8d08609991d7960cdcdab826bd07c3216bfcd3aa8f6fb33ab7a5fd29648d6f5`,
  split
  `sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`.
  It contains 14,693 packed 32K windows from 2,886 verified-success sessions,
  496 training task keys, 176,654 visible assistant responses, and 57,384,881
  unique supervised tokens.  Every eligible target is trained exactly once.
- Context and batch: 32,768-token windows, global batch 8, microbatch 1, one
  eight-GPU node, 1,837 native kept-tail optimizer steps.
- Duration and seed: one epoch and seed `20260921`.
- Optimizer surface exposed by both pinned SkyRL configurations: Adam betas
  `[0.9, 0.999]`, weight decay `0.01`, maximum gradient norm `1.0`,
  `constant_with_warmup`, zero warmup steps, and no post-step optimizer
  offload.  Learning rate is method-specific by design; transferring the dense
  LR directly to LoRA would not be a fair practical method comparison.
- Objective: visible assistant actions only; copied history and tool output are
  loss-masked.  The task-outcome split has no teacher-CE development file.

The test suite compares the compiled plans and rendered override maps, not just
the JSON text, and rejects drift in seed, batch, epoch count, W&B group, or any
optimizer field.

## Unavoidable method-and-runtime confounds

This remains a **method-and-runtime comparison**, not a one-variable adapter
ablation:

1. Dense uses FSDP with eight data-parallel ranks.  LoRA uses Megatron TP8 with
   one data-parallel rank, so the same global batch is realized through
   different sharding, microbatch dispatch, reduction, and accumulation paths.
2. The native optimizer implementations differ.  Dense constructs
   `torch.optim.AdamW`; Megatron constructs its distributed Adam optimizer.
   The common SkyRL schema exposes and now binds betas, weight decay, clipping,
   schedule, warmup, and offload, but not a shared epsilon or identical
   optimizer-state/reduction implementation.
3. The immutable trainer images and source revisions differ.  BF16 parameter
   handling, FP32 reductions/master weights, RNG consumption, fused kernels,
   and collective order therefore need not be bitwise identical even with the
   same seed.
4. Learning rates differ by an intentional factor of ten.  This asks which
   credibly tuned method works better, not whether the same scalar LR behaves
   differently under two parameterizations.
5. Checkpoint cadence and artifact routes remain runtime-appropriate: dense
   keeps two large full-state checkpoints at a 100-step cadence; LoRA keeps
   three adapter/optimizer checkpoints at 20 steps and requires a separate
   merge/reload acceptance path.  This affects reliability and wall time and
   must be reported.

These differences must stay in every result label.  A protected task outcome
can choose a practical method; it cannot support the claim that adapter
parameterization alone caused the difference.

## Small staged LR/batch/epoch study

Do not run a Cartesian sweep.  Run the prepared pair first, then spend at most
five additional one-node runs before re-review (seven total on any chosen
path).  Every later arm needs a fresh immutable config and is conditional on
protected Fleet-development results.  GPU-hours remain unknown until
exact-runtime canaries provide timing; do not substitute timings from the other
runtime.

| Stage | Arms | Purpose and decision |
|---|---|---|
| 1 — paired anchors | Dense `3e-6`, batch 8, one epoch; LoRA rank 64 all-linear `3e-5`, batch 8, one epoch | Run only this prepared 32K pair first.  It tests the practical LoRA hypothesis with exact data, seed, optimizer, and exposure parity. |
| 2 — narrow LR refinement | Fresh dense `5e-6` and LoRA `1e-4` successors, otherwise identical; add dense `1e-5` only if `5e-6` improves the protected metric without instability and the optimum remains boundary-censored | This covers the small literature-supported sets—dense `{3e-6, 5e-6, 1e-5}` and LoRA `{3e-5, 1e-4}`—without guessing a broad sweep.  Stop as soon as an interior result or instability makes another edge uninformative. |
| 3 — choose one diagnosis, not all | Either one LoRA rank-256 all-linear successor at the selected LoRA LR; **or** a fresh dense/LoRA batch-16 pair; **or** a fresh dense/LoRA 96K-context pair | Choose rank only if rank 64 appears capacity-limited, batch only if optimization/noise is the live question, and 96K only after the 32K pair is accepted and longer-context truncation is measured.  Do not pool any branch with the 32K/batch-8 result. |
| 4 — duration branch instead of Stage 3 | Fresh two-epoch dense/LoRA successors at the selected LR/batch/context | Consider this pair only after a one-epoch arm improves the protected Fleet-development metric and neither method is diverging.  At batch 8 each sees 114,769,762 supervised target tokens over 3,674 kept-tail updates.  Freeze a new two-epoch schedule; never relabel an early checkpoint from a shorter horizon. |

This design follows the closest useful evidence without treating it as a
default.  [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) reports
that all-layer, sufficiently high-rank LoRA can track full tuning in some SFT
settings, often prefers roughly ten times the full-tuning LR, and can be less
tolerant of large batches.  [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
finds meaningful code/math gaps and high-rank full-tuning perturbations, so a
positive result is not assumed.  The closest cyber recipe found,
[CTF-Dojo](https://arxiv.org/abs/2508.18370), reports Qwen3-32B full SFT at 32K
context, global batch 16, learning rate `5e-6`, and two epochs; those settings
motivate bounded controls, not direct transfer.  None of these sources studies
this exact cyber-agent corpus, Qwen3.8 revision, or FSDP/Megatron pair;
protected Fleet task outcomes, not imitation loss, decide here.

## Launch prerequisites

No external action is authorized by these configs.  Before minting or launching
the first executable successor:

1. Merge the reviewed source and start from that exact clean commit.  Re-run the
   focused tests and prepare both configs locally into new directories.
2. Run the exact-image, zero-GPU CPU preflight for each arm.  It must prove the
   corpus/model digests, native loader, and the effective betas, weight decay,
   clipping, scheduler, warmup, LR, seed, and topology.
   The LoRA arm must use the named Qwen3.8 rank-64 route and produce its typed
   receipt; the generic SFT receipt rail rejects this exact runtime.
3. Qualify the new versioned wrapper with bounded one-step GPU canaries for both
   FSDP and Megatron.  Require finite loss/gradient, one real update, intended
   changed tensors, recoverable checkpoint state, exact reload, and full
   resource release.  Re-open the accepted LoRA merge/full-model reload chain;
   historical A2 evidence does not qualify this new identity.
4. Prove the new job, output directory, and W&B identities are absent.  Never
   reuse or patch a prior create-once RayJob.
5. Obtain a fresh Jobs API preview.  The maintained direct SFT fallback accepts
   only the generic `workload: fleetai-training-ng-gpu` selector, and only in
   exact production context
   `nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6` adds
   `topology.nebius.com/gpu-cluster-id: computegpucluster-e04x263hvn91b321fq`
   to every head/worker template.  It adds the mandatory root
   `fleet.ai/failure-alerts: "off"` annotation before creation, then requires
   exact API-server dry-run and persisted readback equality.  Development or
   unknown contexts and missing, different, or pre-bound source selectors fail
   closed.  This repair is SFT-only; never patch an already-created RayJob.
6. Run the Fleet-team/account check, cross-namespace GPU census, capacity and
   duplicate checks, then freeze the paired development evaluation protocol,
   primary metric, non-inferiority/reliability margin, exclusions, serving
   parity, and checkpoint rule before results are visible.

Final Fleet and external benchmark results remain sealed confirmation only and
must never flow back into recipe, corpus, or checkpoint selection.
