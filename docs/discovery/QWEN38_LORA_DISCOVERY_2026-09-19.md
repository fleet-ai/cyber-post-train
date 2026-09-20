# Qwen3.8-27B LoRA cyber research discovery

Status: active discovery, 2026-09-19

This is the high-retrieval handoff for the Qwen3.8-27B LoRA research program.
It records what is proved, what is only a candidate, and what must pass before a
paid training or evaluation job is created. The companion machine-readable
index is `configs/discovery/qwen38-lora-evidence-index-v1.json`.

The objective is not to reproduce training loss. For this program, “10% lift”
means at least a 10% **relative** improvement over a byte-identical base-model
control in each predeclared macro pass@1 measurement. Absolute percentage-point
change and uncertainty are reported as well. The two measurements are:

1. WebExploitBench Level-0, using one fixed OpenCode protocol; and
2. lineage-held-out Fleet blackbox-exploit tasks.

Neither WebExploitBench nor the frozen Fleet final set may be used for training,
hyperparameter selection, or checkpoint selection. If a matched base score is
zero, relative lift is undefined and the goal cannot be claimed from that
measurement without a separately predeclared absolute criterion.

## Discovery verdict

Fleet already has strong execution proof for **full-weight** Qwen3.8 training,
but no qualified exact-Qwen3.8 LoRA production recipe:

| Capability | What is proved | What is not proved |
|---|---|---|
| SFT | Qwen3.8 full-weight FSDP completed 186-step and 230-step runs on B300s. The 230-step checkpoint was sealed, exported, reloaded, staged, and registered. | No exact-Qwen3.8 LoRA SFT run has passed forward, backward, optimizer, save, reload, merge, and serving-parity gates. |
| RL | Fleet's maintained Miles path has performed many finite full-weight Qwen3.8 GRPO updates and saved checkpoints. | The maintained Qwen3.8 recipe exposes no LoRA path; its Mooncake pilot explicitly rejects positive LoRA rank. Its agent loop is not OpenCode and does not compact context. |
| LoRA RL | A retired `dataminer_v2` PEFT prototype performed real Qwen3.8 LoRA-GRPO updates. Current SkyRL has a Qwen3.5-architecture Megatron bridge, LoRA, and per-turn training support that make it a strong exact-model candidate. | The old prototype truncated updates, was not a supported Fleet recipe, and could not train compacted histories. Current SkyRL has no exact Fleet Qwen3.8 LoRA cyber receipt or Fleet compaction integration yet. |
| WebExploitBench | TensorLake collection and scoring can be separated. Unmerged commit [`ad7d05c0`](https://github.com/fleet-ai/cyber-post-train/blob/ad7d05c0/docs/WEB_EVAL_RESULTS_2026-09-15.md#fresh75-step-230-complete-pass1-evaluation) reports 15/15 scored Fresh75 attempts (3/110), but its private receipt bytes are not in current main and no matched base exists. | There is no complete matched OpenCode base-versus-trained comparison that can establish lift. |

Therefore the leading implementation candidate is **current SkyRL Megatron
LoRA**, not Miles. It is not currently launchable for exact Qwen3.8. Static
review found two code-level gaps that must be repaired and tested first: the SFT
configuration does not route Qwen3.8 through SkyRL's text-only Qwen3.5 bridge,
and the default Megatron `all-linear` target list has no proof that it covers
Qwen3.8's Gated-DeltaNet projections. It must also use per-turn training records
so context compaction remains valid. The older full-weight recipes remain
controls and infrastructure references, not substitutes for the requested LoRA
experiments.

## Historical training evidence

The machine index currently names 62 executed training jobs grouped into their
scientific lineages: 37 SFT jobs and 25 RL jobs. It also inventories seven
checkpoint candidates, eight WebExploitBench evaluation lineages, and three
Fleet-evaluation lineages. Grouping retries under one lineage prevents a failed
preflight and its corrected successor from being mistaken for two experiments;
every recoverable job name remains listed inside the lineage.

### Successful full-weight SFT

The strongest reusable Qwen3.8 SFT execution proof is Fresh75 V4:

- Run: `chris-q38-f75-max-full-v4-c96ff205`.
- Topology: one node, eight B300 GPUs.
- Recipe: BF16 full-weight FSDP, two epochs, global batch 8, microbatch 1,
  learning rate `1e-5`, and 16,384-token windows.
- Data: 916 windows, 1,036,061 supervised tokens, 4,725 assistant responses,
  115 successful teacher sessions, and 37 exact task versions.
- Completion: 230/230 optimizer steps and 2,072,122 supervised tokens.
- Artifact: sealed 33-file step-230 checkpoint, then accepted BF16 export,
  CPU/GPU reload, staging, and registration.
- Evidence: `configs/runs/qwen38-fresh75-teacher-sft-largest-full-v4.json`,
  `configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json`, and
  `docs/evidence/qwen38-fresh75-*.json`.

The earlier dense teacher V5 run also completed:

- Run root: `/mnt/sfs/jobs/chris-cyber-q38-teacher-dense-v5`.
- 186 optimizer steps and 1,622,545 supervised tokens.
- Held-out teacher-token loss fell from `0.719304` to `0.375388`.
- The step-186 checkpoint was independently sealed. Later, a separate zero-GPU
  BF16 export and one-GPU finite synthetic reload were accepted without taking
  another optimizer step. Serving parity and task-solving lift were not proved.
- This loss reduction measures imitation of held-out teacher text. It must not
  be presented as evidence of better exploitation ability.

A self-trace arm reached step 44 and its 33-file checkpoint was independently
sealed. A separate zero-GPU BF16 export and one-GPU finite synthetic reload were
also accepted. It remains a training artifact rather than an accepted research
result because serving parity and capability lift were not established.

The original Qwen3.8 318-step teacher run used 508 teacher successes across 86
lineages, producing 2,540 windows at batch 8, one epoch, and learning rate
`1e-6`. It completed all training steps but timed out in the final distributed
Hugging Face export. A resumable step-300 checkpoint and raw step-318 export
were reported, but no surviving sealed/reloadable artifact is accepted today.
This is medium-confidence historical evidence because the sanitized record does
not retain the immutable model-lock receipt. Later work deliberately separated
training, checkpointing, export, and reload so an export failure could not erase
a good training outcome.

### Successful full-weight RL references

The maintained Fleet Training Infrastructure (FTI) / Miles recipe is the best
cluster and optimizer reference, but not the LoRA implementation:

- 96K route: one or two 8-B300 nodes, tensor parallel 4, context parallel 2,
  sequence parallelism, 98,304-token context, 81,920-token response budget.
- Native-context route: four 8-B300 nodes, tensor parallel 8, context parallel
  4, 262,144-token context, 245,760-token response budget.
- Typical optimizer: Adam, learning rate `1e-6`, weight decay `0.1`, betas
  `0.9/0.98`, GRPO clip `0.2/0.28`, microbatch 1, temperature 1.
- `deniz-qwen38-tu-074-01` performed 46 finite steps over 4,116 episodes and
  saved two 538 GB checkpoints.
- `deniz-qwen38-tu-075-01` completed a finite 128-sample update.
- `deniz-qwen38-radix-on-01` ran for more than 14 hours without an engine death.
- The earlier two-node 256K run completed six finite steps, but its longest
  response was only 69.3K. A configured limit is not proof of a maximum-length
  training sequence.

Miles cannot be the requested implementation because its
[`mooncake.py`](https://github.com/fleet-ai/theseus/blob/cc18d2cd3e9370abf4f6f19df317d96ce6b619e4/services/fti/src/fti/trainers/miles/mooncake.py#L77-L78)
rejects positive LoRA rank.

### Working but retired LoRA RL prototype

`fleet-ai/dataminer_v2@d267fe246acc805ef72d4962eb79a2feea754b2c`,
`experiments/rl-transfer-v002/`, is genuine prior art:

- Qwen3.8 PEFT targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`.
- Ranks 8, 32, and 64; the main arms used rank 32, alpha 1, dropout 0.
- Main learning-rate arms: `1e-5`, `1e-4`, and `1e-3`.
- Four TP1 vLLM replicas plus three one-GPU update arms.
- It completed a real 1.46-million-token update and later finite multi-epoch
  wave-2 updates.

This proves that exact-model PEFT updates are possible. It does **not** qualify
the production path: generation allowed 262K, the H200 update was truncated at
70K, B300 tests fit about 160K and OOMed around 200K, and the literal-prefix
estimator could not support context compaction.

## Why current SkyRL is the leading LoRA path

The current `fleet-ai/skyrl-fleet-v2` tree contains four relevant pieces:

1. A Megatron bridge for unified Qwen3.5-style checkpoints routed through the
   GPTModel Gated-DeltaNet path. Qwen3.8 appears to share this architecture,
   but exact-model loading and tensor conversion still require qualification.
2. LoRA on both FSDP and Megatron. For Megatron, `all-linear` maps to fused
   QKV, attention output, fused MLP input, and MLP output projections.
3. vLLM adapter synchronization and checkpoint support.
4. **Step-wise training**: every model turn is a separate sample containing
   the exact prompt tokens the model saw, exact output tokens it generated,
   and aligned rollout log-probabilities.

The fourth item supplies the representation needed to train compacted turns;
it does not itself compact a Fleet cyber episode. A compacted turn does not
have to be a literal prefix extension of the prior turn. Each turn is trained
from its own exact state and action, while the final task reward is broadcast to
the turns in that trajectory. This is compatible with outcome-based GRPO/RLOO
only when the environment actually performs compaction, preserves trajectory
boundaries, and records the exact prompt tokens seen on every turn. It is more
expensive than one concatenated forward pass and must be benchmarked.
Retokenizing a rewritten conversation or pretending a summary is the original
prefix is not valid.

Step-wise training also changes weighting: a longer trajectory contributes more
turn samples unless the loss explicitly normalizes by trajectory. That choice
must be predeclared and logged so “longer” is not silently treated as “more
important.” Generic save, reload, and vLLM adapter-sync code is present, but no
exact-Qwen3.8 receipt proves those paths yet.

### Static qualification findings at the pinned SkyRL revision

The core review used
`fleet-ai/skyrl-fleet-v2@fe1ad6c154f6c40bfcfb3c514f5afe3b1dbed5a4`.
The Fleet integration branch at
`a3ab2d6da6ffe897fc00a143013efbaf6c940361` is a descendant, and the four
affected core files are byte-identical across the two revisions. The review
found two confirmed launch blockers and one missing acceptance test:

1. `skyrl/train/config/sft_config.py::build_skyrl_config_for_sft` does not copy
   a `language_model_only` setting into the internal policy configuration, and
   the public `SFTConfig` has no corresponding field. By contrast,
   `megatron_worker.py::init_configs` only forces the text-only
   `Qwen3_5TextForCausalLM` bridge when `language_model_only=True`. Without that
   route, the unified conditional-generation bridge self-packs the model and
   conflicts with SkyRL's packed SFT path. This must be a typed configuration
   field, copied consistently to the trainer and exact-image tested; a command-
   line override that the SFT config silently drops is not a repair.
2. `megatron_worker.py::configure_lora` maps Megatron `all-linear` to only
   `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`. Qwen3.8 has 48
   Gated-DeltaNet linear-attention layers in addition to 16 full-attention
   layers. No exact-model trainable-parameter census currently proves that this
   fixed target list reaches the intended Gated-DeltaNet projections. The
   qualification must enumerate every adapter tensor by layer and fail if any
   intended linear projection is absent or any base tensor is trainable.
3. SkyRL has generic adapter checkpoint and vLLM synchronization code, but its
   trainer-level Megatron LoRA checkpoint test remains excluded/TODO and there
   is no exact-Qwen3.8 save, zero-step reload, merge, or sampler-parity test.
   The current adapter reload and HF export paths use `strict=False`; reload
   rejects unexpected keys but does not reject missing adapter keys. Exact-model
   acceptance must therefore add a complete expected-key manifest rather than
   treating a non-throwing load as success.
4. Neither bundled cyber-facing generator is currently a valid compaction path.
   The Harbor generator explicitly rejects summarization and asserts that an
   episode has one segment. The ThunderAgent integration imports and calls
   parent interfaces that are absent at this revision, does not emit the
   trajectory-boundary fields required by step-wise training, and re-tokenizes
   messages. Core step-wise data structures can represent discontinuous prompts,
   but a repaired environment adapter must produce exact per-turn tokens and
   trajectory IDs after real compaction.

These findings prevent a paid exact-model SFT or RL submission. The correct
next action is a code-and-test repair followed by the CPU/image and synthetic
distributed gates below, not a speculative cluster canary.

### Supported image-build path and its current blockers

Fleet's supported build route is `ftl build` against the development image API,
which creates a batch job in the development cluster's `buildkit` namespace and
publishes an immutable ECR digest. It is not the old SkyRL raw-`kubectl` Docker-
in-Docker script: that script targets a retired AWS cluster, uses a branch and
short mutable tags, injects a GitHub token, and directly creates and deletes
privileged jobs.

The supported route cannot yet build the chosen SkyRL revision:

- The upstream core commit
  `fe1ad6c154f6c40bfcfb3c514f5afe3b1dbed5a4` does not contain the Fleet-v2
  Dockerfile. The Fleet integration commit
  `a3ab2d6da6ffe897fc00a143013efbaf6c940361` is 29 commits ahead and supplies
  the Dockerfile and environment. The qualification fixes must be rebased onto
  that exact descendant so the built image contains both the reviewed core and
  the Fleet runtime; building either ancestor alone would test the wrong source.
- The development BuildKit role currently permits publication only to
  `fleet/miles-trainer`. A dedicated immutable SkyRL repository must be created
  and added to the role through Terraform before a build is attempted.
- The older SkyRL Dockerfile requests a BuildKit secret named `gh_token`, while
  the supported Fleet image API provides private-Git access through an SSH
  mount and does not provide that secret. The Dockerfile must use the server-
  supplied SSH contract, or the API must gain a server-managed secret. A token
  must never be passed as a build argument.

There is no image-build preview endpoint: `ftl build submit` creates a real
development build job and pushes an image. Before that call, resolve every
source and platform reference to a full commit, run the exact-checkout CPU
tests, verify authenticated read-only API routing, verify the repository/IAM
and secret contracts, and predeclare the output tag as the full source commit.
After success, record the returned digest immediately and consume only
`repository@sha256:...`; non-Miles build records disappear after the build-job
retention window.

## Failure lessons that become launch requirements

The historical failures are useful because they define concrete preflight
checks rather than vague caution:

| Failure class | Observed example | Required prevention |
|---|---|---|
| Wrong package or import surface | Miles V4 lacked `megatron.post_training`. | Import every production module in the exact image before requesting a GPU. |
| Runtime user cannot read staged input | SkyRL V3 used root-owned `0700` inputs while the trainer ran as UID 1000. | Run every file-read/write preflight as the real runtime UID and create a file in every output parent. |
| Host RAM underestimated | Miles V5 was killed by host-memory OOM, not GPU VRAM. | Measure model conversion/loading RSS and request the proven high-water mark plus headroom. |
| Device placement defect | Dense teacher V4 failed before training because model residency was wrong. | Construct, shard, and run a synthetic forward in the exact distributed topology before the scientific canary. |
| Long sequence activation OOM | Full-weight 262K attempts OOMed even when ordinary startup fit. | Qualify several real token lengths and bind the longest proved length. Never infer 262K from model loading. |
| Tool/template mismatch | Dataminer V3 showed the `qwen3` TITO template produced the wrong tool syntax; `qwen35` matched Qwen3.8. | Golden-test the exact chat template, tool serialization, `tool_call_id`, parser, and EOS behavior against stored non-secret fixtures. |
| Context end mistaken for model failure | Older RL episodes ended at `context_full` without a reward. | Use step-wise records plus explicit compaction; classify infrastructure truncation separately from genuine task failure. |
| Uniform reward groups | Roughly half of one GRPO campaign's groups had no reward variance. | Log usable/mixed groups and reject a scale-up whose effective gradient-bearing fraction is too low. |
| Checkpoint/export coupled to training | The 318-step run trained successfully but failed during final export synchronization. | Save resumable adapters during training; merge/export/reload in a separate zero-step job. |
| Evaluation controller or judge defect | Historical WebExploitBench runs lost controllers, used a missing judge, lacked images, or OOMed inside a smaller nested container. | Use the exact collection/scoring checklist in `docs/WEBEXPLOITBENCH_TENSORLAKE_FAILURE_CENSUS.md`. |

Operationally, a run is not healthy merely because a Pod is Ready. A training
canary must produce a finite loss, finite non-zero gradient, parameter change,
adapter checkpoint, zero-step reload, merge/export, and served parity. An RL
canary additionally needs real verifier IDs, at least two reward values in one
update group, a finite optimizer update, successful adapter synchronization to
the sampler, and a post-sync behavior check.

## Harness, tools, context, and compaction controls

### Harness

OpenCode is the fixed external-evaluation harness. Training does not become
"OpenCode" merely because its model later runs behind OpenCode. Fleet tasks and
WebExploitBench expose different environments and tools, so perfect tool parity
is impossible. The controlled variables are:

- the same exact Qwen chat template and thinking settings in training and
  serving;
- one versioned mapping from each environment's tools to a common semantic
  action record;
- assistant-only training targets that include decisions and tool calls, not
  observations or wrapper text;
- no silent tool-schema or parser change between experiment arms; and
- matched OpenCode base-versus-checkpoint evaluation under one sealed protocol.

Internal Fleet reward curves answer whether a policy learned on Fleet. Only the
matched OpenCode evaluation answers whether that learning transfers to
WebExploitBench.

### Context and compaction

Qwen3.8 natively supports 262,144 tokens. The project should not pretend that
16K training teaches long-horizon behavior, but it also must not spend four
nodes per arm before proving the algorithm. The ladder is:

1. exact-model short one-step correctness gate;
2. measured 32K and 96K adapter-update gates;
3. a compacted multi-turn gate that rewrites history between two turns and
   proves exact per-turn tokens/log-probabilities are trained;
4. 160K and 262K memory probes in the intended topology; and
5. only then, long-context production arms.

Compaction is not just truncation. It must preserve a structured summary of the
task, discoveries, credentials/tokens only when safe for the environment,
attempted hypotheses, evidence, and next actions. The raw pre-compaction turn
records remain in the training evidence, but each later action is conditioned
on exactly the compacted prompt the policy actually saw.

## External evidence and how it changes the plan

- The official Qwen3.8 model card describes 27B parameters, 64 layers, a hybrid
  Gated-DeltaNet/attention layout, 262,144 native context, thinking enabled by
  default, and optional preservation of earlier thinking blocks. Therefore the
  reasoning and `preserve_thinking` settings are experimental controls, not
  harmless serving switches.
- The Unsloth Qwen3.8 guide confirms text-only LoRA/QLoRA, recommends freezing
  vision layers, targeting both attention and MLP modules, dropout 0, and using
  the same chat template and EOS at deployment. Its `2e-4`, rank-16, 2K example
  is a small tutorial, not a cyber recipe. It is an outer learning-rate prior,
  not a production default.
- Current SkyRL guidance recommends all-linear coverage, rank 32-64 for RL, and
  a LoRA learning rate roughly 10× the comparable full-weight rate.
- CTF-Dojo is the closest public cyber-SFT reference found: verified
  trajectories, Qwen-family models up to 32B, global batch 16, learning rate
  `5e-6`, up to two epochs, and at most 32K sequences. It supports beginning
  conservatively and measuring execution lift rather than copying a large
  generic instruction-tuning learning rate.
- LoRA and QLoRA literature supports all-layer adapters and 4-bit base weights
  when memory is the constraint. On 8×B300, BF16 LoRA is the primary arm;
  QLoRA is a parallelism/cost control, not assumed to be better.

Primary external sources:

- <https://huggingface.co/Qwen/Qwen3.8-27B>
- <https://unsloth.ai/docs/models/qwen3.8/train>
- <https://docs.skyrl.ai/docs/getting-started/supported_models>
- <https://docs.skyrl.ai/docs/examples/lora>
- <https://huggingface.co/docs/peft/en/accelerate/fsdp>
- <https://arxiv.org/abs/2106.09685> (LoRA)
- <https://arxiv.org/abs/2305.14314> (QLoRA)
- <https://thinkingmachines.ai/blog/lora/>
- <https://arxiv.org/abs/2508.18370> (CTF-Dojo)

## Data policy

The current catalog has roughly one thousand blackbox tasks. The 75-task
conservative set is useful for a clean control, but it is not the default
training ceiling. Training pools should include every usable task whose
environment starts, tools work, and verifier returns a real outcome. Minor
task-quality uncertainty is metadata for stratification, not automatic
exclusion.

Splits must be by task lineage, never by rollout or window, so near-duplicate
versions cannot cross train and held-out. The final held-out set is frozen once.
A separate development set may be used for experiment selection. Sampling must
limit repeated successes from one task/session so a prolific lineage cannot
dominate, while preserving broad application, vulnerability, difficulty, and
outcome coverage.

For SFT, the default target is every valid assistant action in a successful
trajectory, not just the final flag submission. Keep the exact preceding state
needed to understand each action. Teacher reasoning may be retained only when
it is actually available to the student at inference under the same thinking
contract; private or unreproducible hidden reasoning is not manufactured.

## Initial experiment plan after qualification

The first sweep is deliberately small enough to diagnose causality and wide
enough to expose a useful region. Every checkpoint immediately triggers the
same Fleet-development evaluation and a matched OpenCode WebExploitBench
collection, as requested. Only the Fleet-development metric is visible to the
experiment selector. WebExploitBench scores and frozen Fleet-final scores stay
sealed until one candidate and its protocol are frozen, so they cannot silently
select a winner. Training loss never selects a winner.

### SFT LoRA

| Priority | Rank / alpha | Learning rate | Data | Epochs | Context arm | Purpose |
|---:|---|---:|---|---:|---|---|
| 1 | 64 / 32 | `3e-5` | broad successful teacher actions | 1 | measured long-window curriculum | literature/Fleet anchor |
| 2 | 32 / 32 | `3e-5` | same | 1 | same | adapter-capacity control |
| 3 | 64 / 32 | `1e-5` | same | 1 | same | lower-update control |
| 4 | 64 / 32 | `1e-4` | same | 1 | same | upper-rate exploration |
| 5 | 64 / 32 | `3e-5` | broad Qwen successes | 1 | same | self-trace comparison |
| 6 | 64 / 32 | `3e-5` | balanced teacher + self | 1 | same | data-mixture comparison |

BF16 LoRA is primary. A rank-64 NF4 QLoRA arm at `1e-4` is a resource control
only after its logits and adapter reload match the BF16 path on a tiny fixture.

### Next broad-data SFT queue

The table below is a proposal, not a set of submitted jobs. It binds the
current, independently counted teacher corpus: 14,693 windows, 2,886 successful
sessions, 57,384,881 unique supervised tokens, and 496 task keys. All rows use
the exact Qwen3.8 revision, assistant-action-only targets, one 32,768-token
window cap, the same task-lineage split, c1 priority, W&B, and the matched Fleet
development evaluation. A checkpoint is not selected from training loss.

| Priority | Method | Learning rate | Global batch | Adapter | Epochs | Updates | Decision |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | BF16 LoRA | `3e-5` | 8 | rank 64 / alpha 32, all linear layers | 1 | 1,837 | Main adapter anchor; exact config prepared after the production checkpoint merge/reload gate passed, but not submitted while the global failure budget is exhausted. |
| 2 | full-weight SFT | `3e-6` | 8 | none | 1 | 1,837 | Main full-weight anchor; run after the native resume check passes. |
| 3 | BF16 LoRA | `1e-5` | 8 | rank 64 / alpha 32, all linear layers | 1 | 1,837 | Lower learning-rate edge; tests whether the anchor changes the model too aggressively. |
| 4 | full-weight SFT | `1e-6` | 8 | none | 1 | 1,837 | Lower learning-rate edge matched to priority 2. |
| 5 | BF16 LoRA | `1e-4` | 8 | rank 64 / alpha 32, all linear layers | 1 | 1,837 | Upper learning-rate edge; stop only for a declared non-finite or divergence rule, not a noisy loss point. |
| 6 | full-weight SFT | `3e-6` | 16 | none | 1 | 919 | Batch interaction at nearly equal token exposure. |
| 7 | BF16 LoRA | `3e-5` | 8 | rank 32 / alpha 32, all linear layers | 1 | 1,837 | Capacity control, launched only after rank 64 proves the adapter path end to end. |
| 8 | best development candidate | inherited | inherited | inherited | 2 | 2× one-epoch updates | Duration test; conditional on the frozen Fleet development metric, never on WebExploitBench. |

The numerical brackets come from three different kinds of evidence. CTF-Dojo
observed `5e-6`, global batch 16, and two epochs for verified cyber trajectories;
this is the closest task evidence, but it used different Qwen models and shorter
sequences. Thinking Machines observed that all-layer LoRA can track full tuning
when it has enough rank, that LoRA often prefers about ten times the full-weight
learning rate, and that large batches can hurt LoRA more; this motivates the
separate `3e-6` versus `3e-5` anchors and the small-batch default, but its tasks
were not interactive cyber work. The official Qwen3.8 guide supplies an outer
`2e-4` tutorial point at rank 16 and 2K context; it supports exploring above
`3e-5`, but does not justify copying that tutorial setting into the production
anchor. The exact Qwen model card makes the 262,144-token inference contract and
thinking controls explicit, but does not publish an SFT recipe. These transfer
limits are why task success, not imitation loss, decides which row advances.

### RL LoRA

Start from the accepted SFT adapter selected by the predeclared Fleet-development
metric only, and compare an untrained-base adapter only after the end-to-end path
works.

| Priority | Rank / alpha | Learning rate | Rollouts per task | Unique tasks per update | Updates | Purpose |
|---:|---|---:|---:|---:|---:|---|
| 1 | 64 / 32 | `1e-5` | 8 | 4-8 | 20 | SkyRL step-wise GRPO anchor |
| 2 | 64 / 32 | `3e-6` | 8 | 4-8 | 20 | conservative-rate control |
| 3 | 32 / 32 | `1e-5` | 8 | 4-8 | 20 | capacity control |
| 4 | 64 / 32 | `3e-5` | 8 | 4-8 | 20 | upper-rate exploration |

Use one policy epoch per collected wave, token-level importance correction,
temperature 1, clip low/high `0.2/0.28`, gradient clip 1, and no update from a
group with one reward value. Log the fraction of collected episodes and tokens
that actually contribute a gradient.

## Qualification and evaluation ladder

Gates 1–7 occur before production experiment arms. Each gate is create-once and
advances only on an independently readable receipt:

1. **Static compatibility:** exact Qwen revision, tokenizer, chat template,
   SkyRL commit/image, Megatron bridge, LoRA target list, and adapter inventory.
2. **CPU/image test:** import and render the exact production command in the
   exact image as the real runtime UID; verify input/output permissions and no
   secrets in command arguments.
3. **Synthetic distributed test:** small Qwen3.5-family fixture proves all-rank
   LoRA gradients, optimizer update, checkpoint, reload, sampler synchronization,
   and merge.
4. **Exact-model one-step SFT:** one real batch, finite loss/gradient, changed
   adapter tensors, unchanged frozen base, W&B telemetry, adapter save/reload,
   merged model reload, and all resources released.
5. **Exact-model RL reward/update:** real Fleet task versions, mixed verifier
   rewards, exact per-turn token/log-prob records, finite update, synchronized
   sampler behavior, checkpoint/reload, complete environment cleanup.
6. **Compaction gate:** a forced context rewrite between turns proves the next
   step trains against exactly the rewritten prompt and the trajectory reward
   maps to all of its steps.
7. **Length/topology gates:** measured 32K, 96K, 160K, and 262K probes. The
   longest successful length, not the configured limit, is the claim.
8. **External evaluation:** before training, seal byte-identical base controls
   for the Fleet final set and the qualified TensorLake collect-once/deferred-
   score OpenCode protocol. Each checkpoint may run the same final evaluations
   immediately, but their results remain unavailable to experiment selection.
   After one candidate is frozen using Fleet development only, unseal and report
   its matched Fleet-final and WebExploitBench comparisons.

Only after gates 1–7 should the cluster be filled with experiment arms. The
project limit is eight active nodes / 64 GPUs, c1 priority, 100 concurrent
TensorLake tasks, no node idle for more than 20 minutes, and at most ten failed
submitted cluster jobs from this goal's reset point. A local static check or API
preview is not a cluster job. Any goal-owned submitted dev or production cluster
job that reaches a terminal failed state does consume the budget, including a
CPU/image qualification job.

## Retrieval map

- Historical run/checkpoint/eval index:
  `configs/discovery/qwen38-lora-evidence-index-v1.json`
- WebExploitBench/TensorLake incidents and exact recovery checklist:
  `docs/WEBEXPLOITBENCH_TENSORLAKE_FAILURE_CENSUS.md`
- Full-weight RL paths: `docs/QWEN38_FLEET_RL_PATHS_2026-09-13.md`
- Cluster launch rules: `docs/TRAINING.md`,
  `docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md`, and
  `docs/GPU_RESOURCE_LIFECYCLE.md`
- Scientific controls: `docs/SCIENTIFIC_PROTOCOL.md`
- Current task inventory:
  `configs/data/fleet-blackbox-training-coverage-20260915-v1.json`
- Fresh75 run and model promotion:
  `docs/QWEN38_FRESH75_TEACHER_SFT_MAX_V1.md`
