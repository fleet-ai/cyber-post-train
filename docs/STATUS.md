# Execution status — 2026-08-31

## Current Qwen 3.6 27B experiment

- Primary student: `Qwen/Qwen3.6-27B` at exact Hugging Face revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (dense 27B, Apache-2.0).
  Fifteen BF16 safetensor shards totaling 55,563,006,400 bytes were downloaded,
  hashed, and promoted on the training SFS. The canonical weight-manifest digest
  is `sha256:14ad10368de9b9e5974ff12a4b70ea7884194b58e670177bbac79daeb81f16b9`.
- The reproducible Fleet corpus remains job
  `a62dd51f-a52b-4941-8207-4679e4b25b51`: 1,265 sessions over 160 exact task
  lineages, including 587 verified successes. The deterministic task split is
  130 train / 10 dev / 20 test; no task lineage crosses a split.
- The opt-in Torch gated-delta trainer is Ready as version
  `4b4dc57c-c7dc-5562-bbdd-9e1d6764ede0`, image
  `q36-torchgdn-6db8d0c9` at digest
  `sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`.
  SFT gate `ft-run-0b877f30` is the first exact Qwen3.6/B300 run to complete a
  BF16 forward/backward (48.52 s), optimizer step (10.07 s), full sharded
  checkpoint, and post-step evaluation. Its held-out loss moved from 0.9575 to
  0.9425 and the RayJob finished `SUCCEEDED`. Full one-epoch SFT
  `ft-run-574bd7b3` is queued. RL compatibility run `ft-run-16d0522e` is live;
  matched-policy/reference successor `ft-run-3a82f8cc` is queued so both sides
  of the KL comparison use the same Torch operator path.
- The live Fleet Training API model catalog resolves the staged base as
  `qwen3.6-27b`. Typed runs now use the direct queue-aware Jobs API at
  `https://api.ft.flt.build`, which renders through the server's authoritative
  preview/submit path into Kueue `training-lq`. The selected immutable trainer
  is the minimal fixed Ready successor at commit `37e76223`, version UUID
  `b4a49402-5e7f-5f94-a5ff-980126d78390`; it includes CUDA 13 / `sm_103a`,
  exact `flash-linear-attention==0.5.2`, verifier hydration, and the corrected
  run-naming contract without importing hundreds of unrelated later commits.
- Four retained SFT compatibility attempts progressively proved corpus staging,
  permissions, VLM-safe sequence parallelism, checkpoint loading, exact 508/33
  train/dev tokenization, and five pre-train eval batches. The latest,
  `ft-run-6b01f3fd`, reached `eval_loss=0.6244` and the first BF16 backward, then
  failed because CUDA 12.8 NVCC cannot compile TileLang's B300 `sm_103a` target.
  A second audit then found that all 508 training records were right-truncated
  at 16,384 tokens: only 2,567/27,438 assistant targets were fully retained and
  zero final-success turns survived. No optimizer step or checkpoint has yet
  been claimed.
- The replacement SFT corpus selects five exact-token windows per verified
  success, always including the final assistant turn. It contains 2,540 train,
  165 dev and 230 untouched test windows, with median 11,573 and maximum 14,334
  tokens and zero oversized exclusions. The API selector
  `chris-cyber-qwen36-windowed-v2` prevents mixing historical unwindowed rows;
  the v3 SFS promotion and all file hashes are recorded by completed CPU job
  `chris-cyber-qwen36-windowed-corpus-v3`. All 587 successful submission turns
  and all 587 final text turns are selected.
- Theseus PRs #27859 and #27873 produced and registered the first short-tag
  image `chris-qwen-b300-3fb68bb8` at digest
  `sha256:231257749cd5e9f53e06dd61be7d357318a3869373faa5836ec9a958cf47e698`.
  Its build compiled CUDA runtime headers, CUB BlockReduce and device math for
  `sm_103a` through `/usr/local/cuda/bin/nvcc`; the catalog manifest also
  includes the fail-closed verifier detail hydration path. It is now also
  superseded: SkyRL's frozen `flash-linear-attention==0.5.1` failed during the
  actual Qwen gated-delta backward despite passing the generic catalog smoke.
- RL-from-base compatibility run `ft-run-ff78ae76` failed before rollout because
  the Fleet list endpoint omitted verifier source. Theseus PR #27850 adds a
  fail-closed detail-endpoint hydration path. That fix and the CUDA 13 B300
  compiler fix are both present in the registered combined trainer image and
  are exercised together by the new RL smoke.
- The full one-epoch SFT request is frozen in
  `configs/runs/qwen36-27b-sft-full.json` but will not be submitted until a
  one-step compatibility arm proves checkpoint load, tokenizer/tool formatting,
  BF16 backward, optimizer step, and checkpoint save. The older one-step arm
  `ft-run-30714d6e` was admitted through Kueue `training-cq`, loaded all 2,540
  train and 165 dev windows with zero filtering, and completed the 21-batch
  pre-train evaluation at `eval_loss=0.9582`. Its first forward/backward then
  failed before the optimizer step in FLA's `prepare_wy_repr_bwd_kernel` Triton
  autotuner with `CUDA: misaligned address`; no checkpoint was created. Its
  fixed successor `ft-run-1c54ba33` was submitted through the Jobs API with the
  exact trainer above. It reproduced the same failure after loading all corpus
  windows and completing pre-train evaluation at `eval_loss=0.9588`: the first
  backward failed in FLA's `prepare_wy_repr_bwd_kernel` with a Triton
  `CUDA: misaligned address`. It produced zero optimizer steps and zero
  checkpoints. Its config digest is
  `sha256:ff90aef21013b0ea68bbadf5be8df43dc08297d9a830ca7a9601663c36011440`.
  A one-GPU, no-secret diagnostic job, `chris-q36-fla-b300-v1`, tested six
  explicit Triton launch configurations against the exact Qwen gated-delta
  tensor shape in the same trainer image. Each candidate runs in a fresh
  process because a CUDA address fault poisons its process context. All six
  candidates passed on an NVIDIA B300 with the exact FLA 0.5.2 / Torch 2.11 /
  Triton 3.6 stack. This isolates the failure to unsafe autotuner benchmarking,
  rather than the selected kernel invocation. Conservative config `2 warps / 4
  stages` was content-hashed at
  `sha256:2166f41ace1a98ec71e623afbb45406324a36a77c9dd13896c95246c605aa143`
  and staged read-only by convention on SFS. Successor gate `ft-run-b786dd74`
  injected that exact directory with FLA's `default` cache mode and proved the
  override bypassed autotuning. It nevertheless failed on the selected kernel
  invocation at the first backward with the same misaligned-address error,
  after pre-evaluation at `eval_loss=0.9586`; it produced zero optimizer steps
  and zero checkpoints. Source inspection then established that Qwen repeats
  its 16 query/key heads threefold before calling FLA, so the earlier synthetic
  probe's 16-head layout did not match the kernel's actual 48-head layout. The
  one-GPU, no-secret corrective diagnostic `chris-q36-fla-h48-v2` exercised the
  exact 48/48 head layout through queue `training-lq`. All six explicit launch
  configurations failed with the same misaligned-address error. This falsifies
  the earlier unsafe-autotuner hypothesis: the FLA 0.5.2 backward kernel itself
  is incompatible with this exact Qwen/B300 layout. The full SFT request remains
  locked pending a kernel-level repair or a proven alternate implementation.
- The RL intent-to-treat split remains 130 train / 10 dev / 20 untouched test.
  One historical train version is archived and server-unrunnable, so an explicit
  signed as-treated request contains 129 train and 10 dev tasks. That exact
  request passed the Training API preview with HTTP 200. Its two-task one-step
  gate `ft-run-e1a5e2dc` was admitted through Kueue `training-cq` but failed
  before actor allocation because the rendered mandatory tracker list included
  MLflow while the Nebius MLflow application and Service are intentionally
  disabled. It produced no steps or checkpoints. Fixed successor
  `ft-run-c89da950` was submitted through the Jobs API with two exact training
  tasks. It completed infrastructure/model initialization but all eight rollout
  episodes failed internally because the rollout service used Fira's
  `getAccessibleAtlasResources` readiness tool against unrelated environments.
  The subsequent reference forward also hit SkyRL's VLM microbatch-padding
  assertion. It produced zero valid training sessions, zero optimizer steps,
  and zero checkpoints; its one recorded step and two metric rows are therefore
  infrastructure evidence, not a model result. The replacement configs disable
  microbatch padding explicitly, but no successor will launch until the
  environment-neutral readiness fix is proven. The full request remains
  unsubmitted until a gate proves rollout, deterministic verifier reward,
  optimizer step, and checkpoint save.
- Theseus PR #27880 makes the tracker list an explicit deployment capability:
  W&B remains mandatory on Nebius, while MLflow remains mandatory only on
  clusters that actually deploy it. The live Nebius deployment is healthy with
  all replicas on `TRAINING_API_TRACKING_BACKENDS=[wandb]`; a fresh server-side
  preview renders exactly `trainer.logger=[wandb]`. PR #27859 installs exact
  `flash-linear-attention==0.5.2` with no dependency changes, incorporating
  upstream's Blackwell gated-delta backward restriction (FLA #913 / PR #1000),
  and records that version in the trainer manifest. Build `33239675544` passed
  from exact merge commit `4fd37536c49f4e81f40c7233bab997bd23589d26`,
  producing image tag `4fd37536` at digest
  `sha256:25e56db0367fa6fc83dcdb14b44d8b7dd7fde4af053b8a87861b942ef28bf75c`;
  its build log proves FLA 0.5.2 and a clean trainer contract. Trainer-pin PR
  #27889 closed without merge. Colleague changes subsequently registered the
  fixed lineage; the experiment deliberately pins the earliest Ready successor,
  `37e76223`, rather than mutable `latest`. Both successor gates are now live.
  No full request will be submitted until its corresponding gate is green.
- Hugging Face access to WebExploitBench is granted. All 15 official Level-0
  packs are digest-verified and pass the official non-inference CAGE checks.
  The formal Qwen baseline pins the same model revision plus an exact SGLang
  serving contract. All 15 images built successfully and the frozen 15-trial
  pass@1 run `webexploit-qwen36-27b-base-6a9e13bd-l0-p1-v1` is terminal: 12
  trials completed and three ended with separately recorded model,
  target-availability, or execution-timeout outcomes. Logs and scores remain
  sealed pending the predeclared adjudication and checkpoint/protocol freeze.
  No benchmark prompt, trace, or result enters training.

The GLM-5.2 work below is retained as historical provenance; it is no longer the
selected primary experiment.

## Fleet baseline

- Source corpus: 160 registered blackbox tasks from job
  `a62dd51f-a52b-4941-8207-4679e4b25b51`.
- GLM-5.2 Agent Runtime canary:
  <https://fleetai.com/dashboard/jobs/7bf2f012-b9de-4d15-be30-f86c25147efc>.
- First six-task full-run tranche:
  <https://fleetai.com/dashboard/jobs/335e8b2b-0105-4e87-bbb9-00bc64c73014>.
- Submitted model identifier: `z-ai/glm-5.2`.
- Full pass@1 plan: 160 sessions in 27 jobs, with at most six sessions per job.
- The canary completed in 131 steps with deterministic verifier score 0. In the
  six-task tranche, two sessions completed with verifier score 1 (69 and 50
  steps); four remain in progress. These seven hosted-model sessions are
  infrastructure shakedowns, not the formal baseline, because the hosted route
  does not expose an immutable checkpoint/tokenizer/template identity.
- Hold the remaining 154 hosted sessions. The formal baseline will deploy the
  same pinned checkpoint and serving/harness stack used for every intervention.

## WebExploitBench

- Individual Hugging Face access request is pending author review.
- The official public Level-0 subset, ComfyUI target and CAGE agent are built and
  digest-pinned locally; no gated data was scraped or reconstructed.
- Direct Fleet inference currently does not route `glm-5.2-fp8`,
  `fleet-glm/glm-5.2-fp8` or `z-ai/glm-5.2`. The model remains available through
  Fleet Agent Runtime. A direct benchmark run therefore waits on gateway routing,
  not on local harness preparation.

## Public fallback benchmarks

- XBEN: official pinned source and isolated XBEN-019 end-to-end harness smoke are
  green. Treat it as engineering validation only because it is dated/saturated.
- CVE-Bench v2.1.0: pinned zero-day smoke path is ready and is the stronger
  temporary transfer benchmark. Its public containers are not yet registered as
  Fleet Agent Runtime tasks, and direct GLM-5.2 gateway routing is unavailable.

## Training

- Export, secret-safe normalization, lineage-aware splitting, SFT, preference,
  online-RL and integrity-gated reward code are implemented and tested.
- The source job exported 1,265 completed trajectories into ignored mode-`0600`
  local storage: 587 score-1 successes and 678 score-0 failures. The normalized
  manifest digest is
  `sha256:05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494`.
- Trainer inputs contain 508 SFT-train trajectories, 33 SFT-dev, 46 SFT-test,
  56 matched preference pairs and 161 distinct online-RL prompts.
- `chris-cyber-glm52-stage-data-05c79281` copied only those five trainer-input
  files plus the manifest to cluster shared storage. The job verified the sealed
  archive, every per-file digest and every row count, then atomically promoted
  them under
  `/mnt/sfs/cyber-post-train/data/05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494`.
  Raw exports and the all-trajectory file were not transferred.
- At task level, 58/160 have zero successful source trajectories, 46/160 have
  only successful trajectories, and 56/160 are mixed-outcome. The latter are
  the cleanest initial preference/learnability stratum; zero-success tasks are
  still useful for fresh online RL but not behavioral cloning.
- Every trajectory is now bound to its exact Fleet `eval_task_version_id`, task
  version, environment/data versions, prompt digest and verifier UUID. The
  historical API returns `verifier_sha: null`, so all 1,265 records explicitly
  retain that one source-checksum gap rather than silently claiming it exists.
- Nebius CLI profile `cyber-post-train` and kubeconfig access are working for
  project `fleeta-bic-prod` (`project-e04e4tz1bc00b2qxfzqsdx`). The main
  `fleetai-training` cluster exposes 192 B300 GPUs across 24 eight-GPU nodes;
  the `fleetai-training-dev` cluster has two one-GPU B300 nodes.
- All Cyber Post Train workloads use the existing `training-lq` Kueue at
  priority 0, cannot preempt peer workloads, and retain completed/failed jobs as
  evidence. No other owner's workload has been modified or cancelled.
- The exact base is `zai-org/GLM-5.2` commit `b4734de4facf877f85769a911abafc5283eab3d9`:
  282 BF16 shards (1,506,667,387,408 bytes), weight-manifest digest
  `sha256:72b917aa2a1664751f99c4f19b1f1b23acb5eff5cffef4ff5522caef767324d4`,
  tokenizer-manifest digest
  `sha256:1858391a47cfc55e452754f8a11b2343037ca0926ddf973297f0cfd877827ad4`,
  and chat-template digest
  `sha256:172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679`.
- `chris-cyber-glm52-stage-b4734de4` completed successfully. It verified the
  remote 282-shard manifest, downloaded all 295 repository files, hashed all
  1,506,667,387,408 weight bytes, wrote the immutable checkpoint lock, and
  atomically promoted revision `b4734de4facf877f85769a911abafc5283eab3d9`.
  The staging job requested no GPU.
- The official NeMo RL GLM-5.2 trainer is pinned to commit
  `63e620046c67f922c4a57dcb65d7e6fceb60f5d4`. The first image-builder manifest
  exposed a Docker-sidecar TLS/port mismatch before executing the build and
  exhausted its two attempts. The corrected priority-0 successor,
  `chris-cyber-glm52-build-nemo-63e62004-v2`, completed and published immutable
  image digest
  `sha256:aec56926a7f0db357c3ac1eaf1576f4113ec7ac1adfbacb29cbfe4d3a7955666`.
  The failed job was retained as evidence; no workload was cancelled or deleted.
- The staged compatibility probe was designed to check the promoted checkpoint
  lock, tokenizer/chat-template roundtrip, exact framework imports, eight B300
  device identities, and NCCL all-reduce/all-to-all before the expensive
  model-load and forward/backward test. The priority-0, non-preempting
  `chris-cyber-glm52-preflight-b4734de4` job was admitted on one eight-B300 node,
  but failed during OCI container creation because the pinned image did not
  contain `/bin/bash`. No probe code ran and no compatibility receipt was
  produced. The GLM experiment is superseded by the Qwen3.6-27B path, so this
  historical job is retained as negative infrastructure evidence rather than
  retried.
- The missing Fleet checkpoint routing path is implemented in an isolated Theseus
  worktree: it sends `fleet/<run>-step-<n>` only to the authenticated
  `inference.flt.build` gateway and refuses both missing credentials and public
  provider fallback. Its focused test suite has 64 passing tests and is open as
  [Theseus PR #27754](https://github.com/fleet-ai/theseus/pull/27754). The PR
  passed the merge queue and landed as commit
  `f4f2a3e6002286e994a494f2fb168802fa171384`; its post-merge Orchestrator Deploy
  workflow completed, including checkpoint API/worker staging rollout and live
  staging smoke tests. Production promotion steps were not part of that push
  workflow, so canonical production behavior still requires verification before
  formal Fleet checkpoint evaluations.
- The public implementation lives in the private Fleet repository
  [fleet-ai/cyber-post-train](https://github.com/fleet-ai/cyber-post-train).
  Git authoring uses `christopher@fleet.so` for this work.
- The upstream GRPO recipe was validated on 64 eight-H100 nodes with TP2, PP8,
  EP64 and CP8. The proposed one-node B300 topology is therefore an unproven
  compatibility hypothesis, not a scaled-down claim. The next gate must exercise
  HF-to-Megatron conversion, expert partitioning, LoRA attachment, optimizer
  construction, BF16 forward/backward, and checkpoint save/resume on the exact
  image and checkpoint before any SFT or RL launch.
- That end-to-end gate is implemented as the still-unsubmitted
  `chris-cyber-glm52-model-probe-b4734de4` job. It performs one real BF16 LoRA
  optimizer step with EP8, saves weights and optimizer state, restarts the
  official trainer, resumes the checkpoint, and executes a second step. It is
  gated on successful preliminary preflight receipts.
- Remaining gates are a green checkpoint compatibility receipt, pinned training
  and inference images, the exact evaluation harness manifest, and the Fleet
  reward-broker contract. The evaluation protocol fails closed unless all
  model, harness, benchmark and sampling bindings share one digest.
