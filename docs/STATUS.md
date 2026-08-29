# Execution status — 2026-08-28

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
- A staged compatibility probe now checks the promoted checkpoint lock,
  tokenizer/chat-template roundtrip, exact framework imports, eight B300 device
  identities, and NCCL all-reduce/all-to-all before the expensive model-load and
  forward/backward test. `chris-cyber-glm52-preflight-b4734de4` is now waiting
  in `training-lq` at priority 0 with the required single-node topology request.
  Kueue reports that all 24 B300 nodes are currently excluded by existing CPU,
  memory or GPU reservations. It will admit the job automatically when one full
  node becomes available and cannot preempt peer workloads. None of these
  preliminary checks can mark the final compatibility receipt green by itself.
- The missing Fleet checkpoint routing path is implemented in an isolated Theseus
  worktree: it sends `fleet/<run>-step-<n>` only to the authenticated
  `inference.flt.build` gateway and refuses both missing credentials and public
  provider fallback. Its focused test suite has 64 passing tests and is open as
  [Theseus PR #27754](https://github.com/fleet-ai/theseus/pull/27754). The PR
  passed the merge queue and landed as commit
  `f4f2a3e6002286e994a494f2fb168802fa171384`; its post-merge Orchestrator Deploy
  workflow is queued behind existing deployment concurrency. Deployed behavior
  still requires verification before formal Fleet checkpoint evaluations.
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
