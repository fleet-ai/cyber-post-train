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
- At the access check, active workloads requested 184/192 main-cluster GPUs
  (166 training, 18 inference), leaving only one eight-GPU node free. Do not
  bypass the existing Kueue policy or alter another owner's workload.
- The exact base is `zai-org/GLM-5.2` commit `b4734de4facf877f85769a911abafc5283eab3d9`:
  282 BF16 shards (1,506,667,387,408 bytes), weight-manifest digest
  `sha256:72b917aa2a1664751f99c4f19b1f1b23acb5eff5cffef4ff5522caef767324d4`,
  tokenizer-manifest digest
  `sha256:1858391a47cfc55e452754f8a11b2343037ca0926ddf973297f0cfd877827ad4`,
  and chat-template digest
  `sha256:172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679`.
- `chris-cyber-glm52-stage-b4734de4` is running at Kueue priority 0. It first
  verified the remote 282-shard manifest, then began resumable download into a
  partial shared-filesystem path. It will hash every local shard and atomically
  promote only after all bytes match. No GPU is requested.
- The official NeMo RL GLM-5.2 trainer is pinned to commit
  `63e620046c67f922c4a57dcb65d7e6fceb60f5d4`. The first image-builder manifest
  exposed a Docker-sidecar TLS/port mismatch before executing the build and
  exhausted its two attempts. The corrected priority-0 successor,
  `chris-cyber-glm52-build-nemo-63e62004-v2`, is building from that exact source
  and a digest-pinned NVIDIA base image. The failed job was retained as evidence;
  no workload was cancelled or deleted.
- A staged compatibility probe now checks the promoted checkpoint lock,
  tokenizer/chat-template roundtrip, exact framework imports, eight B300 device
  identities, and NCCL all-reduce/all-to-all before the expensive model-load and
  forward/backward test. None of these preliminary checks can mark the final
  compatibility receipt green by itself.
- The missing Fleet checkpoint routing path is implemented in an isolated Theseus
  worktree: it sends `fleet/<run>-step-<n>` only to the authenticated
  `inference.flt.build` gateway and refuses both missing credentials and public
  provider fallback. Its focused test suite has 64 passing tests. Review and
  deployment are still required before formal Fleet checkpoint evaluations.
- Remaining gates are a green checkpoint compatibility receipt, pinned training
  and inference images, the exact evaluation harness manifest, and the Fleet
  reward-broker contract. The evaluation protocol fails closed unless all
  model, harness, benchmark and sampling bindings share one digest.
