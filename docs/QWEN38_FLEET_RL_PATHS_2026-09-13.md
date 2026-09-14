# Qwen3.8 Fleet RL paths — 2026-09-13

This note records the three independently reconstructed routes to a real Fleet
cyber RL update. Infrastructure startup alone is not success. A route qualifies
only after exact training-task versions produce real grades with verifier
execution IDs, at least one finite non-zero gradient update, and a durable
checkpoint.

## 1. Maintained FTI/Miles Fleet V1 path (primary)

This is the current company-supported integration in Theseus. The pinned image
contains Qwen3.8-27B at its native 256K context, Fleet V1 task loading, real
instance tools, stored-verifier grading, GRPO, optimizer state, checkpointing,
and W&B tracking. The same model recipe has trained successfully on another
task set. What remained unproved was the complete loop on our exact Fleet cyber
task versions.

The runtime is pinned to FTI 0.9.2, Theseus commit
`0b1af5684310ee244bf7bdb0028e5ef78c08098b`, and image digest
`sha256:0ca02d9bc920e104d67c249172eb0e8b0cc34fc409e738f76f2efee44610dc59`.
An earlier candidate digest was rejected by a zero-GPU preflight because it was
FTI 0.8.8 and did not support Fleet V1. It was never used for a paid run.

The exact 0.9.2-image preflight passed with Pod UID
`595e3364-6223-444d-95c6-8d796092cf4b`. It checked the synchronous grading
contract, preservation of the verifier execution UUID into sample metadata,
grade-before-metadata ordering, exact task-version lookup, replacement of the
placeholder message by the task's real prompt, and parsing of the complete
Miles, custom-generation and W&B argument set. It used no GPU.

The retired A1 and A2 plans are retained as immutable submission evidence at
`configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v1.json` and
`configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v2.json`. The active A3 plan is
`configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v3.json`. All select the same
one exact version from the locked 59-task training split; neither dev nor
final-test tasks can pass the local validator. The task was chosen from
app-balanced candidates because earlier completed Qwen sessions had produced
more than one valid verifier outcome. No earlier score is copied into training.

The canary uses four whole B300 nodes because that is the smallest supported
shape for the native 256K Qwen recipe. It requests c1 priority, runs one GRPO
step from eight samples, checkpoints at step one, and tracks in W&B. A small
adapter retains the verifier execution UUID that the upstream integration
currently discards from sample metadata; generation, tool use, grading and the
optimizer remain upstream FTI/Miles behavior.

The exact 0.9.2 upstream payload does not request privileged containers. The
production Jobs API preview warned when an early local request added that
unnecessary access, so the POST was withheld and the request was corrected to
match upstream before submission.

The first corrected request passed the production preview with four nodes, 32
GPUs, c1/q1 priority, preferred topology and no warnings. A read-only,
zero-GPU SFS gate proved its create-once output path absent before submission.
Exactly one POST created API/RayJob `chris-q38-v1rl-c1-a1-65dfa2c9`
(immutable run ID `65dfa2c9-bbcb-474c-bf2b-2fdb3a3e20d6`, RayJob UID
`c42bef48-badf-4738-9931-e0ad229c4f2c`). Its Workload UID was
`b0a91f67-58ed-4880-b46e-d684a015548e`.

While that run was still suspended and had never created a Pod or output, a
source audit found that upstream Fleet V1 kept its verifier execution UUID only
in memory. That was insufficient for a terminal scientific audit after Ray
cleanup. A1 was therefore deleted once through the Jobs API before admission.
The subsequent API GET returned 404; its exact RayJob and Workload were absent,
and it had used zero GPUs.

The A2 adapter adds one narrow durability layer around the otherwise unchanged
upstream generator. For each successfully graded episode it writes one
create-once receipt containing only the exact task and verifier UUIDs, a finite
reward class, a keyed reward fingerprint, cleanup confirmation, and plan and
receipt digests. It never writes the prompt, interaction, answer, raw reward,
instance UUID, credential, or model score. Multi-segment episodes collapse to
one receipt; malformed or unwritable evidence aborts that sample before Miles
can train on it.

The new exact-image, zero-GPU preflight passed in Pod UID
`47ce9b54-32df-4c0a-a33f-822cb90af636`, including receipt creation,
idempotence, redaction, the earlier task/session contract, and native command
parsing. A clean production preview and a separate zero-GPU output-absence gate
then passed. Exactly one POST created API/RayJob
`chris-q38-v1rl-c1-a2-dbe132a2`, immutable run ID
`dbe132a2-4e38-4ce7-a50d-d796468ea9e7`, RayJob UID
`e0eaafec-17c5-42db-8ff3-669c58a8a79d`, and Workload UID
`1737584f-3dff-406a-8afd-32ad58752bea`. At first readback it was suspended and
unadmitted with no RayCluster, Pod, GPU, output, or W&B run yet. Queueing without
allocation is not training evidence.

Before A2 was admitted, an exact source audit caught a separate historical
Miles defect: its default W&B helper copied the literal environment credential
into the inner Ray command line. A2 was therefore deleted once through the Jobs
API while still suspended, with subsequent API 404 and exact Kubernetes
absence; it also used zero GPUs and created no output or W&B run.

A3 adds a fail-closed startup hook that enables W&B while leaving authentication
only in the environment. Exact-image preflight Pod UID
`747355d1-497d-4f10-a4e4-38fff0586f49` used non-secret marker values and proved
that neither the W&B nor Fleet value entered the Miles training arguments or
any rendered shell command. It also repeated every A2 evidence and runtime
check. A clean production preview and zero-GPU output-absence gate passed, then
one POST created API/RayJob `chris-q38-v1rl-c1-a3-ea459e04`, immutable run ID
`ea459e04-5e4f-432c-9898-1f3cba2e8ed8`, RayJob UID
`bbe48aba-ccba-4e59-b487-2f573b09a546`, and Workload UID
`577b519f-44d3-4ebd-956e-9f49b65cb17e`. At first readback A3 was safely queued
at priority 10000 with no allocation. Its plan, request, and preview-manifest
SHA-256 values are respectively
`9c0765a5bd7893b07fff468add9fe021b27ee33797453fa673bc38f0c6eb124e`,
`1d73c62d0bcdbaa2d23aad806bc4cc72d157d051c6e74fb51a1e78af6c807263`, and
`8112241b111a64e5a145eb75b4288db10471c503792005756c34940aab5f66b8`.

The bounded successor is frozen, but not submitted, at
`configs/runs/qwen38-27b-fti-v1-rl-production-a1.json`. Its digest-bound selector
expands to all 59 training task versions and rejects any change to that exact
list; the 20 development and 10 final-test task versions stay excluded. It uses
eight optimizer steps, eight prompt groups per step, eight samples per prompt,
learning rate `1e-6`, 32 concurrent Fleet environments, a checkpoint every
step, c1/q1, and W&B run ID `chris-q38-v1rl-prod-a1`. The expanded plan SHA-256
is `c235e109e57e757b8a2a90dc090b2187d2d590d1c6972ecbd8c62c294e28d683`;
its evidence-spool and environment-only W&B request SHA-256 is
`8b75f3ed15d78fad62427e97cd1e0d9f2e216fe5ce2ac03057922fbe0b914fe1` and
its clean production-preview manifest SHA-256 is
`17eaa63b877e535f3e7c479a5ea3f9cbbf1abd1551be24c6f866af8d5e85ece3`.
It remains gated on the canary's real reward, verifier-ID, optimizer-gradient,
checkpoint and instance-release evidence.

## 2. Repository Miles adapter (backup)

The repository's earlier custom Miles path is implemented in
`training/miles_training.py`, `training/miles_rollout.py`, and
`training/rl_episode.py`. Its latest real Qwen canary completed a long model
interaction but exhausted the 96K generation envelope before grading. It
therefore produced no verifier reward, optimizer update, or checkpoint and is
not a valid RL result. The code now classifies bounded context/turn exhaustion
truthfully, but its older image and shorter context make it a slower route than
the maintained 256K integration.

## 3. Repository SkyRL adapter (independent backup)

The SkyRL path has local/native adapter tests and a full-weight Qwen layout, but
has not yet shown a real Fleet cyber reward, optimizer update, and checkpoint in
one run. Its main historical operational issue was ownership and permissions on
privately staged input. It remains useful as an independent trainer comparison,
but should not be scaled until a train-only reward-acquisition canary passes the
same evidence gate.

## Scaling rule

No path advances to a multi-step experiment because its pod started or because
the trainer exited successfully. Scale only the path whose canary proves all of
the following: exact task/version bindings, finite mixed rewards, non-empty
verifier execution IDs, one finite non-zero optimizer update, a readable
checkpoint, and release of every created Fleet instance and allocated GPU.
