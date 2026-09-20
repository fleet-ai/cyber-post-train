# Qwen3.8 LoRA zero-step export development qualification — 2026-09-20

## Scope and immutable inputs

- Development Kubernetes context only:
  `nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb`.
- Priority/topology: `c1`, one pod, one eight-GPU B300 node, TP8, PP1,
  CP1, DP1, zero optimizer updates, no external evaluation.
- Exact image:
  `ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317`.
- Accepted input receipt only:
  `/mnt/sfs/jobs/chris-q38-lora-sft-c1-v10/QWEN38_LORA_CHECKPOINT.json`.
- Input receipt file SHA-256:
  `2f914db630534a4cc2f036b646e40244565c68f67f810af5061dff1912e3f062`.
- Input receipt self digest:
  `7a5a8c0ed29c35ee4ddf5430fbcf235885c6a1405a800d0041d1931ba3a6abd6`.
- Every GPU pod had `activeDeadlineSeconds: 1800`, `restartPolicy: Never`,
  no service-account token, no secrets, offline Hugging Face/Transformers,
  and disabled W&B.

No production object or Jobs API request was submitted. No optimizer step,
benchmark/evaluation, raw weight inspection/printing, private runtime log read,
prompt, trace, flag, or credential access occurred.

## Create-once preflight and first two fail-closed runs

The exact-image CPU preflight accepted v1 with output-root absence and exact
source/image identity:

- preflight pod UID `67e6c49e-0fb2-4503-a29c-0a2e1a59e94f`;
- plan SHA-256
  `7b6f861ba502d29d47df796306edec2d9edf5a2e7537c694b1f74cf0d5c51fba`;
- request SHA-256
  `3b8dcd49988d9abfa89e7fbfc57f49d640454a302fd5630dc16fe9267a3fc61c`;
- fresh run root `/mnt/sfs/jobs/chris-q38-lora-export-c1-v1`.

GPU pod UID `5eb02b01-a3b7-4c16-99b2-a7ee781d1817` failed after
1,325.880 seconds with a sanitized `ValueError` receipt and
`optimizer_steps_executed: 0`. It had zero restarts. The exact UID was deleted
immediately and node GPU requests returned to zero. The first failure contract
did not identify a safe stage, so no private logs were read. Commit `56490a57`
added a sanitized failure-stage code before a fresh attempt.

The exact-image CPU preflight accepted v2:

- preflight pod UID `400ae1ee-5f8d-4aae-92a5-b51b6753bae7`;
- plan SHA-256
  `cdb974f3a7b45ebc7e929f3b43b8864e02259122e57bac034af74fac47f6be60`;
- request SHA-256
  `be96df8d73a29118b62da05ac6ab65e7f229fee12b6904f4df389d7f559a0a81`;
- fresh run root `/mnt/sfs/jobs/chris-q38-lora-export-c1-v2`.

GPU pod UID `805afea3-599c-4060-9b56-35d027271df8` failed after
1,316.606 seconds with zero restarts. The strict sanitized failure receipt was:

```json
{"elapsed_seconds":1316.606,"error_class":"ValueError","failure_stage":"base_layout_compare","optimizer_steps_executed":0,"plan_sha256":"cdb974f3a7b45ebc7e929f3b43b8864e02259122e57bac034af74fac47f6be60","status":"failed"}
```

The exact UID was deleted immediately and node GPU requests returned to zero.
Neither run emitted a terminal export receipt and neither failed run was reused.

## Exact native contract and repair

Read-only exact-image source inspection established this pinned call chain:

`WorkerDispatch.save_hf_model` → `MegatronWorker.save_hf_model` →
`MegatronStrategy.save_hf_model` → `bridge.save_hf_weights(strict=False)`.

The source comment makes `strict=False` load-bearing for `language_model_only`
exports. A metadata-only comparison (no tensor values read) of the accepted base
and v2 native output proved:

- exact base: 1,199 indexed tensors;
- native language-only export: 851 indexed tensors;
- export-only tensors: zero;
- common-key dtype/shape mismatches: zero;
- base-only tensors: 348, all under `model.visual.*` or `mtp.*`.

Metadata probe pod UID `a0a06d42-195b-46f5-aeb0-9cee38dd4c8a` succeeded
with zero restarts and was deleted. This explains the strict layout failure:
the native call correctly exports the trained language model but cannot alone
emit the complete multimodal/MTP base layout.

Commit `621d96c9` implements the smallest fail-closed completion path. For each
of the two independent native exports it:

1. requires the native key set to be a subset of the exact accepted base;
2. rejects any omitted tensor outside `model.visual.*` and `mtp.*`;
3. requires every common tensor to retain the base dtype and shape;
4. writes the exact base shard/index layout using native merged values for the
   851 language keys and accepted base values only for the 348 allowlisted
   frozen keys;
5. copies only exact accepted base sidecars;
6. preserves the existing double-export equality, every-tensor BF16 reopen,
   base-versus-merged value comparison, source/checkpoint immutability, full
   model/tokenizer reload, and finite-logit gates.

Forty-six focused local tests pass. Exact-image CPU qualification pod UID
`8ed026d7-1550-41c9-8b89-0f6276122de3` accepted the synthetic strict
layout/value-preservation contract with producer SHA-256
`792b588c8b20e437af4226fbb63020f6d5075e410fb83a54bdb638f649dff89a`;
it was then deleted.

## Fresh repaired successor

The exact-image v3 CPU preflight accepted:

- preflight pod UID `e76b4540-dcec-4c57-b363-b752b21abc00`;
- plan SHA-256
  `cb6afe09ca5f21b1bd45d2b315b57f35179b6e0c40feaf59399ac091dbd73b55`;
- request SHA-256
  `9077bee67a8a56aea3a9615e559c4f8493b5ac91e9f91327ba529ebe41a1f7f0`;
- fresh run root `/mnt/sfs/jobs/chris-q38-lora-export-c1-v3`.

The bounded successor is pod `chris-q38-lora-export-c1-v3`, UID
`010dd15b-58a5-494a-825e-d233b9576aac`. At this evidence checkpoint it is
Pending without a node or GPU allocation behind the project-priority RL
FleetJob. It must not be promoted unless it emits and independently passes the
strict `cyber_qwen38_megatron_lora_merged_hf_export_v1` receipt.

### v3 terminal and I/O-bounded v4 successor

v3 later ran from `12:12:07Z` until the server-enforced 1,800-second deadline.
Both native TP8 exports and both complete base-layout writes finished. The main
process remained active in strict every-tensor reopen/equality validation when
Kubernetes terminated the pod with `reason: DeadlineExceeded`; it had zero
restarts and emitted no accepted export receipt. Exact UID
`010dd15b-58a5-494a-825e-d233b9576aac` was deleted immediately. Its GPU
allocation was absent before any successor was created.

Commit `d0842449` removes only redundant full-layout I/O. It still reopens and
compares both independent native merged exports to prove deterministic merged
language values, but writes the deterministic frozen base completion once. It
then independently reopens every published tensor once, compares the complete
layout and values with the exact base, reloads the complete model/tokenizer,
checks finite logits, and rehashes all immutable inputs. Forty-six focused tests
pass. Exact-image CPU qualification pod UID
`b9b34aa1-7bcd-4fc9-a633-527598470e32` accepted producer SHA-256
`d65b34816ab9db2d57f50c454376dde28e19ba3e792fc98e65690dca322ee49c`
with zero optimizer updates, then was deleted.

The fresh v4 preflight accepted:

- preflight pod UID `c5cd045b-929e-4428-b67f-c51e54a271f7` (deleted);
- plan SHA-256
  `52cb4b09f213b99c6422261ae145d6581f2e24610b654dea4beea43e5ea41f50`;
- request SHA-256
  `3c5c69b350d50b74cb011332c0f9d8eca2be378e98f77c9423286049a6b43625`;
- fresh run root `/mnt/sfs/jobs/chris-q38-lora-export-c1-v4`.

Pod `chris-q38-lora-export-c1-v4`, UID
`2547746a-c422-46a9-97a7-d2285b27afb3`, was created c1/TP8 and was initially
Pending without allocation. It remains fail-closed until an independently
validated terminal export receipt exists.

### v4 terminal and scheduler-independent reload repair

v4 ran from `12:45:05Z` until its fixed `13:15:05Z` development deadline.
It had zero restarts and zero optimizer updates. It completed the native merge
and published the complete candidate tree, then reached the sanitized
`full_model_tokenizer_reload` stage. Kubernetes recorded `DeadlineExceeded`
and stopped the container exactly at 1,800 seconds. The terminal failure
receipt has file SHA-256
`025e555106720cec6c5b7cd6ce949dbc0b6fa38963ac49682403c1bc65cd2e11` and
records `SystemExit`, elapsed time `1799.164`, plan SHA-256
`52cb4b09f213b99c6422261ae145d6581f2e24610b654dea4beea43e5ea41f50`,
and zero optimizer steps. No accepted export receipt exists. The exact Pod was
deleted after its UID was rechecked and is absent.

The failure was a resource-lifecycle defect, not a merge or model outcome. At
the final stage, all eight Megatron actor processes were still alive and held
GPU memory. The independent reload was expressed as a new Ray task requesting
one GPU, so it waited behind the TP8 actors' eight existing Ray GPU leases even
though the dedicated B300 had ample physical memory. Extending the development
deadline would hide that defect and is not permitted.

The next successor instead starts a fresh Python process with only GPU 0
visible immediately after the complete export is published. This preserves an
independent process, CUDA context, complete model/tokenizer reopen and finite
logit check, but it does not ask Ray for a ninth logical GPU lease. The reload
runs concurrently with the existing read-only full-layout verification. Its
stdout and stderr remain in the private run log; only the validated four-field
reload result enters the public receipt. Every existing deterministic merge,
BF16 tensor, base-layout, source-immutability and zero-optimizer gate remains
unchanged. Forty-four focused tests pass, three tensor tests skip locally only
because this worktree's test environment does not include Torch, and Ruff
passes.

Commit `bb19de89` contains that repair and was pushed before execution. The
development ConfigMap was independently byte-compared with the committed
sources and made immutable. Exact-image CPU qualification pod
`chris-q38-lora-export-cpu-qual-v5`, UID
`b5b96b69-209f-4545-9f10-b5a71b0eed1e`, accepted producer SHA-256
`7d6e5926eb528e7f7dc35c038689e2abcebdc15e85b2209bec9cf5711baeb81d`.
Its sanitized receipt SHA-256 is
`5ca089bac3e0523240789fe7e0f975b20301859da76d08ead4c6b9e66c7c928e`.
The pod had zero restarts and was deleted after the exact UID was rechecked.

The fresh v5 preflight then accepted:

- preflight pod UID `4ef148fb-ae0b-4477-9ecf-c9a04420c81b` (deleted);
- preflight receipt SHA-256
  `4b1f0cc2ee8b1da485e773f8b4d427e33088b14d89a124e2bce2feb3dc1ad11c`;
- plan SHA-256
  `16a435a85aa87a2d0b5c99e9ae08c17cd8f06a76400c28d176128f763cbc6240`;
- request SHA-256
  `345cbe67748f5da4f0fb8cc1f82fc8a14c96fd9178be60a0f0f100e4e38bd3b5`;
- fresh run root `/mnt/sfs/jobs/chris-q38-lora-export-c1-v5`.

The bounded v5 GPU pod is `chris-q38-lora-export-c1-v5`, UID
`7e7f7d36-fa85-4026-875f-29e880c86a02`. It started on the development cluster
at `2026-09-20T13:25:29Z`, requests one eight-GPU B300 node at c1, has a fixed
1,800-second deadline, and had zero restarts at admission. An exact-UID cleanup
observer is armed. It remains fail-closed until its terminal export receipt is
independently validated.

### v5 accepted terminal

v5 completed before the fixed development deadline with exit code zero and
zero restarts. The exact-UID observer deleted the terminal Pod and verified its
absence, so the development GPU allocation was released. Unlike v4, the fresh
subprocess completed the full repaired-model and tokenizer reload while the
driver retained the TP8 actors; the result was consumed before terminal
publication, proving the ninth-Ray-lease deadlock was removed rather than
hidden by a longer deadline.

Independent exact-image CPU verifier pod
`chris-q38-lora-export-c1-v5-verify`, UID
`67b47961-64ef-4cae-b49d-8e83bad5233b`, then succeeded with zero restarts and
was deleted. Its sanitized validation receipt SHA-256 is
`e60eb3ecb06b0488b98f558f4a1e873a18a95cba4aa8f5ed8bfcbc3c734c67b2`.
The accepted export identity is:

- logical export receipt SHA-256
  `1561a76907551bf829221af328921a626ddde8c5791cf29fa46ad0c99357c110`;
- export receipt file SHA-256
  `3da93b3304e284205ba8a8939d1dcb8cc4fe749b0f255bbbe35376af5e7c5ff2`;
- source checkpoint receipt SHA-256
  `7a5a8c0ed29c35ee4ddf5430fbcf235885c6a1405a800d0041d1931ba3a6abd6`.

The verifier accepted all terminal gates: zero optimizer steps, deterministic
merge, every output tensor reopened equal, finite logits, complete GPU model
and tokenizer reload, unchanged source checkpoint, and unchanged base model.
The merged artifact is therefore accepted for downstream immutable staging and
matched evaluation; no capability claim follows from export acceptance alone.
