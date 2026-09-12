# SkyRL router process-boundary diagnostic

This is an operational compatibility fix, not RL or model qualification.

## What failed, and what is known

The previous repository wrapper rejected every multiprocessing start method
other than `fork` before entering Ray initialization. It also replaced the
native router target with a local closure, which would not be picklable under
`spawn` even if the guard alone were removed.

The sanitized pre-Ray CPU probe reported `fork` before native imports and
`spawn` afterward. It then stopped at the earlier native resource check because
a CPU-only probe cannot satisfy the requested GPU placement. Consequently that
probe establishes the context incompatibility but did not execute actor or
engine initialization. Without a diagnostic terminal receipt identifying the
exception site, attributing the earlier dev4 `ValueError` specifically to the
fork guard is a strong inference, not independently confirmed incident evidence.

The [exact upstream native router source](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/backends/skyrl_train/inference_servers/vllm_router.py)
constructs `multiprocessing.Process` using its module-global target. It therefore
uses the current process context; it does not force `fork`. The pinned-image
CPU test checks this file's SHA-256 before exercising its actual `start` and
`shutdown` methods.

## Narrow fix

- Preserve the native target as its original importable global function.
- Wrap only the router module's process constructor with a top-level picklable
  entrypoint. Preserve the native process arguments, name, daemon setting,
  lifecycle and returned process object.
- Blank credential-like environment variables during launch and restore the
  parent's exact values afterward. The child checks context and nonempty
  credential presence before invoking the native target. The shared proof
  contains an integer state only, never credential values.
- Support `fork` and `spawn`. Reject `forkserver`: an already-running server can
  retain an older environment, so its inheritance is not assumed safe.
- Preserve exact router/actor/placement-group tracking and cleanup checks.

A specifically recognized contract rejection before any Ray initialization may
return zero only after the empty owned-resource state and unchanged output
bundle are proven. Its receipt is `diagnostic_contract_rejected`, with
`engine_start_qualified=false`, all training qualification flags false and zero
tasks, rewards, optimizer updates and checkpoints. Arbitrary exceptions,
unproven cleanup and changed output still fail. No production failure is
relabelled or erased by this code change.

## Qualification scope

The tests exercise real local `fork` and `spawn` children, positive and negative
credential checks, parent restoration, target/context drift, typed pre-Ray
rejection, and failure of cleanup/output proof. The pinned-image-only test uses
the actual native router start/shutdown methods but a synthetic child target
and a substituted health check. It performs no Ray initialization, model load,
engine startup, rollout, task request, verifier call or optimization.

### Historical process-boundary qualification, 2026-09-12

The root operator's standalone CPU self-test passed in the then-current immutable
training image, with these exact historical bindings:

- Image SHA-256: `ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`.
- Tested `training/skyrl_training.py` SHA-256:
  `30ff8db9f711665ad733449b5c23ae82c64be64bc7d1d1e6ec3abb8fd4126fb3`.
- Installed native router SHA-256:
  `f39f7b00125d670773cf1a461f2bc2e13e32dc40a84936da4d96b2db8e315a34`.
- CPU Pod: `chris-q38-rl-spawn-selftest-v1`, UID
  `ab400729-828f-418c-b3c7-740d5acc1d12`.

The observed native context was `spawn`. One picklable synthetic child passed
credential isolation and exited zero; the router was released, and the original
native target and multiprocessing module reference were restored. No GPUs,
Ray initialization, model loading or HTTP requests were used. This was a
standalone exact-image check, not a claim that every pytest case ran in that
image. Router release is also not a claim that the shared CPU helper Pod was
deleted; its remaining operator work and lifecycle are separate.

Local tests alone were not sufficient evidence for that image. This result
qualified only the tested process boundary. It did not qualify the later first
relay image or the corrected relay image described below.

### Startup-relay correction and exact-image qualification, 2026-09-12

The first relay image
`sha256:9b6f43938f9b28aaff7ba91edd59be3d18b01f9a22078ba5475cdac5e6bfcca6`
passed a zero-GPU image qualification, then dev7 exposed a distinct transport
defect: Ray could serialize but could not reconstruct the two-argument
`FleetVllmStartupError`, replacing its already-sanitized cause with
`UnserializableException`. The
[dev7 terminal observation](evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev7-terminal-v1.json)
classifies that run as infrastructure-invalid, records zero task/training work,
and proves complete eight-GPU release. It does not identify the underlying vLLM
child failure. That image is historical and engine-start disqualified.

Theseus commit `34de8d5753b8dfe44460ff9656db4ddc9a85a62c` adds the
privacy-safe reconstruction contract. Its corrected immutable image is
`sha256:e48827529b1cf5fafa153b2aed1b774c2eec86905baf5ccb62b36300533e252b`.
The [exact-image qualification](evidence/qwen38-study/2026-09-12-skyrl-startup-relay-image-cpu-qualification-v1.json)
binds receipt SHA-256
`4326ec6a28f1f2deee6d6ebaf9c04f80ec8ba5ac8856ae636e91aca5e4e53837`,
helper SHA-256
`077803a51e7c41473315ecf56eb58ecd2e23a5544ae0d3267a7059d3c3b63713`,
and the reviewed unit/integration test hashes. Nine unit cases and six installed
vLLM spawn-integration cases passed, including stdlib pickle,
`ray.cloudpickle`, and the real Ray task-error envelope. The
[terminal evidence](evidence/qwen38-study/2026-09-12-skyrl-startup-relay-image-terminal-v1.json)
proves the runtime image ID matched the request, the zero-GPU helper exited
cleanly with zero restarts, and its Pod was deleted.

This corrected-image result remains an operational CPU gate. It proves neither
Qwen3.8 model loading nor CUDA, two-engine TP4x2 startup, reward, optimization,
checkpoint recovery, or production readiness. At that point, dev8 still had to
prove real engine startup and bounded cleanup before later RL gates could open.
No earlier diagnostic plan was eligible for replay.

Dev8 is now terminal and remains immutable. Its image is also privacy-disqualified
by the later worker-RPC review, so neither renaming dev8 nor repeating it can
qualify the corrected runtime. The current successor is the fresh
[dev9 diagnostic](QWEN38_SKYRL_ENGINE_DIAGNOSTIC_DEV9.md), pinned to the
worker-RPC-sanitized image and its exact zero-GPU qualification. Dev9 preserves
the same 2xTP4 zero-work contract and must independently prove engine startup,
runtime user, exact image IDs, cleanup, terminal state, and GPU release.
