# Qwen3.8 Miles policy observer and reload gate

The Miles dev canary needs an independent, value-sensitive proof that its
sealed checkpoint contains a real policy update. The same proof must also show
that Miles can restore the full native state without performing more work.
`training.miles_policy_observer` provides one create-once `1x8` dev job for
both purposes. A second eight-GPU reload job is neither required nor allowed by
this path.

The observer is post-hoc. The active dev3 training runtime did not need to emit
new evidence fields. Its exact historical authority remains:

- source commit `d69fd01e4b435adc5492c8aedba9cf7f3af5e40e`
- plan SHA-256 `245b404f9507ac2603c53969dd4506aee811e1c02ebf468543994cff76e3e95e`
- request SHA-256 `d2f2514a33bdf05610c5765c11efaecfcd6eea39e4297115cc39ce3c86d3f927`
- runtime-bundle SHA-256 `0e3ff1646344abb1b4be13ea064143014e94c13bc22c6c7e8c1a552f8bbe32a6`

Current source is never used to reconstruct those submitted bytes.

## What the one GPU run proves

The runtime opens no task or verifier and creates no rollout engine. It first
loads the exact base policy on all eight ranks, releases that actor group, then
loads the exact CPU-sealed trained DCP on all eight ranks with optimizer,
scheduler, and RNG restore enabled. It records only high-entropy state
commitments and structural counts. It does not record weights, prompts,
trajectories, rewards, scores, flags, answers, or metric values.

The trained leg probes model, optimizer, scheduler, and RNG state twice with no
operation between probes. It fails if state changes. The runtime also requires
the restored next-rollout index, zero rollout/verifier/forward/backward/update/
save/W&B work, and byte-identical source checkpoints after the load.

A successful process receipt is deliberately not scientific acceptance. A
separate offline join reopens the exact runtime source and request, the
pre-admission UID event journal, terminal controller state, zero-restart Pod,
immutable image, and complete post-terminal absence of the RayJob, Workload,
RayCluster, GPU Pod, quota reservation, and GPUs. A no-change policy produces
an honest observer result but cannot produce `POLICY_DELTA.json`.

## Create-once order

1. Let dev3 finish and prove its training controller and eight GPUs are
   released. Seal its checkpoint on CPU. Preserve the original plan, request,
   source commit, submission response, and two-row submission journal in a
   create-once evidence directory. Compile
   `cyber_miles_submitted_execution_binding_v1`; do not regenerate the request
   from current code.
2. Copy
   `configs/qualification/qwen38-miles-policy-observer-dev3-v1.template.json`
   to a private create-once config. Its three source paths must point to the
   exact staged plan, submission binding, and CPU-sealed checkpoint manifest.
   Compile `compile_observer`, render `job_request`, preview against the dev
   API, and bind the exact request bytes with `compile_submission_binding`.
3. Before admission, call `start_capture` and continuously pass the full
   sanitized RayJob, Workload, RayCluster, and Pod objects to
   `training.miles_event_evidence.record_event`. The watcher must retain
   owner UIDs, RayJob `priorityClassName`, shutdown/TTL fields, immutable Pod
   image ID, restart count, timestamps, and Pod GPU limits. Never read logs or
   environment values. Start capture before the Workload admission timestamp;
   TTL-zero objects may disappear immediately after completion.
4. Submit exactly once only after absence/duplicate checks. The request is dev
   `c1`, one worker, eight GPUs, no secret, W&B disabled, and no requeue. An
   ambiguous POST is reconciled, never repeated.
5. After process success, compile `CONTROLLER.json` from the retained event
   journal, then compile `RELEASE.json` only after every exact UID and GPU
   allocation is absent. Create `POLICY_DELTA.json` with
   `accept_policy_delta`.
6. Create the training run's `MILES_TERMINAL_ACCEPTED.json` with
   `training.miles_acceptance.accept_terminal`. This binds the policy proof to
   genuine task interaction, authoritative verifier identities, private reward
   variance, exact W&B update-zero telemetry, exactly one optimizer update,
   checkpoint seal, controller success, and training-resource release.
7. Finally call `accept_observer_reload` with that terminal receipt. It writes
   `/mnt/sfs/jobs/chris-q38-miles-reload-dev3/RELOAD_ACCEPTED.json`. This is a
   CPU-only evidence join: it performs no API call and allocates no GPU. The
   generic production gate reopens it with
   `training.miles_reload_acceptance.validate_accepted(..., check_files=True)`.

The dependency is acyclic:

```
observer result + observer release -> policy delta -> training terminal
training terminal + same observer evidence -> native reload acceptance
```

The final reload receipt exposes the generic
`source_manifest_sha256` and `source_terminal_acceptance_sha256` bindings used
by promotion and export. It records `observer_gpu_jobs: 1` and
`additional_reload_gpu_jobs: 0`.

## Promotion boundary

Production remains closed unless both the training terminal and consolidated
reload receipts revalidate from their referenced files. Promotion must then
bind the exact dev3 identity, the frozen production data manifest, and the same
qualified native topology. Export, serving parity, held-out Fleet evaluation,
and paired Tensorlake WebExploitBench evaluation remain later gates. Only the
weight payload may differ between paired base and post-training evaluation.
