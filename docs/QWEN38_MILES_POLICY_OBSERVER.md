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

The runtime opens no task or verifier and creates no rollout engine. It performs
three sequential, isolated all-rank restores: exact base once, then the exact
CPU-sealed trained DCP twice. The base leg sentinelizes and reloads the policy
while proving its freshly initialized optimizer, scheduler, and RNG did not
change under the intentional no-load flags. Each trained leg installs a
different model/optimizer/scheduler/RNG sentinel immediately before Miles'
actual checkpoint loader and proves that every component was overwritten.
The two trained restore commitments must match exactly. Policy, optimizer, and
scheduler must differ from the exact base state. RNG must be restored exactly
and independently in both trained legs, but this no-dropout recipe does not
claim that one update necessarily advances RNG.

Every leg probes model, optimizer, scheduler, and RNG state twice with no update
between probes and fails if the state changes. The second trained restore also
runs exactly one fixed, non-task, no-grad prediction forward over token IDs
1 through 32. It exposes only the input digest and the ordered top-16 prediction
digest, and requires a 0.01 logit margin at the top-16 boundary. This lets the
later HF reload prove semantic equivalence without exposing logits or text.
The runtime requires the restored next-rollout index, zero rollout/verifier/
backward/update/save/W&B work, exactly one forward, and byte-identical source
checkpoints after all three loads.

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
   Compile `compile_observer` and render `job_request` into immutable `plan.json`
   and `request.json` files.
3. Use the dedicated create-once operator rail below. It completes all local
   source/request/output checks, commits `CAPTURE_INTENT.json`, opens
   resource-version Kubernetes watches for all four lifecycle kinds, and only
   then calls the Jobs API's one-POST `submit_once` rail. It derives API identity
   and submission time exclusively from the exact two-row journal, writes the
   bound `STARTED.json`, and persists only sanitized UID-bound events. It drops
   an early Pod event until the main container has a projectable status.

   ```sh
   cyber-miles-policy-observer submit \
     --plan /absolute/prepared/plan.json \
     --request /absolute/prepared/request.json \
     --journal /absolute/evidence/SUBMISSION.jsonl \
     --output /absolute/evidence/SUBMITTED_EXECUTION.json \
     --watch-directory /absolute/evidence/watch \
     --source-commit <exact-40-character-commit>
   ```

   The request is dev `c1`, one worker, eight GPUs, no secret, W&B disabled,
   and no requeue. The watch may remain open for up to 12 hours while queued;
   queued without allocation is not idle. An ambiguous POST is reconciled,
   never repeated.
4. After process success, derive terminal state and external release through
   read-only API/Kubernetes queries:

   ```sh
   cyber-miles-policy-observer controller \
     --plan /absolute/prepared/plan.json \
     --submission /absolute/evidence/SUBMITTED_EXECUTION.json \
     --watch-directory /absolute/evidence/watch \
     --output /absolute/evidence/CONTROLLER.json

   cyber-miles-policy-observer release \
     --plan /absolute/prepared/plan.json \
     --submission /absolute/evidence/SUBMITTED_EXECUTION.json \
     --controller /absolute/evidence/CONTROLLER.json \
     --query-output /absolute/evidence/RELEASE_QUERY.json \
     --output /absolute/evidence/RELEASE.json
   ```

   Release acceptance requires the Jobs API to report exact success and live
   queries to prove the exact RayJob, Workload, RayCluster, and Pod UIDs absent;
   caller-supplied absence booleans are not accepted. Then create
   `POLICY_DELTA.json` with `accept_policy_delta`.
5. Create the training run's `MILES_TERMINAL_ACCEPTED.json` with
   `training.miles_acceptance.accept_terminal`. This binds the policy proof to
   genuine task interaction, authoritative verifier identities, private reward
   variance, exact W&B update-zero telemetry, exactly one optimizer update,
   checkpoint seal, controller success, and training-resource release.
6. Finally call `accept_observer_reload` with that terminal receipt. It writes
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
`additional_reload_gpu_jobs: 0`, and carries the fixed prediction-probe contract
that the independently reloaded HF export must match.

## Promotion boundary

Production remains closed unless both the training terminal and consolidated
reload receipts revalidate from their referenced files. Promotion must then
bind the exact dev3 identity, the frozen production data manifest, and the same
qualified native topology. Export, serving parity, held-out Fleet evaluation,
and paired Tensorlake WebExploitBench evaluation remain later gates. Only the
weight payload may differ between paired base and post-training evaluation.
