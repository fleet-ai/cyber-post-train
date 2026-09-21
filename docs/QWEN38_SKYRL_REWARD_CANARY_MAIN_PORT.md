# Qwen3.8 SkyRL reward-canary main port

Status: **prepared, not submitted; external preview and submission are blocked.**

The independent offline launch-readiness audit is
[`2026-09-20-skyrl-launch-readiness-audit-v1.json`](evidence/qwen38-study/2026-09-20-skyrl-launch-readiness-audit-v1.json).
It recompiles the exact V17 and prod4 plan/request identities without reading
private payloads or contacting external services, and checks the five queued
production configurations. It is a fail-closed audit, not launch authority.

This is the current-main port of the scientifically valid one-node reward
canary from runtime commit `8b5e8a00521b4fe412e3e7d0bce9b8ea6855b005`,
with preflight provenance from
`6c0bc541e136fc114a3f7a17ce6ba07e9178bb78`. The prior preflight is
historical evidence only. It certifies neither these rebased runtime bytes nor
the new create-once identities.

The next immutable inputs are:

- topology config:
  [`qwen38-skyrl-topology-probe-dev-v2.json`](../configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json);
- data config:
  [`qwen38-rl-reward-canary-data-prod-v4.json`](../configs/qualification/qwen38-rl-reward-canary-data-prod-v4.json);
- run config:
  [`qwen38-rl-reward-canary-prod-v4.json`](../configs/qualification/qwen38-rl-reward-canary-prod-v4.json);
- source and submission closure:
  [`qwen38-rl-reward-canary-port-v4.json`](../configs/qualification/qwen38-rl-reward-canary-port-v4.json).

The port preserves the exact train and dev task versions, ordered
`bash`/`submit_report` surface, fractional partial reward contract,
98,304/81,920-token context and response limits, 600-turn ceiling,
2,400-second episode limit, 330-second tool limit, one node, eight GPUs, two
TP4 engines, eight samples, one optimizer step, learning rate `1e-6`,
pre/post evaluation, step-1 checkpoint, and seed 42. The only runtime repair is
canonical ordering of the already digest-verified live tool schema before Qwen
prompt rendering.

The v4 configuration selects immutable image
`sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`
and sets `VLLM_USE_FLASHINFER_SAMPLER=0`. Its retained engine receipt proves
two-by-four engine startup and release only; it is not one-node training
qualification. The `v17` topology successor must independently produce and
validate `TOPOLOGY_PROBE.json` after its fixed 30-second terminal-receipt grace.

The scientific canary is production-routed because the reviewed experiment has
a 2,400-second episode limit. The development cluster's 30-minute workload
limit cannot contain that episode plus model startup and cleanup. The
development cluster remains the bounded place for the zero-update topology
check.

The exact prepared topology packet is bound to:

- plan SHA-256 `b398e73ba3124c4f22e53aceb9ac7924916ba2a3027a3f780261d184505ae678`;
- request SHA-256 `def61728802c80a304a5f2ffdc0c1c22dae785737f2f23c03e9455b1712bda83`;
- FleetJob manifest SHA-256 `0f8c7f3f5a6772fd4aaef706d8d2a638e6e7b24cb87edc6c3b0937d95bd6f389`;
- CPU-preflight manifest SHA-256 `64f06f9a9ed4db4de4cd2ce3afa2b84b25b8f7d4be471c32dd96e3842004ef14`;
- receipt-verifier manifest SHA-256 `1409b89a9c113ff6ca2b7cb1afcfdc8428c0bd475b123d0820f185c684e8edfe`.

The locally compiled scientific packet is bound to plan SHA-256
`25d0abf30da462a6ba67c6ac8a3fc95f3f89a9a08e3ea295b21e8a22481630b3`
and request SHA-256
`98a83dbbab61f360a12ec6389adb55de637ded64dcceaa1debf0d791e754cef4`.
Those digests do not authorize submission.

## Why submission is blocked

The command surface refuses external preview and submission before creating a
Jobs client. There is no longer a failure-count gate. Four unresolved gates
remain:

1. accepted `v17` topology receipt plus proof that its eight GPUs were released;
2. create-once staging and digest verification of the exact one-train/one-dev
   data manifest under the `prod4` data path;
3. an exact-image CPU preflight and Jobs API preview for `prod4`; and
4. execution of the encoded read-only Jobs/Kubernetes/SFS/W&B absence guard
   immediately before the one allowed create.

The outer FleetJob is not a Kubernetes Job or RayJob and therefore carries no
alert annotation. The mandatory root `fleet.ai/failure-alerts: "off"`
annotation remains on its embedded RayJob and on both zero-GPU Jobs. Exact
server-preview evidence is recorded in
[`2026-09-20-skyrl-alert-optout-launch-repair-v1.md`](evidence/qwen38-study/2026-09-20-skyrl-alert-optout-launch-repair-v1.md).

The offline defects for the five full arms are repaired. Each arm now has an
exact sanitized staged-manifest candidate, a runnable UID/mode/file-set/digest
validator, a full immutable plan, an exact request digest, a local no-submit
preview, and a two-phase UID-bound release-observer contract. These artifacts
contain counts and digests but no task prompts. They do not claim that the
private bytes have been copied to SFS or that a server preview has run.

The production-only runtime leaves the 2,400-second scientific episode limit
unchanged. Its plan-bound hard watchdog is 43,200 seconds for the four ten-step
arms and 172,800 seconds for the fifty-step arm. Those bounds exceed the legal
38,400/163,200-second episode schedules plus startup and drain allowance; the
shared one-step-canary runtime and its frozen request are unchanged.

The production submit path no longer relies on the generic duplicate check.
After the server preview and immediately before its create-once POST, it
requires a freshly armed release observer and rechecks the complete Jobs
history, the exact production Kubernetes namespace, the SFS staged tree and
absent output root, and the exact W&B run ID. The checks must finish within 120
seconds and are recorded in the durable no-retry POST intent. Historical clean
observations remain non-reusable.

External gates remain: prod4 acceptance, failure-budget reset, create-once SFS
staging, exact-image CPU preflights, authenticated server previews, and live
observer arming/release receipts. Nothing in the offline packet authorizes a
launch.

Review the deterministic packet without external access:

```sh
uv run --locked python scripts/prepare_qwen38_skyrl_production_queue.py --check
uv run --locked python scripts/audit_qwen38_skyrl_launch_readiness.py --check
```

After separately authorized SFS staging, preparing an arm through
`cyber-post-train rl` reopens that exact manifest and writes
`RELEASE_OBSERVER_CONTRACT.json` plus `OFFLINE_PREVIEW.json` beside the plan and
request. Running `cyber-post-train preflight` in the pinned CPU image invokes
the staged-data validator. `preview` and `submit` remain blocked by the sealed
qualification until the external gates above are replaced by accepted evidence.

After those gates pass, the scientific run may be created once. Acceptance
requires eight genuine rollouts, non-constant verifier rewards, one finite
optimizer update, a reloadable step-1 checkpoint, and complete resource release.
All-zero valid rewards are a truthful experiment outcome but do not establish a
learning update. Missing or truncated reward evidence is an infrastructure
failure, not a zero-reward model result.

## A successor data package must change its embedded run identity

The prod7 V2 data-stage Job completed correctly, but its exact-image CPU
preflight rejected the package before any GPU allocation. The public manifest
named prod7 while the private train and development rows still named prod6
inside `cyber_config_json.run_id`. Changing only the manifest does not create a
valid successor: the embedded episode identity is what the environment,
recorder, and checkpoint lineage use during training.

The V2 destination is immutable and retired. A replacement uses a new V3 data
destination and `rebind_private_source_run_id`. That helper permits exactly two
changes per row: the run ID and the row's checksum after that run-ID change. It
canonicalizes the row again, recomputes the train/development payload hashes,
and recomputes the manifest checksum. Any prompt, task, environment, reward,
model, split, tool, or horizon change fails closed. The exact-image CPU
preflight must then reopen the staged package and pass before a GPU RayJob may
be authorized.

Sanitized rejection and release evidence is
[`2026-09-21-skyrl-prod7-stale-episode-run-id-preflight-rejection-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod7-stale-episode-run-id-preflight-rejection-v1.json).
It records zero collected episodes, zero rewards, zero optimizer updates, zero
checkpoints, and complete release. It is not a capability result.

## Cleanup observers must prove a cleanup reason

Prod6 established a separate operational failure class. The RL process was
still generating its pre-update development episode, the Pod was Ready with
zero restarts, and GPUs were actively computing. The local cleanup observer
then suffered five consecutive observation errors and its old `finally` block
deleted the exact RayJob. SFS contained only the run and batch `STARTED`
markers: no collection, reward, update, checkpoint, runtime failure, rejection,
or completion marker existed. This was an observer-induced infrastructure
failure, not a model result.

An observer exists to release a workload after there is evidence that cleanup
is required. Failure of the observer's own Kubernetes read is not that
evidence. The observer may now delete only when at least one positive condition
is true:

1. the exact bound workload reports a terminal state;
2. the exact bound workload violates its reviewed GPU-resource contract; or
3. the plan-bound maximum runtime is reached.

Transient or repeated read failures are recorded using a short sanitized code
and the observer keeps watching. They never authorize deletion of a
nonterminal run. The result receipt records the total and maximum consecutive
observation failures, the last sanitized failure code, and the positive reason
that eventually authorized cleanup. The regression deliberately exhausts the
old five-cycle threshold while an eight-GPU run remains active, proves no
delete occurs, then proves normal exact-UID cleanup after a genuine terminal
status.

Sanitized incident and release evidence is recorded in
[`2026-09-21-skyrl-prod6-observer-induced-release-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod6-observer-induced-release-v1.json).
It records zero collected training batches, zero optimizer updates, zero
checkpoints, and complete release of the eight GPUs. It makes no capability
claim.
