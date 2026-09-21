# Operational lessons

This is the routing index for experiment failures that changed how this
repository operates. It is not a live job-status page and it does not replace
the linked receipts. Exact run identities and outcomes belong in sanitized,
immutable evidence; reusable checks belong in code and regression tests.

Last reconciled against `main` commit
`d376b4ccd36a20ec9b45197307464ba6b2e81424` on 2026-09-20.

## What “closed” means

A repeated failure class is closed only when all four durable layers exist:

1. a sanitized receipt or immutable config records what actually happened;
2. code rejects or correctly handles the failure at the narrowest deterministic
   boundary;
3. a regression test proves that boundary; and
4. this index or the focused runbook explains the reusable rule.

If the boundary is owned by an external service, local code must fail closed at
preview or preflight and the exact upstream repair must be linked. A successful
test proves only its stated boundary; it does not prove a model trained, a
checkpoint reloaded, or an evaluation produced a valid capability result.

Status terms below are deliberate:

- **enforced on main** — code and a focused regression are in the main branch;
- **repair in review** — the incident is proven and an exact repair commit is
  linked, but main must not be treated as enforcing it yet;
- **historical context** — evidence explains an incident but is not itself an
  executable guard.

## Closed lessons on main

| Failure boundary | Reusable rule | Durable anchors |
|---|---|---|
| Root failed-job alert setting | A request flag is not enough. Before create, require the rendered root `Job` or `RayJob` annotation `fleet.ai/failure-alerts: "off"`; a Pod-template annotation does not qualify. | **Enforced on main:** [`cyber_post_train/jobs.py`](../cyber_post_train/jobs.py), [`tests/test_generic_jobs.py`](../tests/test_generic_jobs.py), and the repository-wide manifest census in [`tests/test_cluster_failure_alert_opt_out.py`](../tests/test_cluster_failure_alert_opt_out.py). Procedure and limits: [`CLUSTER_ALERTS_AND_INFERENCE_SERVING.md`](CLUSTER_ALERTS_AND_INFERENCE_SERVING.md). |
| Jobs API release | A successful delete is exactly an empty HTTP 204. Do not parse JSON, and never replay an uncertain delete after a transport error; reconcile API and Kubernetes identity first. | **Enforced on main:** `Jobs.delete` in [`cyber_post_train/jobs.py`](../cyber_post_train/jobs.py) and `test_delete_accepts_exact_empty_204_and_never_parses_json` in [`tests/test_generic_jobs.py`](../tests/test_generic_jobs.py). |
| SFT per-token output retention | Scalar SFT must call native forward/backward with `return_per_token_outputs=False`. Retaining token-level outputs across steps can create monotonic GPU-memory growth and a delayed OOM even though early steps are finite. | **Enforced on main:** [`training/sft_runtime.py`](../training/sft_runtime.py) and [`tests/test_sft_runtime.py`](../tests/test_sft_runtime.py). The affected and preserved checkpoint states are recorded in [`configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json`](../configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json). |
| Qwen Megatron-LoRA checkpoint recovery | A TP8 Megatron checkpoint is not an FSDP or generic PEFT checkpoint. Seal its exact all-rank layout under the Qwen Megatron schema, then require a native all-rank reload before resume or promotion. | **Enforced on main:** [`training/checkpoints.py`](../training/checkpoints.py), [`training/recovery.py`](../training/recovery.py), [`tests/test_checkpoints.py`](../tests/test_checkpoints.py), and [`tests/test_recovery.py`](../tests/test_recovery.py). Exact sanitized qualification: [`qwen38-lora-lr1-step40-megatron-checkpoint-v1.json`](../configs/qualification/qwen38-lora-lr1-step40-megatron-checkpoint-v1.json). |
| LoRA recovery source identity | A recovery successor needs a fresh run, output, and tracking identity, but it must not be looked up as if it were the original production-qualified run. Validate the checkpoint's embedded source plan against the accepted production binding, and separately prove that the successor preserves every required scientific, data, model, and topology field. | **Enforced on main:** merge commit [`897842f7`](https://github.com/fleet-ai/cyber-post-train/commit/897842f70334cf958c2ab33bbe0c3c382b278d72), [`training/sft_runtime.py`](../training/sft_runtime.py), and `test_recovery_runtime_reopens_the_embedded_production_source` in [`tests/test_qwen38_lora_first_config_set.py`](../tests/test_qwen38_lora_first_config_set.py). The pre-repair failure is recorded in [`qwen38-lora-recovery-runtime-gate-failure-20260920.json`](evidence/qwen38-lora-recovery-runtime-gate-failure-20260920.json) (receipt SHA-256 `7e736de27805d84dd5620f953fded34a23e2e198fda02f4bee5380c6812a4e87`). This row records code enforcement only; it does not claim that a live recovery run has passed. |

## Proven lessons whose repairs are not yet main authority

These rows are safe historical guidance, but the linked repair is not a main-branch
invariant until it merges and this index is reconciled again.

| Failure boundary | Reusable rule | Exact evidence or repair |
|---|---|---|
| Pinned-image SFT dataset loader | Route the dataset adapter by the exact pinned trainer family. One Qwen full-weight SkyRL image consumes prepared `list[dict]` rows, while the Megatron-LoRA image expects `TextDataset` and `sequence_lengths`. Preflight must invoke the same loader the staged image will use; import success alone is insufficient. | **Repair in review:** exact head [`cde81ae7`](https://github.com/fleet-ai/cyber-post-train/commit/cde81ae7ad0e3d18986e300cbca365573c3cc87b), containing repair commit `47624df5`, plan-family routing in `training/sft_runtime.py`, an exact-image loader preflight in `training/sft.py`, focused `test_sft_runtime.py` / `test_sft_config.py` regressions, and sanitized evidence at `docs/evidence/qwen38-next-sft-intentional-launch-20260920.json`. Until merged, current main’s global `TextDataset` assumption is not a valid full-weight qualification. |
| WebExploitBench prompt bundle | A Python benchmark module may depend on sibling prompt files. Seal every sibling as an individually hashed source record, materialize the complete bound set remotely, and reject any launch manifest whose local view differs from that set. Do not include evaluator-only prompt files in exported rollout bundles. | **Enforced on main:** [`benchmark_snapshot.py`](../evals/webexploitbench/tensorlake/benchmark_snapshot.py), [`collection_snapshot_qualification_worker.py`](../evals/webexploitbench/tensorlake/collection_snapshot_qualification_worker.py), [`collection_launcher.py`](../evals/webexploitbench/tensorlake/collection_launcher.py), and their focused tests under [`evals/webexploitbench/tests`](../evals/webexploitbench/tests). Historical acceptance of the zero-model qualification is recorded in the [checkpoint/eval ledger](../configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json); it is an operational gate, not a benchmark result. |
| Serving parity beyond weights | Matching checkpoint weights is necessary but insufficient. A matched comparison must clone and verify the live base image, command, arguments, resources, harness/tool parser, catalog metadata, and deterministic finite probes; only model path, exact revision, owned name, lifecycle, and reviewed priority may differ. | **Repair in review:** exact commit [`5ddab4f9`](https://github.com/fleet-ai/cyber-post-train/commit/5ddab4f937f37cce2ca82c89d44ef6b61eafae42). Historical artifact readiness, which does not claim live parity, is in [`qwen38-checkpoint-serving-readiness-v1.json`](../configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json). |
| OCI index versus platform image identity | An OCI index digest and Docker’s resolved `linux/amd64` child image ID identify different layers of the same pull. Compare each value to the matching manifest layer; never declare drift by comparing the index directly with the child. | **Repair in review:** exact commit [`3f35ac4d`](https://github.com/fleet-ai/cyber-post-train/commit/3f35ac4da9c794a5d9a7b4254afb1db6db638eae) and sanitized evidence path `docs/evidence/qwen38-fresh75-fleet-dev17-failure-v1-20260920.json` in that commit. |
| Docker-in-Docker bind mounts | The evaluator and Docker daemon have separate mount namespaces. Every nested-container bind source must be mounted at the same absolute path in both containers, and a positive startup diagnostic must prove read/write access before model calls. | **Repair in review:** exact commit [`3f35ac4d`](https://github.com/fleet-ai/cyber-post-train/commit/3f35ac4da9c794a5d9a7b4254afb1db6db638eae) and sanitized evidence path `docs/evidence/qwen38-fresh75-fleet-dev17-failure-v2-20260920.json` in that commit. |
| RL turns with more than one tool call | Preserve every valid call in emitted order, execute in order, record one deterministic result per call, and encode all results as one masked observation group with exactly one next-assistant header. Stop after a successful final submission; malformed calls or results remain invalid episodes and must never receive fabricated reward. | **Repair in review:** exact code commit [`f285ed72`](https://github.com/fleet-ai/cyber-post-train/commit/f285ed72a3047292a39cb02a9da18fec81215f01) with ordered parser, observation, and execution regressions. Sanitized terminal context: [`2026-09-20-skyrl-prod4-direct-rayjob-qualification-v1.md`](https://github.com/fleet-ai/cyber-post-train/blob/2c4e9cd027c3c2d567bb3ad6a91bf2d4bbcfdfc3/docs/evidence/qwen38-study/2026-09-20-skyrl-prod4-direct-rayjob-qualification-v1.md). |

## Adding a lesson

Add only a proven failure class. Link sanitized evidence, name the deterministic
boundary, and state exactly what the test proves. Put current job IDs, timestamps,
result counts, and digests in the immutable receipt or versioned config—not here.
If a later repair replaces an earlier rule, update the row instead of stacking a
contradictory instruction below it.
