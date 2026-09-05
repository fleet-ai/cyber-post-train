# Operational lessons and prevention map

This is the living index of cross-run failures that changed how cyber
experiments are operated. It maps each reusable lesson to the mechanism that
now prevents or detects it. Exact Job UIDs, timestamps, endpoint state, and
one-run outcomes belong in sanitized receipts; this file is not launch
authority and does not replace those receipts.

Read this file before changing a Fleet evaluation controller, cluster-serving
package, duplicate preflight, or acceptance path.

## Enforced lessons

| Failure class | Durable decision | Enforcement and regression |
|---|---|---|
| The two Fleet Jobs APIs were treated as one service | Determine the live service by its deployed contract: managed agent evaluations use `/v1/jobs`; general immutable image-plus-command cluster workloads use `/v1/runs`. Never infer the cluster API's capability from a convenience client's catalog. | [`cyber-cluster-jobs-operator`](../skills/cyber-cluster-jobs-operator/SKILL.md), [`check_contract.py`](../skills/cyber-cluster-jobs-operator/scripts/check_contract.py), [`test_cluster_jobs_contract.py`](../tests/test_cluster_jobs_contract.py) |
| Source YAML looked valid but the rendered workload combined an incompatible priority and preemption policy | Validate the rendered Pod fields together and reject an installed `PriorityClass` whose behavior contradicts the requested non-preemption contract. | [`priority_preemption_guard.py`](../evals/fleet/priority_preemption_guard.py), [`test_priority_preemption_guard.py`](../evals/fleet/tests/test_priority_preemption_guard.py) |
| A model route advertised the expected model but did not obey both required tools | Model identity and an HTTP 200 are insufficient. Before a scored cell, make content-free forced-tool probes for every required structured tool and fail closed on malformed or missing tool calls. | [`hosted_behavioral_preflight.py`](../evals/fleet/hosted_behavioral_preflight.py), [`test_hosted_behavioral_preflight.py`](../tests/test_hosted_behavioral_preflight.py), [`hosted-behavioral-preflight.md`](hosted-behavioral-preflight.md) |
| Serving location was silently treated as irrelevant to scientific identity | Hosted and dedicated servers are separate treatment blocks until exact model, route, image, precision, context, harness, tools, task versions, and retry policy pass the parity gate. Partition at complete task boundaries. | [`treatment_parity.py`](../evals/fleet/treatment_parity.py), [`test_fleet_treatment_parity.py`](../tests/test_fleet_treatment_parity.py), [`CURRENT_FLEET_PASS4_EVAL.md`](CURRENT_FLEET_PASS4_EVAL.md) |
| Session metadata omitted an optional projection field even though the sealed execution chain was exact | Accept only an omitted/null optional projection when the sealed config-to-acceptance-to-session/verifier chain proves identity; continue to reject any contradictory non-null value. Never repeat the cell merely to work around a list-projection omission. | [`exact_pass4_bulk_runtime_v3.py`](../evals/fleet/exact_pass4_bulk_runtime_v3.py), [`qwen38_dedicated_scored_canary_v1.py`](../evals/fleet/qwen38_dedicated_scored_canary_v1.py), [`test_exact_pass4_bulk_v3.py`](../tests/test_exact_pass4_bulk_v3.py), [`test_qwen38_dedicated_scored_canary_v1.py`](../evals/fleet/tests/test_qwen38_dedicated_scored_canary_v1.py) |
| Duplicate reconciliation was too broad, slow, or unbounded | Query only exact task keys, bound concurrency and request time, retry only safe transient reads, enforce a real cancellation deadline, and preserve failed preflights as non-consuming tombstones. Never auto-retry a mutating request. | [`qwen_bulk_generation16_preflight.py`](../evals/fleet/qwen_bulk_generation16_preflight.py), [`test_qwen_bulk_generation16.py`](../tests/test_qwen_bulk_generation16.py) |
| A server-side dry run admitted Jobs whose Pods referenced a missing Secret | A rendered manifest is not runnable until every referenced Secret and external ConfigMap is proven present in the target namespace immediately before create. Enumerate references from `env`, `envFrom`, volumes, projected volumes, and image-pull secrets; do not rely on API dry-run to resolve them. | [`qwen_bulk_generation16_renderer.py`](../evals/fleet/qwen_bulk_generation16_renderer.py), [`test_qwen_bulk_generation16.py`](../tests/test_qwen_bulk_generation16.py) |
| A create-once diagnostic reused one filename for the BEFORE and AFTER sides of a stage | Evidence filenames must encode both stage and phase, and the complete receipt namespace must be enumerated and proven collision-free before rendering or submission. | [`qwen_bootstrap_diagnostic.py`](../evals/fleet/qwen_bootstrap_diagnostic.py), [`test_qwen_bootstrap_diagnostic.py`](../tests/test_qwen_bootstrap_diagnostic.py) |
| A Kubernetes-projected runtime plan was passed directly to a loader that correctly rejects symlinks | Keep symlink rejection in the authority loader. Validate the projected plan schema and self-digest, copy it create-once into the private workspace as a regular file, and pass only that regular file to runtime. | [`projected_runtime_plan.py`](../evals/fleet/projected_runtime_plan.py), [`run_qwen_bulk_generation16.sh`](../evals/fleet/scripts/run_qwen_bulk_generation16.sh), [`test_projected_runtime_plan.py`](../tests/test_projected_runtime_plan.py) |
| Controller directories, sessions, claims, and acceptance receipts could disagree | Use one score-blind 800-cell ledger, classify every cell as accepted, active, retryable, blocked, or unstarted, and reject contradictory evidence. A directory count or completed process is not a valid outcome. | [`exact_pass4_ledger.py`](../evals/fleet/exact_pass4_ledger.py), [`test_exact_pass4_ledger.py`](../tests/test_exact_pass4_ledger.py) |
| A read-only preflight spelled a global claim path differently from the runtime writer | One canonical helper maps `sha256:<hex>` only to `<hex>.json`; runtime writes, preflight collision checks, and the global ledger all call that helper and reject invalid execution ids. | [`exact_pass4_bulk_runtime_v3.py`](../evals/fleet/exact_pass4_bulk_runtime_v3.py), [`qwen_bulk_generation16_preflight.py`](../evals/fleet/qwen_bulk_generation16_preflight.py), [`exact_pass4_ledger.py`](../evals/fleet/exact_pass4_ledger.py) |
| A failed process could still have created a fully ingested, cleaned, non-repeatable session | Preserve model-started or verifier-backed cells as non-repeatable even when they do not qualify for accepted credit. A reviewed acceptance rule may later reclassify them; a retry must never erase or duplicate them. | [`exact_pass4_ledger.py`](../evals/fleet/exact_pass4_ledger.py), [`test_exact_pass4_ledger.py`](../tests/test_exact_pass4_ledger.py) |
| A healthy dedicated server could reserve expensive GPUs while the controller was broken or absent | Loading is productive startup. Once Ready, require fresh real traffic or controller progress; after the user-approved idle window, preserve state and release the server before debugging or resubmitting. Never fabricate a heartbeat. | [`cyber-cluster-jobs-operator`](../skills/cyber-cluster-jobs-operator/SKILL.md), [`CURRENT_FLEET_PASS4_EVAL.md`](CURRENT_FLEET_PASS4_EVAL.md) |
| Repository-wide evidence searches surfaced protected historical material | Treat search output as disclosure. Search only named safe source, config, test, or sanitized-receipt paths; never recursively search raw outputs, mutations, transcripts, benchmark material, or a repository parent. | [`AGENTS.md`](../AGENTS.md), [`cyber-run-evidence`](../skills/cyber-run-evidence/SKILL.md) |
| A successful successor was mistaken for incident closure | Close an incident only after sanitized evidence, a named prevention boundary, enforcing code/test where possible, updated living guidance, focused validation, and an explicit residual-risk statement. | [`AGENTS.md`](../AGENTS.md), [`cyber-experiment-maintainer`](../skills/cyber-experiment-maintainer/SKILL.md) |
| Moving an unchanged harness to a laptop was assumed to repair hosted model tool behavior | Execution location is not a parity substitute. Require fresh exact endpoint identity plus content-free behavioral checks from the intended execution location; if a required tool contract fails, preserve a sanitized terminal receipt and keep all scored creates closed. | [`laptop_opencode_lane_v1.py`](../evals/fleet/laptop_opencode_lane_v1.py), [`laptop_secret_launcher_v1.py`](../evals/fleet/laptop_secret_launcher_v1.py), [`test_laptop_opencode_lane_v1.py`](../tests/test_laptop_opencode_lane_v1.py), [`2026-09-05-qwen38-laptop-rank3-tool-parity-terminal-v1.json`](evidence/qwen38-study/2026-09-05-qwen38-laptop-rank3-tool-parity-terminal-v1.json) |

## Adding a lesson

Add a row only when the lesson changes future decisions across runs. The
closeout must answer all of the following:

1. What failed, and did it create any paid, scored, model-started, or otherwise
   non-repeatable side effect?
2. What is the narrowest deterministic boundary that should reject or detect
   the failure earlier?
3. Which code path enforces that boundary, and which regression exercises the
   real rendered or API-facing behavior?
4. Which operator decision cannot be fully enforced in code and therefore
   belongs in a focused skill or living runbook?
5. What remains unenforced, and which scale gate stays closed because of it?

Do not add credentials, prompts, traces, flags, scores, mutable endpoint state,
or current workload identifiers. Link sanitized immutable evidence from the
incident-specific receipt when that evidence is publishable.
