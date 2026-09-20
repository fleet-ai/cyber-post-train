# Qwen3.8 evaluation and TensorLake capacity audit

Observed at **2026-09-20 09:32 UTC** in read-only mode. The machine-readable
record is
[`qwen38-eval-capacity-audit-20260920.json`](qwen38-eval-capacity-audit-20260920.json).
No training, serving, evaluation, or sandbox resource was launched, resumed,
cancelled, or changed.

## Bottom line

The durable inventory contains four accepted Qwen3.8 checkpoints, but it does
not contain an accepted matched WebExploitBench comparison or an accepted
held-out Fleet evaluation for any trained checkpoint. There is no scored
evaluation that can be launched immediately without first completing its
identity and serving checks.

TensorLake has **no active WebExploitBench sandbox belonging to this project**.
The account-wide inventory contained 299 running sandboxes named `fleet-tlp-*`
and 63 suspended sandboxes. The 299 running sandboxes appear to be a separate
fixed pool and are not counted as this project's evaluation usage without an
owner mapping. This means the project is using zero visible TensorLake
evaluation slots, but the account-wide list alone does not prove exact spare
capacity.

## Accepted checkpoints

| Priority | Checkpoint | Durable model files | Current evaluation readiness |
|---|---|---|---|
| 1 | Fresh75 step 230 | checkpoint and final export both present | Serving qualification passed, but the route is paused. It is the closest checkpoint to a matched evaluation. |
| 2 | Teacher dense v5 step 186 | checkpoint and final export both present | Serving has not been qualified. |
| 3 | Self-SFT step 44 | checkpoint and final export both present | Serving has not been qualified. |
| 4 | LR30 step 76 | checkpoint and final export both present | Serving has not been qualified. |

Fresh75 is registered as `chris-q38-fresh75-step230-v1`. A read-only Fleet API
check found it paused with request routing disabled. Its accepted serving check
is preserved separately and binds model UID
`d51cf7c4-3ab8-4912-8f73-dfe7153acdb4`.

## What has already been evaluated

- Fresh75 has one valid standalone OpenCode pass@1 result:
  `fresh75-step230-opencode-web-p1-gpt55-v1`. It collected and scored all 15
  attempts and found 3 successful objectives out of 110. Do not repeat that
  exact run. It cannot establish training lift because it did not run a matched
  base model at the same time with the same settings.
- Fresh75's wider `fresh75-step230-opencode-web-p8-v3` campaign is partial:
  104 of 120 attempts across 13 of 15 targets. It is not a complete comparison.
- The teacher step-186 `v28` pair ended and released its resources, but no
  accepted score record was found. Its old identity must be reconciled before a
  successor is created.
- The historical OpenCode base result is partial at 44 of 60 attempts. Other
  matched lineages reached engineering canaries but not a complete matched
  result.
- No trained checkpoint has an accepted held-out Fleet result. The earlier
  Fresh75 `dev17/final8` plan was either not launched or left no accepted
  receipt.

## Unknown TensorLake lineage that must be reconciled

Thirteen suspended sandboxes use the prefix `q38-s10-web-p4-v1`. Twelve look
like task sandboxes and one is a download helper. All stopped because their
timeouts expired on September 18. No matching local plan, accepted result, or
entry in the durable evidence index was found. The checkpoint, task coverage,
and result claims are therefore unknown.

Before dispatching a similar campaign, recover or explicitly classify this
lineage. The JSON record lists every sandbox name and ID. This prevents an
apparently new run from silently repeating work that may already have produced
partial evidence.

## Ranked next evaluations

1. **Fresh75 matched OpenCode WebExploitBench comparison.** Create or freeze a
   matched base arm, resume the Fresh75 arm, prove both routes are serving the
   exact intended model, seal a new paired plan, then run one task and one
   attempt per arm before the full 15-task pass@4 comparison.
2. **Fresh75 representative Fleet holdout.** Bind exact held-out task versions,
   freeze the OpenCode and scoring settings, and use the same accepted Fresh75
   model identity.
3. **Teacher step 186 matched WebExploitBench and Fleet holdout.** First qualify
   its export for serving and reconcile the old `v28` pair.
4. **Self-SFT step 44 matched WebExploitBench and Fleet holdout.** First qualify
   its export for serving.
5. **LR30 step 76 matched WebExploitBench and Fleet holdout.** First qualify its
   export for serving.

The small autoresearch engineering checkpoint is intentionally omitted. It is
useful for testing the machinery, not for measuring cyber capability.

## How duplicate work is prevented

The current WebExploitBench tooling already has the right safety structure:

1. Seal a plan that fixes the exact model, harness, task set, and scorer.
2. Give the campaign, sandboxes, model calls, results, and snapshots unique
   identities.
3. Record a claim before each remote create or score request.
4. If a request outcome is unclear, stop and reconcile it; never send the same
   request again just because the local process crashed.
5. Check live TensorLake inventory and accepted terminal receipts before
   dispatching.
6. For comparisons, bind symmetric base and trained-model arms to exact model
   and serving evidence.
7. Run a one-task paired canary before the full 15-task pass@4 comparison.

Historical WebExploitBench records do not all carry the newer experiment
fingerprints used by the autoresearch database. Until they are backfilled,
duplicate prevention must use the sealed plan, request journals, TensorLake
inventory, result roots, and the durable evidence index together.

## Sources checked

- Repository commit `607b7b0bb30d798a3af1ba5f17e1ebc9f2366a77`.
- `configs/discovery/qwen38-lora-evidence-index-v1.json`.
- `evals/webexploitbench/tensorlake/` plan, collection, supervision, scoring,
  and acceptance code.
- `evals/fleet/` evaluation and rollout-ledger code.
- Read-only Fleet inference metadata for the registered Fresh75 model.
- Read-only TensorLake `GET /sandboxes` inventory.
- Sanitized local acceptance records for Fresh75 serving and the v14
  score-free TensorLake runtime snapshot qualification. The v14 qualification
  proves the runtime snapshot works; it is not a scored model evaluation.

No credentials, prompts, traces, flags, answers, model outputs, or private
trainer logs were read into this report.
