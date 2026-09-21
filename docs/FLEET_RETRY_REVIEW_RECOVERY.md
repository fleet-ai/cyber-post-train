# Recovering incomplete Fleet evaluation cells

This procedure is for a cell that the evaluator placed in `retry_review`. It
does not grant permission to launch a job. It applies only after the exact
campaign owner has reviewed the private evidence and approved the specific
create-once action.

The central rule is simple: **never generate a second model trajectory when a
stored trajectory already exists.** Repeating generation after seeing which
cells had infrastructure trouble can bias a comparison, even if nobody reads a
score.

## Find the rule by symptom

| Sanitized symptom | Safe path |
| --- | --- |
| `output_limit` with a stored session | Reconcile its existing score or run one scoring-only recovery; never regenerate. |
| `process_error` with a stored session | Reconcile its existing score or run one scoring-only recovery; never regenerate. |
| `output_limit` or `process_error` without a completed stored session | Manual terminal review; never regenerate. |
| `post_claim.connecterror` with no generation artifact anywhere | One rollout retry after exact absence proof. |
| `post_claim.operationalerror` with no generation artifact anywhere | One rollout retry after exact absence proof. |
| Any evidence conflict, unknown failure, or exhausted limit | Manual terminal review; do not create work. |

## Two separate recovery paths

1. **A stored session exists.** Verify that it belongs to the exact task
   version, model route, harness, seed, and evaluator plan. If it already has a
   complete authoritative score, reconcile that existing result without
   reading or recording the score value. If scoring is incomplete, run at most
   one scoring-only recovery against the immutable stored session. This path
   never calls the model.
2. **No session or generation artifact exists.** A rollout retry is possible
   only for the allowlisted post-claim transport failures and only after proving
   absence from the local result store, authoritative session store, trace
   manifest, and scoring-intent record. Run at most one retry with the exact
   original task, model, harness, seed, budgets, and route.

An output limit or agent-process error never authorizes another rollout. Use the
stored session path if its immutable session exists; otherwise leave the cell
in manual terminal review.

## Review order

1. Wait for all four primary seed-43 arms to become terminal. Do not change an
   active frozen run.
2. Read only the allowlisted lifecycle metadata. Do not inspect score values,
   prompts, traces, flags, responses, or rewards.
3. Produce a private, self-digesting cell manifest. It binds every selected
   cell to its original frozen plan and records exactly one action:
   `accept_existing_scored_session`, `scoring_only_recovery`, `rollout_retry`,
   or `manual_terminal_review`.
4. Classify every arm with the same rules before applying any action. Accepted
   cells are final and never replayed. Do not choose an action from model
   success, failure, or score.
5. Run scoring-only actions before considering any rollout retry. Reobserve
   each selected cell immediately before a create.

## Create-once gate

For each approved scoring or rollout recovery, prove all of the following
immediately before submission:

- the reserved Job, ConfigMap, Pod, Workload, output directory, and database
  identity are absent;
- the rendered root Job has `fleet.ai/failure-alerts: "off"`, priority `c1`,
  zero GPUs, and the reviewed CPU and memory request;
- a server dry-run of the exact object succeeds;
- the original model route is Ready with the frozen revision and parity
  receipt;
- the exact frozen task versions, OpenCode harness, sampling seed, budgets,
  and verifier binding are unchanged; and
- the cluster has capacity for immediate useful work.

Create each object once. If the create response is uncertain, reobserve the
exact identities; do not submit again. A failure or missing receipt remains
truthful evidence and must not be hidden by moving the cell back to pending.

The comparison is usable only when each complete arm independently reaches
17 accepted cells. Never splice cells from an older campaign or a different
arm. The final eight tasks remain sealed.

The machine-readable review-only contract is
`configs/evaluation/qwen38-fleet-dev17-seed43-reviewed-recovery-v1.json`.
Its classifier is `evals/fleet/retry_review_policy.py`. The classifier rejects
unknown fields, including a score value, and fails closed on contradictory
evidence.
