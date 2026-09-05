# Generation-8 optimized canary fallback (held)

Generation 8 is a held-only fallback for the two Generation-7 canaries. It
exists because Generation 7 repeatedly reconstructs and semantically validates
the same immutable JSON graph before creating a claim. On a one-CPU evaluator,
that redundant work can dominate startup.

No Generation-8 release or Kubernetes object is included. If either exact
Generation-7 global execution claim appears, this fallback is obsolete and must
not be released.

## Scientific identity

The Qwen rank-4 attempt-1 and GLM rank-13 attempt-1 statistical cells are
unchanged. Model revision, task/environment/verifier bindings, OpenCode 1.18.27,
`bash` plus `submit_report`, 262,144-token context, 20,000-token compaction
headroom, timeout, sampling, scoring, and hosted routes are copied byte for byte
from their Generation-7 plans. Only the create-once execution generation,
execution ID, run ID, network, Job, ConfigMap, and SFS root advance.

## Stop tombstone and release sequence

Before rendering either append-only release, one sanitized tombstone must prove
all of the following for both exact Generation-7 UIDs:

1. The Job is no longer active and is exclusively terminal Failed; its exact
   Pod is terminal.
2. Every global execution-claim path for generations 1 through 8 is absent.
3. Every known output root for the cell's generations 1 through 8 is absent,
   except the two preserved Generation-1 `canary1-v2` roots. If either preserved
   root exists, it must contain exactly the canonical Generation-1 plan and
   scoring release, one exact task claim, one exact attempt claim, the empty
   claim-gate lock, and empty attempt/result/quarantine directories. Its claim
   digests must match the reviewed Generation-1 pre-model tombstone, which
   proves zero model calls, attempts, sessions, verifier executions, terminals,
   and accepted outcomes. The originally owning Job and Pod must also remain
   Failed under their exact reviewed UIDs. Any extra, missing, changed, or
   symlinked entry or live-identity drift fails closed; the preserved root is
   never deleted or reused.
4. Fleet task-session metadata contains none of the cell, execution, or run
   identities. The observer never requests transcripts, scores, prompts, flags,
   or logs.

The release renderer validates that same two-model tombstone before producing
either model release. The runtime repeats the live Kubernetes, SFS, claim, and
Fleet-session observation immediately before taking the global O_EXCL claim.
It creates no output before the claim. A present claim, output, session, active
Generation-7 Job, changed UID, or ambiguous terminal state fails closed.

### Reviewed controlled-stop option

The natural path preserves both exact G7 Jobs and Pods and requires them to be
terminal Failed. A separate explicit controlled-stop path is available if the
operator decides to stop the still-active, pre-claim G7 work:

1. `observe-prestop` freezes both exact active Job/Pod UIDs and resource
   versions and proves every G1-G8 execution claim, non-historical output root,
   and authoritative Fleet session identity absent, while revalidating either
   preserved Generation-1 root under the exact contract above.
2. `render-delete-authorization` emits a HELD receipt containing exactly two
   Kubernetes DELETE requests. Each request names one owned G7 Job, carries its
   exact UID as a server-enforced precondition, and uses foreground propagation.
   This command only renders evidence; it performs no deletion.
3. Only the explicit `execute-controlled-stop` command may mutate the cluster.
   It requires the pre-snapshot and authorization to be no more than 15 minutes
   old. Under the shared global execution-claim lock it repeats the full
   pre-snapshot, requires byte-identical Job/Pod identity and resource-version
   fields, writes that fresh snapshot, sends the two UID-preconditioned deletes,
   waits for both Jobs and Pods to disappear, repeats the exhaustive
   claim/output/session scan, and freezes separate post-stop and deletion
   receipts. The lock stays held across the fresh check, deletion, and post-check.
4. `observe-tombstone --protocol-dir <dir>` repeats post-delete live absence and
   embeds all five controlled-stop receipts in the tombstone. Both G8 releases
   bind that tombstone. The G8 runtime repeats post-delete absence immediately
   before its O_EXCL execution claim.

The operator-facing command shapes are:

```sh
python -m evals.fleet.autocontinue_generation8_optimized_v1 observe-prestop \
  --repo <exact-commit-checkout> --output <unused-prestop.json>
python -m evals.fleet.autocontinue_generation8_optimized_v1 \
  render-delete-authorization --repo <exact-commit-checkout> \
  --prestop <prestop.json> --output <unused-delete-authorization.json>
python -m evals.fleet.autocontinue_generation8_optimized_v1 \
  execute-controlled-stop --repo <exact-commit-checkout> \
  --prestop <prestop.json> --delete-authorization <authorization.json> \
  --protocol-dir <unused-protocol-directory>
```

These commands require an in-cluster service-account context with read access,
DELETE permission for only the exact owned Jobs, SFS access, and the Fleet key
for metadata-only session reconciliation. They never print or persist the
service-account token or Fleet key. The checked-in submit wrapper remains HELD
and calls none of them.

## Single-pass validation

The runtime parses each immutable spec, plan, held receipt, predecessor release,
package manifest, stop tombstone, and scoring release once into one typed
`ValidatedContext`. Claim and terminal operations receive that context and
assert canonical byte equality with the originally validated values. They never
re-run the recursive Generation-1-through-7 validators. The cache exists only
inside that process; a new process or changed input is validated from scratch.

The bounded local benchmark against the exact pushed Generation-7 bytes took
61.53 seconds for one model's recursive `validate-spec`. The Generation-8
two-model held preview took 0.06 seconds, a roughly 1,025x reduction in static
pre-claim validation wall time. The focused regression test requires at least a
10x improvement, leaving substantial slower-runner headroom.

The mounted package still checks every packaged source byte and object-manifest
digest before execution. The Docker CLI, buildx plugin, evaluator image, and
resource policy remain pinned exactly as in Generation 7.

## Current state

Run the module and package previews only after these files are committed in a
reviewed successor. The submit wrapper intentionally exits with status 78 in
`submit` mode until a separate reviewed change adds the accepted live tombstone
and append-only releases. Preview does not create cluster objects.
