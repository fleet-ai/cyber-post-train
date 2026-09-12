# Qwen teacher LR dev pre-POST audit — September 12, 2026

The old prepared LR arms are internally consistent but do **not** bind the
current available-A provenance named in frozen V5. Fresh V2 configs now pin a
create-once metadata successor. Nothing in this audit stages data, submits a
job, or establishes live capacity/authentication. Production remains blocked.

## Authority and limits

[V5](../configs/studies/qwen-blackbox-teacher-staged-search-v5.json) explicitly
opens offline preflight and reviewed dev preview, not an automatic POST. Its
cadence acceptance is operational only and retains the irrecoverable source-Pod
UID/restart evidence gap. It is not a zero-restart proof for that source run.

[Draft V6](../configs/studies/qwen-blackbox-teacher-staged-search-v6.json)
records the parent's review of the explicit user launch instruction and active
goal in thread `01a08909-b4fa-7fb1-b207-68f0b494e78d`. This is **conversation
evidence, not a signed or cryptographic authorization receipt**. Its conditional
scope is only the three named bounded dev LR qualifications. It preserves
dev-first, normal c1/q1 admission, no requeue, and the aggregate eight-node ceiling.
It grants no peer mutation, production launch, or separate RL operation.

## Reproduced provenance discrepancy

The previous three `.operator/*lr*-prepared-v1` directories have valid
PREPARED/PREFLIGHT seals; their requests regenerate exactly from runtime
`7b0787891acdf9ebe0b07475339248dce64b659a546691a7c445f99b52a20a8f`.
The CPU receipts bind 602 rows, 29 represented training tasks and 700,359
supervised tokens. No local submission journal was present at audit time;
that is **not** a live duplicate/output-absence check.

Their corpus digest, `28376c692ec7caf23300b2cd80b85a0d9bbe156eedcc175daef36a0dea327a51`,
is exactly the old `teacher-corpus-a-v1` manifest with the current outcome-protocol
pointer. V5 instead names current available-A
`c4945ddb7cd8fd0b7b9ebbd9bad9b96215b0905c059ef1b22d71d4f214b4eb8e`.
Comparing local metadata proved equal train-file entries, source digest, split
and tokenizer. The differences are the `study_data.py` builder pin,
source-selection receipt and target-policy fingerprint. Thus this audit found
no changed training bytes, but identical row counts/Parquet hashes cannot make
different provenance receipts interchangeable.

The old configs, prepared directories and receipts remain unchanged. The current
source manifest receives **only** a new `fleet_dev_protocol_sha256` pointer and
self-seal. Its current builder/selection/fingerprint fields are preserved in
the [sanitized successor receipt](evidence/qwen38-study/2026-09-12-available-a-protocol-successor-v1.json).
The new manifest digest is
`1b083613c16362daf7c86fd8b458d99d7e167133df919602c21916314142c4dc`;
file digest is `d7fcb3e0c8c7726f5d1175f831fa0e88af632f1c6b68c6101c8db3e2cf4b85f7`.
Its intended fresh SFS path is
`/mnt/sfs/jobs/chris-q38-study-corpora-v1/teacher-a/manifest-available-a-outcome-v2.json`.
This audit did not stage it. Local private publication is under
`data/private/qwen-study-20260911/teacher-corpus-a-available-outcome-v2/`.

`python -m training.corpus_protocol` provides a small metadata-only create-once
publication utility. It requires exact source/protocol digests and changes no
Parquet or source record. The SFT compiler now accepts an optional
`data.manifest_sha256`; these three successors require it, so pointing them at
a valid but wrong-provenance manifest fails before preparation.

## Exact successor arms

| Config | Fresh run/output/W&B basename | LR | Pause | Checkpoints | Wave |
| --- | --- | ---: | ---: | --- | ---: |
| [LR1 V2](../configs/qualification/qwen38-teacher-lr1-cosine-layout-dev-v2.json) | `chris-q38-ta8-lr1-dev-v2` | 1e-6 | 6 | 6 | 1 |
| [LR30 V2](../configs/qualification/qwen38-teacher-lr30-cosine-layout-dev-v2.json) | `chris-q38-ta8-lr30-dev-v2` | 3e-5 | 6 | 6 | 1 |
| [LR100 V2](../configs/qualification/qwen38-teacher-lr100-cosine-layout-dev-v2.json) | `chris-q38-ta8-lr100-dev-v2` | 1e-4 | 21 | 20, 21 | 2 |

Every arm remains fresh-base Qwen3.8-27B, one node/eight GPUs, global batch 8,
microbatch 1/GPU, 16,384 tokens, one-epoch **76-step scheduler horizon**, cosine,
four warmup steps, seed 42, latest three checkpoints. A pause does not shorten the
scheduler horizon. Requests stay 64 CPU, 512Gi memory requested/768Gi limit, c1,
no requeue, and the existing immutable SkyRL image. No reference-CE dev file is
loaded. Scalar-only W&B goes to `thefleet/cyber-post-train`, group
`q38-blackbox-sft-dev-lr-v2`; the actual fresh run IDs must still be checked live.

## Exact pre-POST checklist

1. Review V6's explicit conditional scope against current user authorization;
   do not infer production permission from it.
2. Create-once stage the new manifest at the fresh path. Independently verify
   its file/self digests and current builder, source-selection, target-policy,
   split, tokenizer and train-file bindings. Preserve the old staged manifest.
3. Prepare new plan/request directories on shared durable storage. Recompile
   from the V2 config and require the pinned manifest digest. Do not reuse,
   copy or modify old PREFLIGHT receipts or intent journals.
4. Run fresh zero-GPU preflight in the exact image for **each new** plan/request:
   native source hashes, model/sidecars, data hash, native config, actual
   tokenization/target accounting. Capture helper UID, imageID, zero-GPU status
   and terminal evidence. CPU success is not distributed CUDA qualification.
5. Confirm authenticated Jobs API **dev** origin and matching Kubernetes context,
   including working access to monitor exact UIDs and verify release. A kubecontext
   change does not change the HTTP destination. Do not POST while monitoring auth
   is unavailable.
6. Obtain/review authenticated dev preview with no errors or warnings: exact
   image, entrypoint, resources, output, Secret references, normal `training-lq`,
   c1 and effective derived q1 value 10,000; requeue disabled. A priority label is
   not immunity from infrastructure interruption.
7. Exhaustively reconcile canonical-cell producers, exact/prefix names, titles,
   output roots, existing journals and ambiguous submissions. Independently
   verify the new output/runtime paths are absent and W&B ID unused. Local
   absence alone proves none of these live facts.
8. Immediately before each POST, count admitted/reserved campaign allocations
   across clusters and stay within eight nodes **including the requested node**.
   Respect available dev capacity and the two-node first-wave ceiling; run LR100
   only after first-wave allocations release. Any unexpected defect holds later
   submissions for review. Do not alter peer work or bypass admission.
9. Predeclare monitoring: poll 60s, startup 30m, confirmed no-progress plus low-GPU
   and no meaningful I/O for 20m, hard 8h, at most 5m advancing-checkpoint drain.
   Missing telemetry is not proof of idle. Capture controller, Workload,
   RayCluster and Pod UIDs/imageID/restarts **before** TTL cleanup.
10. Use one durable `POST_INTENT_DO_NOT_RETRY` journal and exactly one Jobs API
    submit per approved cell. Reconcile an uncertain response; never replay it.

## Acceptance and remaining blockers

Before promotion, each new dev arm still requires finite update scalars through
its intended pause, complete per-step scalar-only W&B history and acknowledged
sync, an exact checkpoint seal, eight-rank zero-update checkpoint reload,
zero-restart proof and allocation release. LR100 must exercise periodic step 20
and planned-pause step 21. These are operational gates, not evidence of lift.

Production also remains blocked on the split-A base-control outcome lifecycle,
exact production preview and an explicit production decision. The canonical
LR1e-5 cell must not be replayed under a new name. Fleet-dev outcomes remain the
only HPO signal; training loss is diagnostic, and Fleet final/WebExploitBench
stay sealed from selection.

Local compiler/config/receipt regression tests cover the provenance mismatch,
exact digest guard, preservation, create-once publication and V6's scope.
The focused Python 3.12 suite passed **323 tests**; Ruff and scoped formatting
checks passed. Current `origin/main` was fetched; the bound split-A, split-B and
common-final files are byte-identical to it. The shared active branch has separate
ancestry and was not rebased beneath other agents.
They do not prove current remote mounts, W&B authentication, CPU execution,
available GPUs, or that any V2 training step has run.
