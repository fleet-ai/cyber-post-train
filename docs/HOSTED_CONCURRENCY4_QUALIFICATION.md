# Hosted endpoint concurrency-4 qualification

This is a **held, non-scoring operational gate** for the exact hosted routes
`qwen3.8-27b` and `glm-5.3`. It does not authorize a bulk evaluation and cannot
be submitted by its checked-in launcher.

For each model, one CPU-only Job would send two synthetic structured-tool
requests per stream at concurrency 2, then concurrency 4. The tool surface is
exactly `bash` plus `submit_report`; neither tool is executed. No Fleet task,
instance, session, verifier, or scoring route is called. Receipts retain only
UID-bound identity, counts, aggregate latency, and protocol-validity facts.

The concurrency-4 decision was fixed before observation. Both waves must have
zero request or structured-tool errors, every request must finish within 300
seconds, each wave within 600 seconds, concurrency 4 throughput must be at least
1.25 times concurrency 2 throughput, and its p95 stream latency may be at most
2.5 times the concurrency 2 p95. A failure or rejection is descriptive and must
not be retried merely to obtain a pass.

## Exact release gates

Before a human prepares a separately reviewed launch release, all of the
following must hold:

1. The generation-6 Qwen and GLM canaries are exclusively Complete and have
   digest-valid accepted receipts under their exact Job and Pod UIDs. The
   superseded generation-5 canaries failed before model execution, so their
   generation-5-specific launch release is permanently invalid and must not be
   reused.
2. No scored generation-6 controller is active, and all exact-pass@4 bulk
   Jobs, Pods, ConfigMaps, and output roots are absent. This
   qualification uses its own `hosted-concurrency-qualification-v1` lease
   namespace and must not overlap scored endpoint traffic.
3. The Fleet credential resolves to team id
   `a1025f0b-ad67-49fc-a023-51800ab43e84`; the exact secret UID is still
   `e0febd8e-94a2-46b0-a0bf-dd6b3154187b`; each hosted served id appears once.
4. The exact package commit is clean, pushed, reviewed, and the immutable
   ConfigMap, intent, Job, Pods, output root, and qualification lease are absent.
5. The output root is create-once and the Job remains CPU-only,
   `fleet-serve-low`, `preemptionPolicy: Never`, `backoffLimit: 0`.
6. A release changes the held annotations and launcher only; it must not change
   request payloads, model revisions, tool schemas, waves, or thresholds.

Even a passing terminal receipt only supports a proposal to raise the hosted
stream cap from two to four. It does not prove task capability, exact hosted
weight bytes, or the 262144-token context when the hosted roster omits context
metadata. Existing accepted generation-6 harness evidence remains required,
and any bulk concurrency change needs its own reviewed, task-boundary-safe plan.

After both Generation-7 Jobs are exclusively complete, use the score-blind
evidence wrapper to prepare the append-only v3 release:

```sh
evals/fleet/scripts/prepare_hosted_concurrency4_qualification_release_v3.sh preview
evals/fleet/scripts/prepare_hosted_concurrency4_qualification_release_v3.sh render-release
```

The wrapper binds the exact live `allie-dev` UID, reads only the two exact G7
Job/Pod objects and their allowlisted terminal/claim receipts, and validates
the complete acceptance chain before writing the release once. Commit and
review that new receipt before using the existing v3 submitter. It never reads
session transcripts, prompts, flags, or scores.

Create-only Kubernetes objects may be applied through
`evals.fleet.kubernetes_create_relay`. Its input is a separately frozen
kind/name/digest allowlist, and it supports only internal ConfigMaps, CPU-only
Jobs, and ClusterIP Services. When the local identity cannot create an object,
the exact live `allie-dev` Pod can relay the same immutable bytes using its
projected service-account identity; the token never leaves the Pod. Partial
creates are preserved for reconciliation and are never retried blindly.

The G6-gated v2 successor remains held until both canaries meet the first gate.
Its held preview performs no model, task, session, scoring, or verifier call and
creates no object. Once the terminal evidence exists, render the append-only v2
launch release from the clean implementation commit; only the v2 launcher may
consume it.
