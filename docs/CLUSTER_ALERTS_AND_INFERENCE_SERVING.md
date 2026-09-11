# Cluster alerts and persistent inference serving

This guide records the stable operational boundary between Fleet training Jobs and
persistent inference services. Recheck the deployed monitor and inference-control-plane
versions before acting; thresholds and delivery routing can change.

## Alert sources

The implementation lives in `fleet-ai/theseus`:

- `services/ftl/src/ftl/report_status.py` implements failed-Job reporting.
- `services/ftl/src/ftl/report.py` implements idle-GPU reporting.
- `services/ftl/src/ftl/verify.py` performs the optional second-stage idle check.
- `k8s/training/charts/ft-idle-monitor/` defines the deployed schedules, thresholds,
  exclusions, ownership map, and delivery settings.

At the 2026-09-06 audit, the deployed FTL image was built from Theseus commit
`91212928e107ff949899f37a563d7cd0b9c9d123` and matched current main for these files.

### Failed Jobs

The status monitor runs every five minutes and watches Kubernetes `Job` and `RayJob`
objects in `fleet-train-jobs`.

It reports a normal Job when it has a terminal `Failed=True` condition. It reports a
RayJob when `jobStatus` or `jobDeploymentStatus` is `FAILED`. The KubeRay submitter Job
owned by a RayJob is excluded so one RayJob failure is not reported twice.

Each newly observed failed-object UID is copied to the training Slack channel and sent
to Better Stack's noncritical email/SMS route. Durable ConfigMap state deduplicates
accepted incidents by UID. Ownership comes from `fleet.ai/submitted-by`, then the
`owner` label, then an explicitly mapped job-name prefix. A `chris-*` or
`chrisisaverted-*` prefix maps to Chris when no stronger owner metadata exists.

Do not hide a real failure by deleting it before the monitor observes it, relabeling it,
moving it to an unrelated namespace, or forcing a false zero exit. Prevent avoidable
alerts by previewing the exact entrypoint, making expected handled outcomes exit cleanly,
using bounded canaries, and reserving terminal failure for a real defect.

### Idle GPUs

The idle monitor runs every thirty minutes. Its rule-based suspect set is GPU workloads
that are at least one hour old and averaged below 1% GPU utilization during the previous
thirty minutes. Missing utilization is not interpreted as idle. A total telemetry outage
is itself reported.

The optional verifier can suppress a suspect only when recent logs positively establish
legitimate bursty work. `stuck`, `confirmed-idle`, `insufficient-evidence`, verifier
failure, and suspects beyond the verification cap remain visible. The verifier is a
backstop, not an operating strategy.

The `inference` namespace is excluded because a routed inference Deployment may
legitimately wait for requests. That exclusion does not justify an experiment-owned
endpoint without a live consumer: follow `docs/GPU_RESOURCE_LIFECYCLE.md` and release
dedicated capacity after the bounded campaign or when useful consumption stops.

Free-GPU capacity messages were disabled in the audited deployment. They remain visible
through on-demand FTL status commands and can be re-enabled by a future platform change.

### Inference health rules

Theseus also installs Prometheus rules for model availability, missing metrics, persistent
queues, cache pressure, request retractions or aborts, HTTP errors, time to first token,
container restarts, staging failures, and stale inference-control-plane state. At the
audit date, managed Alertmanager delivery was disabled, so these rules exposed firing
state but did not send Slack or PagerDuty messages. Never assume that remains true without
checking the live deployment.

## Resource choice

### Jobs API policy checked 2026-09-10

The deployed `https://api.ft.flt.build/v1/openapi.json` exposes generic
`POST /v1/runs/preview` and `POST /v1/runs`: image, command, workers, resources,
environment and Secret references are configurable. It is not limited to the old
typed model catalog. Matching Theseus schema:
`services/fleet-train-api/src/fleet_train_api/schemas/rl_job.py` at
`a70a7ee85e34f9858f5897b53c79e0f7f2e07b5e`.

- Request pod priority `c1` for the authorized high-priority training class or
  `c2` for backfill. Queue priority is now derived; explicit
  `queue_priority_class` is rejected. Do not copy old `q1` payload overrides.
- The schema documents Kueue workload preemption as disabled. This is not a
  guarantee against node failures, administrative cancellation or every other
  interruption. Keep checkpoints and verify actual rendered/effective policy.
- The generic API requires at least one GPU per worker. Do not allocate a GPU
  merely to run CPU data validation or copy files. Use an authorized CPU path.
- Each POST creates a fresh run name; it is not an idempotency key. Preserve a
  durable pre-POST intent and exhaustively reconcile uncertain responses.
- Inject credentials via existing Secret references, never literal request env
  values or command arguments. Do not print server failure bodies: they may embed
  private trainer logs.

These are dated observations. The CLI validates the live preview and stops on
drift; never bypass that stop by manually unsuspending or relabeling a workload.

A 2026-09-11 generic preview for the pinned Miles image rendered `HTTPMode`
without a `rayVersion` field or a separate submitter Pod template. Do not impose
the old typed launcher's Ray 2.56 default on every generic image. Verify the
actual preview and the image's installed Ray/runtime on every node; this preview
alone did not qualify Miles GPU startup or training.

Runtime bundles prepared on macOS must reproduce byte-identically during Linux
preflight. Python 3.12's gzip header includes an OS byte; the shared bundler
normalizes it before hashing. Keep each environment string below Linux's 128-KiB
process-start limit (including its name and terminator). The request validator
rejects oversized values before preview; a large CPU-test bundle must be chunked
or staged, not passed as one environment value.

Use the Fleet inference control plane for a persistent routed model. It owns one
`InferenceModel`, one ready-only Service, one Deployment, gateway backends and routes,
exact model staging, explicit pause/resume/retire actions, and serving metrics in the
`inference` namespace.

Do not implement a persistent endpoint as a never-ending Job or RayJob in
`fleet-train-jobs`. That shape mixes serving lifecycle with batch success/failure,
becomes eligible for idle alerts, and creates noisy failure pages when a wrapper or
cleanup command exits nonzero.

Use the Jobs API for bounded batch work such as training, conversion, validation, or an
evaluation controller. Persistent serving and batch consumers are separate resources and
must have separate health and terminal contracts.

## Exact-replica checklist

Before adding an experiment-owned replica of a shared model, bind and compare:

- Hugging Face repository and immutable revision;
- complete weight and tokenizer manifests;
- serving image repository, immutable digest, engine, and command;
- weight precision, quantization, KV-cache dtype, and speculative decoding;
- tensor/data/expert parallel shape and GPU type/topology;
- context length, memory fraction, attention backends, reasoning parser, and tool parser;
- served model name, gateway route, capabilities, and sampling defaults; and
- non-scored health, model-information, tokenization, structured-tool, and concurrency
  probes.

A different public route name is operationally necessary for a separately addressable
replica. Record shared and dedicated endpoints as distinct serving blocks. Bind every
rollout to its serving block and report block-level outcomes before pooling. Partition
only complete task/attempt cells and never repeat an accepted or active cell during a
serving transition.

## Launch and release sequence

1. Reconcile current API and Kubernetes state, including object UIDs, routes, ready
   endpoints, image IDs, restarts, requests, queues, and accelerator utilization.
2. Diagnose any stale or unhealthy shared control plane before adding replicas.
3. Freeze a create-once registration payload and duplicate check.
4. Register one non-scored replica canary and wait for exclusive readiness.
5. Prove exact identity and behavior before attaching scored consumers.
6. Start bounded consumers and continuously reconcile useful request and acceptance
   progress using score-blind evidence.
7. Pause new claims, drain within the frozen allowance, preserve evidence, and release
   experiment-owned capacity when consumption ends or stalls outside its allowance.

Never preempt, reprioritize, unsuspend, cancel, or alter peer workloads to obtain serving
capacity. A persistent serving request may wait for normal admission.
