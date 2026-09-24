# First runtime check for the 33 pending Fleet blackbox tasks

This change prepares one small, model-free runtime check. It does **not** launch
the check.

## Why this task was selected

The 33 candidates have exact task versions and reviewed source lineage, but none
has the complete runtime receipt needed to admit it to training or evaluation.
Eight carry the discovery label `clean`. Seven of those eight have the most
common candidate shape: one application, one vulnerability atom, and medium
difficulty. Exactly one of those seven also has no exact-version or shared-atom
overlap with the bound teacher3k training corpus. The packet selects that one:

- application: `concur`
- task version: `ae236a07-128d-4556-ba2b-25a248818497`
- atom: `cyber/atoms/concur/extract-job-status-missing-auth@0:atom_source`

This is only a low-ambiguity first check. The `clean` label does not prove that
the environment starts, that the tools work, or that the grader runs.

## What the later check must prove

The existing canonical controller will create no model request. For this one
exact task version it must:

1. re-read and match the exact task, environment, seed, grader, task graph, and
   atom bindings;
2. start one environment exactly once;
3. see exactly the `bash` and `submit_report` tools and call both with fixed,
   harmless inputs;
4. run the task's real grader on the fixed `no_flag` negative control and receive
   a finite zero result;
5. write only a zero-message metadata receipt, never a prompt, response, flag,
   tool output, score, or credential; and
6. delete the exact environment and prove cleanup completed.

The grader call proves that grading is reachable and finite. It does not prove
that the vulnerability is exploitable.

## Outer Job safety

When explicitly authorized later, the controller should run in one CPU-only Job
on the development cluster. The future launcher must preserve these rules:

- c1 priority, zero GPUs, at most 30 minutes, no restart, and no Job retry;
- the root Job—not merely its Pod—must have
  `fleet.ai/failure-alerts: "off"`;
- the exact Job, ConfigMap, Secret, and owned Pod names must be absent before
  previews and again after two identical server previews;
- the reviewed controller files must come from one exact merged commit, with
  their bytes sealed in an immutable ConfigMap; the private one-task plan must
  be sealed separately in an immutable Secret;
- a file and its parent directory must be synced after writing the exclusive
  create intent and before the single create request;
- the root authorization names one local create journal. The same authorization
  cannot be pointed at a second journal to create the Job again;
- the server may add ordinary Kubernetes defaults, but it may not add another
  container, an initialization container, `envFrom`, a lifecycle hook, or any
  other unreviewed container field that could read the private plan;
- if the live create result differs from both reviewed previews, bind all three
  returned object UIDs durably, delete the exact Job first and the exact source
  objects second, prove they are absent, and fail the launch; it must never
  continue or retry that create;
- an uncertain create response is reconciled and never retried; a partial
  three-object create is first observed without change for at least 10 seconds,
  then requires explicit cleanup confirmation, fresh UID and resource-version
  reads, exact-UID deletion, and final absence proof; if all three objects are
  present but differ from the previews, explicit confirmation may delete them
  only when every stable exact-name object retains this packet's annotation;
  and
- cleanup must freshly read each object, prove its UID still matches, then use
  that read's current resource version with the exact name and UID for deletion;
  finally it must prove the Job, ConfigMap, Secret, owned Pods, and queue
  Workload are gone and no GPU allocation remains.

The create journal is newline-framed. A crash may leave only its final line
unfinished; recovery discards that unfinished tail but rejects corruption in
any earlier synced line. Outer cleanup also has its own synced intent, so a
local observer crash after the Job has already been deleted can resume without
recreating or guessing any object. Every cleanup poll performs a fresh
UID-scoped child search rather than trusting only the children seen initially.

Environment cleanup and Kubernetes Job cleanup are separate. Both must finish.
A completed Job proves the packaged entry point passed its environment-cleanup
gate. A failed Job is still removed from Kubernetes, but its outer receipt must
say that environment cleanup is unproven and that an external resource leak is
possible; Kubernetes absence alone is not a complete release claim.

## Why this packet cannot launch

The packet is deliberately marked `prepared_no_launch`, `launchable: false`, and
`launch_authorized: false`. No server preview, absence check, Job, ConfigMap,
Secret, or environment was created. A real run requires a fresh private plan
from merged source using the packet's exact task key and version selector, two
live server previews, exact name absence, pre-armed cleanup, and explicit root
authorization. The local launcher is deliberately fail-closed until those
inputs exist; its tests do not authorize a live create.

The machine-readable packet is
[`configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json`](../configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json).

## Offline verification

The launcher, packaged entry point, and canary contract can be checked without
contacting Kubernetes or Fleet:

```sh
uv run pytest -q \
  tests/test_task_quality_qualification_job.py \
  tests/test_task_quality_qualification.py \
  tests/test_task_quality_cleanup_recovery.py \
  tests/test_fleet_blackbox_qa33_zero_model_canary.py
uv run ruff check \
  evals/fleet/task_quality_qualification.py \
  evals/fleet/task_quality_qualification_job.py \
  evals/fleet/task_quality_qualification_job_entry.py
```

These checks exercise render validation and a fake Kubernetes control plane.
They do not perform a server preview, create a Job, or authorize a launch.
