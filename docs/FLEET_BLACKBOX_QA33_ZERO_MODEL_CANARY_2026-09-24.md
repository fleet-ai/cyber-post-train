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
- the exact Job and ConfigMap names must be absent before previews and again
  after two identical server previews;
- a file and its parent directory must be synced after writing the exclusive
  create intent and before the single create request;
- an uncertain create response is reconciled and never retried; and
- cleanup must freshly read each object, prove its UID still matches, then use
  that read's current resource version with the exact name and UID for deletion;
  finally it must prove the Job, ConfigMap, owned Pods, and queue Workload are
  gone and no GPU allocation remains.

Environment cleanup and Kubernetes Job cleanup are separate. Both must finish.

## Why this packet cannot launch

The packet is deliberately marked `prepared_no_launch`, `launchable: false`, and
`launch_authorized: false`. No server preview, absence check, Job, ConfigMap, or
environment was created. A real run requires a fresh private plan from merged
source using the packet's exact task key and version selector, a reviewed narrow
Job launcher, two live server previews, exact name absence, pre-armed cleanup,
and explicit root authorization.

The machine-readable packet is
[`configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json`](../configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json).
