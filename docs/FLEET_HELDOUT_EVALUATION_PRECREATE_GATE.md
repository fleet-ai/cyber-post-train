# One new Fleet held-out evaluation: launch gate

This note describes the smallest safe path to launch **one new** Qwen
held-out Fleet evaluation after its model is ready. It is a checklist, not a
launcher and not permission to create a cluster object.

## Reuse the existing evaluator, not an older campaign

The worker runtime already exists in
[`evals/fleet/cluster_entry.py`](../evals/fleet/cluster_entry.py), with its
bootstrap script in
[`evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh`](../evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh).
The next campaign should make a **new** immutable config, ConfigMap, Job name,
output directory, and database name for the selected checkpoint. Do not copy
or apply an old campaign's Job YAML: its names and evidence belong to that
campaign.

The current checked-in final template is deliberately unresolved, and the
prepared seed-44 comparison is explicitly not launchable. Existing seed-43
campaigns are historical records, not reusable inputs. Therefore no new Fleet
held-out evaluation is launch-ready today.

## The small missing launch mechanism

Once a candidate has an accepted checkpoint and serving-route proof, add one
small, dedicated CPU-evaluator create command. It should accept one new launch
packet containing the exact config, ConfigMap, Job, checkpoint provenance,
serving-route proof, output directory, and database name. It must not be a
generic training launcher or a clone of an earlier campaign.

Immediately before its one `kubectl create`, the command must do only these
two live safety checks:

1. **Fresh duplicate check.** Read the exact new Job, ConfigMap, Kueue
   Workload, Pod, output directory, database, and evaluation ledger. Stop if
   any exists, or if the ledger already contains the same complete evaluation
   identity. This protects both against name reuse and against an accidental
   scientific duplicate.
2. **Server-rendered alert proof.** Send the exact Job to Kubernetes with a
   server dry-run, then inspect the returned root `batch/v1 Job`. Require:

   ```yaml
   metadata:
     annotations:
       fleet.ai/failure-alerts: "off"
   ```

   A source YAML value, request flag, or Pod-template annotation alone does
   not count. Stop if the returned root object lacks that exact value.

After both checks, write a durable create-intent record and make exactly one
create request. If its response is uncertain, re-read the exact names; never
repeat the create request blindly.

## Required regression tests for that command

- reject an existing object or matching evaluation ledger;
- reject a server-rendered Job missing the root annotation even when its Pod
  template has it; and
- prove that a successful dry-run is followed by one create at most.

This is deliberately narrow. It preserves the existing evaluator runtime and
adds only the two safeguards required before a new Fleet object can exist.
