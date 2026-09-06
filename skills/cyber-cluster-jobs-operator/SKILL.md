---
name: cyber-cluster-jobs-operator
description: Preview, submit, monitor, and release general containerized GPU workloads through Fleet's Nebius cluster Jobs API. Use for model serving, staging, training, or GPU diagnostics through /v1/runs; not for managed Fleet agent-evaluation jobs through /v1/jobs.
---

# Cyber Cluster Jobs Operator

Operate the cluster compute service without confusing it with Fleet's managed evaluation service.

## Establish the live contract

Read the repository `AGENTS.md`, then run:

```sh
python3 skills/cyber-cluster-jobs-operator/scripts/check_contract.py
```

The check is GET-only. It must prove that the deployed `POST /v1/runs` request body is a general job shape requiring an immutable container `image` and a `command`, with explicit worker/GPU/resource fields. Treat drift as a stop condition and inspect the current live OpenAPI schema. Do not infer the deployed contract from a local client, an old Theseus checkout, or a previous receipt.

## Choose the correct service

- Use `POST https://api.ft.flt.build/v1/runs` for a containerized cluster workload such as an exact model server, training process, model staging operation, or GPU diagnostic. This API renders a RayJob and admits it through the cluster queue.
- Use `POST https://orchestrator.fleetai.com/v1/jobs` for a managed Fleet agent evaluation whose model and harness are supported by that platform's catalog.
- Do not substitute a catalog model for an exact checkpoint merely because names look similar. Pool results only when model bytes/revision, serving treatment, harness, tools, task versions, and retry policy are all proven compatible.

## Preview and submit safely

1. Pin the image by immutable digest and bind the exact command, model path/revision, resources, run directory, priority, preemption policy, secrets, and topology.
2. Use `POST /v1/runs/preview` first and validate the rendered image, command, pull secret, GPU count, priority, preemption behavior, mounts, and run directory.
3. Search every page of `GET /v1/runs` plus live Kubernetes objects and the shared run directory before submission. A title alone is not a duplicate key because the service mints the run name.
4. Submit once. Persist a sanitized receipt containing the request digest, returned API run name, run directory, and later the RayJob, Workload, and Pod UIDs. Never persist the bearer token.
5. Prove model identity and non-scored behavioral parity before directing scored controllers to a new server.

## Resource discipline

- A general `/v1/runs` worker currently requests at least one GPU; do not use it for a controller that only calls a remote endpoint when laptop or queued CPU execution is sufficient.
- Never claim both highest cluster priority and non-preemption without checking the live PriorityClass behavior. Choose the highest class whose deployed preemption policy actually satisfies the user's instruction; preserve any tradeoff explicitly.
- `preemptionPolicy: Never` covers Kubernetes Pod scheduling only. It does not protect a Kueue Workload from ClusterQueue quota preemption. Monitor the exact Workload UID and its complete condition history. On any `Preempted`, preemption-backed `Requeued`, or `Evicted` transition, treat every recreated RayCluster, Pod, and Service as a new unqualified endpoint, stop assigning work, preserve sanitized evidence, and release it through the Jobs API.
- Monitor startup, readiness, GPU utilization, traffic heartbeats, and controller health. If an admitted server has no useful GPU work for the user-approved idle window, terminate it through the supported API and preserve a terminal receipt before creating a corrected successor.
- Never cancel, reprioritize, or mutate peer workloads.

## Evidence boundary

Status and identity evidence may include object names and UIDs, resource shape, image digest, command digest, readiness, utilization, and sanitized terminal reasons. Do not read or expose benchmark prompts, flags, private traces, scores, or credentials merely to operate the server.
