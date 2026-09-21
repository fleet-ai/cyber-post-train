"""Cross-namespace census for project-owned Fleet GPU capacity.

Capacity is allocated by Pods, not by chat summaries or a hand-maintained list of
experiments.  This module deliberately scans every namespace and recognizes both
training and Fleet inference ownership labels.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from typing import Any

GPU_RESOURCE = "nvidia.com/gpu"
ACTIVE_PHASES = {"Pending", "Running", "Unknown"}
ROLE_LABELS = {
    "training": "fleet.ai/run-name",
    "serving": "inference.fleet.ai/model",
}


class CapacityError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _items(document: dict, label: str) -> list[dict]:
    rows = document.get("items")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CapacityError(f"{label} response is not a Kubernetes list")
    return rows


def _gpu_quantity(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise CapacityError("GPU quantity must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CapacityError("GPU quantity must be an integer") from exc
    if result < 0 or str(result) != str(value):
        raise CapacityError("GPU quantity must be a nonnegative integer")
    return result


def _container_quantity(container: dict, field: str) -> int:
    resources = container.get("resources") or {}
    return _gpu_quantity((resources.get(field) or {}).get(GPU_RESOURCE))


def _effective_pod_quantity(spec: dict, field: str) -> int:
    """Return Kubernetes' effective Pod quantity for one resource field.

    Ordinary init containers run one at a time, so they contribute their peak,
    not their sum.  An init container with ``restartPolicy: Always`` is a
    restartable sidecar: it remains alive for later init containers and the app,
    so its quantity is cumulative.  This is the scheduler formula for the Pod
    shapes used by Fleet; Pod overhead is additive when declared.
    """
    app_total = sum(
        _container_quantity(container, field) for container in spec.get("containers") or []
    )
    running_sidecars = 0
    init_peak = 0
    for container in spec.get("initContainers") or []:
        quantity = _container_quantity(container, field)
        if container.get("restartPolicy") == "Always":
            running_sidecars += quantity
            init_peak = max(init_peak, running_sidecars)
        else:
            init_peak = max(init_peak, running_sidecars + quantity)
    overhead = _gpu_quantity((spec.get("overhead") or {}).get(GPU_RESOURCE))
    return max(app_total + running_sidecars, init_peak) + overhead


def _gpu_request(pod: dict) -> tuple[int, list[str]]:
    spec = pod.get("spec") or {}
    requested = _effective_pod_quantity(spec, "requests")
    limited = _effective_pod_quantity(spec, "limits")
    problems = []
    if requested != limited:
        problems.append("GPU requests and limits differ")
    return max(requested, limited), problems


def _starts_with(value: Any, prefixes: tuple[str, ...]) -> bool:
    return isinstance(value, str) and any(value.startswith(prefix) for prefix in prefixes)


def _identity(pod: dict, prefixes: tuple[str, ...]) -> tuple[str | None, str | None, list[str]]:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    matches = [
        (role, labels.get(key))
        for role, key in ROLE_LABELS.items()
        if _starts_with(labels.get(key), prefixes)
    ]
    problems: list[str] = []
    if matches:
        identities = {identity for _, identity in matches}
        if len(identities) != 1 or len(matches) != 1:
            problems.append("project ownership labels are ambiguous")
        return matches[0][0], matches[0][1], problems

    name = metadata.get("name")
    label_values = [value for value in labels.values() if isinstance(value, str)]
    name_matches = _starts_with(name, prefixes) or any(
        isinstance(name, str) and name.startswith("inference-" + prefix) for prefix in prefixes
    )
    label_matches = any(_starts_with(value, prefixes) for value in label_values)
    if name_matches or label_matches:
        problems.append("project GPU Pod has no recognized training or serving ownership label")
        return "unclassified", str(name), problems
    return None, None, []


def _claim_from_pod_sets(pod_sets: Any) -> tuple[int, int, list[str]]:
    """Return the logical node/GPU claim encoded by Kueue-style PodSets."""
    if not isinstance(pod_sets, list) or not pod_sets:
        return 0, 0, ["queued GPU claim has no PodSets"]
    nodes = gpus = 0
    problems: list[str] = []
    for pod_set in pod_sets:
        if not isinstance(pod_set, dict):
            return 0, 0, ["queued GPU claim PodSet is malformed"]
        count = pod_set.get("count", 1)
        template = pod_set.get("template") or {}
        spec = template.get("spec") if isinstance(template, dict) else None
        if type(count) is not int or count < 0 or not isinstance(spec, dict):
            return 0, 0, ["queued GPU claim PodSet is malformed"]
        quantity, quantity_problems = _gpu_request({"spec": spec})
        if quantity:
            nodes += count
            gpus += count * quantity
            problems.extend(quantity_problems)
    if not gpus:
        problems.append("queued claim contains no GPU request")
    return nodes, gpus, problems


def _rayjob_claim(rayjob: dict) -> tuple[int, int, list[str]]:
    cluster = (rayjob.get("spec") or {}).get("rayClusterSpec") or {}
    pod_sets: list[dict] = []
    head = cluster.get("headGroupSpec")
    if isinstance(head, dict):
        pod_sets.append(
            {
                "count": head.get("numOfHosts", 1),
                "template": head.get("template"),
            }
        )
    workers = cluster.get("workerGroupSpecs") or []
    if not isinstance(workers, list):
        return 0, 0, ["queued RayJob worker groups are malformed"]
    for worker in workers:
        if not isinstance(worker, dict):
            return 0, 0, ["queued RayJob worker group is malformed"]
        pod_sets.append(
            {
                "count": worker.get("replicas", 0),
                "template": worker.get("template"),
            }
        )
    return _claim_from_pod_sets(pod_sets)


def build_capacity_census(
    pods: dict,
    inference_models: dict,
    rayjobs: dict | None = None,
    workloads: dict | None = None,
    *,
    owner_prefixes: tuple[str, ...] = ("chris-q38-",),
    max_nodes: int = 8,
    max_gpus: int = 64,
    planned_nodes: int = 0,
    planned_gpus: int = 0,
    observed_at: str | None = None,
) -> dict:
    """Return a self-digesting, score-blind capacity receipt.

    ``planned_*`` represents the one create or resume being considered.  The
    receipt is qualified only when both the current and projected capacity are
    inside their limits and every owned allocation has an unambiguous role.
    """
    if (
        not owner_prefixes
        or any(not prefix for prefix in owner_prefixes)
        or min(max_nodes, max_gpus) <= 0
        or min(planned_nodes, planned_gpus) < 0
        or (planned_nodes == 0) != (planned_gpus == 0)
        or planned_gpus < planned_nodes
    ):
        raise CapacityError("capacity limits, planned resources and owner prefixes are invalid")

    problems: list[str] = []
    allocated: list[dict] = []
    queued: list[dict] = []
    role_counts = {"training": 0, "serving": 0, "unclassified": 0}

    for pod in _items(pods, "Pod"):
        gpus, pod_problems = _gpu_request(pod)
        if not gpus:
            continue
        role, identity, ownership_problems = _identity(pod, owner_prefixes)
        if role is None:
            continue
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        spec = pod.get("spec") or {}
        phase = status.get("phase", "Unknown")
        if phase not in ACTIVE_PHASES:
            continue
        restarts = sum(
            int(row.get("restartCount") or 0) for row in status.get("containerStatuses") or []
        )
        row = {
            "namespace": metadata.get("namespace", "default"),
            "name": metadata.get("name"),
            "uid": metadata.get("uid"),
            "resource_version": metadata.get("resourceVersion"),
            "role": role,
            "identity": identity,
            "phase": phase,
            "node": spec.get("nodeName"),
            "gpus": gpus,
            "restarts": restarts,
            "terminating": metadata.get("deletionTimestamp") is not None,
        }
        row_problems = [*pod_problems, *ownership_problems]
        if row["terminating"]:
            row_problems.append("owned GPU Pod is terminating and still consumes capacity")
        for problem in row_problems:
            problems.append(f"{row['namespace']}/{row['name']}: {problem}")
        role_counts[role] += 1
        (allocated if row["node"] else queued).append(row)

    allocated.sort(key=lambda row: (row["role"], row["namespace"], row["name"]))
    queued.sort(key=lambda row: (row["role"], row["namespace"], row["name"]))

    models: list[dict] = []
    serving_pods: dict[str, int] = {}
    for pod in [*allocated, *queued]:
        if pod["role"] == "serving":
            serving_pods[pod["identity"]] = serving_pods.get(pod["identity"], 0) + 1

    for model in _items(inference_models, "InferenceModel"):
        metadata = model.get("metadata") or {}
        name = metadata.get("name")
        if not _starts_with(name, owner_prefixes):
            continue
        spec = model.get("spec") or {}
        status = model.get("status") or {}
        row = {
            "namespace": metadata.get("namespace", "default"),
            "name": name,
            "uid": metadata.get("uid"),
            "resource_version": metadata.get("resourceVersion"),
            "desired_state": spec.get("desiredState"),
            "phase": status.get("phase"),
            "active_pods": int(status.get("activePods") or 0),
            "ready_replicas": int(status.get("readyReplicas") or 0),
            "observed_generation": status.get("observedGeneration"),
        }
        actual = serving_pods.pop(name, 0)
        row["observed_gpu_pods"] = actual
        if row["active_pods"] != actual:
            problems.append(
                f"{row['namespace']}/{name}: InferenceModel activePods={row['active_pods']} "
                f"but cross-namespace Pod census found {actual}"
            )
        models.append(row)
    for name, count in sorted(serving_pods.items()):
        problems.append(f"serving identity {name}: {count} GPU Pod(s) have no InferenceModel")
    models.sort(key=lambda row: (row["namespace"], row["name"]))

    pod_nodes_by_identity: dict[str, int] = {}
    pod_gpus_by_identity: dict[str, int] = {}
    for pod in [*allocated, *queued]:
        identity = pod["identity"]
        if pod["role"] == "training" and isinstance(identity, str):
            pod_nodes_by_identity[identity] = pod_nodes_by_identity.get(identity, 0) + 1
            pod_gpus_by_identity[identity] = pod_gpus_by_identity.get(identity, 0) + pod["gpus"]

    owned_rayjobs: dict[str, dict] = {}
    for rayjob in _items(rayjobs or {"items": []}, "RayJob"):
        role, identity, identity_problems = _identity(rayjob, owner_prefixes)
        if role is None:
            continue
        metadata = rayjob.get("metadata") or {}
        namespace, name = metadata.get("namespace", "default"), metadata.get("name")
        for problem in identity_problems:
            problems.append(f"{namespace}/{name}: {problem}")
        if role != "training" or not isinstance(identity, str) or identity != name:
            problems.append(f"{namespace}/{name}: owned RayJob identity is not exact")
            continue
        if identity in owned_rayjobs:
            problems.append(f"{namespace}/{name}: duplicate owned RayJob identity")
            continue
        status = rayjob.get("status") or {}
        if status.get("jobStatus") in {"SUCCEEDED", "FAILED", "STOPPED"}:
            continue
        nodes_claimed, gpus_claimed, claim_problems = _rayjob_claim(rayjob)
        for problem in claim_problems:
            problems.append(f"{namespace}/{name}: {problem}")
        owned_rayjobs[identity] = {
            "namespace": namespace,
            "name": name,
            "uid": metadata.get("uid"),
            "resource_version": metadata.get("resourceVersion"),
            "nodes": nodes_claimed,
            "gpus": gpus_claimed,
            "source": "rayjob",
        }

    workload_claims: dict[str, dict] = {}
    for workload in _items(workloads or {"items": []}, "Workload"):
        metadata = workload.get("metadata") or {}
        conditions = (workload.get("status") or {}).get("conditions") or []
        if any(
            isinstance(condition, dict)
            and condition.get("type") == "Finished"
            and condition.get("status") == "True"
            for condition in conditions
        ):
            continue
        owners = [
            owner
            for owner in metadata.get("ownerReferences") or []
            if isinstance(owner, dict) and owner.get("kind") == "RayJob"
        ]
        owner_name = owners[0].get("name") if len(owners) == 1 else None
        labels = metadata.get("labels") or {}
        labelled = labels.get(ROLE_LABELS["training"])
        identity = owner_name if _starts_with(owner_name, owner_prefixes) else labelled
        if not _starts_with(identity, owner_prefixes):
            continue
        namespace, name = metadata.get("namespace", "default"), metadata.get("name")
        nodes_claimed, gpus_claimed, claim_problems = _claim_from_pod_sets(
            (workload.get("spec") or {}).get("podSets")
        )
        for problem in claim_problems:
            problems.append(f"{namespace}/{name}: {problem}")
        if len(owners) != 1 or identity not in owned_rayjobs:
            problems.append(f"{namespace}/{name}: owned queued Workload has no exact RayJob owner")
        elif (
            owners[0].get("uid")
            and owned_rayjobs[identity].get("uid")
            and owners[0]["uid"] != owned_rayjobs[identity]["uid"]
        ):
            problems.append(f"{namespace}/{name}: queued Workload owner UID differs from RayJob")
        if identity in workload_claims:
            problems.append(f"{namespace}/{name}: duplicate queued Workload claim")
            continue
        workload_claims[identity] = {
            "namespace": namespace,
            "name": name,
            "uid": metadata.get("uid"),
            "resource_version": metadata.get("resourceVersion"),
            "nodes": nodes_claimed,
            "gpus": gpus_claimed,
            "source": "workload",
        }

    queued_claims: list[dict] = []
    for identity in sorted(set(owned_rayjobs) | set(workload_claims)):
        rayjob = owned_rayjobs.get(identity)
        claim = workload_claims.get(identity) or rayjob
        if claim is None:
            continue
        if (
            workload_claims.get(identity)
            and rayjob
            and (claim["nodes"] != rayjob["nodes"] or claim["gpus"] != rayjob["gpus"])
        ):
            problems.append(f"{claim['namespace']}/{claim['name']}: Workload/RayJob claim differs")
        remaining_nodes = max(0, claim["nodes"] - pod_nodes_by_identity.get(identity, 0))
        remaining_gpus = max(0, claim["gpus"] - pod_gpus_by_identity.get(identity, 0))
        if (remaining_nodes == 0) != (remaining_gpus == 0):
            problems.append(
                f"{claim['namespace']}/{claim['name']}: queued claim is only partially represented"
            )
        if remaining_nodes or remaining_gpus:
            queued_claims.append(
                {
                    **claim,
                    "identity": identity,
                    "nodes": remaining_nodes,
                    "gpus": remaining_gpus,
                }
            )

    nodes = sorted({row["node"] for row in allocated})
    current_nodes = len(nodes) + len(queued) + sum(row["nodes"] for row in queued_claims)
    current_gpus = (
        sum(row["gpus"] for row in allocated)
        + sum(row["gpus"] for row in queued)
        + sum(row["gpus"] for row in queued_claims)
    )
    projected_nodes = current_nodes + planned_nodes
    projected_gpus = current_gpus + planned_gpus
    if projected_nodes > max_nodes:
        problems.append(f"projected GPU nodes {projected_nodes} exceed limit {max_nodes}")
    if projected_gpus > max_gpus:
        problems.append(f"projected GPUs {projected_gpus} exceed limit {max_gpus}")

    receipt = {
        "schema": "cyber_project_gpu_capacity_census_v1",
        "observed_at": observed_at
        or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope": {
            "kubernetes_namespaces": "all",
            "owner_prefixes": list(owner_prefixes),
            "ownership_labels": ROLE_LABELS,
        },
        "limits": {"nodes": max_nodes, "gpus": max_gpus},
        "planned": {"nodes": planned_nodes, "gpus": planned_gpus},
        "current": {
            "nodes": current_nodes,
            "gpus": current_gpus,
            "node_names": nodes,
            "role_pod_counts": role_counts,
            "allocated_pods": allocated,
            "queued_pods": queued,
            "queued_claims": queued_claims,
            "inference_models": models,
        },
        "projected": {"nodes": projected_nodes, "gpus": projected_gpus},
        "problems": sorted(set(problems)),
    }
    receipt["qualified"] = not receipt["problems"]
    receipt["sha256"] = _digest(receipt)
    return receipt


def _kubectl_json(context: str, *arguments: str) -> dict:
    command = [
        "kubectl",
        "--context",
        context,
        "--request-timeout=60s",
        *arguments,
        "-o",
        "json",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=75)
    if result.returncode:
        raise CapacityError("Kubernetes capacity read failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CapacityError("Kubernetes capacity read returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CapacityError("Kubernetes capacity read returned a non-object")
    return value


def live_capacity_census(context: str, **kwargs: Any) -> dict:
    """Read Pods plus queued GPU claims from every namespace, without mutation."""
    if not context:
        raise CapacityError("an explicit Kubernetes context is required")
    pods = _kubectl_json(context, "get", "pods", "--all-namespaces")
    models = _kubectl_json(
        context,
        "get",
        "inferencemodels.inference.fleet.ai",
        "--all-namespaces",
    )
    rayjobs = _kubectl_json(context, "get", "rayjobs.ray.io", "--all-namespaces")
    workloads = _kubectl_json(
        context,
        "get",
        "workloads.kueue.x-k8s.io",
        "--all-namespaces",
    )
    return build_capacity_census(pods, models, rayjobs, workloads, **kwargs)
