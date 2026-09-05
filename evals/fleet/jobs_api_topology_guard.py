"""Fail-closed checks for Jobs API topology and Kueue flavor compatibility."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import yaml

TOPOLOGY_ANNOTATIONS = {
    "kueue.x-k8s.io/podset-preferred-topology": "preferred",
    "kueue.x-k8s.io/podset-required-topology": "required",
}

_QUANTITY_SUFFIXES = {
    "m": Decimal("0.001"),
    "Ki": Decimal(2**10),
    "Mi": Decimal(2**20),
    "Gi": Decimal(2**30),
    "Ti": Decimal(2**40),
}


def _quantity(value: Any) -> Decimal:
    """Parse the bounded Kubernetes quantities used by the Jobs API gate."""
    raw = str(value)
    for suffix, multiplier in _QUANTITY_SUFFIXES.items():
        if raw.endswith(suffix):
            return Decimal(raw[: -len(suffix)]) * multiplier
    return Decimal(raw)


def _rendered_podset_requests(manifest_yaml: str) -> tuple[str, dict[str, dict[str, Any]]]:
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    if len(manifests) != 1 or manifests[0].get("kind") != "RayJob":
        raise ValueError("preview must render exactly one RayJob")
    manifest = manifests[0]
    queue = ((manifest.get("metadata") or {}).get("labels") or {}).get("kueue.x-k8s.io/queue-name")
    if not isinstance(queue, str) or not queue:
        raise ValueError("preview omitted the LocalQueue")
    cluster = (manifest.get("spec") or {}).get("rayClusterSpec") or {}
    templates: list[tuple[str, dict[str, Any]]] = [
        ("head", ((cluster.get("headGroupSpec") or {}).get("template") or {}))
    ]
    for index, group in enumerate(cluster.get("workerGroupSpecs") or []):
        templates.append((f"worker-{index}", (group or {}).get("template") or {}))
    result: dict[str, dict[str, Any]] = {}
    for name, template in templates:
        spec = template.get("spec") or {}
        containers = spec.get("containers") or []
        if len(containers) != 1:
            raise ValueError(f"pod set {name} must render exactly one container")
        requests = (containers[0].get("resources") or {}).get("requests") or {}
        result[name] = {
            "requests": {key: _quantity(value) for key, value in requests.items()},
            "node_selector": spec.get("nodeSelector") or {},
        }
    return queue, result


def _cluster_queue_flavors(cluster_queue: dict[str, Any]) -> dict[str, dict[str, Decimal]]:
    result: dict[str, dict[str, Decimal]] = {}
    for group in (cluster_queue.get("spec") or {}).get("resourceGroups") or []:
        for flavor in (group or {}).get("flavors") or []:
            name = flavor.get("name")
            if not isinstance(name, str) or not name or name in result:
                raise ValueError("ClusterQueue flavor identity is invalid")
            result[name] = {
                row["name"]: _quantity(row.get("nominalQuota", "0"))
                for row in flavor.get("resources") or []
                if isinstance(row, dict) and isinstance(row.get("name"), str)
            }
    return result


def _reserved_by_flavor(cluster_queue: dict[str, Any]) -> dict[str, dict[str, Decimal]]:
    result: dict[str, dict[str, Decimal]] = {}
    for flavor in (cluster_queue.get("status") or {}).get("flavorsReservation") or []:
        name = flavor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("ClusterQueue reservation flavor identity is invalid")
        result[name] = {
            row["name"]: _quantity(row.get("total", "0"))
            for row in flavor.get("resources") or []
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
    return result


def _pod_requests(pod: dict[str, Any]) -> dict[str, Decimal]:
    """Return the effective scheduler request for one already-bound Pod."""
    spec = pod.get("spec") or {}

    def container_requests(container: dict[str, Any]) -> dict[str, Decimal]:
        resources = container.get("resources") or {}
        requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        return {
            resource: _quantity(requests.get(resource, limit))
            for resource, limit in {**limits, **requests}.items()
        }

    summed: dict[str, Decimal] = {}
    for container in spec.get("containers") or []:
        for resource, amount in container_requests(container).items():
            summed[resource] = summed.get(resource, Decimal(0)) + amount
    init_max: dict[str, Decimal] = {}
    for container in spec.get("initContainers") or []:
        for resource, amount in container_requests(container).items():
            init_max[resource] = max(init_max.get(resource, Decimal(0)), amount)
    result = {
        resource: max(summed.get(resource, Decimal(0)), init_max.get(resource, Decimal(0)))
        for resource in set(summed) | set(init_max)
    }
    for resource, amount in (spec.get("overhead") or {}).items():
        result[resource] = result.get(resource, Decimal(0)) + _quantity(amount)
    return result


def _tolerates(taint: dict[str, Any], tolerations: list[dict[str, Any]]) -> bool:
    if taint.get("effect") not in {"NoSchedule", "NoExecute"}:
        return True
    for toleration in tolerations:
        if toleration.get("effect") not in {None, "", taint.get("effect")}:
            continue
        operator = toleration.get("operator", "Equal")
        if operator == "Exists" and toleration.get("key") in {None, "", taint.get("key")}:
            return True
        if (
            operator == "Equal"
            and toleration.get("key") == taint.get("key")
            and toleration.get("value", "") == taint.get("value", "")
        ):
            return True
    return False


def require_single_pod_node_capacity(
    manifest_yaml: str,
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    resource_flavor: dict[str, Any],
    *,
    expected_podset: str = "head",
) -> dict[str, Any]:
    """Require one currently schedulable node that fits the complete GPU Pod.

    ClusterQueue quota does not prove that a multi-GPU Pod fits one node.  This
    gate closes that gap using live allocatable resources and requests of
    non-terminal Pods already bound to each candidate node.  Admission remains
    authoritative and this snapshot must be refreshed immediately before create.
    """
    _, podsets = _rendered_podset_requests(manifest_yaml)
    if tuple(podsets) != (expected_podset,):
        raise ValueError("node-capacity gate requires exactly one rendered pod set")
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    cluster = (manifests[0].get("spec") or {}).get("rayClusterSpec") or {}
    template = (cluster.get("headGroupSpec") or {}).get("template") or {}
    pod_spec = template.get("spec") or {}
    selector = {
        **((resource_flavor.get("spec") or {}).get("nodeLabels") or {}),
        **(pod_spec.get("nodeSelector") or {}),
    }
    tolerations = pod_spec.get("tolerations") or []
    requested = podsets[expected_podset]["requests"]
    active_phases = {None, "", "Pending", "Running", "Unknown"}
    eligible: list[dict[str, Any]] = []
    for node in nodes:
        metadata = node.get("metadata") or {}
        spec = node.get("spec") or {}
        status = node.get("status") or {}
        if spec.get("unschedulable") is True:
            continue
        ready = next(
            (
                row.get("status")
                for row in status.get("conditions") or []
                if row.get("type") == "Ready"
            ),
            None,
        )
        if ready != "True":
            continue
        labels = metadata.get("labels") or {}
        if any(labels.get(key) != value for key, value in selector.items()):
            continue
        if any(not _tolerates(taint, tolerations) for taint in spec.get("taints") or []):
            continue
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("candidate node identity is absent")
        used: dict[str, Decimal] = {}
        for pod in pods:
            if (pod.get("spec") or {}).get("nodeName") != name:
                continue
            if (pod.get("status") or {}).get("phase") not in active_phases:
                continue
            for resource, amount in _pod_requests(pod).items():
                used[resource] = used.get(resource, Decimal(0)) + amount
        allocatable = {
            resource: _quantity(amount)
            for resource, amount in (status.get("allocatable") or {}).items()
        }
        free = {
            resource: allocatable.get(resource, Decimal(0)) - used.get(resource, Decimal(0))
            for resource in requested
        }
        if all(free[resource] >= amount for resource, amount in requested.items()):
            eligible.append(
                {
                    "name": name,
                    "uid": str(metadata.get("uid") or ""),
                    "free": {resource: str(amount) for resource, amount in free.items()},
                }
            )
    if not eligible:
        raise RuntimeError("no currently schedulable node fits the complete rendered GPU pod")
    return {
        "rendered_podset": expected_podset,
        "requested": {resource: str(amount) for resource, amount in requested.items()},
        "eligible_nodes": eligible,
        "must_recheck_immediately_before_submit": True,
    }


def require_exact_topology_candidate(
    manifest_yaml: str,
    local_queue: dict[str, Any],
    cluster_queue: dict[str, Any],
    resource_flavors: dict[str, dict[str, Any]],
    topologies: dict[str, dict[str, Any]],
    *,
    expected_queue: str,
    expected_level: str,
    expected_podsets: tuple[str, ...],
) -> dict[str, Any]:
    """Require an in-quota topology-capable flavor for every rendered GPU pod set.

    The Jobs API cannot name a ResourceFlavor.  A required TAS annotation makes
    non-topology flavors ineligible; this gate additionally proves that at least
    one topology-aware flavor has enough unreserved quota for the complete
    request immediately before submit.  Admission remains authoritative, so the
    check must be repeated after preview and a later inadmissible Workload must
    be released rather than patched.
    """
    queue, podsets = _rendered_podset_requests(manifest_yaml)
    if queue != expected_queue or (local_queue.get("metadata") or {}).get("name") != queue:
        raise ValueError("rendered and live LocalQueue identities disagree")
    cluster_queue_name = (local_queue.get("spec") or {}).get("clusterQueue")
    if cluster_queue_name != (cluster_queue.get("metadata") or {}).get("name"):
        raise ValueError("LocalQueue and ClusterQueue identities disagree")
    if tuple(podsets) != expected_podsets:
        raise ValueError("rendered Ray pod-set shape drifted")

    topology_requests = rendered_topology_requests(manifest_yaml)
    if len(topology_requests) != len(expected_podsets) or any(
        row["mode"] != "required" or row["level"] != expected_level for row in topology_requests
    ):
        raise ValueError("every GPU pod set must require the exact topology level")

    quota = _cluster_queue_flavors(cluster_queue)
    reserved = _reserved_by_flavor(cluster_queue)
    aggregate: dict[str, Decimal] = {}
    for podset in podsets.values():
        for resource, amount in podset["requests"].items():
            aggregate[resource] = aggregate.get(resource, Decimal(0)) + amount

    candidates: list[dict[str, Any]] = []
    for name, limits in quota.items():
        flavor = resource_flavors.get(name)
        if not flavor:
            raise ValueError(f"ResourceFlavor evidence is absent for: {name}")
        flavor_spec = flavor.get("spec") or {}
        topology_name = flavor_spec.get("topologyName")
        if not isinstance(topology_name, str) or not topology_name:
            continue
        topology = topologies.get(topology_name)
        if not topology:
            raise ValueError(f"Topology evidence is absent for: {topology_name}")
        levels = [row.get("nodeLabel") for row in (topology.get("spec") or {}).get("levels") or []]
        if expected_level not in levels:
            continue
        labels = flavor_spec.get("nodeLabels") or {}
        if any(
            labels.get(key) not in {None, value}
            for podset in podsets.values()
            for key, value in podset["node_selector"].items()
        ):
            continue
        available = {
            resource: limits.get(resource, Decimal(0))
            - reserved.get(name, {}).get(resource, Decimal(0))
            for resource in aggregate
        }
        if all(available[resource] >= amount for resource, amount in aggregate.items()):
            candidates.append(
                {
                    "name": name,
                    "uid": str((flavor.get("metadata") or {}).get("uid") or ""),
                    "topology_name": topology_name,
                    "topology_uid": str((topology.get("metadata") or {}).get("uid") or ""),
                    "available": {key: str(value) for key, value in available.items()},
                }
            )
    if not candidates:
        raise RuntimeError("no topology-aware flavor has enough unreserved quota")
    return {
        "local_queue": queue,
        "local_queue_uid": str((local_queue.get("metadata") or {}).get("uid") or ""),
        "cluster_queue": cluster_queue_name,
        "cluster_queue_uid": str((cluster_queue.get("metadata") or {}).get("uid") or ""),
        "required_topology_level": expected_level,
        "rendered_podsets": list(podsets),
        "aggregate_requests": {key: str(value) for key, value in aggregate.items()},
        "eligible_flavors": candidates,
        "must_recheck_immediately_before_submit": True,
    }


def rendered_topology_requests(manifest_yaml: str) -> list[dict[str, str]]:
    """Return every topology request rendered onto a Ray pod-set template."""
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    if len(manifests) != 1 or manifests[0].get("kind") != "RayJob":
        raise ValueError("preview must render exactly one RayJob")
    cluster = (manifests[0].get("spec") or {}).get("rayClusterSpec") or {}
    templates: list[tuple[str, dict[str, Any]]] = [
        ("head", ((cluster.get("headGroupSpec") or {}).get("template") or {}))
    ]
    for index, group in enumerate(cluster.get("workerGroupSpecs") or []):
        templates.append((f"worker-{index}", (group or {}).get("template") or {}))

    requests: list[dict[str, str]] = []
    for pod_set, template in templates:
        annotations = (template.get("metadata") or {}).get("annotations") or {}
        present = [key for key in TOPOLOGY_ANNOTATIONS if key in annotations]
        if len(present) > 1:
            raise ValueError(f"pod set {pod_set} has conflicting topology requests")
        if present:
            key = present[0]
            value = annotations[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"pod set {pod_set} has an invalid topology level")
            requests.append(
                {
                    "pod_set": pod_set,
                    "mode": TOPOLOGY_ANNOTATIONS[key],
                    "level": value,
                }
            )
    return requests


def queue_topology_evidence(
    local_queue: dict[str, Any],
    cluster_queue: dict[str, Any],
    resource_flavors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove every usable flavor in the queue supports topology-aware scheduling.

    Kueue may combine different flavors for resources in one pod set. A rendered
    topology request is therefore safe only when every flavor with positive
    quota in the referenced ClusterQueue has a topology definition.
    """
    cluster_queue_name = (local_queue.get("spec") or {}).get("clusterQueue")
    if cluster_queue_name != (cluster_queue.get("metadata") or {}).get("name"):
        raise ValueError("LocalQueue and ClusterQueue identities disagree")

    usable_names: set[str] = set()
    for group in (cluster_queue.get("spec") or {}).get("resourceGroups") or []:
        for flavor in (group or {}).get("flavors") or []:
            resources = (flavor or {}).get("resources") or []
            has_quota = any(
                str((resource or {}).get("nominalQuota", "0")) not in {"0", "0m"}
                for resource in resources
            )
            if has_quota:
                name = (flavor or {}).get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("ClusterQueue contains an unnamed usable flavor")
                usable_names.add(name)

    missing = sorted(usable_names - set(resource_flavors))
    if missing:
        raise ValueError(f"ResourceFlavor evidence is absent for: {', '.join(missing)}")
    flavors = [
        {
            "name": name,
            "uid": str((resource_flavors[name].get("metadata") or {}).get("uid") or ""),
            "topology_name": (resource_flavors[name].get("spec") or {}).get("topologyName"),
        }
        for name in sorted(usable_names)
    ]
    incompatible = [row["name"] for row in flavors if not row["topology_name"]]
    return {
        "local_queue": (local_queue.get("metadata") or {}).get("name"),
        "local_queue_uid": (local_queue.get("metadata") or {}).get("uid"),
        "cluster_queue": cluster_queue_name,
        "cluster_queue_uid": (cluster_queue.get("metadata") or {}).get("uid"),
        "usable_flavors": flavors,
        "incompatible_flavors": incompatible,
        "all_usable_flavors_topology_aware": not incompatible,
    }


def require_rendered_topology_compatibility(
    manifest_yaml: str,
    local_queue: dict[str, Any],
    cluster_queue: dict[str, Any],
    resource_flavors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reject a rendered topology request the live queue cannot always honor."""
    requests = rendered_topology_requests(manifest_yaml)
    evidence = queue_topology_evidence(local_queue, cluster_queue, resource_flavors)
    evidence["rendered_topology_requests"] = requests
    if requests and not evidence["all_usable_flavors_topology_aware"]:
        names = ", ".join(evidence["incompatible_flavors"])
        raise RuntimeError(
            "Jobs API rendered topology-aware scheduling but the live queue has "
            f"usable ResourceFlavors without topology support: {names}"
        )
    return evidence
