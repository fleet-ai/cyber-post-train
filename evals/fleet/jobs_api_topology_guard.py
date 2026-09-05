"""Fail-closed checks for Jobs API topology and Kueue flavor compatibility."""

from __future__ import annotations

from typing import Any

import yaml

TOPOLOGY_ANNOTATIONS = {
    "kueue.x-k8s.io/podset-preferred-topology": "preferred",
    "kueue.x-k8s.io/podset-required-topology": "required",
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
