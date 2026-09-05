import pytest
import yaml

from evals.fleet import jobs_api_topology_guard as topology


def _preview(annotation: str | None = "preferred") -> str:
    annotations = {}
    if annotation == "preferred":
        annotations["kueue.x-k8s.io/podset-preferred-topology"] = "tier-1"
    elif annotation == "required":
        annotations["kueue.x-k8s.io/podset-required-topology"] = "hostname"
    return yaml.safe_dump(
        {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "spec": {
                "rayClusterSpec": {
                    "headGroupSpec": {
                        "template": {"metadata": {"annotations": annotations}}
                    },
                    "workerGroupSpecs": [],
                }
            },
        }
    )


def _queue(topology_name: str | None) -> tuple[dict, dict, dict]:
    local = {
        "metadata": {"name": "training-lq", "uid": "local-uid"},
        "spec": {"clusterQueue": "training-cq"},
    }
    cluster = {
        "metadata": {"name": "training-cq", "uid": "cluster-uid"},
        "spec": {
            "resourceGroups": [
                {
                    "flavors": [
                        {
                            "name": "b300",
                            "resources": [
                                {"name": "nvidia.com/gpu", "nominalQuota": "8"}
                            ],
                        }
                    ]
                }
            ]
        },
    }
    flavors = {
        "b300": {
            "metadata": {"name": "b300", "uid": "flavor-uid"},
            "spec": {"topologyName": topology_name},
        }
    }
    return local, cluster, flavors


def test_guard_rejects_rendered_request_on_non_topology_flavor() -> None:
    local, cluster, flavors = _queue(None)
    with pytest.raises(RuntimeError, match="without topology support: b300"):
        topology.require_rendered_topology_compatibility(
            _preview(), local, cluster, flavors
        )


def test_guard_accepts_compatible_live_queue() -> None:
    local, cluster, flavors = _queue("ib-topology")
    value = topology.require_rendered_topology_compatibility(
        _preview("required"), local, cluster, flavors
    )
    assert value["all_usable_flavors_topology_aware"] is True
    assert value["rendered_topology_requests"] == [
        {"pod_set": "head", "mode": "required", "level": "hostname"}
    ]


def test_guard_allows_no_rendered_topology_request() -> None:
    local, cluster, flavors = _queue(None)
    value = topology.require_rendered_topology_compatibility(
        _preview(None), local, cluster, flavors
    )
    assert value["rendered_topology_requests"] == []


def test_guard_requires_complete_flavor_evidence() -> None:
    local, cluster, _ = _queue("ib-topology")
    with pytest.raises(ValueError, match="ResourceFlavor evidence is absent"):
        topology.require_rendered_topology_compatibility(_preview(), local, cluster, {})
