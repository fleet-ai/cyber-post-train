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
                    "headGroupSpec": {"template": {"metadata": {"annotations": annotations}}},
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
                            "resources": [{"name": "nvidia.com/gpu", "nominalQuota": "8"}],
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
        topology.require_rendered_topology_compatibility(_preview(), local, cluster, flavors)


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


def _live_shape(
    *,
    mode: str = "required",
    level: str = "topology.nebius.com/tier-1",
    workers: int = 1,
) -> str:
    key = f"kueue.x-k8s.io/podset-{mode}-topology"
    cluster = {
        "headGroupSpec": {
            "template": {
                "metadata": {"annotations": {key: level}},
                "spec": {
                    "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
                    "containers": [
                        {
                            "resources": {
                                "requests": {
                                    "cpu": "96",
                                    "memory": "1Ti",
                                    "nvidia.com/gpu": 8,
                                }
                            }
                        }
                    ],
                },
            }
        },
        "workerGroupSpecs": [],
    }
    if workers == 2:
        cluster["workerGroupSpecs"] = [
            {
                "template": {
                    "metadata": {"annotations": {key: level}},
                    "spec": {
                        "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
                        "containers": [
                            {
                                "resources": {
                                    "requests": {
                                        "cpu": "96",
                                        "memory": "1Ti",
                                        "nvidia.com/gpu": 8,
                                    }
                                }
                            }
                        ],
                    },
                }
            }
        ]
    return yaml.safe_dump(
        {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "training-lq"}},
            "spec": {"rayClusterSpec": cluster},
        }
    )


def _live_queue(*, reserved_gpus: int = 104) -> tuple[dict, dict, dict, dict]:
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
                            "name": "b300-training",
                            "resources": [
                                {"name": "nvidia.com/gpu", "nominalQuota": "128"},
                                {"name": "cpu", "nominalQuota": "2944"},
                                {"name": "memory", "nominalQuota": "40Ti"},
                            ],
                        },
                        {
                            "name": "b300-unconstrained",
                            "resources": [
                                {"name": "nvidia.com/gpu", "nominalQuota": "64"},
                                {"name": "cpu", "nominalQuota": "1472"},
                                {"name": "memory", "nominalQuota": "20Ti"},
                            ],
                        },
                        {
                            "name": "cpu-head",
                            "resources": [
                                {"name": "nvidia.com/gpu", "nominalQuota": "0"},
                                {"name": "cpu", "nominalQuota": "36"},
                                {"name": "memory", "nominalQuota": "120Gi"},
                            ],
                        },
                    ]
                }
            ]
        },
        "status": {
            "flavorsReservation": [
                {
                    "name": "b300-training",
                    "resources": [
                        {"name": "nvidia.com/gpu", "total": str(reserved_gpus)},
                        {"name": "cpu", "total": "1480"},
                        {"name": "memory", "total": "22720Gi"},
                    ],
                },
                {
                    "name": "b300-unconstrained",
                    "resources": [
                        {"name": "nvidia.com/gpu", "total": "0"},
                        {"name": "cpu", "total": "0"},
                        {"name": "memory", "total": "0"},
                    ],
                },
                {
                    "name": "cpu-head",
                    "resources": [
                        {"name": "nvidia.com/gpu", "total": "0"},
                        {"name": "cpu", "total": "18200m"},
                        {"name": "memory", "total": "71168Mi"},
                    ],
                },
            ]
        },
    }
    flavors = {
        "b300-training": {
            "metadata": {"name": "b300-training", "uid": "training-flavor-uid"},
            "spec": {
                "nodeLabels": {"workload": "fleetai-training-ng-gpu"},
                "topologyName": "ib-topology",
            },
        },
        "b300-unconstrained": {
            "metadata": {"name": "b300-unconstrained", "uid": "unconstrained-uid"},
            "spec": {
                "nodeLabels": {"workload": "fleetai-training-ng-gpu"},
                "topologyName": None,
            },
        },
        "cpu-head": {
            "metadata": {"name": "cpu-head", "uid": "cpu-flavor-uid"},
            "spec": {"nodeLabels": {"workload": "fleetai-training-ng-cpu"}},
        },
    }
    topologies = {
        "ib-topology": {
            "metadata": {"name": "ib-topology", "uid": "topology-uid"},
            "spec": {
                "levels": [
                    {"nodeLabel": "topology.nebius.com/gpu-cluster-id"},
                    {"nodeLabel": "topology.nebius.com/tier-2"},
                    {"nodeLabel": "topology.nebius.com/tier-1"},
                    {"nodeLabel": "kubernetes.io/hostname"},
                ]
            },
        }
    }
    return local, cluster, flavors, topologies


def test_exact_candidate_uses_only_in_quota_topology_flavor() -> None:
    local, cluster, flavors, topologies = _live_queue()
    value = topology.require_exact_topology_candidate(
        _live_shape(),
        local,
        cluster,
        flavors,
        topologies,
        expected_queue="training-lq",
        expected_level="topology.nebius.com/tier-1",
        expected_podsets=("head",),
    )
    assert [row["name"] for row in value["eligible_flavors"]] == ["b300-training"]
    assert value["aggregate_requests"]["nvidia.com/gpu"] == "8"
    assert value["must_recheck_immediately_before_submit"] is True


def test_exact_candidate_rejects_preferred_topology() -> None:
    local, cluster, flavors, topologies = _live_queue()
    with pytest.raises(ValueError, match="must require the exact topology"):
        topology.require_exact_topology_candidate(
            _live_shape(mode="preferred"),
            local,
            cluster,
            flavors,
            topologies,
            expected_queue="training-lq",
            expected_level="topology.nebius.com/tier-1",
            expected_podsets=("head",),
        )


def test_exact_candidate_rejects_when_topology_flavor_has_no_free_gpu_quota() -> None:
    local, cluster, flavors, topologies = _live_queue(reserved_gpus=128)
    with pytest.raises(RuntimeError, match="no topology-aware flavor"):
        topology.require_exact_topology_candidate(
            _live_shape(),
            local,
            cluster,
            flavors,
            topologies,
            expected_queue="training-lq",
            expected_level="topology.nebius.com/tier-1",
            expected_podsets=("head",),
        )


def test_two_node_shape_needs_double_quota_and_is_a_different_runtime_contract() -> None:
    local, cluster, flavors, topologies = _live_queue(reserved_gpus=116)
    with pytest.raises(RuntimeError, match="no topology-aware flavor"):
        topology.require_exact_topology_candidate(
            _live_shape(workers=2),
            local,
            cluster,
            flavors,
            topologies,
            expected_queue="training-lq",
            expected_level="topology.nebius.com/tier-1",
            expected_podsets=("head", "worker-0"),
        )
