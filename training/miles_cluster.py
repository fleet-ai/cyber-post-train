"""Exact cluster identities used by Miles evidence and operator rails."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from cyber_post_train.jobs import API_URLS


class ClusterProfile(NamedTuple):
    api_base_url: str
    kube_context: str
    namespace: str
    namespace_uid: str


_PROFILES = {
    "dev": ClusterProfile(
        api_base_url=API_URLS["dev"],
        kube_context="nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
        namespace="fleet-train-jobs",
        namespace_uid="10394b76-e1d4-40b1-a8e2-7575e95df216",
    ),
    "prod": ClusterProfile(
        api_base_url=API_URLS["prod"],
        kube_context="nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
        namespace="fleet-train-jobs",
        namespace_uid="fd6d2fcd-687a-4257-9dba-a034bb381e6b",
    ),
}


def cluster_profile(target: object) -> ClusterProfile:
    """Resolve one immutable cluster identity; never accept caller-supplied URLs."""

    if not isinstance(target, str) or target not in _PROFILES:
        raise ValueError("Miles cluster target must be exactly dev or prod")
    return _PROFILES[target]


def plan_cluster_target(plan: Mapping[str, Any]) -> str:
    """Read the explicit target, retaining dev for pre-target historical plans."""

    execution = plan.get("execution")
    target = execution.get("cluster_target") if isinstance(execution, Mapping) else None
    if target is None:
        return "dev"
    cluster_profile(target)
    return str(target)
