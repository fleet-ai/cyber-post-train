"""Fail-closed policy checks for Nebius training workload manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

EXPECTED_NAMESPACE = "fleet-train-jobs"
EXPECTED_QUEUE = "training-lq"
EXPECTED_CLUSTER_QUEUE = "training-cq"
EXPECTED_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAME_PATTERN = re.compile(r"^chris-cyber-[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
MODEL_SLUG = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
OWNER_LABEL = "cyber-post-train.fleet.ai/owner"
EXPERIMENT_LABEL = "cyber-post-train.fleet.ai/experiment"
MODEL_LABEL = "cyber-post-train.fleet.ai/model"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pod_specs(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    spec = _mapping(manifest.get("spec"))
    kind = manifest.get("kind")
    if kind == "Job":
        return [_mapping(_mapping(_mapping(spec.get("template")).get("spec")))]
    if kind != "RayJob":
        return []
    cluster = _mapping(spec.get("rayClusterSpec"))
    head = _mapping(_mapping(_mapping(cluster.get("headGroupSpec")).get("template")).get("spec"))
    workers = []
    for group in cluster.get("workerGroupSpecs") or []:
        group = _mapping(group)
        workers.append(_mapping(_mapping(group.get("template")).get("spec")))
    return [head, *workers]


def validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject workloads that could bypass the queue or impersonate priority."""
    errors: list[str] = []
    kind = manifest.get("kind")
    if kind not in {"Job", "RayJob"}:
        errors.append("kind must be Job or RayJob")
    metadata = _mapping(manifest.get("metadata"))
    name = metadata.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name) or len(name) > 63:
        errors.append("metadata.name must match chris-cyber-* and fit DNS-63")
    if metadata.get("namespace") != EXPECTED_NAMESPACE:
        errors.append(f"metadata.namespace must be {EXPECTED_NAMESPACE}")
    labels = _mapping(metadata.get("labels"))
    if labels.get("kueue.x-k8s.io/queue-name") != EXPECTED_QUEUE:
        errors.append(f"workload must use Kueue local queue {EXPECTED_QUEUE}")
    if labels.get(OWNER_LABEL) != "chris":
        errors.append(f"metadata.labels[{OWNER_LABEL}] must be chris")
    if not labels.get(EXPERIMENT_LABEL):
        errors.append(f"metadata.labels[{EXPERIMENT_LABEL}] is required")
    model = labels.get(MODEL_LABEL)
    if model is not None:
        if not isinstance(model, str) or not MODEL_SLUG.fullmatch(model):
            errors.append(f"metadata.labels[{MODEL_LABEL}] must be a DNS label")
        elif isinstance(name, str) and not name.startswith(f"chris-cyber-{model}-"):
            errors.append(f"metadata.name must agree with metadata.labels[{MODEL_LABEL}]")
    for index, pod_spec in enumerate(_pod_specs(manifest)):
        if pod_spec.get("priorityClassName"):
            errors.append(f"pod spec {index} must not set priorityClassName")
        if pod_spec.get("preemptionPolicy") not in {None, "Never"}:
            errors.append(f"pod spec {index} preemptionPolicy must be Never or unset")
        if pod_spec.get("schedulerName") not in {None, "default-scheduler"}:
            errors.append(f"pod spec {index} must not bypass the default scheduler")
    if not _pod_specs(manifest):
        errors.append("workload contains no recognized pod specs")
    if errors:
        raise ValueError("invalid Nebius training workload: " + "; ".join(errors))


def validate_scheduler_state(
    cluster_queue: Mapping[str, Any], local_queue: Mapping[str, Any]
) -> None:
    """Prove that the selected queue cannot preempt equal-priority workloads."""
    errors: list[str] = []
    cluster_metadata = _mapping(cluster_queue.get("metadata"))
    cluster_spec = _mapping(cluster_queue.get("spec"))
    local_metadata = _mapping(local_queue.get("metadata"))
    local_spec = _mapping(local_queue.get("spec"))
    if cluster_metadata.get("name") != EXPECTED_CLUSTER_QUEUE:
        errors.append(f"cluster queue must be {EXPECTED_CLUSTER_QUEUE}")
    if local_metadata.get("name") != EXPECTED_QUEUE:
        errors.append(f"local queue must be {EXPECTED_QUEUE}")
    if local_spec.get("clusterQueue") != EXPECTED_CLUSTER_QUEUE:
        errors.append("local queue points to an unexpected cluster queue")
    if local_spec.get("stopPolicy") not in {None, "None"}:
        errors.append("local queue is stopped")
    preemption = _mapping(cluster_spec.get("preemption"))
    if preemption.get("withinClusterQueue") != "LowerPriority":
        errors.append("within-queue preemption must be LowerPriority-only")
    if preemption.get("reclaimWithinCohort") not in {None, "Never"}:
        errors.append("cohort reclaim must be disabled")
    borrow = _mapping(preemption.get("borrowWithinCohort"))
    if borrow.get("policy") not in {None, "Never"}:
        errors.append("cohort borrow preemption must be disabled")
    if errors:
        raise ValueError("unsafe or unexpected Kueue state: " + "; ".join(errors))
