"""Inert contract for the Qwen3.8 Miles 32-GPU startup qualification.

Acceptance is deliberately unavailable. A future implementation must render
the exact startup command and collect raw Jobs API and Kubernetes evidence;
self-digested summaries are not an independent evidence source.
"""

from __future__ import annotations

from pathlib import Path

PLAN_SCHEMA = "cyber_miles_distributed_startup_qualification_plan_v2"
RECEIPT_SCHEMA = "cyber_miles_distributed_startup_qualification_v2"
GATE_VERSION = 2
RUNTIME_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "dc1a41ac386c9f7377e7a6f7b92a402830e308855aa2632413af25917f38dd93"
)
EXPECTED_COUNTERS = {
    "rollouts": 0,
    "reward_calls": 0,
    "optimizer_updates": 0,
    "checkpoint_writes": 0,
}
CONTRACT = {
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "native_profile": "qwen3.8-27b-256k",
    "runtime_image": RUNTIME_IMAGE,
    "miles_commit": "9e178ca16839b0600155f3927f57ce0670b8f453",
    "megatron_commit": "e8f574511105db3b58a61f90da178101d8ab2d45",
    "megatron_optimizer_sha256": (
        "sha256:2bad8558ca6b818406186fd9df6b052bb77ae82c9336d340e1c6e3ed71d9e66a"
    ),
    "distributed_backend": "cpu:gloo,cuda:nccl",
    "nodes": 4,
    "gpus_per_node": 8,
    "world_size": 32,
    "tensor_parallel_size": 8,
    "pipeline_parallel_size": 1,
    "context_parallel_size": 4,
    "context_tokens": 262144,
    "response_tokens": 245760,
}
EXECUTION = {
    "cluster": "prod",
    "priority_class": "c1",
    "queue_priority": "q1",
    "active_deadline_seconds": 3600,
    "requeue_if_preempted": False,
    "create_once": True,
}
DISABLED_REASON = (
    "Miles startup acceptance is disabled until an exact renderer and independent "
    "raw Jobs API/Kubernetes evidence collector are implemented"
)


def _same_json_type(left: object, right: object) -> bool:
    """Compare JSON-like values without Python's bool/int or int/float aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_type(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_type(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _template() -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "gate_version": GATE_VERSION,
        "status": "not_run",
        "launchable": False,
        "contract": CONTRACT,
        "bindings": {
            "runtime_source_sha256": None,
            "checkpoint_sha256": None,
            "output_root": None,
        },
        "execution": EXECUTION,
        "expected_counters": EXPECTED_COUNTERS,
        "required_external_observations": {
            "controller": None,
            "counters": None,
            "release": None,
        },
        "receipt_schema": RECEIPT_SCHEMA,
        "successor_reward_canary_allowed": False,
    }


def validate_template(value: dict) -> dict:
    """Accept only the exact, strictly typed, non-launchable template."""
    if not _same_json_type(value, _template()):
        raise ValueError("startup template is not the exact inert contract")
    return value


def validate_plan(value: dict) -> dict:
    """Reject concrete plans until the independent evidence rail exists."""
    raise ValueError(DISABLED_REASON)


def validate_receipt(receipt: dict, *, plan: dict, request: dict) -> dict:
    """Reject all purported acceptances; no trusted receipt schema exists yet."""
    raise ValueError(DISABLED_REASON)


def seal_receipt(
    *,
    plan: dict,
    request: dict,
    controller: dict,
    counters: dict,
    release: dict,
    output: Path,
) -> dict:
    """Never create an acceptance from caller-supplied observation summaries."""
    raise ValueError(DISABLED_REASON)
