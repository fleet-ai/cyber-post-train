"""Exact four-node derivative of the recovered Qwen3.8 262K parent.

This compiler accepts no scientific menu.  It reopens the retained v12 plan,
verifies its canonical digest, changes only the reviewed four-node hypothesis,
and binds the current runtime bytes.  The resulting plan may be previewed but
remains submission-blocked until its zero-GPU preflight and root review exist.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from cyber_post_train.jobs import API_URLS, digest, validate_request

from . import sft
from .sft_262k_runtime import QUALIFICATION, validate_plan

VARIANT = "qwen38_sft_262k_4node_v1"
PARENT_PLAN_SHA256 = "3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690"
PARENT_RUNTIME_SHA256 = "b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3"
IMAGE = sft.IMAGE


def _known(value: dict, names: set[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() - names:
        raise ValueError(f"unknown fields in {label}; this qualification is immutable")


def compile_sft(config: dict, *, relative_to: Path) -> dict:
    _known(
        config,
        {
            "runtime_variant",
            "parent_plan",
            "name",
            "output_root",
            "wandb",
            "cluster",
        },
        "four-node 262K SFT",
    )
    if config.get("runtime_variant") != VARIANT:
        raise ValueError("four-node 262K runtime variant changed")
    parent_path = relative_to / config["parent_plan"]
    parent = sft.read_mapping(parent_path)
    if digest(parent) != PARENT_PLAN_SHA256:
        raise ValueError("recovered v12 parent plan digest changed")
    if (
        parent.get("runtime_sha256") != PARENT_RUNTIME_SHA256
        or parent.get("execution", {}).get("image") != IMAGE
    ):
        raise ValueError("recovered v12 runtime or image changed")
    cluster = config.get("cluster")
    if cluster != {"priority": "c1", "target": "prod"}:
        raise ValueError("four-node 262K candidate requires exact production c1 binding")
    name = config.get("name")
    output_root = config.get("output_root")
    if (
        name != "chris-q38-t3k262-4n-can-v1"
        or output_root != "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1"
    ):
        raise ValueError("four-node 262K create-once run identity changed")
    wandb = config.get("wandb")
    if not isinstance(wandb, dict) or wandb.get("run_id") != name or wandb.get("name") != name:
        raise ValueError("four-node 262K W&B identity changed")

    plan = copy.deepcopy(parent)
    plan.update(
        {
            "runtime_variant": VARIANT,
            "run_name": name,
            "output_root": output_root,
            "runtime_sha256": hashlib.sha256(
                Path(sft.__file__).with_name("sft_262k_runtime.py").read_bytes()
            ).hexdigest(),
            "wandb": copy.deepcopy(wandb),
            "long_context_qualification": copy.deepcopy(QUALIFICATION),
            "qualification": {
                "schema": "qwen38_262k_four_node_submission_gate_v1",
                "submission_gate": {
                    "preview_authorized": True,
                    "preflight_authorized": True,
                    "submission_authorized": False,
                    "blockers": [
                        "zero-GPU preflight receipt absent",
                        "four-node GPU launch has not received root review",
                    ],
                },
            },
        }
    )
    plan["recipe"].update({"nodes": 4, "batch_size": 32, "max_steps": 4})
    plan["execution"].update(
        {
            "priority": "c1",
            "cluster_target": "prod",
            "jobs_api_base_url": API_URLS["prod"],
        }
    )
    validate_plan(plan, check_files=False)
    request = job_request(plan)
    validate_request(request)
    return plan


def job_request(plan: dict) -> dict:
    if plan.get("runtime_variant") != VARIANT:
        raise ValueError("four-node 262K plan selected the wrong compiler")
    validate_plan(plan, check_files=False)
    return sft.job_request(plan)


def preflight(plan: dict) -> dict:
    from .sft_262k_runtime import install_runtime

    validate_plan(plan, check_files=True)
    install_runtime()
    return sft.preflight(plan)
