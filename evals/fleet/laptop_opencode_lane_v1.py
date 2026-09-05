"""Fail-closed qualification and partitioning for a laptop OpenCode lane.

This module performs no scored work.  It binds one complete pass@4 task,
checks the immutable local harness, and emits a sanitized qualification receipt.
Authenticated network qualification is deliberately separate so an absent key
cannot accidentally turn a local inspection into a task or scoring mutation.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import hosted_behavioral_preflight, self_hosted

SCHEMA = "fleet-laptop-opencode-qualification-v1"
PLAN_SCHEMA = "fleet-laptop-opencode-held-task-v1"
CAMPAIGN_PATH = Path(
    "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
)
DOCKERFILE_PATH = Path("evals/fleet/Dockerfile.opencode")
IMAGE = "chris/opencode:1.18.27-cyber-v1"
IMAGE_ID = "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb"
OPENCODE_BINARY_SHA256 = (
    "bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
)
NODE_BASE_DIGEST = (
    "sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96"
)
RELEASE_ASSET_SHA256 = exact.EXPECTED_TREATMENT["release_asset_sha256"]
RESERVED_MODEL = "qwen3.8-27b"
RESERVED_RANK = 3
EXPECTED_TASK = {
    "task_key": "cysec1-2-current-gen_blackbox-607e28ace466c1f71ece3d32__blackbox_ctf_v1",
    "task_version_id": "33d37078-0669-478e-af39-43cd245f0da8",
    "environment_version_id": "d70c4fe9-70c5-4020-91b1-a23d886a1e22",
    "env_key": "cysec1-2-current-gen",
    "env_version": "v0.0.3",
}
EXPECTED_CELLS = [
    "sha256:f728f5712478f233b05c0f2acfd1aba5ae9f63862e1b759ea40a508e188557d8",
    "sha256:6edac950ae44ff62c07074a775afdb65cc9b394047883e0b57fcad84e74d2fa7",
    "sha256:ffb5b0fb6a3eb702b78c75aca66446eb39668d0949bc92b4628ad77c2102bcae",
    "sha256:47e1093f8b03d4179d334a4feac13aba526c5c36f8eb98bc29a681d27aa68e36",
]


class QualificationError(RuntimeError):
    """A content-free, stable local qualification failure."""


Runner = Callable[[Sequence[str]], str]
ModelProbe = Callable[[str, str], dict[str, Any]]


def _run(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _qualification_config() -> dict[str, Any]:
    treatment = exact.EXPECTED_TREATMENT
    return {
        "harness": {
            "name": treatment["harness"],
            "version": treatment["harness_version"],
            "release_asset_sha256": treatment["release_asset_sha256"],
            "provider_adapter": treatment["provider_adapter"],
            "context_management": treatment["context_management"],
            "context_window_size": treatment["context_window_size"],
            "compaction_headroom_tokens": treatment["compaction_headroom_tokens"],
            "max_output_tokens": treatment["max_output_tokens"],
            "max_model_requests": treatment["max_model_requests"],
            "timeout_seconds": treatment["timeout_seconds"],
        },
        "model": exact.EXPECTED_MODELS[RESERVED_MODEL],
    }


def held_plan(repo_root: Path) -> dict[str, Any]:
    """Bind rank 3 as one indivisible, sequential laptop task."""
    campaign = exact.read_object(repo_root / CAMPAIGN_PATH)
    universe = exact.build_universe(campaign, repo_root)
    selected_task = exact.validate_campaign(campaign, repo_root)[RESERVED_RANK - 1]
    if selected_task.get("rank") != RESERVED_RANK or any(
        selected_task.get(key) != value for key, value in EXPECTED_TASK.items()
    ):
        raise QualificationError("reserved_task_identity_drift")
    rows = [
        row
        for row in universe["cells"]
        if row["model"] == RESERVED_MODEL and row["selection_rank"] == RESERVED_RANK
    ]
    rows.sort(key=lambda row: row["attempt"])
    if [row["attempt"] for row in rows] != [1, 2, 3, 4]:
        raise QualificationError("reserved_task_is_not_complete_pass4")
    if [row["cell_id"] for row in rows] != EXPECTED_CELLS:
        raise QualificationError("reserved_cell_identity_drift")
    if any(
        row.get(key) != value
        for row in rows
        for key, value in EXPECTED_TASK.items()
        if key in {"task_key", "task_version_id"}
    ):
        raise QualificationError("reserved_task_identity_drift")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "universe_sha256": universe["universe_sha256"],
        "model": exact.EXPECTED_MODELS[RESERVED_MODEL],
        "selection_rank": RESERVED_RANK,
        "task": EXPECTED_TASK,
        "cells": [
            {"attempt": row["attempt"], "cell_id": row["cell_id"]} for row in rows
        ],
        "execution": {
            "scored_launch_authorized": False,
            "whole_task_partition": True,
            "attempt_order": [1, 2, 3, 4],
            "maximum_concurrent_attempts": 1,
            "automatic_retry": False,
        },
        "required_before_first_scored_create": [
            "digest_valid_laptop_qualification_receipt",
            "immutable_hosted_successor_exclusion_for_all_four_cells",
            "fresh_global_ledger_all_four_cells_unaccepted_unclaimed_inactive",
            "fresh_authoritative_session_inventory_has_no_matching_session",
            "globally_visible_create_once_claim_or_equivalent_atomic_reservation",
        ],
        "glm_hosted_laptop_lane": {
            "authorized": False,
            "reason": "known_invalid_hosted_submit_report_route",
            "location_change_repairs_model_tool_behavior": False,
        },
        "privacy": {
            "prompts_traces_flags_scores_or_model_outputs_included": False,
            "credentials_included": False,
        },
    }
    plan["plan_sha256"] = crypto.digest_without(plan, "plan_sha256")
    return plan


def qualify_local(repo_root: Path, *, runner: Runner = _run) -> dict[str, Any]:
    """Inspect the laptop and exact image without contacting Fleet APIs."""
    docker_version = json.loads(runner(["docker", "version", "--format", "{{json .}}"])).copy()
    image = json.loads(
        runner(["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"])
    )
    version = runner(
        ["docker", "run", "--rm", "--platform", "linux/amd64", IMAGE, "opencode", "--version"]
    )
    binary_line = runner(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64", IMAGE,
            "sh", "-c", "sha256sum /usr/local/bin/opencode",
        ]
    )
    dockerfile = (repo_root / DOCKERFILE_PATH).read_text()
    settings = self_hosted.opencode_settings(_qualification_config())
    limits = settings["provider"]["fleet-cluster"]["models"]["qwen3.8-27b"]["limit"]
    checks = {
        "docker_server_available": (docker_version.get("Server") or {}).get("Os") == "linux",
        "docker_server_architecture_known": (docker_version.get("Server") or {}).get("Arch")
        in {"arm64", "amd64"},
        "exact_image_id": image.get("Id") == IMAGE_ID,
        "image_linux_amd64": (image.get("Os"), image.get("Architecture")) == ("linux", "amd64"),
        "image_nonroot_node": (image.get("Config") or {}).get("User") == "node",
        "image_workdir": (image.get("Config") or {}).get("WorkingDir") == "/workspace",
        "emulated_release_version": version == "1.18.27",
        "runtime_binary_sha256": binary_line.split(maxsplit=1)[0] == OPENCODE_BINARY_SHA256,
        "dockerfile_base_digest": NODE_BASE_DIGEST in dockerfile,
        "dockerfile_release_asset_sha256": RELEASE_ASSET_SHA256.removeprefix("sha256:")
        in dockerfile,
        "native_compaction_autocontinue": settings.get("compaction")
        == {"auto": True, "reserved": 20000},
        "exact_context_limits": limits
        == {"context": 262144, "input": 229376, "output": 32768},
        "exact_mcp_permission": settings.get("permission") == {"*": "deny", "fleet_*": "allow"},
        "builtin_tools_disabled": all(value is False for value in settings["tools"].values()),
        "exact_campaign_tools": exact.EXPECTED_TREATMENT["tools"] == ["bash", "submit_report"],
    }
    if not all(checks.values()):
        raise QualificationError("local_harness_qualification_failed")
    credential_present = bool(os.environ.get("FLEET_API_KEY"))
    receipt = {
        "schema_version": SCHEMA,
        "status": "LOCAL_PASSED_NETWORK_BLOCKED" if not credential_present else "LOCAL_PASSED",
        "scored_launch_authorized": False,
        "local_checks": checks,
        "runtime": {
            "docker_client_os": (docker_version.get("Client") or {}).get("Os"),
            "docker_client_arch": (docker_version.get("Client") or {}).get("Arch"),
            "docker_server_os": (docker_version.get("Server") or {}).get("Os"),
            "docker_server_arch": (docker_version.get("Server") or {}).get("Arch"),
            "image": IMAGE,
            "image_id": IMAGE_ID,
            "image_arch": "amd64",
            "opencode_version": version,
            "opencode_binary_sha256": "sha256:" + OPENCODE_BINARY_SHA256,
        },
        "network_qualification": {
            "credential_present": credential_present,
            "fleet_team_verified": False,
            "authoritative_routes_verified_get_only": False,
            "qwen_served_id_and_structured_tools_verified": False,
            "requests_made": 0,
            "mutations_made": 0,
        },
        "held_plan": held_plan(repo_root),
        "privacy": {
            "prompts_traces_flags_scores_or_model_outputs_included": False,
            "credentials_included": False,
            "response_content_persisted": False,
        },
    }
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt


def qualify_network(
    repo_root: Path,
    api_key: str,
    *,
    client: Any,
    model_probe: ModelProbe | None = None,
) -> dict[str, Any]:
    """Run only account, route, and generic hosted-model parity checks.

    The client must already carry the Fleet bearer header.  No task instance,
    session, verifier, or scoring call is made.  The two model requests are tiny
    synthetic forced-tool probes and persist no response content.
    """
    if not api_key:
        raise QualificationError("fleet_credential_absent")
    plan = held_plan(repo_root)
    account = self_hosted._request(client, "GET", "/v1/account")
    if account != {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}:
        raise QualificationError("fleet_team_identity_mismatch")
    config = {
        "authority": {
            "provisioning_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
            ),
            "scoring_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}"
            ),
        },
        "task": {
            "key": EXPECTED_TASK["task_key"],
            "version_id": EXPECTED_TASK["task_version_id"],
        },
    }
    routes = self_hosted.assert_authoritative_routes_deployed(client, config)
    if model_probe is None:
        model_probe = hosted_behavioral_preflight.probe
    behavior = model_probe("qwen3.8-27b", api_key)
    if behavior.get("passed") is not True:
        raise QualificationError("qwen_hosted_structured_tool_parity_failed")
    observed_tools = [row.get("tool") for row in behavior.get("observations") or []]
    if observed_tools != ["bash", "submit_report"]:
        raise QualificationError("qwen_hosted_tool_probe_set_drifted")
    receipt = {
        "schema_version": SCHEMA,
        "status": "NETWORK_PASSED_LAUNCH_STILL_HELD",
        "scored_launch_authorized": False,
        "fleet_account": account,
        "authoritative_routes": routes,
        "qwen_hosted_behavior": {
            "served_id": behavior.get("served_id"),
            "exact_served_id_advertised": behavior.get("exact_served_id_advertised"),
            "tools": observed_tools,
            "structured_tools_available": True,
            "source_receipt_sha256": behavior.get("receipt_sha256"),
            "response_content_persisted": False,
        },
        "held_plan_sha256": plan["plan_sha256"],
        "request_policy": {
            "task_instance_session_verifier_or_scoring_mutations": 0,
            "benchmark_requests": 0,
            "generic_non_benchmark_model_requests": 2,
        },
        "privacy": {
            "prompts_traces_flags_scores_or_model_outputs_included": False,
            "credentials_included": False,
            "response_content_persisted": False,
        },
    }
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt
