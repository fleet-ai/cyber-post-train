"""Render and validate the HELD GLM5.3 dedicated-serving v6 package.

This module has no submit operation.  It freezes the currently proven runtime,
the current Jobs API request shape, a post-Ready idle watchdog, and a whole-task
serving partition.  Live work remains impossible until a later append-only
release binds accepted generation-5, reconciliation, authenticated preview,
UID-bound parity, and dedicated-canary evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v6-held.json"
MODULE_PATH = "evals/fleet/glm53_dedicated_v6.py"
LIFECYCLE_PATH = "evals/fleet/scripts/glm53_dedicated_v6_lifecycle.sh"
HEARTBEAT_PATH = "evals/fleet/scripts/glm53_dedicated_v6_controller_heartbeat.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_glm53_dedicated_v6.sh"
DOC_PATH = "evals/fleet/DEDICATED_GLM53_V6.md"
HELD_PATH = "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v6-held.json"
SCHEMA = "fleet-glm53-dedicated-serving-v6-held-spec-v1"
HELD_SCHEMA = "fleet-glm53-dedicated-serving-v6-held-receipt-v1"
PREVIEW_SCHEMA = "fleet-glm53-dedicated-serving-v6-authenticated-preview-v1"
DUPLICATE_SCHEMA = "fleet-glm53-dedicated-serving-v6-create-once-gate-v1"
PARITY_SCHEMA = "fleet-glm53-dedicated-serving-v6-parity-gate-v1"
RUNTIME_SCHEMA = "fleet-glm53-dedicated-serving-v6-runtime-gate-v1"
CANARY_SCHEMA = "fleet-glm53-dedicated-serving-v6-cell-accepted-v1"
IMAGE = (
    "ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@"
    "sha256:ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec"
)
MODEL_REVISION = "30333038ada1f1dacb294a93270305a890b50c14"
MODEL_PATH = "/mnt/sfs/models/glm-5.3-30333038"
STAGING_RECEIPT_SHA = "sha256:9a680f714de5c72e60dc6b08f363a8c82658dc1540bbb9aab90a3b8939445869"
TOOL_CATALOG_SHA = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
EXPECTED_API_FIELDS = {
    "image",
    "command",
    "workers",
    "gpus_per_worker",
    "env",
    "secrets",
    "resources",
    "priority_class",
    "privileged",
    "run_dir",
    "title",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON file: {path}")
    value = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a UUID") from exc
    if parsed.int == 0:
        raise ValueError(f"{label} must be a nonzero UUID")
    return str(parsed)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate_spec(value, root)
    return value


def validate_spec(value: dict[str, Any], root: Path) -> None:
    model = value.get("model") or {}
    runtime = value.get("runtime") or {}
    resources = value.get("resources") or {}
    lifecycle = value.get("lifecycle") or {}
    replicas = value.get("replicas") or {}
    gates = value.get("gates") or {}
    privacy = value.get("privacy") or {}
    if (
        value.get("schema_version") != SCHEMA
        or value.get("status") != "HELD"
        or value.get("launch_authorized") is not False
        or model
        != {
            "repository": "zai-org/GLM-5.3",
            "revision": MODEL_REVISION,
            "served_id": "glm-5.3",
            "sfs_path": MODEL_PATH,
            "context_length": 262144,
            "staging_receipt_sha256": STAGING_RECEIPT_SHA,
        }
        or runtime.get("image") != IMAGE
        or runtime.get("base_image")
        != "lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1"
        or runtime.get("ray_version") != "2.56.0"
        or runtime.get("image_pull_secret_required") != "ghcr-pull"
        or set(runtime.get("jobs_api_request_fields") or []) != EXPECTED_API_FIELDS
        or runtime.get("jobs_api_image_pull_field_available") is not False
        or runtime.get("authenticated_preview_must_prove_pull_secret") is not True
        or resources
        != {
            "workers": 1,
            "gpus_per_worker": 8,
            "maximum_nodes": 2,
            "maximum_gpus": 16,
            "cpu_request": "96",
            "cpu_limit": "192",
            "memory_request": "1Ti",
            "memory_limit": "2Ti",
            "privileged": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        }
        or lifecycle
        != {
            "script": LIFECYCLE_PATH,
            "controller_heartbeat_script": HEARTBEAT_PATH,
            "post_ready_idle_seconds": 600,
            "poll_seconds": 5,
            "maximum_streams_per_endpoint": 2,
            "heartbeat_files": [
                "lifecycle/traffic-stream-1",
                "lifecycle/traffic-stream-2",
            ],
            "drain_file": "lifecycle/DRAIN",
            "startup_is_outside_idle_budget": True,
            "release_on_controller_terminal": True,
        }
        or set(replicas) != {"A", "B"}
        or gates
        != {
            "generation5_canary_accepted_required": True,
            "prebulk_reconciliation_accepted_required": True,
            "fresh_exact_cell_inventory_required": True,
            "jobs_api_authenticated_preview_required_per_replica": True,
            "jobs_api_preview_exact_image_pull_required": True,
            "uid_bound_readiness_and_parity_required_per_replica": True,
            "one_scored_dedicated_canary_before_second_stream": True,
            "replica_a_canary_before_replica_b_submit": True,
            "create_once_duplicate_checks_required": True,
        }
        or privacy
        != {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        }
    ):
        raise ValueError("GLM dedicated v6 held specification drifted")
    if not (root / LIFECYCLE_PATH).is_file() or (root / LIFECYCLE_PATH).is_symlink():
        raise ValueError("GLM dedicated v6 lifecycle script is absent or unsafe")
    if not (root / HEARTBEAT_PATH).is_file() or (root / HEARTBEAT_PATH).is_symlink():
        raise ValueError("GLM dedicated v6 heartbeat script is absent or unsafe")
    _validate_replicas(replicas)
    _validate_server_arguments(value.get("server_arguments"))
    _validate_partition(value.get("partition"), root)


def _validate_replicas(replicas: dict[str, Any]) -> None:
    for replica in ("A", "B"):
        lower = replica.lower()
        expected = {
            "title": f"chris-cyber-evalserve-glm53-tp8-{lower}-v6",
            "run_dir": f"/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-{lower}-v6",
            "serving_block": f"glm-dedicated-{lower}-v6",
        }
        if replicas.get(replica) != expected:
            raise ValueError(f"GLM dedicated v6 replica {replica} drifted")


def _validate_server_arguments(arguments: Any) -> None:
    if not isinstance(arguments, list) or not all(isinstance(row, str) for row in arguments):
        raise ValueError("GLM dedicated v6 server arguments are invalid")
    required_subsequences = [
        ["--model-path", MODEL_PATH],
        ["--served-model-name", "glm-5.3"],
        ["--tp-size", "8"],
        ["--dp-size", "8"],
        ["--ep-size", "8"],
        ["--context-length", "262144"],
        ["--quantization", "fp8"],
        ["--kv-cache-dtype", "bfloat16"],
        ["--attention-backend", "dsa"],
        ["--dsa-prefill-backend", "trtllm"],
        ["--dsa-decode-backend", "trtllm"],
        ["--moe-a2a-backend", "deepep"],
        ["--speculative-algorithm", "EAGLE"],
        ["--speculative-draft-model-path", MODEL_PATH],
        ["--reasoning-parser", "glm45"],
        ["--tool-call-parser", "glm47"],
    ]
    if arguments[:3] != ["python3", "-m", "sglang.launch_server"]:
        raise ValueError("GLM dedicated v6 server entrypoint drifted")
    for pair in required_subsequences:
        if not any(arguments[index : index + 2] == pair for index in range(len(arguments) - 1)):
            raise ValueError(f"GLM dedicated v6 argument drifted: {pair[0]}")


def _validate_partition(partition: Any, root: Path) -> None:
    if not isinstance(partition, dict):
        raise ValueError("GLM dedicated v6 partition is absent")
    canary = partition.get("canary")
    if canary != {
        "serving_block": "glm-hosted-autocontinue-v1",
        "selection_rank": 13,
        "attempt": 1,
    }:
        raise ValueError("GLM dedicated v6 canary binding drifted")
    if partition.get("dedicated_canaries") != {
        "A": {"selection_rank": 51, "attempt": 1},
        "B": {"selection_rank": 76, "attempt": 1},
    }:
        raise ValueError("GLM dedicated v6 dedicated canary bindings drifted")
    if (
        partition.get("whole_task_serving_block_required") is not True
        or partition.get("pooling_across_serving_blocks_allowed") is not False
    ):
        raise ValueError("GLM dedicated v6 treatment separation drifted")
    blocks = partition.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError("GLM dedicated v6 block count drifted")

    expected_ranges = [range(1, 26), range(26, 51), range(51, 76), range(76, 101)]
    expected_names = [
        "glm-hosted-autocontinue-v1",
        "glm-hosted-autocontinue-v1",
        "glm-dedicated-a-v6",
        "glm-dedicated-b-v6",
    ]
    rank_owner: dict[int, str] = {}
    stream_names: set[str] = set()
    for index, (block, expected_range, expected_name) in enumerate(
        zip(blocks, expected_ranges, expected_names, strict=True)
    ):
        ranks = set(block.get("full_ranks") or []) | {
            int(rank) for rank in (block.get("partial_attempts") or {})
        }
        if ranks != set(expected_range) or block.get("serving_block") != expected_name:
            raise ValueError("GLM dedicated v6 block rank or treatment drifted")
        if block.get("kind") != ("hosted" if index < 2 else "dedicated"):
            raise ValueError("GLM dedicated v6 block kind drifted")
        streams = block.get("streams")
        if not isinstance(streams, list) or not 1 <= len(streams) <= 2:
            raise ValueError("GLM dedicated v6 stream count drifted")
        streamed: set[int] = set()
        for stream in streams:
            name = stream.get("name")
            stream_ranks = stream.get("ranks")
            if not isinstance(name, str) or name in stream_names:
                raise ValueError("GLM dedicated v6 stream identity drifted")
            if not isinstance(stream_ranks, list) or streamed.intersection(stream_ranks):
                raise ValueError("GLM dedicated v6 stream rank overlap")
            stream_names.add(name)
            streamed.update(stream_ranks)
        if streamed != ranks:
            raise ValueError("GLM dedicated v6 streams do not cover their block")
        for rank in ranks:
            if rank in rank_owner and rank_owner[rank] != expected_name:
                raise ValueError("a task is split across serving treatments")
            rank_owner[rank] = expected_name
    if set(rank_owner) != set(range(1, 101)):
        raise ValueError("GLM dedicated v6 blocks do not cover easiest-100")

    campaign = exact.read_object(root / exact_pass4_campaign_path())
    universe = exact.build_universe(campaign, root)
    expected_cells = {
        (row["selection_rank"], row["attempt"]): row["cell_id"]
        for row in universe["cells"]
        if row["model"] == "glm-5.3"
    }
    assigned = controller_cells(partition, expected_cells)
    if set(assigned) != set(expected_cells) - {(13, 1)}:
        raise ValueError("GLM dedicated v6 partition does not cover universe minus canary")
    if any(assigned[key] != expected_cells[key] for key in assigned):
        raise ValueError("GLM dedicated v6 statistical cell identity drifted")


def exact_pass4_campaign_path() -> str:
    return "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"


def controller_cells(
    partition: dict[str, Any], expected_cells: dict[tuple[int, int], str]
) -> dict[tuple[int, int], str]:
    assigned: dict[tuple[int, int], str] = {}
    for block in partition["blocks"]:
        partial = {int(key): value for key, value in block["partial_attempts"].items()}
        ranks = sorted(set(block["full_ranks"]) | set(partial))
        for rank in ranks:
            attempts = partial.get(rank, [1, 2, 3, 4])
            for attempt in attempts:
                key = (rank, attempt)
                if key in assigned:
                    raise ValueError("GLM dedicated v6 cell overlap")
                assigned[key] = expected_cells[key]
    return assigned


def server_payload(value: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    validate_spec(value, root)
    if replica not in {"A", "B"}:
        raise ValueError("replica must be A or B")
    selected = value["replicas"][replica]
    lifecycle = (root / LIFECYCLE_PATH).read_text()
    command = "bash -lc " + shlex.quote(lifecycle) + " -- " + shlex.join(value["server_arguments"])
    payload = {
        "image": IMAGE,
        "command": command,
        "workers": 1,
        "gpus_per_worker": 8,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NVSHMEM_REMOTE_TRANSPORT": "none",
            "GLM53_RUN_DIR": selected["run_dir"],
            "GLM53_REPLICA": replica,
        },
        "secrets": [],
        "resources": {
            "cpu_request": "96",
            "cpu_limit": "192",
            "memory_request": "1Ti",
            "memory_limit": "2Ti",
        },
        "priority_class": "fleet-serve-low",
        "privileged": True,
        "run_dir": selected["run_dir"],
        "title": selected["title"],
    }
    if set(payload) != EXPECTED_API_FIELDS:
        raise AssertionError("internal Jobs API payload field drift")
    return payload


def controller_specs(value: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    validate_spec(value, root)
    campaign = exact.read_object(root / exact_pass4_campaign_path())
    universe = exact.build_universe(campaign, root)
    cells = {
        (row["selection_rank"], row["attempt"]): row
        for row in universe["cells"]
        if row["model"] == "glm-5.3"
    }
    result: dict[str, dict[str, Any]] = {}
    for block in value["partition"]["blocks"]:
        partial = {int(key): item for key, item in block["partial_attempts"].items()}
        for stream_number, stream in enumerate(block["streams"], 1):
            name = stream["name"]
            rows = [
                {
                    "cell_id": cells[(rank, attempt)]["cell_id"],
                    "selection_rank": rank,
                    "attempt": attempt,
                    "task_key": cells[(rank, attempt)]["task_key"],
                    "task_version_id": cells[(rank, attempt)]["task_version_id"],
                }
                for rank in stream["ranks"]
                for attempt in partial.get(rank, [1, 2, 3, 4])
            ]
            dedicated = block["kind"] == "dedicated"
            result[name] = {
                "model": "glm-5.3",
                "serving_block": block["serving_block"],
                "serving_kind": block["kind"],
                "task_ranks": stream["ranks"],
                "cells": rows,
                "new_session_count": len(rows),
                "maximum_endpoint_streams": 2,
                "heartbeat_relative_path": (
                    f"lifecycle/traffic-stream-{stream_number}" if dedicated else None
                ),
                "uid_bound_runtime_gate_required": dedicated,
                "dedicated_canary_required_before_stream": dedicated and stream_number == 2,
                "launch_authorized": False,
                "treatment": exact.EXPECTED_TREATMENT,
            }
    return result


def dedicated_canary_cell(value: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    if replica not in {"A", "B"}:
        raise ValueError("replica must be A or B")
    expected = value["partition"]["dedicated_canaries"][replica]
    stream = controller_specs(value, root)[f"dedicated-{replica.lower()}-1"]
    matches = [
        cell
        for cell in stream["cells"]
        if cell["selection_rank"] == expected["selection_rank"]
        and cell["attempt"] == expected["attempt"]
    ]
    if len(matches) != 1:
        raise ValueError("GLM dedicated v6 canary is not unique in its first stream")
    return matches[0]


def validate_authenticated_preview(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    selected = value["replicas"][replica]
    expected = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "authenticated": True,
        "route": "/v1/runs/preview",
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "requested_image": IMAGE,
        "rendered_image": IMAGE,
        "rendered_image_pull_secrets": ["ghcr-pull"],
        "rendered_workers": 1,
        "rendered_gpus_per_worker": 8,
        "rendered_priority_class": "fleet-serve-low",
        "rendered_preemption_policy": "Never",
        "rendered_run_dir": selected["run_dir"],
        "post_put_patch_delete": 0,
        "credentials_included": False,
    }
    if {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    } != expected or receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise ValueError("GLM dedicated v6 authenticated preview gate is invalid")


def validate_duplicate_gate(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    selected = value["replicas"][replica]
    controllers = controller_specs(value, root)
    prefix = f"dedicated-{replica.lower()}-"
    cells = [
        cell
        for name, controller in controllers.items()
        if name.startswith(prefix)
        for cell in controller["cells"]
    ]
    prebulk_sha = _sha256(
        receipt.get("prebulk_reconciliation_receipt_sha256"),
        "prebulk_reconciliation_receipt_sha256",
    )
    expected = {
        "schema_version": DUPLICATE_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "title": selected["title"],
        "run_dir": selected["run_dir"],
        "jobs_api_title_matches": 0,
        "kubernetes_label_or_name_matches": 0,
        "sfs_run_dir_exists": False,
        "checked_sources": [
            "fleet_sessions",
            "kubernetes_jobs_and_pods",
            "sfs_attempt_outputs",
            "global_cell_claims",
            "model_started_receipts",
        ],
        "prebulk_reconciliation_receipt_sha256": prebulk_sha,
        "assigned_cell_count": len(cells),
        "assigned_cell_ids_sha256": self_hosted.sha256(
            self_hosted.canonical_json(sorted(cell["cell_id"] for cell in cells))
        ),
        "accepted_active_claimed_or_model_started_cell_collisions": 0,
        "checked_immediately_before_create": True,
        "post_put_patch_delete": 0,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    if {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    } != expected or receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise ValueError("GLM dedicated v6 create-once gate is invalid")


def validate_parity_gate(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    selected = value["replicas"][replica]
    for field in ("api_run_uid", "ray_job_uid", "ray_cluster_uid", "head_pod_uid", "service_uid"):
        _uuid(receipt.get(field), field)
    if (
        receipt.get("schema_version") != PARITY_SCHEMA
        or receipt.get("status") != "PASSED"
        or receipt.get("replica") != replica
        or receipt.get("request_sha256") != self_hosted.sha256(self_hosted.canonical_json(payload))
        or receipt.get("run_dir") != selected["run_dir"]
        or receipt.get("image") != IMAGE
        or receipt.get("resolved_image_id") != IMAGE
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("model_path") != MODEL_PATH
        or receipt.get("served_id") != "glm-5.3"
        or receipt.get("context_length") != 262144
        or receipt.get("priority_class") != "fleet-serve-low"
        or receipt.get("preemption_policy") != "Never"
        or receipt.get("gpus") != 8
        or receipt.get("pod_restarts") != 0
        or receipt.get("health_http_200") is not True
        or receipt.get("service_port") != 8000
        or receipt.get("service_health_http_200") is not True
        or receipt.get("server_arguments_sha256")
        != self_hosted.sha256(" ".join(value["server_arguments"]).encode())
        or receipt.get("rendered_command_sha256") != self_hosted.sha256(payload["command"].encode())
        or receipt.get("structured_tools") != ["bash", "submit_report"]
        or receipt.get("tool_catalog_sha256") != TOOL_CATALOG_SHA
        or receipt.get("content_blind_probe") is not True
        or receipt.get("non_scored_model_requests") != 1
        or receipt.get("scored_sessions") != 0
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("GLM dedicated v6 UID-bound parity gate is invalid")


def validate_canary_acceptance(
    receipt: dict[str, Any],
    parity_receipt: dict[str, Any],
    value: dict[str, Any],
    root: Path,
    replica: str,
) -> None:
    validate_parity_gate(parity_receipt, value, root, replica)
    cell = dedicated_canary_cell(value, root, replica)
    session_id = _uuid(receipt.get("session_id"), "session_id")
    verifier_id = _uuid(receipt.get("verifier_execution_id"), "verifier_execution_id")
    claim_sha = _sha256(receipt.get("claim_receipt_sha256"), "claim_receipt_sha256")
    run_id = receipt.get("run_id")
    agent_exit_code = receipt.get("agent_exit_code")
    if not isinstance(run_id, str) or not run_id.startswith("chris-cyber-"):
        raise ValueError("GLM dedicated v6 scored canary run_id is invalid")
    if type(agent_exit_code) is not int:
        raise ValueError("GLM dedicated v6 scored canary exit code is invalid")
    expected = {
        "schema_version": CANARY_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "replica": replica,
        "serving_block": value["replicas"][replica]["serving_block"],
        "cell_id": cell["cell_id"],
        "selection_rank": cell["selection_rank"],
        "attempt": cell["attempt"],
        "task_key": cell["task_key"],
        "task_version_id": cell["task_version_id"],
        "run_id": run_id,
        "session_id": session_id,
        "verifier_execution_id": verifier_id,
        "claim_receipt_sha256": claim_sha,
        "session_model": exact.EXPECTED_MODELS["glm-5.3"]["session_model"],
        "model_revision": MODEL_REVISION,
        "authoritative_metadata_run_id": run_id,
        "api_run_uid": parity_receipt["api_run_uid"],
        "ray_job_uid": parity_receipt["ray_job_uid"],
        "ray_cluster_uid": parity_receipt["ray_cluster_uid"],
        "head_pod_uid": parity_receipt["head_pod_uid"],
        "service_uid": parity_receipt["service_uid"],
        "authoritative_session_matches": 1,
        "agent_exit_code": agent_exit_code,
        "agent_process_exit_success": agent_exit_code == 0,
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "scores_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    if {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    } != expected or receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise ValueError("GLM dedicated v6 scored canary acceptance is invalid")


def validate_runtime_gate(
    receipt: dict[str, Any],
    parity_receipt: dict[str, Any],
    canary_receipt: dict[str, Any],
    value: dict[str, Any],
    root: Path,
    replica: str,
) -> None:
    validate_parity_gate(parity_receipt, value, root, replica)
    validate_canary_acceptance(canary_receipt, parity_receipt, value, root, replica)
    uid_fields = (
        "api_run_uid",
        "ray_job_uid",
        "ray_cluster_uid",
        "head_pod_uid",
        "service_uid",
    )
    for field in uid_fields:
        _uuid(receipt.get(field), field)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "parity_receipt_sha256": parity_receipt["receipt_sha256"],
        **{field: parity_receipt[field] for field in uid_fields},
        "scored_canary_accepted": True,
        "scored_canary_cell_id": canary_receipt["cell_id"],
        "scored_canary_session_id": canary_receipt["session_id"],
        "scored_canary_verifier_execution_id": canary_receipt["verifier_execution_id"],
        "scored_canary_receipt_sha256": canary_receipt["receipt_sha256"],
        "maximum_streams_released": 2,
        "prompts_traces_flags_or_scores_included": False,
    }
    if {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    } != expected or receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise ValueError("GLM dedicated v6 UID-bound runtime gate is invalid")


def held_preview(root: Path) -> dict[str, Any]:
    value = spec(root)
    controllers = controller_specs(value, root)
    payloads = {replica: server_payload(value, root, replica) for replica in ("A", "B")}
    return {
        "status": "HELD",
        "launch_authorized": False,
        "server_request_sha256": {
            replica: self_hosted.sha256(self_hosted.canonical_json(payload))
            for replica, payload in payloads.items()
        },
        "controller_session_counts": {
            key: item["new_session_count"] for key, item in controllers.items()
        },
        "total_remaining_glm_cells": sum(
            item["new_session_count"] for item in controllers.values()
        ),
        "dedicated_nodes": 2,
        "dedicated_gpus": 16,
        "maximum_streams_per_endpoint": 2,
        "authenticated_preview_required": True,
        "uid_bound_parity_and_canary_required": True,
        "post_ready_idle_seconds": 600,
        "no_objects_created": True,
    }


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    value = spec(root)
    preview = held_preview(root)
    expected = {
        "schema_version": HELD_SCHEMA,
        "status": "HELD",
        "launch_authorized": False,
        "spec_path": SPEC_PATH,
        "spec_file_sha256": file_sha256(root / SPEC_PATH),
        "authority_module_path": MODULE_PATH,
        "authority_module_file_sha256": file_sha256(root / MODULE_PATH),
        "lifecycle_path": LIFECYCLE_PATH,
        "lifecycle_file_sha256": file_sha256(root / LIFECYCLE_PATH),
        "controller_heartbeat_path": HEARTBEAT_PATH,
        "controller_heartbeat_file_sha256": file_sha256(root / HEARTBEAT_PATH),
        "submit_path": SUBMIT_PATH,
        "submit_file_sha256": file_sha256(root / SUBMIT_PATH),
        "documentation_path": DOC_PATH,
        "documentation_file_sha256": file_sha256(root / DOC_PATH),
        "staging_receipt_sha256": STAGING_RECEIPT_SHA,
        "server_request_sha256": preview["server_request_sha256"],
        "controller_session_counts": preview["controller_session_counts"],
        "total_remaining_glm_cells": 399,
        "release_blockers": [
            "generation5_glm_canary_must_be_accepted",
            "fresh_prebulk_reconciliation_must_accept_exact_unstarted_cells",
            "authenticated_jobs_api_preview_must_render_ghcr_pull_for_each_replica",
            "replica_a_uid_bound_parity_and_scored_canary_before_second_stream_or_replica_b",
            "replica_b_uid_bound_parity_and_scored_canary_before_its_second_stream",
        ],
        "treatment_policy": {
            "whole_task_serving_blocks": True,
            "hosted_and_dedicated_results_pooled": False,
            "maximum_streams_per_endpoint": 2,
        },
        "resource_policy": {
            "maximum_nodes": 2,
            "maximum_gpus": 16,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "post_ready_no_heartbeat_shutdown_seconds": 600,
        },
        "privacy": value["privacy"],
    }
    if {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    } != expected or receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise ValueError("GLM dedicated v6 held receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("held-preview", "server-preview", "validate"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--replica", choices=("A", "B"))
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.command == "validate":
        validate_held(load(root / HELD_PATH), root)
        print(json.dumps({"status": "HELD", "valid": True}, sort_keys=True))
    elif args.command == "server-preview":
        if args.replica is None:
            parser.error("server-preview requires --replica A or B")
        print(json.dumps(server_payload(spec(root), root, args.replica), sort_keys=True))
    else:
        print(json.dumps(held_preview(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
