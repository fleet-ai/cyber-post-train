"""Fail-closed authority for the HELD GLM5.3 dedicated-serving v7 path.

The module deliberately has no network or submit operation.  It binds the
generation-7 corrected protocol to a whole-task hosted/dedicated partition and
provides validators for the evidence that a later append-only release must
collect before either GPU replica can be created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation7_authority_v1 as g7_authority
from evals.fleet import autocontinue_generation7_canary as g7
from evals.fleet import exact_pass4_bulk_v3 as bulk_v3
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import glm53_dedicated_v6 as v6
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v7-held.json"
MODULE_PATH = "evals/fleet/glm53_dedicated_v7.py"
SUBMIT_PATH = "evals/fleet/scripts/submit_glm53_dedicated_v7.sh"
DOC_PATH = "evals/fleet/DEDICATED_GLM53_V7.md"
HELD_PATH = "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v7-held.json"
JOBS_API_OBSERVATION_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v7-jobs-api-observation.json"
)
LIFECYCLE_PATH = v6.LIFECYCLE_PATH
HEARTBEAT_PATH = "evals/fleet/scripts/glm53_dedicated_v7_controller_heartbeat.sh"

SCHEMA = "fleet-glm53-dedicated-serving-v7-held-spec-v1"
HELD_SCHEMA = "fleet-glm53-dedicated-serving-v7-held-receipt-v1"
PREVIEW_SCHEMA = "fleet-glm53-dedicated-serving-v7-authenticated-preview-v1"
DUPLICATE_SCHEMA = "fleet-glm53-dedicated-serving-v7-create-once-gate-v1"
PARITY_SCHEMA = "fleet-glm53-dedicated-serving-v7-parity-gate-v1"
CANARY_SCHEMA = "fleet-glm53-dedicated-serving-v7-cell-accepted-v1"
RUNTIME_SCHEMA = "fleet-glm53-dedicated-serving-v7-runtime-gate-v1"
NAMESPACE = "fleet-train-jobs"
CANONICAL_SERVER_ARGUMENTS = (
    "python3", "-m", "sglang.launch_server",
    "--model-path", "/mnt/sfs/models/glm-5.3-30333038",
    "--served-model-name", "glm-5.3",
    "--host", "0.0.0.0", "--port", "8000",
    "--trust-remote-code",
    "--tp-size", "8", "--dp-size", "8", "--ep-size", "8",
    "--enable-dp-attention",
    "--context-length", "262144",
    "--quantization", "fp8",
    "--kv-cache-dtype", "bfloat16",
    "--mem-fraction-static", "0.85",
    "--attention-backend", "dsa",
    "--dsa-prefill-backend", "trtllm",
    "--dsa-decode-backend", "trtllm",
    "--moe-a2a-backend", "deepep",
    "--speculative-algorithm", "EAGLE",
    "--speculative-draft-model-path", "/mnt/sfs/models/glm-5.3-30333038",
    "--speculative-num-steps", "1",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "2",
    "--reasoning-parser", "glm45",
    "--tool-call-parser", "glm47",
    "--enable-metrics", "--enable-mfu-metrics", "--enable-metrics-for-all-schedulers",
)
CANONICAL_SERVER_ARGUMENTS_SHA256 = (
    "sha256:5969c3805b0473da3c445bebe89f96e3a2f7e7318b555eebd9135dd0eddd4903"
)

IMAGE = v6.IMAGE
MODEL_REVISION = v6.MODEL_REVISION
MODEL_PATH = v6.MODEL_PATH
STAGING_RECEIPT_SHA = v6.STAGING_RECEIPT_SHA
TOOL_CATALOG_SHA = v6.TOOL_CATALOG_SHA
PACKAGE_COMMIT = "b2934446d93cf34facf5fd007216e6dedcc9c2b4"
G7_PACKAGE_COMMIT = "e549ed9588159af9aaf6186e5b458a9eb070c114"
G7_SPEC_PATH = g7.G7_SPEC_PATHS["glm-5.3"]
G7_RELEASE_PATH = g7_authority.RELEASE_PATHS["glm-5.3"]
G7_GATE_PATH = str(bulk_v3.RECONCILIATION_ROOT / "glm-5.3-generation7-gate.json")
PREBULK_V4_ROOT = "/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v4"

EXPECTED_API_FIELDS = v6.EXPECTED_API_FIELDS
UID_FIELDS = (
    "api_run_uid",
    "ray_job_uid",
    "workload_uid",
    "ray_cluster_uid",
    "head_pod_uid",
    "service_uid",
    "probe_job_uid",
    "probe_pod_uid",
)


def load(path: Path) -> dict[str, Any]:
    return v6.load(path)


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
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _ranks(pair: Any, label: str) -> list[int]:
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(type(item) is int for item in pair)
        or not 1 <= pair[0] <= pair[1] <= 100
    ):
        raise ValueError(f"{label} must be an inclusive rank pair")
    return list(range(pair[0], pair[1] + 1))


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate_spec(value, root)
    return value


def _validate_generation7_binding(value: Any, root: Path) -> None:
    release_path = root / G7_RELEASE_PATH
    spec_path = root / G7_SPEC_PATH
    release = load(release_path)
    canary_spec = load(spec_path)
    # The complete release validator is intentionally reserved for the live
    # generation-7 acceptance gate: rebuilding its full historical package on
    # every local dedicated-server preview takes minutes.  The HELD package
    # binds the already-reviewed immutable file and self-digest here.
    if (
        release.get("schema_version") != g7_authority.RELEASE_SCHEMA
        or release.get("status") != "RELEASED"
        or release.get("model") != "glm-5.3"
        or release.get("package_commit") != G7_PACKAGE_COMMIT
        or release.get("generation7_spec_sha256") != canary_spec.get("generation7_spec_sha256")
        or release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256")
    ):
        raise ValueError("GLM dedicated v7 generation-7 release is invalid")
    expected = {
        "model": "glm-5.3",
        "selection_rank": 13,
        "attempt": 1,
        "job_name": g7.EXPECTED["glm-5.3"]["job_name"],
        "run_id": g7.EXPECTED["glm-5.3"]["run_id"],
        "spec_path": G7_SPEC_PATH,
        "generation7_spec_sha256": canary_spec["generation7_spec_sha256"],
        "release_path": G7_RELEASE_PATH,
        "release_file_sha256": file_sha256(release_path),
        "release_receipt_sha256": release["receipt_sha256"],
        "package_commit": G7_PACKAGE_COMMIT,
        "accepted_gate_path": G7_GATE_PATH,
        "accepted_gate_required": True,
    }
    if value != expected:
        raise ValueError("GLM dedicated v7 generation-7 binding drifted")


def _validate_partition(partition: Any, root: Path) -> None:
    if not isinstance(partition, dict):
        raise ValueError("GLM dedicated v7 partition is absent")
    if partition.get("accepted_generation7_hosted_canary") != {
        "selection_rank": 13,
        "attempt": 1,
    } or partition.get("dedicated_canaries") != {
        "A": {"selection_rank": 51, "attempt": 1},
        "B": {"selection_rank": 76, "attempt": 1},
    }:
        raise ValueError("GLM dedicated v7 canary bindings drifted")
    if (
        partition.get("whole_task_serving_block_required") is not True
        or partition.get("pooling_across_serving_blocks_allowed") is not False
    ):
        raise ValueError("GLM dedicated v7 treatment policy drifted")
    blocks = partition.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError("GLM dedicated v7 block count drifted")
    expected = [
        ("glm-hosted-autocontinue-v1", "hosted", (1, 25), [("hosted-1", 1, 25)]),
        ("glm-hosted-autocontinue-v1", "hosted", (26, 50), [("hosted-2", 26, 50)]),
        (
            "glm-dedicated-a-v7",
            "dedicated",
            (51, 75),
            [("dedicated-a-1", 51, 63), ("dedicated-a-2", 64, 75)],
        ),
        (
            "glm-dedicated-b-v7",
            "dedicated",
            (76, 100),
            [("dedicated-b-1", 76, 88), ("dedicated-b-2", 89, 100)],
        ),
    ]
    owner: dict[int, str] = {}
    for block, (name, kind, endpoints, streams) in zip(blocks, expected, strict=True):
        ranks = _ranks(block.get("ranks"), "block ranks")
        if block.get("serving_block") != name or block.get("kind") != kind:
            raise ValueError("GLM dedicated v7 block identity drifted")
        if ranks != list(range(endpoints[0], endpoints[1] + 1)):
            raise ValueError("GLM dedicated v7 block rank drifted")
        rows = block.get("streams")
        if not isinstance(rows, list) or len(rows) != len(streams):
            raise ValueError("GLM dedicated v7 stream count drifted")
        streamed: list[int] = []
        for row, (stream_name, start, end) in zip(rows, streams, strict=True):
            if row.get("name") != stream_name:
                raise ValueError("GLM dedicated v7 stream identity drifted")
            stream_ranks = _ranks(row.get("ranks"), "stream ranks")
            if stream_ranks != list(range(start, end + 1)):
                raise ValueError("GLM dedicated v7 stream ranks drifted")
            streamed.extend(stream_ranks)
        if sorted(streamed) != ranks or len(streamed) != len(set(streamed)):
            raise ValueError("GLM dedicated v7 streams do not partition their block")
        for rank in ranks:
            if rank in owner:
                raise ValueError("GLM dedicated v7 task rank crosses serving blocks")
            owner[rank] = name
    if set(owner) != set(range(1, 101)):
        raise ValueError("GLM dedicated v7 partition does not cover easiest-100")

    # Exact cell IDs are checked when controller_specs materializes the frozen
    # campaign.  Structural validation stays cheap enough to run before every
    # preview.


def validate_spec(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != SCHEMA or value.get("status") != "HELD":
        raise ValueError("GLM dedicated v7 held identity drifted")
    if value.get("launch_authorized") is not False:
        raise ValueError("GLM dedicated v7 unexpectedly authorizes launch")
    if value.get("model") != {
        "repository": "zai-org/GLM-5.3",
        "revision": MODEL_REVISION,
        "served_id": "glm-5.3",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": STAGING_RECEIPT_SHA,
    }:
        raise ValueError("GLM dedicated v7 model binding drifted")
    if value.get("runtime") != {
        "image": IMAGE,
        "base_image": (
            "lmsysorg/sglang@"
            "sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1"
        ),
        "ray_version": "2.56.0",
        "image_pull_secret_required": "ghcr-pull",
    }:
        raise ValueError("GLM dedicated v7 runtime binding drifted")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 8,
        "maximum_nodes": 2,
        "maximum_gpus": 16,
        "cpu_request": "96",
        "cpu_limit": "192",
        "memory_request": "1Ti",
        "memory_limit": "2Ti",
        "privileged": True,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
    }:
        raise ValueError("GLM dedicated v7 resource policy drifted")
    jobs_api = value.get("jobs_api") or {}
    if (
        jobs_api.get("base_url") != "https://api.ft.flt.build"
        or jobs_api.get("identity_route") != "GET /v1/whoami"
        or jobs_api.get("preview_route") != "POST /v1/runs/preview"
        or jobs_api.get("submit_route") != "POST /v1/runs"
        or jobs_api.get("authentication") != "github_cli_token_in_memory_bearer_header"
        or jobs_api.get("openapi_priority_classes") != ["fleet-train-high", "fleet-infra-quiet"]
        or jobs_api.get("requested_priority_class") != "fleet-infra-quiet"
        or jobs_api.get("requested_priority_accepted") is not True
        or jobs_api.get("requested_preemption_policy") != "Never"
        or jobs_api.get("authenticated_preview_priority_class") != "fleet-infra-quiet"
        or jobs_api.get("authenticated_preview_rendered_pull_secret") != "ghcr-pull"
        or jobs_api.get("authenticated_preview_launch_authorized") is not False
        or jobs_api.get("observation_path") != JOBS_API_OBSERVATION_PATH
        or jobs_api.get("observation_receipt_sha256")
        != "sha256:fdfcdd6490d507b6068f9947b9847121d9c9eb2ba6905aa2c04dcb5629462fb0"
    ):
        raise ValueError("GLM dedicated v7 Jobs API observation drifted")
    observation = load(root / JOBS_API_OBSERVATION_PATH)
    if (
        observation.get("receipt_sha256") != jobs_api["observation_receipt_sha256"]
        or observation.get("receipt_sha256")
        != self_hosted.digest_without(observation, "receipt_sha256")
        or observation.get("policy_preview", {}).get("accepted") is not True
        or observation.get("policy_preview", {}).get("http_status") != 200
        or observation.get("policy_preview", {}).get("launch_authorized") is not False
        or observation.get("mutations")
        != {"objects_created": False, "put_patch_delete_requests": 0, "submit_requests": 0}
    ):
        raise ValueError("GLM dedicated v7 Jobs API evidence is invalid")
    _validate_server_arguments(value.get("server_arguments"))
    if value.get("server_arguments_sha256") != CANONICAL_SERVER_ARGUMENTS_SHA256:
        raise ValueError("GLM dedicated v7 canonical server argument digest drifted")
    if value.get("lifecycle") != {
        "script": LIFECYCLE_PATH,
        "controller_heartbeat_script": HEARTBEAT_PATH,
        "post_ready_idle_seconds": 600,
        "poll_seconds": 5,
        "maximum_streams_per_endpoint": 2,
        "heartbeat_files": ["lifecycle/traffic-stream-1", "lifecycle/traffic-stream-2"],
        "drain_file": "lifecycle/DRAIN",
        "startup_is_outside_idle_budget": True,
        "release_on_controller_terminal": True,
    }:
        raise ValueError("GLM dedicated v7 lifecycle drifted")
    expected_replicas = {
        replica: {
            "title": f"chris-cyber-evalserve-glm53-tp8-{replica.lower()}-v7",
            "run_dir": f"/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-{replica.lower()}-v7",
            "serving_block": f"glm-dedicated-{replica.lower()}-v7",
            "service_name_template": "{ray_cluster_name}-head-svc",
            "service_dns_template": (
                "{service_name}.fleet-train-jobs.svc.cluster.local"
            ),
            "probe_job_name": (
                f"chris-cyber-glm53-dedicated-{replica.lower()}-v7-parity"
            ),
            "qualified_controller_concurrency": 2,
            "parity_receipt_path": (
                f"{PREBULK_V4_ROOT}/glm-dedicated-{replica.lower()}-parity.json"
            ),
            "canary_receipt_path": (
                f"{PREBULK_V4_ROOT}/glm-dedicated-{replica.lower()}-canary.json"
            ),
            "runtime_receipt_path": (
                f"{PREBULK_V4_ROOT}/glm-dedicated-{replica.lower()}-runtime.json"
            ),
        }
        for replica in ("A", "B")
    }
    if value.get("replicas") != expected_replicas:
        raise ValueError("GLM dedicated v7 replica identities drifted")
    _validate_generation7_binding(value.get("generation7"), root)
    if value.get("bulk_handoff") != {
        "current_generation7_bulk_commit": PACKAGE_COMMIT,
        "current_generation7_bulk_serving_block": "glm-hosted-autocontinue-v1",
        "current_generation7_bulk_is_dedicated_compatible": False,
        "current_generation7_bulk_must_remain_unreleased_for_dedicated_path": True,
        "dedicated_aware_append_only_successor_required": True,
        "fresh_prebulk_reconciliation_required": True,
        "whole_task_partition_required": True,
        "pooling_across_serving_blocks_allowed": False,
    }:
        raise ValueError("GLM dedicated v7 bulk handoff drifted")
    if value.get("acceptance_contract") != {
        "jobs_api_run_and_rayjob_binding_required": True,
        "workload_uid_and_owner_required": True,
        "service_dns_selector_target_required": True,
        "probe_job_and_pod_uid_required": True,
        "endpoint_traversal_required": True,
        "parallel_non_scored_concurrency_qualification_required": True,
        "source_binding_required": True,
        "global_cell_claim_required": True,
        "claim_execution_binding_required": True,
        "authoritative_session_match_count": 1,
        "verifier_execution_id_required": True,
        "session_ingest_completed_required": True,
        "cleanup_completed_required": True,
        "job_and_pod_uid_required": True,
        "zero_restarts_required": True,
        "scores_included": False,
    }:
        raise ValueError("GLM dedicated v7 acceptance contract drifted")
    if value.get("privacy") != {
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }:
        raise ValueError("GLM dedicated v7 privacy policy drifted")
    _validate_partition(value.get("partition"), root)
    for path in (LIFECYCLE_PATH, HEARTBEAT_PATH, SUBMIT_PATH, DOC_PATH):
        candidate = root / path
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"GLM dedicated v7 required file is absent or unsafe: {path}")


def _validate_server_arguments(arguments: Any) -> None:
    if arguments != list(CANONICAL_SERVER_ARGUMENTS):
        raise ValueError("GLM dedicated v7 canonical server argv drifted")
    if self_hosted.sha256(self_hosted.canonical_json(arguments)) != (
        CANONICAL_SERVER_ARGUMENTS_SHA256
    ):
        raise AssertionError("GLM dedicated v7 canonical server argv constant drifted")


def controller_cells(
    partition: dict[str, Any], expected_cells: dict[tuple[int, int], str]
) -> dict[tuple[int, int], str]:
    assigned: dict[tuple[int, int], str] = {}
    for block in partition["blocks"]:
        partial = {int(key): attempts for key, attempts in block["partial_attempts"].items()}
        for rank in _ranks(block["ranks"], "block ranks"):
            for attempt in partial.get(rank, [1, 2, 3, 4]):
                key = (rank, attempt)
                if key in assigned:
                    raise ValueError("GLM dedicated v7 cell overlap")
                assigned[key] = expected_cells[key]
    return assigned


def controller_specs(value: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    validate_spec(value, root)
    campaign = exact.read_object(root / v6.exact_pass4_campaign_path())
    universe = exact.build_universe(campaign, root)
    cells = {
        (row["selection_rank"], row["attempt"]): row
        for row in universe["cells"]
        if row["model"] == "glm-5.3"
    }
    result: dict[str, dict[str, Any]] = {}
    for block in value["partition"]["blocks"]:
        partial = {int(key): attempts for key, attempts in block["partial_attempts"].items()}
        for index, stream in enumerate(block["streams"], 1):
            ranks = _ranks(stream["ranks"], "stream ranks")
            dedicated = block["kind"] == "dedicated"
            canary = value["partition"]["dedicated_canaries"].get(
                block["serving_block"].removeprefix("glm-dedicated-").removesuffix("-v7").upper()
            )
            rows = [
                {
                    "cell_id": cells[(rank, attempt)]["cell_id"],
                    "selection_rank": rank,
                    "attempt": attempt,
                    "task_key": cells[(rank, attempt)]["task_key"],
                    "task_version_id": cells[(rank, attempt)]["task_version_id"],
                }
                for rank in ranks
                for attempt in partial.get(rank, [1, 2, 3, 4])
                if not (
                    dedicated
                    and canary is not None
                    and rank == canary["selection_rank"]
                    and attempt == canary["attempt"]
                )
            ]
            result[stream["name"]] = {
                "model": "glm-5.3",
                "serving_block": block["serving_block"],
                "serving_kind": block["kind"],
                "task_ranks": ranks,
                "cells": rows,
                "new_session_count": len(rows),
                "maximum_endpoint_streams": 2,
                "qualified_controller_concurrency": 2 if dedicated else None,
                "heartbeat_relative_path": (
                    f"lifecycle/traffic-stream-{index}" if dedicated else None
                ),
                "uid_bound_runtime_gate_required": dedicated,
                "dedicated_canary_required_before_stream": dedicated,
                "launch_authorized": False,
                "treatment": exact.EXPECTED_TREATMENT,
            }
    return result


def server_payload(value: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    validate_spec(value, root)
    if replica not in {"A", "B"}:
        raise ValueError("replica must be A or B")
    selected = value["replicas"][replica]
    lifecycle = (root / LIFECYCLE_PATH).read_text()
    payload = {
        "image": IMAGE,
        "command": (
            "bash -lc "
            + shlex.quote(lifecycle)
            + " -- "
            + shlex.join(CANONICAL_SERVER_ARGUMENTS)
        ),
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
        "priority_class": "fleet-infra-quiet",
        "privileged": True,
        "run_dir": selected["run_dir"],
        "title": selected["title"],
    }
    if set(payload) != EXPECTED_API_FIELDS:
        raise AssertionError("GLM dedicated v7 Jobs API payload shape drifted")
    return payload


def dedicated_canary_cell(value: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    target = value["partition"]["dedicated_canaries"][replica]
    campaign = exact.read_object(root / v6.exact_pass4_campaign_path())
    universe = exact.build_universe(campaign, root)
    matches = [
        {
            "cell_id": cell["cell_id"],
            "selection_rank": cell["selection_rank"],
            "attempt": cell["attempt"],
            "task_key": cell["task_key"],
            "task_version_id": cell["task_version_id"],
            "execution_id": cell["initial_execution"]["execution_id"],
        }
        for cell in universe["cells"]
        if cell["model"] == "glm-5.3"
        and cell["selection_rank"] == target["selection_rank"]
        and cell["attempt"] == target["attempt"]
    ]
    if len(matches) != 1:
        raise ValueError("GLM dedicated v7 canary is not unique")
    return matches[0]


def dedicated_canary_plan(
    value: dict[str, Any], root: Path, replica: str
) -> dict[str, Any]:
    """Return the one scored cell allowed after parity and before the runtime gate."""
    validate_spec(value, root)
    if replica not in {"A", "B"}:
        raise ValueError("replica must be A or B")
    cell = dedicated_canary_cell(value, root, replica)
    body = {
        "schema_version": "fleet-glm53-dedicated-serving-v7-pre-canary-plan-v1",
        "replica": replica,
        "serving_block": value["replicas"][replica]["serving_block"],
        "cell": cell,
        "precondition": "uid_bound_non_scored_parity_only",
        "parity_gate": (
            f"{PREBULK_V4_ROOT}/glm-dedicated-{replica.lower()}-parity.json"
        ),
        "runtime_gate_required_before_canary": False,
        "post_canary_runtime_gate_required_before_bulk": True,
        "launch_authorized": False,
    }
    return {**body, "plan_sha256": self_hosted.sha256(self_hosted.canonical_json(body))}


def validate_generation7_gate(
    receipt: dict[str, Any], root: Path, *, evidence_root: Path | None = None
) -> None:
    bulk_v3.validate_canary_gate(receipt, "glm-5.3", root, evidence_root=evidence_root)


def validate_authenticated_preview(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    expected = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "authenticated": True,
        "base_url": "https://api.ft.flt.build",
        "route": "/v1/runs/preview",
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "requested_image": IMAGE,
        "rendered_image": IMAGE,
        "rendered_image_pull_secrets": ["ghcr-pull"],
        "rendered_workers": 1,
        "rendered_gpus_per_worker": 8,
        "rendered_priority_class": "fleet-infra-quiet",
        "rendered_preemption_policy": "Never",
        "rendered_run_dir": value["replicas"][replica]["run_dir"],
        "post_put_patch_delete": 0,
        "credentials_included": False,
    }
    if {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected:
        raise ValueError("GLM dedicated v7 authenticated preview gate is invalid")
    if receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise ValueError("GLM dedicated v7 authenticated preview digest is invalid")


def validate_duplicate_gate(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    prefix = f"dedicated-{replica.lower()}-"
    cells = [
        cell
        for name, controller in controller_specs(value, root).items()
        if name.startswith(prefix)
        for cell in controller["cells"]
    ]
    cells.append(dedicated_canary_plan(value, root, replica)["cell"])
    prebulk_sha = _sha256(
        receipt.get("prebulk_reconciliation_receipt_sha256"),
        "prebulk_reconciliation_receipt_sha256",
    )
    expected = {
        "schema_version": DUPLICATE_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "title": value["replicas"][replica]["title"],
        "run_dir": value["replicas"][replica]["run_dir"],
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
    if {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected:
        raise ValueError("GLM dedicated v7 create-once gate is invalid")
    if receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise ValueError("GLM dedicated v7 create-once gate digest is invalid")


def _name(value: Any, label: str, *, prefix: str | None = None) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError(f"{label} must be a nonempty Kubernetes-safe identity")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{label} has the wrong identity prefix")
    return value


def _parity_runtime_identity(
    receipt: dict[str, Any], value: dict[str, Any], replica: str
) -> dict[str, Any]:
    selected = value["replicas"][replica]
    for field in UID_FIELDS:
        _uuid(receipt.get(field), field)
    api_run_id = _name(receipt.get("api_run_id"), "api_run_id", prefix="ft-run-")
    ray_job_name = _name(receipt.get("ray_job_name"), "ray_job_name")
    workload_name = _name(receipt.get("workload_name"), "workload_name")
    ray_cluster_name = _name(receipt.get("ray_cluster_name"), "ray_cluster_name")
    head_pod_name = _name(receipt.get("head_pod_name"), "head_pod_name")
    service_name = _name(receipt.get("service_name"), "service_name")
    probe_pod_name = _name(receipt.get("probe_pod_name"), "probe_pod_name")
    if ray_job_name != api_run_id:
        raise ValueError("Jobs API run and RayJob name are not the same exact run")
    expected_service_name = f"{ray_cluster_name}-head-svc"
    expected_service_dns = f"{expected_service_name}.{NAMESPACE}.svc.cluster.local"
    if (
        selected["service_name_template"] != "{ray_cluster_name}-head-svc"
        or selected["service_dns_template"]
        != "{service_name}.fleet-train-jobs.svc.cluster.local"
        or service_name != expected_service_name
    ):
        raise ValueError("dedicated Service identity drifted")
    service_selector = receipt.get("service_selector")
    if service_selector != {
        "ray.io/cluster": ray_cluster_name,
        "ray.io/node-type": "head",
    }:
        raise ValueError("dedicated Service selector drifted")
    endpoint_traversal = receipt.get("endpoint_traversal")
    if endpoint_traversal != {
        "source_kind": "Pod",
        "source_namespace": NAMESPACE,
        "source_name": probe_pod_name,
        "via_service_dns": expected_service_dns,
        "scheme": "http",
        "port": 8000,
        "path": "/v1/chat/completions",
        "direct_pod_bypass": False,
    }:
        raise ValueError("dedicated parity endpoint traversal drifted")
    expected = {
        "api_run_id": api_run_id,
        "api_run_uid": receipt["api_run_uid"],
        "kubernetes_namespace": NAMESPACE,
        "ray_job_kind": "RayJob",
        "ray_job_name": ray_job_name,
        "ray_job_uid": receipt["ray_job_uid"],
        "workload_kind": "Workload",
        "workload_name": workload_name,
        "workload_uid": receipt["workload_uid"],
        "workload_owner_ray_job_uid": receipt["ray_job_uid"],
        "ray_cluster_kind": "RayCluster",
        "ray_cluster_name": ray_cluster_name,
        "ray_cluster_uid": receipt["ray_cluster_uid"],
        "ray_cluster_owner_ray_job_uid": receipt["ray_job_uid"],
        "head_pod_kind": "Pod",
        "head_pod_name": head_pod_name,
        "head_pod_uid": receipt["head_pod_uid"],
        "head_pod_owner_ray_cluster_uid": receipt["ray_cluster_uid"],
        "service_kind": "Service",
        "service_name": service_name,
        "service_uid": receipt["service_uid"],
        "service_owner_ray_cluster_uid": receipt["ray_cluster_uid"],
        "service_dns": expected_service_dns,
        "service_selector": service_selector,
        "service_target": {"port": 8000, "target_port": 8000, "protocol": "TCP"},
        "probe_job_kind": "Job",
        "probe_job_namespace": NAMESPACE,
        "probe_job_name": selected["probe_job_name"],
        "probe_job_uid": receipt["probe_job_uid"],
        "probe_pod_kind": "Pod",
        "probe_pod_name": probe_pod_name,
        "probe_pod_uid": receipt["probe_pod_uid"],
        "probe_pod_owner_job_uid": receipt["probe_job_uid"],
        "endpoint_traversal": endpoint_traversal,
    }
    return expected


def validate_parity_gate(
    receipt: dict[str, Any], value: dict[str, Any], root: Path, replica: str
) -> None:
    payload = server_payload(value, root, replica)
    identity = _parity_runtime_identity(receipt, value, replica)
    expected = {
        "schema_version": PARITY_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "run_dir": value["replicas"][replica]["run_dir"],
        "image": IMAGE,
        "resolved_image_id": IMAGE,
        "model_revision": MODEL_REVISION,
        "model_path": MODEL_PATH,
        "served_id": "glm-5.3",
        "context_length": 262144,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
        "gpus": 8,
        "pod_restarts": 0,
        "health_http_200": True,
        "service_port": 8000,
        "service_health_http_200": True,
        "server_arguments_sha256": CANONICAL_SERVER_ARGUMENTS_SHA256,
        "rendered_command_sha256": self_hosted.sha256(payload["command"].encode()),
        "structured_tools": ["bash", "submit_report"],
        "tool_catalog_sha256": TOOL_CATALOG_SHA,
        "content_blind_probe": True,
        "non_scored_model_requests": 2,
        "scored_sessions": 0,
        "concurrency_qualification": {
            "mode": "parallel_barrier",
            "requested_concurrency": 2,
            "maximum_observed_inflight": 2,
            "completed_requests": 2,
            "failed_requests": 0,
            "scored_requests": 0,
            "endpoint_traversal": identity["endpoint_traversal"],
        },
        "probe_resource_policy": {
            "cpu_only": True,
            "gpus": 0,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        **identity,
        "prompts_traces_flags_or_scores_included": False,
    }
    if (
        {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("GLM dedicated v7 UID-bound parity gate is invalid")


def validate_canary_acceptance(
    receipt: dict[str, Any],
    parity: dict[str, Any],
    value: dict[str, Any],
    root: Path,
    replica: str,
) -> None:
    validate_parity_gate(parity, value, root, replica)
    cell = dedicated_canary_cell(value, root, replica)
    for field in ("session_id", "verifier_execution_id", "claim_job_uid", "claim_pod_uid"):
        _uuid(receipt.get(field), field)
    _sha256(receipt.get("claim_receipt_sha256"), "claim_receipt_sha256")
    run_id = receipt.get("run_id")
    exit_code = receipt.get("agent_exit_code")
    claim_job_name = _name(receipt.get("claim_job_name"), "claim_job_name")
    expected_claim_job_name = (
        f"chris-cyber-exact100-glm-dedicated-{replica.lower()}-canary-v4"
    )
    if claim_job_name != expected_claim_job_name:
        raise ValueError("GLM dedicated v7 canary claim Job identity is invalid")
    claim_pod_name = _name(receipt.get("claim_pod_name"), "claim_pod_name")
    expected_claim_path = (
        f"{bulk_v3.CLAIM_ROOT}/"
        f"{cell['execution_id'].removeprefix('sha256:')}.json"
    )
    runtime_identity = _parity_runtime_identity(parity, value, replica)
    expected = {
        "schema_version": CANARY_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "replica": replica,
        "serving_block": value["replicas"][replica]["serving_block"],
        **cell,
        "run_id": run_id,
        "session_id": receipt["session_id"],
        "verifier_execution_id": receipt["verifier_execution_id"],
        "claim_execution_id": cell["execution_id"],
        "claim_path": expected_claim_path,
        "claim_job_kind": "Job",
        "claim_job_namespace": NAMESPACE,
        "claim_job_name": claim_job_name,
        "claim_job_uid": receipt["claim_job_uid"],
        "claim_pod_kind": "Pod",
        "claim_pod_name": claim_pod_name,
        "claim_pod_uid": receipt["claim_pod_uid"],
        "claim_pod_owner_job_uid": receipt["claim_job_uid"],
        "claim_receipt_sha256": receipt["claim_receipt_sha256"],
        "source_binding_valid": True,
        "session_model": exact.EXPECTED_MODELS["glm-5.3"]["session_model"],
        "model_revision": MODEL_REVISION,
        "authoritative_metadata_run_id": run_id,
        "serving_parity_receipt_sha256": parity["receipt_sha256"],
        "serving_runtime": runtime_identity,
        "authoritative_session_id": receipt["session_id"],
        "authoritative_verifier_execution_id": receipt["verifier_execution_id"],
        "authoritative_session_matches": 1,
        "agent_exit_code": exit_code,
        "agent_process_exit_success": exit_code == 0,
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "cleanup_scope": "exact_session_and_task_environment",
        "scores_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    if not isinstance(run_id, str) or not run_id.startswith("chris-cyber-"):
        raise ValueError("GLM dedicated v7 canary run identity is invalid")
    if type(exit_code) is not int:
        raise ValueError("GLM dedicated v7 canary process exit is invalid")
    if {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected:
        raise ValueError("GLM dedicated v7 canary acceptance is invalid")
    if receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise ValueError("GLM dedicated v7 canary digest is invalid")


def validate_runtime_gate(
    receipt: dict[str, Any],
    parity: dict[str, Any],
    canary: dict[str, Any],
    value: dict[str, Any],
    root: Path,
    replica: str,
) -> None:
    validate_canary_acceptance(canary, parity, value, root, replica)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "serving_runtime": _parity_runtime_identity(parity, value, replica),
        "scored_canary_accepted": True,
        "scored_canary_cell_id": canary["cell_id"],
        "scored_canary_execution_id": canary["execution_id"],
        "scored_canary_claim_path": canary["claim_path"],
        "scored_canary_claim_receipt_sha256": canary["claim_receipt_sha256"],
        "scored_canary_run_id": canary["run_id"],
        "scored_canary_session_id": canary["session_id"],
        "scored_canary_verifier_execution_id": canary["verifier_execution_id"],
        "scored_canary_authoritative_session_matches": 1,
        "scored_canary_session_ingest_completed": True,
        "scored_canary_cleanup_completed": True,
        "scored_canary_receipt_sha256": canary["receipt_sha256"],
        "dependent_bulk_release_authorized": True,
        "qualified_controller_concurrency": 2,
        "released_streams": [
            f"dedicated-{replica.lower()}-1",
            f"dedicated-{replica.lower()}-2",
        ],
        "maximum_streams_released": 2,
        "prompts_traces_flags_or_scores_included": False,
    }
    if {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected:
        raise ValueError("GLM dedicated v7 runtime gate is invalid")
    if receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise ValueError("GLM dedicated v7 runtime gate digest is invalid")


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
            name: controller["new_session_count"] for name, controller in controllers.items()
        }
        | {f"dedicated-{replica.lower()}-canary": 1 for replica in ("A", "B")},
        "total_post_generation7_glm_cells": sum(
            controller["new_session_count"] for controller in controllers.values()
        )
        + 2,
        "dedicated_nodes_maximum": 2,
        "dedicated_gpus_maximum": 16,
        "replica_a_before_replica_b": True,
        "post_ready_idle_seconds": 600,
        "jobs_api_priority_preview_passed": True,
        "dedicated_aware_bulk_successor_required": True,
        "objects_created": False,
    }


def held_receipt(root: Path) -> dict[str, Any]:
    value = spec(root)
    preview = held_preview(root)
    receipt = {
        "schema_version": HELD_SCHEMA,
        "status": "HELD",
        "launch_authorized": False,
        "spec_path": SPEC_PATH,
        "spec_file_sha256": file_sha256(root / SPEC_PATH),
        "module_path": MODULE_PATH,
        "module_file_sha256": file_sha256(root / MODULE_PATH),
        "submit_path": SUBMIT_PATH,
        "submit_file_sha256": file_sha256(root / SUBMIT_PATH),
        "documentation_path": DOC_PATH,
        "documentation_file_sha256": file_sha256(root / DOC_PATH),
        "generation7_release": value["generation7"],
        "jobs_api_observation": {
            "path": JOBS_API_OBSERVATION_PATH,
            "file_sha256": file_sha256(root / JOBS_API_OBSERVATION_PATH),
            "receipt_sha256": value["jobs_api"]["observation_receipt_sha256"],
        },
        "server_request_sha256": preview["server_request_sha256"],
        "controller_session_counts": preview["controller_session_counts"],
        "total_post_generation7_glm_cells": 399,
        "release_blockers": [
            "generation7_glm_canary_gate_must_be_accepted",
            "all_hosted_generation7_bulk_v3_must_remain_unreleased_for_this_path",
            "fresh_dedicated_aware_bulk_and_prebulk_successor_must_be_accepted",
            "replica_a_and_b_exact_authenticated_previews_must_be_fresh_at_release",
            "replica_a_uid_bound_parity_and_scored_canary_before_second_stream_or_replica_b",
            "replica_b_uid_bound_parity_and_scored_canary_before_its_second_stream",
        ],
        "resource_policy": value["resources"],
        "treatment_policy": {
            "whole_task_serving_blocks": True,
            "hosted_and_dedicated_results_pooled": False,
            "maximum_streams_per_endpoint": 2,
        },
        "privacy": value["privacy"],
        "objects_created": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    return receipt


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    if receipt != held_receipt(root):
        raise ValueError("GLM dedicated v7 held receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "held-preview",
            "render-held",
            "server-preview",
            "validate",
            "validate-preview",
            "validate-duplicate",
            "validate-parity",
            "validate-canary",
            "validate-runtime",
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--replica", choices=("A", "B"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--canary", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.command == "validate":
        validate_held(load(root / HELD_PATH), root)
        print(json.dumps({"status": "HELD", "valid": True}, sort_keys=True))
    elif args.command == "render-held":
        print(json.dumps(held_receipt(root), indent=2, sort_keys=True))
    elif args.command == "server-preview":
        if args.replica is None:
            parser.error("server-preview requires --replica")
        print(json.dumps(server_payload(spec(root), root, args.replica), sort_keys=True))
    elif args.command.startswith("validate-"):
        if args.replica is None or args.receipt is None:
            parser.error(f"{args.command} requires --replica and --receipt")
        value = spec(root)
        receipt = load(args.receipt.resolve(strict=True))
        if args.command == "validate-preview":
            validate_authenticated_preview(receipt, value, root, args.replica)
        elif args.command == "validate-duplicate":
            validate_duplicate_gate(receipt, value, root, args.replica)
        elif args.command == "validate-parity":
            validate_parity_gate(receipt, value, root, args.replica)
        else:
            if args.parity is None:
                parser.error(f"{args.command} requires --parity")
            parity = load(args.parity.resolve(strict=True))
            if args.command == "validate-canary":
                validate_canary_acceptance(receipt, parity, value, root, args.replica)
            else:
                if args.canary is None:
                    parser.error("validate-runtime requires --canary")
                canary = load(args.canary.resolve(strict=True))
                validate_runtime_gate(receipt, parity, canary, value, root, args.replica)
        print(json.dumps({"status": "PASSED", "valid": True}, sort_keys=True))
    else:
        print(json.dumps(held_preview(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
