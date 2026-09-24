"""Fleet held-out adapter for :mod:`evals.campaign`.

One Fleet evaluator Job owns eight exact tasks for one model and one seed.  The
generic campaign controller still tracks the 192 statistical cells separately;
the adapter elects one cell to create each of the 24 source Jobs and lets the
other seven cells bind to that same exact Job UID.  Fleet grading already
happens inside the evaluator, so the score phase only seals the accepted native
grading receipt.  It never repeats a rollout or calls another judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

from evals.campaign import RECEIPT_SCHEMA, canonical, digest
from evals.fleet import heldout_launch

BINDINGS_SCHEMA = "cyber_fleet_eval_campaign_bindings_v1"
GROUP_PREVIEW_SCHEMA = "cyber_fleet_eval_campaign_group_preview_v1"
GROUP_LAUNCH_SCHEMA = "cyber_fleet_eval_campaign_group_launch_v1"
UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
FINAL8_SPLIT_FILE_SHA256 = "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb"
FINAL8_SPLIT_SHA256 = "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
FINAL8_SELECTION_SHA256 = "sha256:9623149c4a021bc13ed2cf94ca26e107b30c18816cf3c02d76b5020cab5066f4"
FINAL8_EVIDENCE_BOUNDARY = "historically_exposed_locked_confirmation_set"
FINAL8_MATRIX_SHA256 = "sha256:72d310d0e8a8f5d949c6e6066feb8cefeef182cb86cb8f1dd1083f6b5f527be7"
FINAL8_OPERATIONAL_ARMS = {
    "base": "qwen38-27b-base-control",
    "step1000": "q38-teacher3k-32k-step1000",
    "b16_step200": "q38-d32-b16-lr5e6-step200",
    "context64_step225": "q38-t3k64-b8-step225",
    "lr1_step500": "q38-t3k32-lr1-step500",
    "context96_step300": "q38-t3k96-b8-step300",
}
FINAL8_SELECTION_RECEIPTS = {
    "base": "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352",
    "step1000": "sha256:8a1d9e093fa1f856c35d357cf34861862317491ee83a9268613f1cf86160eef0",
    "b16_step200": "sha256:88edcde39542f3e32fc895acf93999230c421e93e794f242958d6f675400a72e",
    "context64_step225": "sha256:4e09a634106991b0f4b50b1a10f2c65a36c2eb3cf167514ea53c6d6cbae1a0b7",
    "lr1_step500": "sha256:fc568b2b58c744788b7ad361d3bbe57b72e5974ddd7082d52f5ffe99db4fc65e",
    "context96_step300": "sha256:e7585657109ce728a2daa9ac1a1c01e5d7c07176ebf278d0818f6992942810ac",
}
FINAL8_SERVING_MODEL_IDS = {
    "base": "chris-q38-base-pass4-v1",
    "step1000": "chris-q38-t3k32-s1000-v1",
    "b16_step200": "chris-q38-d32-b16-s200-v1",
    "context64_step225": "chris-q38-t3k64-s225-v1",
    "lr1_step500": "chris-q38-t3k32-lr1-s500-v1",
    "context96_step300": "chris-q38-t3k96-s300-v1",
}
FINAL8_HARNESS = {
    "compaction_headroom_tokens": 20000,
    "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
    "context_window_size": 262144,
    "harness": "opencode",
    "harness_version": "1.18.27",
    "max_model_requests": 600,
    "max_output_tokens": 32768,
    "provider_adapter": "@ai-sdk/openai-compatible",
    "release_asset_sha256": (
        "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    ),
    "timeout_seconds": 28800,
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
    "tools": ["bash", "submit_report"],
}
FINAL8_IMAGES = {
    "agent": "sha256:c7d048c98e6b8e52e5b76ab4006a7626b1ccf63a37bfa4b47ecd0fe9028e1f92",
    "proxy": (
        "ghcr.io/astral-sh/uv:python3.12-bookworm@"
        "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
    ),
}
FINAL8_EXPOSURE_AUDIT = {
    "path": "docs/evidence/qwen38-fleet-final8-exposure-audit-20260923.json",
    "file_sha256": "sha256:41fe6614eadec151919902ed27641ebd1ea19b890c72192f263ec3d9361aedd2",
    "sha256": "sha256:0c4d85028026a3eac4ab0a9dee4689854a55ad74122bdc048407e759585bbb35",
}
FINAL8_TASK_VERSIONS = frozenset(
    {
        "e33aaed0-491b-46e8-8891-55d99c80ef56",
        "ba912601-1184-4dcd-9e9d-2fce22543e4b",
        "9375a9b9-04e5-4f6f-ad47-286121278992",
        "5287e180-64ea-487b-982a-d08a38c05fa2",
        "8c0b249c-8946-4ec3-8609-d74631af547e",
        "f8533754-39ad-4683-b9c1-d26bc24dfd14",
        "8e5c7c45-7259-420d-ba12-0381465d864e",
        "ee77cfbf-ceed-487f-a19d-3922aaaf4bad",
    }
)
ATTEMPT_SEEDS = (46, 47, 48, 49)
QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"


class FleetCampaignError(ValueError):
    """The campaign cell is not exactly bound to a sealed Fleet evaluator."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FleetCampaignError("invalid_json_file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetCampaignError("invalid_json_file") from exc
    if not isinstance(value, dict):
        raise FleetCampaignError("invalid_json_file")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = canonical(value) + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise FleetCampaignError("immutable_group_record_changed")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise FleetCampaignError("invalid_launch_packet_path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise FleetCampaignError("invalid_launch_packet_path")
    candidate = root / Path(relative)
    if candidate.is_symlink() or not candidate.is_file():
        raise FleetCampaignError("invalid_launch_packet_path")
    path = candidate.resolve(strict=True)
    if root.resolve() not in path.parents:
        raise FleetCampaignError("invalid_launch_packet_path")
    return path


def _load_bindings(path: Path) -> dict[str, Any]:
    value = _load(path)
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    required = {
        "schema",
        "matrix_sha256",
        "benchmark_id",
        "task_set_sha256",
        "scoring_protocol_sha256",
        "budgets_sha256",
        "matched_treatment_receipt_sha256",
        "split_manifest_file_sha256",
        "split_manifest_sha256",
        "evidence_boundary_label",
        "exposure_audit",
        "harness",
        "sampling",
        "model_bindings",
        "target_identity_receipts",
        "groups",
        "cells",
        "sha256",
    }
    if (
        set(value) != required
        or value["schema"] != BINDINGS_SCHEMA
        or value["sha256"] != digest(unsigned)
    ):
        raise FleetCampaignError("invalid_fleet_campaign_bindings")
    for field in (
        "matrix_sha256",
        "task_set_sha256",
        "scoring_protocol_sha256",
        "budgets_sha256",
        "matched_treatment_receipt_sha256",
        "split_manifest_file_sha256",
        "split_manifest_sha256",
    ):
        if not isinstance(value[field], str) or not SHA256.fullmatch(value[field]):
            raise FleetCampaignError("invalid_fleet_campaign_bindings")
    if (
        value["matrix_sha256"] != FINAL8_MATRIX_SHA256
        or value["task_set_sha256"] != FINAL8_SELECTION_SHA256
        or value["split_manifest_file_sha256"] != FINAL8_SPLIT_FILE_SHA256
        or value["split_manifest_sha256"] != FINAL8_SPLIT_SHA256
        or value["evidence_boundary_label"] != FINAL8_EVIDENCE_BOUNDARY
        or value["exposure_audit"] != FINAL8_EXPOSURE_AUDIT
    ):
        raise FleetCampaignError("bindings_do_not_select_locked_final8")
    harness = value["harness"]
    sampling = value["sampling"]
    if (
        not isinstance(harness, dict)
        or set(harness) != {"name", "identity_receipt_sha256"}
        or not isinstance(harness["name"], str)
        or not SHA256.fullmatch(str(harness["identity_receipt_sha256"]))
        or not isinstance(sampling, dict)
        or set(sampling) != {"temperature", "top_p", "attempt_seeds"}
        or sampling.get("temperature") != 0.6
        or sampling.get("top_p") != 0.95
        or not isinstance(sampling["attempt_seeds"], list)
        or tuple(sampling["attempt_seeds"]) != ATTEMPT_SEEDS
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in sampling["attempt_seeds"])
        or len(set(sampling["attempt_seeds"])) != 4
    ):
        raise FleetCampaignError("invalid_fleet_campaign_bindings")
    models = value["model_bindings"]
    targets = value["target_identity_receipts"]
    groups = value["groups"]
    cells = value["cells"]
    if (
        not isinstance(models, dict)
        or set(models) != set(FINAL8_OPERATIONAL_ARMS)
        or not isinstance(targets, dict)
        or len(targets) != 8
        or not isinstance(groups, dict)
        or len(groups) != 24
        or not isinstance(cells, dict)
        or len(cells) != 192
    ):
        raise FleetCampaignError("fleet_campaign_requires_six_models_final8_pass4")
    if set(targets) != FINAL8_TASK_VERSIONS:
        raise FleetCampaignError("bindings_do_not_select_locked_final8")
    for task_version, receipt in targets.items():
        if (
            not UUID.fullmatch(task_version)
            or not isinstance(receipt, str)
            or not SHA256.fullmatch(receipt)
        ):
            raise FleetCampaignError("invalid_final8_target_binding")
    model_fields = {
        "artifact_id",
        "checkpoint_id",
        "weights_sha256",
        "model_revision",
        "checkpoint_provenance_sha256",
        "serving_route_proof_sha256",
        "serving_route_receipt_sha256",
        "live_parity_receipt_sha256",
        "selection_artifact_receipt_sha256",
        "serving_model_id",
    }
    for model_id, model in models.items():
        if (
            not isinstance(model_id, str)
            or not isinstance(model, dict)
            or set(model) != model_fields
            or model.get("artifact_id") != FINAL8_OPERATIONAL_ARMS[model_id]
            or model.get("checkpoint_id") != FINAL8_OPERATIONAL_ARMS[model_id]
            or model.get("selection_artifact_receipt_sha256") != FINAL8_SELECTION_RECEIPTS[model_id]
            or model.get("serving_model_id") != FINAL8_SERVING_MODEL_IDS[model_id]
            or not isinstance(model["model_revision"], str)
            or not model["model_revision"]
            or any(
                not isinstance(model[field], str) or not SHA256.fullmatch(model[field])
                for field in (
                    "weights_sha256",
                    "checkpoint_provenance_sha256",
                    "serving_route_proof_sha256",
                    "serving_route_receipt_sha256",
                    "live_parity_receipt_sha256",
                )
            )
        ):
            raise FleetCampaignError("invalid_model_binding")
    group_fields = {
        "comparison_protocol_sha256",
        "leader_experiment_key",
        "launch_packet",
        "launch_packet_sha256",
    }
    for group_id, group in groups.items():
        if (
            not isinstance(group_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", group_id)
            or not isinstance(group, dict)
            or set(group) != group_fields
            or group["leader_experiment_key"] not in cells
            or not SHA256.fullmatch(str(group["comparison_protocol_sha256"]))
            or not SHA256.fullmatch(str(group["launch_packet_sha256"]))
        ):
            raise FleetCampaignError("invalid_group_binding")
        _safe_relative(path.parent, group["launch_packet"])
    combinations: set[tuple[str, str, int]] = set()
    by_group: dict[str, list[str]] = {group_id: [] for group_id in groups}
    for key, cell in cells.items():
        if (
            not SHA256.fullmatch(key)
            or not isinstance(cell, dict)
            or set(cell)
            != {
                "campaign_identity_sha256",
                "group_id",
                "model_id",
                "task_version_id",
                "attempt",
                "canary",
            }
            or cell["campaign_identity_sha256"] != key
            or cell["group_id"] not in groups
            or cell["model_id"] not in models
            or cell["task_version_id"] not in targets
            or type(cell["attempt"]) is not int
            or not 1 <= cell["attempt"] <= 4
            or cell["canary"] is not False
        ):
            raise FleetCampaignError("invalid_cell_binding")
        combination = (cell["model_id"], cell["task_version_id"], cell["attempt"])
        if combination in combinations:
            raise FleetCampaignError("duplicate_cell_binding")
        combinations.add(combination)
        by_group[cell["group_id"]].append(key)
    grouped_model_attempts: set[tuple[str, int]] = set()
    protocol_by_attempt: dict[int, str] = {}
    for group_id, members in by_group.items():
        group = groups[group_id]
        rows = [cells[key] for key in members]
        if (
            len(rows) != 8
            or {row["task_version_id"] for row in rows} != set(targets)
            or len({(row["model_id"], row["attempt"]) for row in rows}) != 1
            or group["leader_experiment_key"] not in members
        ):
            raise FleetCampaignError("invalid_source_job_group")
        model_attempt = (rows[0]["model_id"], rows[0]["attempt"])
        if model_attempt in grouped_model_attempts:
            raise FleetCampaignError("duplicate_source_job_group")
        grouped_model_attempts.add(model_attempt)
        attempt = rows[0]["attempt"]
        protocol_sha256 = group["comparison_protocol_sha256"]
        if attempt in protocol_by_attempt and protocol_by_attempt[attempt] != protocol_sha256:
            raise FleetCampaignError("attempt_groups_have_different_comparison_protocols")
        protocol_by_attempt[attempt] = protocol_sha256
    if grouped_model_attempts != {
        (model_id, attempt) for model_id in models for attempt in range(1, 5)
    }:
        raise FleetCampaignError("incomplete_source_job_groups")
    if set(protocol_by_attempt) != {1, 2, 3, 4} or len(set(protocol_by_attempt.values())) != 4:
        raise FleetCampaignError("comparison_protocols_are_not_distinct_per_attempt")
    return value


def _binding(
    campaign_packet: Path, bindings_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], heldout_launch.Package, Path]:
    packet = _load(campaign_packet)
    bindings = _load_bindings(bindings_path)
    key = packet.get("experiment_key")
    cell = bindings["cells"].get(key)
    if (
        not isinstance(key, str)
        or not isinstance(cell, dict)
        or packet.get("matrix_sha256") != bindings["matrix_sha256"]
        or digest(packet.get("identity")) != key
        or digest(packet["identity"]) != cell["campaign_identity_sha256"]
    ):
        raise FleetCampaignError("campaign_cell_is_not_bound")
    identity = packet["identity"]
    model = bindings["model_bindings"][cell["model_id"]]
    expected_model = {
        field: model[field]
        for field in (
            "checkpoint_id",
            "weights_sha256",
        )
    } | {
        "id": cell["model_id"],
        "matched_treatment_receipt_sha256": bindings["matched_treatment_receipt_sha256"],
    }
    expected_benchmark = {
        "id": bindings["benchmark_id"],
        "task_set_sha256": bindings["task_set_sha256"],
        "harness": bindings["harness"],
        "scoring_protocol_sha256": bindings["scoring_protocol_sha256"],
    }
    if any(
        (
            identity.get("model") != expected_model,
            identity.get("benchmark") != expected_benchmark,
            identity.get("target")
            != {
                "id": cell["task_version_id"],
                "identity_receipt_sha256": bindings["target_identity_receipts"][
                    cell["task_version_id"]
                ],
            },
            identity.get("attempt") != cell["attempt"],
            identity.get("seed") != bindings["sampling"]["attempt_seeds"][cell["attempt"] - 1],
            identity.get("sampling")
            != {field: bindings["sampling"][field] for field in ("temperature", "top_p")},
            identity.get("budgets_sha256") != bindings["budgets_sha256"],
            packet.get("canary") is not cell["canary"],
            packet.get("serving_evidence")
            != {
                "serving_route_receipt_sha256": model["serving_route_receipt_sha256"],
                "live_parity_receipt_sha256": model["live_parity_receipt_sha256"],
            },
        )
    ):
        raise FleetCampaignError("campaign_cell_differs_from_fleet_binding")
    group = bindings["groups"][cell["group_id"]]
    launch_packet = _safe_relative(bindings_path.parent, group["launch_packet"])
    if _file_sha256(launch_packet) != group["launch_packet_sha256"]:
        raise FleetCampaignError("heldout_launch_packet_digest_changed")
    package = heldout_launch.build_package(launch_packet)
    selection = _load(package.packet.files["task_set"])
    comparison_protocol = _load(package.packet.files["comparison_protocol"])
    routes = package.evaluation_config["routes"]
    if len(routes) != 1 or not isinstance(selection.get("tasks"), list):
        raise FleetCampaignError("heldout_source_job_differs_from_eight_cell_group")
    route = next(iter(routes.values()))
    sampling = package.evaluation_config["sampling"]
    harness = package.evaluation_config.get("harness")
    images = package.evaluation_config.get("images")
    configured_models = package.evaluation_config.get("models")
    configured_model = (
        next(iter(configured_models.values()))
        if isinstance(configured_models, dict) and len(configured_models) == 1
        else {}
    )
    job_labels = package.job.get("metadata", {}).get("labels", {})
    if any(
        (
            package.packet.identity["arm_id"] != cell["model_id"],
            package.packet.identity["model_revision"] != model["model_revision"],
            package.packet.identity["comparison_protocol_sha256"]
            != group["comparison_protocol_sha256"],
            comparison_protocol.get("sha256") != group["comparison_protocol_sha256"],
            comparison_protocol.get("model_revisions")
            != {
                model_id: model_binding["model_revision"]
                for model_id, model_binding in bindings["model_bindings"].items()
            },
            package.packet.identity["harness"] != bindings["harness"]["name"],
            package.packet.identity.get("harness_version") != "1.18.27",
            package.packet.identity.get("context_management")
            != FINAL8_HARNESS["context_management"],
            package.packet.identity["checkpoint_provenance_sha256"]
            != model["checkpoint_provenance_sha256"],
            package.packet.identity["serving_route_proof_sha256"]
            != model["serving_route_proof_sha256"],
            set(package.packet.identity["comparison_arms"]) != set(bindings["model_bindings"]),
            package.packet.identity["pass_k"] != 1,
            package.packet.identity["retry_limit"] != 0,
            package.evaluation_config["pass_k"] != 1,
            package.evaluation_config["max_reviewed_infrastructure_retries"] != 0,
            package.evaluation_config.get("concurrency") != 8,
            harness != FINAL8_HARNESS,
            images != FINAL8_IMAGES,
            not isinstance(configured_models, dict) or len(configured_models) != 1,
            configured_model.get("revision") != model["model_revision"],
            route.get("served_id") != model["serving_model_id"],
            route.get("catalog")
            != {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 1},
            route.get("server_info", {}).get("context_length") != 262144,
            route.get("server_info", {}).get("dp_size") != 8,
            route.get("server_info", {}).get("tp_size") != 1,
            route.get("server_info", {}).get("kv_cache_dtype") != "fp8_e4m3",
            route.get("server_info", {}).get("quantization") is not None,
            route.get("server_info", {}).get("reasoning_parser") != "qwen3",
            route.get("server_info", {}).get("tool_call_parser") != "qwen3_coder",
            route.get("server_info", {}).get("load_balance_method") != "total_tokens",
            package.packet.file_sha256["split_manifest"] != FINAL8_SPLIT_FILE_SHA256,
            package.packet.identity["split_manifest_sha256"] != FINAL8_SPLIT_SHA256,
            selection.get("sha256") != FINAL8_SELECTION_SHA256,
            selection.get("selection_role") != "final_test",
            {
                row.get("task_version_id")
                for row in selection.get("tasks", [])
                if isinstance(row, dict)
            }
            != FINAL8_TASK_VERSIONS,
            package.packet.identity["sampling_seed"] != identity["seed"],
            sampling.get("seed") != identity["seed"],
            sampling.get("temperature") != identity["sampling"]["temperature"],
            sampling.get("top_p") != identity["sampling"]["top_p"],
            set(route["task_versions"]) != set(bindings["target_identity_receipts"]),
            not isinstance(job_labels, dict),
            job_labels.get(QUEUE_NAME_LABEL) != "training-lq",
            job_labels.get(QUEUE_PRIORITY_LABEL) != "q1",
        )
    ):
        raise FleetCampaignError("heldout_source_job_differs_from_eight_cell_group")
    return packet, bindings, cell, package, launch_packet


def _group_root(campaign_packet: Path, group_id: str) -> Path:
    # campaign state / targets / <key> / packet.json -> campaign state
    try:
        state = campaign_packet.resolve(strict=True).parents[2]
    except (OSError, IndexError) as exc:
        raise FleetCampaignError("campaign_packet_path_is_not_canonical") from exc
    return state / "fleet-source-jobs" / group_id


def _receipt(packet: dict[str, Any], phase: str, action: str, **fields: Any) -> dict[str, Any]:
    value = {
        "schema": RECEIPT_SCHEMA,
        "experiment_key": packet["experiment_key"],
        "phase": phase,
        "action": action,
        "provider": "fleet" if phase == "rollout" else "local",
        **fields,
    }
    return {**value, "receipt_sha256": digest(value)}


def _read_previous(
    path: Path,
    *,
    action: str,
    phase: str,
    experiment_key: str,
    statuses: set[str],
) -> dict[str, Any]:
    value = _load(path)
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("experiment_key") != experiment_key
        or value.get("action") != action
        or value.get("phase") != phase
        or value.get("status") not in statuses
    ):
        raise FleetCampaignError("previous_campaign_receipt_mismatch")
    if value.get("receipt_sha256") != digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        raise FleetCampaignError("previous_campaign_receipt_mismatch")
    return value


def _preview(package: heldout_launch.Package, cluster: heldout_launch.Cluster) -> dict[str, Any]:
    first = cluster.server_dry_run(package.packet.namespace, package.bundle)
    second = cluster.server_dry_run(package.packet.namespace, package.bundle)
    first_digest = heldout_launch._validate_server_preview(first, package)  # noqa: SLF001
    if heldout_launch._validate_server_preview(second, package) != first_digest:  # noqa: SLF001
        raise FleetCampaignError("fleet_server_preview_changed")
    roots = [
        item
        for item in first.get("items", [])
        if isinstance(item, dict) and item.get("kind") in {"Job", "RayJob"}
    ]
    if len(roots) != 1:
        raise FleetCampaignError("fleet_server_preview_root_count_changed")
    root = roots[0]
    return {
        "server_preview_sha256": first_digest,
        "rendered_objects": [
            {
                "apiVersion": root.get("apiVersion"),
                "kind": root.get("kind"),
                "metadata": {
                    "name": root.get("metadata", {}).get("name"),
                    "namespace": root.get("metadata", {}).get("namespace"),
                    "labels": {
                        QUEUE_NAME_LABEL: root.get("metadata", {})
                        .get("labels", {})
                        .get(QUEUE_NAME_LABEL),
                        QUEUE_PRIORITY_LABEL: root.get("metadata", {})
                        .get("labels", {})
                        .get(QUEUE_PRIORITY_LABEL),
                    },
                    "annotations": {
                        heldout_launch.FAILURE_ALERT_ANNOTATION: root.get("metadata", {})
                        .get("annotations", {})
                        .get(heldout_launch.FAILURE_ALERT_ANNOTATION)
                    },
                },
                "spec": {
                    "template": {
                        "spec": {
                            "priorityClassName": root.get("spec", {})
                            .get("template", {})
                            .get("spec", {})
                            .get("priorityClassName")
                        }
                    }
                },
            }
        ],
    }


def _terminal_file(
    path: Path,
    package: heldout_launch.Package,
    job_uid: str,
    config_map_uid: str,
) -> dict[str, Any]:
    value = _load(path)
    if (
        value.get("schema") != heldout_launch.TERMINAL_SCHEMA
        or value.get("sha256")
        != heldout_launch._canonical_digest(  # noqa: SLF001
            {key: item for key, item in value.items() if key != "sha256"}
        )
        or value.get("evaluation_identity_sha256") != package.packet.identity_sha256
        or value.get("job", {}).get("uid") != job_uid
        or value.get("config_map", {}).get("uid") != config_map_uid
        or value.get("privacy")
        != {
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "score_values_included": False,
            "credentials_included": False,
        }
    ):
        raise FleetCampaignError("invalid_fleet_terminal_receipt")
    return value


def _source_launch(path: Path, package: heldout_launch.Package, group_id: str) -> dict[str, Any]:
    value = _load(path)
    if (
        set(value)
        != {
            "schema",
            "group_id",
            "launch_packet_sha256",
            "job_name",
            "job_uid",
            "config_map_name",
            "config_map_uid",
        }
        or value.get("schema") != GROUP_LAUNCH_SCHEMA
        or value.get("group_id") != group_id
        or value.get("launch_packet_sha256") != _file_sha256(package.packet.path)
        or not UUID.fullmatch(str(value.get("job_uid")))
        or not UUID.fullmatch(str(value.get("config_map_uid")))
        or value.get("job_name") != package.packet.job_name
        or value.get("config_map_name") != package.packet.config_map_name
    ):
        raise FleetCampaignError("invalid_source_job_launch_record")
    return value


def _source_preview(path: Path, package: heldout_launch.Package) -> dict[str, Any]:
    value = _load(path)
    objects = value.get("rendered_objects")
    root = objects[0] if isinstance(objects, list) and len(objects) == 1 else None
    if (
        set(value)
        != {
            "schema",
            "launch_packet_sha256",
            "server_preview_sha256",
            "rendered_objects",
        }
        or value.get("schema") != GROUP_PREVIEW_SCHEMA
        or value.get("launch_packet_sha256") != _file_sha256(package.packet.path)
        or not SHA256.fullmatch(str(value.get("server_preview_sha256")))
        or not isinstance(root, dict)
        or root.get("apiVersion") != "batch/v1"
        or root.get("kind") != "Job"
        or root.get("metadata", {}).get("name") != package.packet.job_name
        or root.get("metadata", {}).get("namespace") != package.packet.namespace
        or root.get("metadata", {})
        .get("annotations", {})
        .get(heldout_launch.FAILURE_ALERT_ANNOTATION)
        != "off"
        or root.get("metadata", {}).get("labels", {}).get(QUEUE_NAME_LABEL) != "training-lq"
        or root.get("metadata", {}).get("labels", {}).get(QUEUE_PRIORITY_LABEL) != "q1"
        or root.get("spec", {}).get("template", {}).get("spec", {}).get("priorityClassName") != "c1"
    ):
        raise FleetCampaignError("invalid_source_job_preview_record")
    return value


def _campaign_cell_status(
    database: Any,
    database_name: str,
    *,
    dsn_env: str,
    task_version_id: str,
    model_revision: str,
) -> dict[str, Any]:
    """Read one complete score-blind cell without expanding shared DB APIs."""
    if hasattr(database, "campaign_cell_status"):
        value = database.campaign_cell_status(
            database_name,
            task_version_id=task_version_id,
            model_revision=model_revision,
            attempt=1,
        )
    else:
        original = urlsplit(os.environ.get(dsn_env, ""))
        query_keys = {
            key.casefold() for key, _ in parse_qsl(original.query, keep_blank_values=True)
        }
        if (
            heldout_launch.DATABASE_NAME.fullmatch(database_name) is None
            or original.scheme not in {"postgres", "postgresql"}
            or not original.netloc
            or original.fragment
            or query_keys & {"database", "dbname"}
        ):
            raise FleetCampaignError("database_cell_status_connection_is_invalid")
        dsn = urlunsplit(
            (
                original.scheme,
                original.netloc,
                "/" + quote(database_name, safe=""),
                original.query,
                "",
            )
        )
        from evals.fleet import rollout_postgres  # noqa: PLC0415

        with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
            rows = connection.execute(
                """
                SELECT cell_id, state, result_class, receipt_digest, failure_code,
                       reconciliation_digest, retry_count, max_retries
                FROM rollout_cells
                WHERE task_version_id = %s AND model_revision = %s AND attempt = 1
                LIMIT 2
                """,
                (task_version_id, model_revision),
            ).fetchall()
            if len(rows) != 1:
                raise FleetCampaignError("database_cell_status_is_not_unique")
            value = dict(rows[0])
            value["local_result_present"] = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM rollout_local_results WHERE cell_id = %s) "
                    "AS present",
                    (value["cell_id"],),
                ).fetchone()["present"]
            )
    expected = {
        "cell_id",
        "state",
        "result_class",
        "receipt_digest",
        "failure_code",
        "reconciliation_digest",
        "retry_count",
        "max_retries",
        "local_result_present",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or SHA256.fullmatch(str(value.get("cell_id"))) is None
        or any(
            type(value.get(field)) is not int or value[field] < 0
            for field in ("retry_count", "max_retries")
        )
        or type(value.get("local_result_present")) is not bool
    ):
        raise FleetCampaignError("database_cell_status_is_invalid")
    return value


def run_action(
    action: str,
    phase: str,
    campaign_packet: Path,
    receipt_path: Path,
    *,
    bindings_path: Path,
    context: str,
    preview_receipt: Path | None = None,
    readiness_receipt: Path | None = None,
    launch_receipt: Path | None = None,
    terminal_receipt: Path | None = None,
    dsn_env: str = "ROLLOUT_DATABASE_URL",
    cluster: heldout_launch.Cluster | None = None,
    database: Any | None = None,
    output_exists: Callable[[str], bool] = heldout_launch._output_exists,  # noqa: SLF001
) -> dict[str, Any]:
    """Run one adapter transition.  Only rollout/launch may create remote work."""
    if action not in {"preview", "ready", "launch", "observe"} or phase not in {
        "rollout",
        "score",
    }:
        raise FleetCampaignError("invalid_campaign_adapter_action")
    packet, _bindings, cell, package, _launch_packet = _binding(campaign_packet, bindings_path)
    group = _group_root(campaign_packet, cell["group_id"])
    group_launch_path = group / "LAUNCH.json"
    group_terminal_path = group / "TERMINAL_OBSERVATION.json"
    cluster = cluster or heldout_launch.KubectlCluster(context)
    database = database or heldout_launch.PostgresDatabase(dsn_env)

    if phase == "score":
        if action == "preview":
            if terminal_receipt is None:
                raise FleetCampaignError("rollout_terminal_receipt_is_required")
            rollout = _read_previous(
                terminal_receipt,
                action="observe",
                phase="rollout",
                experiment_key=packet["experiment_key"],
                statuses={"accepted"},
            )
            result = _receipt(
                packet,
                phase,
                action,
                status="accepted",
                collection_terminal_receipt_sha256=rollout["receipt_sha256"],
                native_grading_receipt_sha256=rollout["terminal_evidence_sha256"],
            )
        elif action == "ready":
            preview = _read_previous(
                preview_receipt or Path(),
                action="preview",
                phase=phase,
                experiment_key=packet["experiment_key"],
                statuses={"accepted"},
            )
            result = _receipt(
                packet,
                phase,
                action,
                status="ready",
                preview_receipt_sha256=preview["receipt_sha256"],
                collection_terminal_receipt_sha256=preview["collection_terminal_receipt_sha256"],
                native_grading_receipt_sha256=preview["native_grading_receipt_sha256"],
            )
        elif action == "launch":
            preview = _read_previous(
                preview_receipt or Path(),
                action="preview",
                phase=phase,
                experiment_key=packet["experiment_key"],
                statuses={"accepted"},
            )
            readiness = _read_previous(
                readiness_receipt or Path(),
                action="ready",
                phase=phase,
                experiment_key=packet["experiment_key"],
                statuses={"ready"},
            )
            source = _source_launch(group_launch_path, package, cell["group_id"])
            result = _receipt(
                packet,
                phase,
                action,
                status="created",
                remote_id=f"native-grade:{source['job_uid']}",
                preview_receipt_sha256=preview["receipt_sha256"],
                readiness_receipt_path=str(readiness_receipt or Path()),
                readiness_receipt_sha256=readiness["receipt_sha256"],
                collection_terminal_receipt_sha256=preview["collection_terminal_receipt_sha256"],
            )
        else:
            launch = _read_previous(
                launch_receipt or Path(),
                action="launch",
                phase=phase,
                experiment_key=packet["experiment_key"],
                statuses={"created"},
            )
            rollout = _read_previous(
                terminal_receipt or Path(),
                action="observe",
                phase="rollout",
                experiment_key=packet["experiment_key"],
                statuses={"accepted"},
            )
            result = _receipt(
                packet,
                phase,
                action,
                status="accepted",
                remote_id=launch["remote_id"],
                launch_receipt_sha256=launch["receipt_sha256"],
                collection_terminal_receipt_sha256=rollout["receipt_sha256"],
                terminal_evidence_sha256=rollout["terminal_evidence_sha256"],
            )
        _write_once(receipt_path, result)
        return result

    if action == "preview":
        group.mkdir(mode=0o700, parents=True, exist_ok=True)
        group_preview = group / "PREVIEW.json"
        if group_preview.exists():
            shared_preview = _source_preview(group_preview, package)
        else:
            preview = _preview(package, cluster)
            shared_preview = {
                "schema": GROUP_PREVIEW_SCHEMA,
                "launch_packet_sha256": _file_sha256(package.packet.path),
                **preview,
            }
            _write_once(group_preview, shared_preview)
        preview = {
            field: shared_preview[field] for field in ("server_preview_sha256", "rendered_objects")
        }
        result = _receipt(packet, phase, action, status="accepted", **preview)
    elif action == "ready":
        preview = _read_previous(
            preview_receipt or Path(),
            action="preview",
            phase=phase,
            experiment_key=packet["experiment_key"],
            statuses={"accepted"},
        )
        if group_launch_path.exists():
            _source_launch(group_launch_path, package, cell["group_id"])
            result = _receipt(
                packet,
                phase,
                action,
                status="ready",
                preview_receipt_sha256=preview["receipt_sha256"],
            )
        elif (
            packet["experiment_key"]
            != _bindings["groups"][cell["group_id"]]["leader_experiment_key"]
        ):
            result = _receipt(
                packet,
                phase,
                action,
                status="deferred_not_ready",
                defer_reason_code="dependency_not_ready",
                preview_receipt_sha256=preview["receipt_sha256"],
            )
        else:
            heldout_launch.duplicate_census(
                package, cluster=cluster, database=database, output_exists=output_exists
            )
            result = _receipt(
                packet,
                phase,
                action,
                status="ready",
                preview_receipt_sha256=preview["receipt_sha256"],
            )
    elif action == "launch":
        preview = _read_previous(
            preview_receipt or Path(),
            action="preview",
            phase=phase,
            experiment_key=packet["experiment_key"],
            statuses={"accepted"},
        )
        readiness = _read_previous(
            readiness_receipt or Path(),
            action="ready",
            phase=phase,
            experiment_key=packet["experiment_key"],
            statuses={"ready"},
        )
        group.mkdir(mode=0o700, parents=True, exist_ok=True)
        if group_launch_path.exists():
            source = _source_launch(group_launch_path, package, cell["group_id"])
        else:
            if (
                packet["experiment_key"]
                != _bindings["groups"][cell["group_id"]]["leader_experiment_key"]
            ):
                raise FleetCampaignError("only_group_leader_may_create_source_job")
            group_binding = _bindings["groups"][cell["group_id"]]
            created = heldout_launch.launch_once(
                package.packet.path,
                validated_package=package,
                expected_packet_sha256=group_binding["launch_packet_sha256"],
                cluster=cluster,
                database=database,
                journal=group / "CREATE_INTENT.jsonl",
                output_exists=output_exists,
            )
            source = {
                "schema": GROUP_LAUNCH_SCHEMA,
                "group_id": cell["group_id"],
                "launch_packet_sha256": _file_sha256(package.packet.path),
                "job_name": created["job_name"],
                "job_uid": created["job_uid"],
                "config_map_name": created["config_map_name"],
                "config_map_uid": created["config_map_uid"],
            }
            _write_once(group_launch_path, source)
        result = _receipt(
            packet,
            phase,
            action,
            status="created",
            remote_id=source["job_uid"],
            preview_receipt_sha256=preview["receipt_sha256"],
            readiness_receipt_path=str(readiness_receipt or Path()),
            readiness_receipt_sha256=readiness["receipt_sha256"],
        )
    else:
        launch = _read_previous(
            launch_receipt or Path(),
            action="launch",
            phase=phase,
            experiment_key=packet["experiment_key"],
            statuses={"created"},
        )
        source = _source_launch(group_launch_path, package, cell["group_id"])
        job = cluster.get("jobs.batch", package.packet.namespace, package.packet.job_name)
        if job.get("metadata", {}).get("uid") != source["job_uid"]:
            raise FleetCampaignError("source_job_uid_changed")
        conditions = job.get("status", {}).get("conditions", [])
        terminal = any(
            isinstance(condition, dict)
            and condition.get("status") == "True"
            and condition.get("type") in {"Complete", "Failed"}
            for condition in conditions
        )
        if not terminal:
            result = _receipt(
                packet,
                phase,
                action,
                status="running",
                remote_id=source["job_uid"],
                launch_receipt_sha256=launch["receipt_sha256"],
            )
        else:
            if group_terminal_path.exists():
                fleet_terminal = _terminal_file(
                    group_terminal_path,
                    package,
                    source["job_uid"],
                    source["config_map_uid"],
                )
            else:
                heldout_launch.collect_terminal(
                    package.packet.path,
                    cluster=cluster,
                    database=database,
                    receipt_path=group_terminal_path,
                    output_exists=output_exists,
                )
                fleet_terminal = _terminal_file(
                    group_terminal_path,
                    package,
                    source["job_uid"],
                    source["config_map_uid"],
                )
            cell_status = _campaign_cell_status(
                database,
                package.packet.database,
                dsn_env=dsn_env,
                task_version_id=cell["task_version_id"],
                model_revision=package.packet.identity["model_revision"],
            )
            accepted = (
                cell_status.get("state") == "accepted"
                and cell_status.get("result_class") == "valid"
                and cell_status.get("local_result_present") is True
                and cell_status.get("retry_count") == 0
                and cell_status.get("max_retries") == 0
                and SHA256.fullmatch(str(cell_status.get("receipt_digest"))) is not None
            )
            evidence = digest(
                {
                    "fleet_terminal_sha256": fleet_terminal["sha256"],
                    "task_version_id": cell["task_version_id"],
                    "model_revision": package.packet.identity["model_revision"],
                    "cell_status": cell_status,
                }
            )
            result = _receipt(
                packet,
                phase,
                action,
                status="accepted" if accepted else "infrastructure_invalid",
                remote_id=source["job_uid"],
                launch_receipt_sha256=launch["receipt_sha256"],
                terminal_evidence_sha256=evidence,
                fleet_terminal_receipt_sha256=fleet_terminal["sha256"],
            )
    _write_once(receipt_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preview", "ready", "launch", "observe"))
    parser.add_argument("phase", choices=("rollout", "score"))
    parser.add_argument("packet", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--preview-receipt", type=Path)
    parser.add_argument("--readiness-receipt", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--dsn-env", default="ROLLOUT_DATABASE_URL")
    args = parser.parse_args()
    result = run_action(
        args.action,
        args.phase,
        args.packet,
        args.receipt,
        bindings_path=args.bindings,
        context=args.context,
        preview_receipt=args.preview_receipt,
        readiness_receipt=args.readiness_receipt,
        launch_receipt=args.launch_receipt,
        terminal_receipt=args.terminal_receipt,
        dsn_env=args.dsn_env,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
