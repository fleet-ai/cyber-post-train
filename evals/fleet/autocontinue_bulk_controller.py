"""Materialize and execute the held corrected-treatment bulk shards.

The immutable campaign deliberately predates executable bulk plans.  This
module leaves those campaign bytes and the frozen legacy controller untouched.
It can materialize the eight bulk shard plans only after both one-cell canaries
have authoritative post-exit acceptance receipts and the inventory, parity,
controller, and launch-authorization gates all pass.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import hosted_sweep_controller as hosted

PACKAGE_SCHEMA = "fleet-opencode-autocontinue-bulk-held-package-v1"
HELD_AUTHORIZATION_SCHEMA = "fleet-opencode-autocontinue-bulk-held-authorization-v1"
AUTHORIZATION_SCHEMA = "fleet-opencode-autocontinue-bulk-authorization-v1"
OBSERVER_SCHEMA = "fleet-opencode-autocontinue-canary-post-exit-v1"
COMPATIBILITY_SCHEMA = "fleet-opencode-autocontinue-bulk-controller-compatibility-v1"
PLAN_SCHEMA = "fleet-hosted-opencode-task-boundary-shard-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-bulk-shard-release-v1"
CAMPAIGN_SHA = "sha256:63946f224a33eb0d2c2a6fdba34358ebf9ca379e5f0f156cd137c95a9b98097e"
MAPPING_SHA = "sha256:de30c778e4337c79dea921b7463aa74760da59f972c928927c4359e828f1f009"
TASK_IDENTITY_SHA = "sha256:c4eca5c58aa7e43eaf85ffb94179a57dbaa320a54a5e530c71795a1467e50fc7"
INVENTORY_EXPECTED_SHA = "sha256:2af86c5626001e44a6d3b540ec40c7efbe681781d495bf23c11b4e1352e7f82f"
INVENTORY_TERMINAL_SHA = "sha256:c71604d22b1d52a7091727b957e0f56f2e8e270e461c844fb57d48b392d831c5"
PARITY_SHA = "sha256:07e733b4b161ed7c36113fc3688f04a42938d93a499b1bcc10c7cd21342900df"
HOSTED_HEALTH_SHA = "sha256:d81a01ffe1087d7d85fe511bc9c978461ba3e24a2c15675871cfb8668c49bd85"
FLOCK_SHA = "sha256:e9f34d35ac3c9e60d2698ebc1685374b98f13e41ac305df0609fb051f35c416c"
FROZEN_CONTROLLER_SHA = "sha256:e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a"
CAMPAIGN_PATH = "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
EVIDENCE_PATHS = {
    "inventory_expected": (
        "evals/fleet/configs/q38-glm53-opencode-autocontinue-task-inventory-v1.json"
    ),
    "inventory_terminal": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-task-inventory-v2-terminal.json"
    ),
    "dedicated_parity": (
        "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-autocontinue-parity-v2-pass.json"
    ),
    "hosted_health": (
        "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-hosted-health-v1.json"
    ),
    "shared_pvc_flock": (
        "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-flock-release-gate-v1.json"
    ),
    "bulk_controller_compatibility": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-bulk-controller-compatibility-v1.json"
    ),
}
CONTEXT_POLICY = "opencode_1.18.27_native_compaction_autocontinue_v1"
LEASE_ROOT = "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1"
GLOBAL_CLAIM_ROOT = "/mnt/sfs/corrected-treatment-cell-claims/opencode11827-autocontinue-primary-v1"
PRIORITY = "fleet-train-high"
TOOL_NAMES = ["bash", "submit_report"]
TOOL_CATALOG_SHA = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
SCHEDULE = [
    {
        "accepted_outcomes_at_least": 0,
        "workers": 1,
        "headroom_gate": "fixed_at_launch_no_automatic_widening",
    }
]

CANARIES = {
    "qwen3.8-27b": {
        "plan_sha256": "sha256:54e4f8b0b9cdd3e4f7c72fabcb6b5d1f1e0154da1559d6e6989ff1facdeb2c2a",
        "source_rank": 4,
        "attempt": 1,
        "task_version_id": "02dd4e3f-d85d-4bf8-9976-eae2f102384d",
        "job_name": "chris-q38-ac-canary1-v1",
    },
    "glm-5.3": {
        "plan_sha256": "sha256:ab8b504b452974e334406524b9ffd8ead64a20954e6d1b381f987fdc3d47dc8c",
        "source_rank": 13,
        "attempt": 1,
        "task_version_id": "9375a9b9-04e5-4f6f-ad47-286121278992",
        "job_name": "chris-glm53-ac-canary1-v1",
    },
}

PARTITIONS = {
    "qwen-hosted-a99": {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "full": {6, 7, 8, 9, *range(11, 31)},
        "partial": {4: {2, 3, 4}},
        "tasks": 25,
        "cells": 99,
        "job": "chris-q38-ac-hosted-a99-v1",
    },
    "qwen-hosted-b100": {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "full": {*range(31, 51), 52, 53, 54, 55, 56},
        "partial": {},
        "tasks": 25,
        "cells": 100,
        "job": "chris-q38-ac-hosted-b100-v1",
    },
    "glm-hosted-a91": {
        "model": "glm-5.3",
        "serving_block": "glm-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "full": {*range(15, 58, 2)},
        "partial": {13: {2, 3, 4}},
        "tasks": 23,
        "cells": 91,
        "job": "chris-glm53-ac-hosted-a91-v1",
    },
    "glm-hosted-b92": {
        "model": "glm-5.3",
        "serving_block": "glm-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "full": {*range(59, 100, 2), 108, 109},
        "partial": {},
        "tasks": 23,
        "cells": 92,
        "job": "chris-glm53-ac-hosted-b92-v1",
    },
    "glm-dedicated-a52": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-a-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "full": {8, 10, 12, *range(16, 35, 2)},
        "partial": {},
        "tasks": 13,
        "cells": 52,
        "job": "chris-glm53-ac-ded-a52-v1",
    },
    "glm-dedicated-a56": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-a-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "full": {*range(36, 53, 2), 102, 104, 110, 112, 113},
        "partial": {},
        "tasks": 14,
        "cells": 56,
        "job": "chris-glm53-ac-ded-a56-v1",
    },
    "glm-dedicated-b52": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-b-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "full": {*range(58, 83, 2)},
        "partial": {},
        "tasks": 13,
        "cells": 52,
        "job": "chris-glm53-ac-ded-b52-v1",
    },
    "glm-dedicated-b56": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-b-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "full": {*range(84, 101, 2), 101, 103, 105, 107, 114},
        "partial": {},
        "tasks": 14,
        "cells": 56,
        "job": "chris-glm53-ac-ded-b56-v1",
    },
}

MODEL_SPECS = {
    "qwen3.8-27b": {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "served_id": "qwen3.8-27b",
        "session_model": (
            "fleet-cluster-opencode-1.18.27/qwen3.8-27b-opencode11827-autocontinue-v1"
        ),
        "settings_canonical_sha256": (
            "sha256:0458cabed67de9e2001661e61707fb16d91669774add3adbf4c97dd559fb5531"
        ),
        "settings_file_sha256": (
            "sha256:1328f6eb97861443b712625d9038a924de9c220c146667565dd4c5bb73a6d8f8"
        ),
    },
    "glm-5.3": {
        "repository": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
        "served_id": "glm-5.3",
        "session_model": ("fleet-cluster-opencode-1.18.27/glm-5.3-opencode11827-autocontinue-v1"),
        "settings_canonical_sha256": (
            "sha256:9614806454c86784e54fde31a8acc3f51adda11593fad3e762142cd04a4010f2"
        ),
        "settings_file_sha256": (
            "sha256:84a1ca763a297bd8badb184391a02c82d57069b9a211ecef4f5e53dd79e20e20"
        ),
    },
}

DEDICATED = {
    "glm-dedicated-a-v5-autocontinue-v1": {
        "replica": "A",
        "endpoint_origin": (
            "http://ft-run-98c32208-5gpb2-head-svc.fleet-train-jobs.svc.cluster.local:8000"
        ),
        "service_name": "ft-run-98c32208-5gpb2-head-svc",
        "serving_config_sha256": (
            "sha256:7815bf97cb23df73332c011cd86140bd0d87ab35ca1db4582a3452f8841d5875"
        ),
        "ray_job_uid": "ef7cb0f2-84d4-4017-ae31-bf34ebb70d0d",
        "ray_cluster_uid": "5107d72e-ae6a-4582-a490-55b349421d06",
        "service_uid": "2c0e64de-c4a0-4f70-ae07-d15b1adad0b3",
        "head_pod_uid": "3f37ab91-4afa-4e4d-8f29-a3eeda70a774",
    },
    "glm-dedicated-b-v5-autocontinue-v1": {
        "replica": "B",
        "endpoint_origin": (
            "http://ft-run-9e92209d-pppzg-head-svc.fleet-train-jobs.svc.cluster.local:8000"
        ),
        "service_name": "ft-run-9e92209d-pppzg-head-svc",
        "serving_config_sha256": (
            "sha256:f93b2f73d9d42fa16e72a9e1f9af638623fe514550602359b9d6a3906021967e"
        ),
        "ray_job_uid": "2bbfd34a-d74a-4497-ab28-48cada0c9753",
        "ray_cluster_uid": "f7f7ae34-6a79-482a-b771-4639b786f66a",
        "service_uid": "14ae906a-60ef-4a02-b6bf-814a00c2c89f",
        "head_pod_uid": "bfeb4c6a-9e4a-4f14-adb2-3f516dd25fdb",
    },
}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON file: {path}")
    value = json.loads(path.read_text(), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def _file_sha(path: Path) -> str:
    return self_hosted.sha256(path.read_bytes())


def _validate_self_digest(value: dict[str, Any], field: str, expected: str | None = None) -> None:
    digest = value.get(field)
    if digest != digest_without(value, field) or (expected is not None and digest != expected):
        raise ValueError(f"{field} mismatch")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)


def validate_compatibility(value: dict[str, Any], root: Path) -> None:
    _validate_self_digest(value, "receipt_sha256")
    implementation = value.get("implementation") or {}
    coverage = value.get("coverage") or {}
    if (
        value.get("schema_version") != COMPATIBILITY_SCHEMA
        or value.get("status") != "PASSED"
        or value.get("campaign_sha256") != CAMPAIGN_SHA
        or value.get("scientific_mapping_sha256") != MAPPING_SHA
        or value.get("task_identity_sha256") != TASK_IDENTITY_SHA
        or implementation.get("bulk_controller_path")
        != "evals/fleet/autocontinue_bulk_controller.py"
        or implementation.get("bulk_controller_sha256")
        != _file_sha(root / implementation["bulk_controller_path"])
        or implementation.get("frozen_controller_sha256") != FROZEN_CONTROLLER_SHA
        or coverage.get("all_eight_campaign_partitions") is not True
        or coverage.get("qwen_new_cells") != 199
        or coverage.get("glm_new_cells") != 399
        or coverage.get("legacy_credit") != 0
        or coverage.get("complete_task_boundaries") is not True
        or coverage.get("canary_credit_only_after_authoritative_post_exit_observer") is not True
        or coverage.get("endpoint_maximum_streams") != 2
        or coverage.get("global_atomic_corrected_treatment_cell_claim") is not True
        or coverage.get("preflight_and_scored_configmaps_distinct") is not True
        or coverage.get("uid_bound_preflight_post_exit_observer_required") is not True
        or coverage.get("required_tools") != TOOL_NAMES
        or coverage.get("context_management") != CONTEXT_POLICY
        or coverage.get("offline_tamper_tests_passed") is not True
        or value.get("launch_performed") is not False
        or value.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("bulk controller compatibility receipt drifted")


def _partition_cells(row: dict[str, Any]) -> set[tuple[int, int]]:
    cells = {
        (int(rank), attempt)
        for rank in row.get("full_source_ranks") or []
        for attempt in range(1, 5)
    }
    for partial in row.get("partial_source_rank_attempts") or []:
        rank = int(partial["source_rank"])
        additions = {(rank, int(attempt)) for attempt in partial.get("attempts") or []}
        if cells & additions:
            raise ValueError("partition repeats a cell")
        cells |= additions
    return cells


def validate_package(package: dict[str, Any], root: Path) -> None:
    _validate_self_digest(package, "package_sha256")
    if (
        package.get("schema_version") != PACKAGE_SCHEMA
        or package.get("status") != "HELD"
        or package.get("launch_authorized") is not False
        or package.get("campaign_sha256") != CAMPAIGN_SHA
        or package.get("scientific_mapping_sha256") != MAPPING_SHA
        or package.get("task_identity_sha256") != TASK_IDENTITY_SHA
        or package.get("campaign_path") != CAMPAIGN_PATH
        or package.get("legacy_credit") != 0
    ):
        raise ValueError("bulk held package envelope drifted")
    refs = package.get("evidence") or {}
    expected_refs = {
        "inventory_expected": INVENTORY_EXPECTED_SHA,
        "inventory_terminal": INVENTORY_TERMINAL_SHA,
        "dedicated_parity": PARITY_SHA,
        "hosted_health": HOSTED_HEALTH_SHA,
        "shared_pvc_flock": FLOCK_SHA,
    }
    for key, digest in expected_refs.items():
        ref = refs.get(key) or {}
        if ref.get("sha256") != digest or ref.get("path") != EVIDENCE_PATHS[key]:
            raise ValueError("bulk evidence reference drifted")
        value = load_object(root / ref["path"])
        digest_field = "expected_sha256" if key == "inventory_expected" else "receipt_sha256"
        _validate_self_digest(value, digest_field, digest)
    compatibility_ref = refs.get("bulk_controller_compatibility") or {}
    if (
        not _is_sha256(compatibility_ref.get("sha256"))
        or compatibility_ref.get("path") != EVIDENCE_PATHS["bulk_controller_compatibility"]
    ):
        raise ValueError("bulk compatibility reference drifted")
    compatibility = load_object(root / compatibility_ref["path"])
    _validate_self_digest(compatibility, "receipt_sha256", compatibility_ref["sha256"])
    validate_compatibility(compatibility, root)
    campaign = load_object(root / package["campaign_path"])
    _validate_self_digest(campaign, "campaign_sha256", CAMPAIGN_SHA)
    if campaign.get("task_identity_sha256") != TASK_IDENTITY_SHA:
        raise ValueError("bulk package campaign identity drifted")
    frozen = root / "evals/fleet/hosted_sweep_controller.py"
    if _file_sha(frozen) != FROZEN_CONTROLLER_SHA:
        raise ValueError("frozen controller bytes drifted")
    campaign_parts = {
        row["id"]: row for row in campaign.get("partitions") or [] if row.get("phase") != "canary"
    }
    if set(campaign_parts) != set(PARTITIONS) or set(package.get("partitions") or []) != set(
        PARTITIONS
    ):
        raise ValueError("bulk partition set drifted")
    totals = {"qwen3.8-27b": 0, "glm-5.3": 0}
    all_cells: dict[str, set[tuple[int, int]]] = {
        "qwen3.8-27b": set(),
        "glm-5.3": set(),
    }
    jobs: set[str] = set()
    for partition_id, expected in PARTITIONS.items():
        row = campaign_parts[partition_id]
        packaged = package["partitions"][partition_id]
        cells = _partition_cells(row)
        expected_cells = {
            *{(rank, attempt) for rank in expected["full"] for attempt in range(1, 5)},
            *{
                (rank, attempt)
                for rank, attempts in expected["partial"].items()
                for attempt in attempts
            },
        }
        if (
            cells != expected_cells
            or len(cells) != expected["cells"]
            or len({rank for rank, _ in cells}) != expected["tasks"]
            or row.get("model") != expected["model"]
            or row.get("serving_block") != expected["serving_block"]
            or row.get("scored_job_name") != expected["job"]
            or row.get("preflight_job_name") != expected["job"] + "-preflight"
            or row.get("sfs_root") != expected["job"]
            or row.get("workers") != 1
            or row.get("priority_class") != PRIORITY
            or row.get("create_once") is not True
            or row.get("launch_authorized") is not False
            or packaged
            != {
                "model": expected["model"],
                "serving_block": expected["serving_block"],
                "planned_task_count": expected["tasks"],
                "planned_cell_count": expected["cells"],
                "scored_job_name": expected["job"],
                "preflight_job_name": expected["job"] + "-preflight",
                "preflight_configmap_name": expected["job"] + "-preflight-cm",
                "scored_configmap_name": expected["job"] + "-scored-cm",
                "sfs_root": expected["job"],
                "endpoint_maximum_streams": 2,
                "required_priority_class": PRIORITY,
                "create_once": True,
                "launch_authorized": False,
            }
        ):
            raise ValueError("bulk partition binding drifted")
        if all_cells[expected["model"]] & cells:
            raise ValueError("bulk partitions overlap")
        all_cells[expected["model"]] |= cells
        totals[expected["model"]] += len(cells)
        jobs |= {expected["job"], expected["job"] + "-preflight"}
    if totals != {"qwen3.8-27b": 199, "glm-5.3": 399} or len(jobs) != 16:
        raise ValueError("bulk denominator drifted")
    privacy = package.get("privacy") or {}
    if privacy != {
        "credentials_included": False,
        "prompts_or_traces_included": False,
        "scores_included": False,
    }:
        raise ValueError("bulk package privacy drifted")


def validate_canary_observer(value: dict[str, Any], model: str) -> dict[str, Any]:
    _validate_self_digest(value, "receipt_sha256")
    expected = CANARIES[model]
    cell = value.get("cell") or {}
    job = value.get("job") or {}
    pod = value.get("pod") or {}
    evidence = value.get("evidence") or {}
    try:
        uuid.UUID(str(job.get("uid")))
        uuid.UUID(str(pod.get("uid")))
        uuid.UUID(str(cell.get("session_id")))
        uuid.UUID(str(cell.get("verifier_execution_id")))
    except ValueError as exc:
        raise ValueError("canary observer UUID binding drifted") from exc
    if (
        value.get("schema_version") != OBSERVER_SCHEMA
        or value.get("status") != "PASSED"
        or value.get("campaign_sha256") != CAMPAIGN_SHA
        or value.get("plan_sha256") != expected["plan_sha256"]
        or not _is_sha256(value.get("final_release_receipt_sha256"))
        or not _is_sha256(value.get("canary_terminal_receipt_sha256"))
        or value.get("bulk_release_eligible") is not True
        or job.get("name") != expected["job_name"]
        or job.get("succeeded") != 1
        or job.get("failed") != 0
        or not isinstance(pod.get("name"), str)
        or not pod.get("name")
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("restart_count") != 0
        or cell.get("source_rank") != expected["source_rank"]
        or cell.get("attempt") != 1
        or cell.get("task_version_id") != expected["task_version_id"]
        or cell.get("accepted") is not True
        or cell.get("credited") is not True
        or cell.get("retry_allowed") is not False
        or not isinstance(cell.get("run_id"), str)
        or not _is_sha256(cell.get("claim_sha256"))
        or not _is_sha256(cell.get("acceptance_receipt_sha256"))
        or evidence.get("exact_claim_count") != 1
        or evidence.get("exact_accepted_count") != 1
        or evidence.get("exact_session_count") != 1
        or evidence.get("exact_verifier_execution_count") != 1
        or evidence.get("claim_sha256") != cell.get("claim_sha256")
        or evidence.get("acceptance_receipt_sha256") != cell.get("acceptance_receipt_sha256")
        or evidence.get("session_id") != cell.get("session_id")
        or evidence.get("verifier_execution_id") != cell.get("verifier_execution_id")
        or evidence.get("final_release_receipt_sha256") != value.get("final_release_receipt_sha256")
        or evidence.get("canary_terminal_receipt_sha256")
        != value.get("canary_terminal_receipt_sha256")
        or evidence.get("quarantine_count") != 0
        or evidence.get("session_ingest_completed") is not True
        or evidence.get("cleanup_completed") is not True
        or value.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("canary post-exit observer drifted")
    return value


def validate_held_authorization(value: dict[str, Any], package: dict[str, Any]) -> None:
    _validate_self_digest(value, "receipt_sha256")
    gates = value.get("gates") or {}
    if (
        value.get("schema_version") != HELD_AUTHORIZATION_SCHEMA
        or value.get("status") != "HELD"
        or value.get("campaign_sha256") != CAMPAIGN_SHA
        or value.get("package_sha256") != package["package_sha256"]
        or value.get("launch_authorized") is not False
        or gates.get("qwen_canary_authoritative_post_exit_acceptance") is not False
        or gates.get("glm_canary_authoritative_post_exit_acceptance") is not False
        or gates.get("append_only_bulk_authorization") is not False
        or gates.get("fresh_inventory_receipt_sha256") != INVENTORY_TERMINAL_SHA
        or gates.get("dedicated_parity_receipt_sha256") != PARITY_SHA
        or gates.get("bulk_controller_compatibility_receipt_sha256")
        != package["evidence"]["bulk_controller_compatibility"]["sha256"]
        or value.get("planned_new_sessions") != {"qwen3.8-27b": 199, "glm-5.3": 399}
        or value.get("legacy_credit") != 0
        or value.get("privacy") != package["privacy"]
    ):
        raise ValueError("held bulk authorization drifted")


def authorization_statement(
    package: dict[str, Any], qwen: dict[str, Any], glm: dict[str, Any]
) -> str:
    return (
        "Authorize create-once corrected-treatment bulk materialization for package "
        f"{package['package_sha256']} after Qwen observer {qwen['receipt_sha256']} "
        f"and GLM observer {glm['receipt_sha256']}; launch remains per-shard preflight gated."
    )


def validate_authorization(
    value: dict[str, Any], package: dict[str, Any], qwen: dict[str, Any], glm: dict[str, Any]
) -> None:
    _validate_self_digest(value, "receipt_sha256")
    evidence = value.get("evidence") or {}
    authorization = value.get("authorization") or {}
    if (
        value.get("schema_version") != AUTHORIZATION_SCHEMA
        or value.get("append_only") is not True
        or value.get("status") != "RELEASED"
        or value.get("campaign_sha256") != CAMPAIGN_SHA
        or value.get("package_sha256") != package["package_sha256"]
        or evidence.get("inventory_expected_sha256") != INVENTORY_EXPECTED_SHA
        or evidence.get("inventory_terminal_receipt_sha256") != INVENTORY_TERMINAL_SHA
        or evidence.get("dedicated_parity_receipt_sha256") != PARITY_SHA
        or evidence.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or evidence.get("shared_pvc_flock_receipt_sha256") != FLOCK_SHA
        or evidence.get("qwen_canary_observer_receipt_sha256") != qwen["receipt_sha256"]
        or evidence.get("glm_canary_observer_receipt_sha256") != glm["receipt_sha256"]
        or evidence.get("qwen_canary_session_id") != qwen["cell"]["session_id"]
        or evidence.get("qwen_canary_verifier_execution_id")
        != qwen["cell"]["verifier_execution_id"]
        or evidence.get("glm_canary_session_id") != glm["cell"]["session_id"]
        or evidence.get("glm_canary_verifier_execution_id") != glm["cell"]["verifier_execution_id"]
        or evidence.get("bulk_controller_compatibility_receipt_sha256")
        != package["evidence"]["bulk_controller_compatibility"]["sha256"]
        or authorization.get("launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("must_not_repeat") is not True
        or authorization.get("required_priority_class") != PRIORITY
        or authorization.get("author") != "/root"
        or authorization.get("statement") != authorization_statement(package, qwen, glm)
        or value.get("privacy") != package["privacy"]
    ):
        raise ValueError("bulk launch authorization drifted")


def _inventory_rows(package: dict[str, Any], root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    expected_ref = package["evidence"]["inventory_expected"]
    terminal_ref = package["evidence"]["inventory_terminal"]
    expected = load_object(root / expected_ref["path"])
    terminal = load_object(root / terminal_ref["path"])
    _validate_self_digest(expected, "expected_sha256", INVENTORY_EXPECTED_SHA)
    _validate_self_digest(terminal, "receipt_sha256", INVENTORY_TERMINAL_SHA)
    if (
        terminal.get("schema_version") != "fleet-opencode-autocontinue-task-inventory-terminal-v1"
        or terminal.get("status") != "PASSED"
        or terminal.get("campaign_sha256") != CAMPAIGN_SHA
        or terminal.get("scientific_mapping_sha256") != MAPPING_SHA
        or terminal.get("task_identity_sha256") != TASK_IDENTITY_SHA
        or terminal.get("expected_inventory_sha256") != INVENTORY_EXPECTED_SHA
        or terminal.get("binding_mismatch_count") != 0
        or (terminal.get("request_counts") or {}).get("model_or_scoring_calls") != 0
        or (terminal.get("request_counts") or {}).get("post_put_patch_delete") != 0
    ):
        raise ValueError("fresh task inventory gate drifted")
    rows = {(row["model"], int(row["source_rank"])): row for row in expected.get("tasks") or []}
    if len(rows) != 150:
        raise ValueError("fresh task inventory row set drifted")
    return rows


def _task(row: dict[str, Any], rank: int) -> dict[str, Any]:
    content = row["task_content_sha256"]
    return {
        "rank": rank,
        "source_rank": int(row["source_rank"]),
        "environment": copy.deepcopy(row["environment"]),
        "task": {
            "key": row["task_key"],
            "version_id": row["task_version_id"],
            "prompt_sha256": content["prompt"],
            "env_variables_sha256": content["env_variables"],
            "output_json_schema_sha256": content["output_json_schema"],
            "cyber_contract": copy.deepcopy(row["cyber_contract"]),
        },
        "verifier": copy.deepcopy(row["verifier"]),
    }


def _harness(model: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    return {
        "name": "opencode",
        "version": "1.18.27",
        "release_asset_sha256": (
            "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
        ),
        "provider_adapter": "@ai-sdk/openai-compatible",
        "context_management": CONTEXT_POLICY,
        "context_window_size": 262144,
        "compaction_headroom_tokens": 20000,
        "max_model_requests": 600,
        "max_output_tokens": 32768,
        "timeout_seconds": 28800,
        "settings_canonical_sha256": spec["settings_canonical_sha256"],
        "settings_file_sha256": spec["settings_file_sha256"],
    }


def _model(model: str, serving_block: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    endpoint = DEDICATED.get(serving_block, {}).get(
        "endpoint_origin", "https://inference.flt.build"
    )
    return {"endpoint_origin": endpoint, **copy.deepcopy(spec)}


def _treatment(model: str, serving_block: str, harness: dict[str, Any]) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    base = {
        "kind": (
            "dedicated_inference_endpoint_v1"
            if serving_block in DEDICATED
            else "hosted_inference_endpoint_v1"
        ),
        "serving_block": serving_block,
        "campaign_sha256": CAMPAIGN_SHA,
        "endpoint_origin": _model(model, serving_block)["endpoint_origin"],
        "served_id": spec["served_id"],
        "model_revision": spec["revision"],
        "session_model": spec["session_model"],
        "harness": copy.deepcopy(harness),
        "required_task_tools": TOOL_NAMES,
        "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA,
    }
    if serving_block in DEDICATED:
        base.update(copy.deepcopy(DEDICATED[serving_block]))
        base.update(
            {
                "runtime_image_digest": (
                    "ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@"
                    "sha256:ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec"
                ),
                "serving_generation": "v5",
                "new_treatment_parity_receipt_sha256": PARITY_SHA,
            }
        )
    return base


def build_plan(
    partition_id: str,
    package: dict[str, Any],
    rows: dict[tuple[str, int], dict[str, Any]],
    qwen_observer: dict[str, Any],
    glm_observer: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    expected = PARTITIONS[partition_id]
    model = expected["model"]
    observer = qwen_observer if model == "qwen3.8-27b" else glm_observer
    source_ranks = sorted({*expected["full"], *expected["partial"]})
    tasks = [
        _task(rows[(model, source_rank)], rank) for rank, source_rank in enumerate(source_ranks, 1)
    ]
    rank_by_source = {int(row["source_rank"]): int(row["rank"]) for row in tasks}
    attempts: list[dict[str, Any]] = []
    for source_rank in source_ranks:
        wanted = expected["partial"].get(source_rank, {1, 2, 3, 4})
        task = tasks[rank_by_source[source_rank] - 1]
        suffix = task["task"]["version_id"][:8]
        for attempt in sorted(wanted):
            attempts.append(
                {
                    "rank": rank_by_source[source_rank],
                    "source_rank": source_rank,
                    "attempt": attempt,
                    "ordinal": len(attempts) + 1,
                    "run_id": f"{expected['job']}-sr{source_rank:03d}-a{attempt}-{suffix}",
                    "network": f"{partition_id}-sr{source_rank:03d}-a{attempt}-{suffix}",
                }
            )
    credits: list[dict[str, Any]] = []
    if expected["partial"]:
        cell = observer["cell"]
        source_rank = int(cell["source_rank"])
        credits.append(
            {
                "rank": rank_by_source[source_rank],
                "source_rank": source_rank,
                "attempt": 1,
                "session_id": cell["session_id"],
                "verifier_execution_id": cell["verifier_execution_id"],
                "source_run_id": cell["run_id"],
                "classification": "ACCEPTED",
                "source_receipt_sha256": cell["acceptance_receipt_sha256"],
                "canary_observer_receipt_sha256": observer["receipt_sha256"],
            }
        )
    harness = _harness(model)
    inventory_policy = (
        "immutable_plan_claim_and_endpoint_uid_v1"
        if expected["serving_block"] in DEDICATED
        else "conservative_no_same_model_session_for_task_key_v1"
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": partition_id,
        "campaign_id": expected["job"],
        "source_job_id": expected["job"],
        "serving_block": expected["serving_block"],
        "preflight_job_name": expected["job"] + "-preflight",
        "scored_job_name": expected["job"],
        "preflight_configmap_name": expected["job"] + "-preflight-cm",
        "scored_configmap_name": expected["job"] + "-scored-cm",
        "sfs_root": expected["job"],
        "source": {
            "campaign_sha256": CAMPAIGN_SHA,
            "scientific_mapping_sha256": MAPPING_SHA,
            "task_identity_sha256": TASK_IDENTITY_SHA,
            "package_sha256": package["package_sha256"],
            "bulk_authorization_receipt_sha256": authorization["receipt_sha256"],
            "inventory_expected_sha256": INVENTORY_EXPECTED_SHA,
            "inventory_terminal_receipt_sha256": INVENTORY_TERMINAL_SHA,
            "dedicated_parity_receipt_sha256": PARITY_SHA,
            "hosted_health_receipt_sha256": HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": FLOCK_SHA,
            "bulk_controller_compatibility_receipt_sha256": package["evidence"][
                "bulk_controller_compatibility"
            ]["sha256"],
            "qwen_canary_observer_receipt_sha256": qwen_observer["receipt_sha256"],
            "glm_canary_observer_receipt_sha256": glm_observer["receipt_sha256"],
        },
        "model": _model(model, expected["serving_block"]),
        "harness": harness,
        "treatment_block": _treatment(model, expected["serving_block"], harness),
        "authority": {
            "multi_app_aggregation_mode": "fractional",
            "provisioning_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
            ),
            "scoring_mode": "partial",
            "scoring_route_template": ("/v1/rollout-rewards/{task_key}/versions/{task_version_id}"),
            "required_cyber_contract": {
                "evidence_schema": "1.0.0",
                "submission_protocol": "2.0.0",
                "verifier_contract": "3.0.0",
            },
        },
        "task_count": expected["tasks"],
        "pass_k": 4,
        "total_session_count": expected["tasks"] * 4,
        "new_session_count": expected["cells"],
        "credited_sessions": credits,
        "legacy_credited_sessions": 0,
        "execution": {
            "task_partition": "complete_task_boundary_after_exact_canary_credit_v1",
            "same_task_max_inflight": 1,
            "retry_policy": "never_repeat_any_verifier_backed_outcome",
            "future_nonzero_exit_policy": (
                "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
            ),
            "infrastructure_failure_policy": ("quarantine_only_exact_cell_no_automatic_retry"),
            "training_data_eligible": True,
            "required_task_tools": TOOL_NAMES,
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA,
            "score_blind_concurrency_schedule": SCHEDULE,
            "inventory_policy": inventory_policy,
            "required_priority_class": PRIORITY,
            "true_non_preemptible_available": False,
            "priority_class_is_not_preemption_immunity": True,
            "launch_authorized": False,
            "global_cell_claim": {
                "claim_root": GLOBAL_CLAIM_ROOT,
                "key_fields": [
                    "campaign_sha256",
                    "serving_block",
                    "model_revision",
                    "session_model",
                    "task_version_id",
                    "attempt",
                ],
                "atomicity": "task_flock_then_o_excl_per_cell_v1",
                "before_model_or_task_instance_side_effect": True,
            },
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": expected["serving_block"],
                "maximum_streams": 2,
            },
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": {
            "credentials_included": False,
            "prompts_included": False,
            "scores_included": False,
            "transcripts_included": False,
        },
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    _validate_self_digest(plan, "plan_sha256")
    expected = PARTITIONS.get(plan.get("shard_key"))
    if expected is None:
        raise ValueError("bulk shard identity drifted")
    tasks = plan.get("tasks") or []
    attempts = plan.get("attempts") or []
    credits = plan.get("credited_sessions") or []
    execution = plan.get("execution") or {}
    model = plan.get("model") or {}
    harness = plan.get("harness") or {}
    treatment = plan.get("treatment_block") or {}
    source = plan.get("source") or {}
    source_ranks = sorted({*expected["full"], *expected["partial"]})
    cells = {(int(row["source_rank"]), int(row["attempt"])) for row in attempts}
    expected_cells = {
        *{(rank, attempt) for rank in expected["full"] for attempt in range(1, 5)},
        *{(rank, attempt) for rank, values in expected["partial"].items() for attempt in values},
    }
    credited_cells = {(int(row["source_rank"]), int(row["attempt"])) for row in credits}
    expected_credits = {(rank, 1) for rank in expected["partial"]}
    spec = MODEL_SPECS[expected["model"]]
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("campaign_id") != expected["job"]
        or plan.get("source_job_id") != expected["job"]
        or plan.get("serving_block") != expected["serving_block"]
        or plan.get("preflight_job_name") != expected["job"] + "-preflight"
        or plan.get("scored_job_name") != expected["job"]
        or plan.get("preflight_configmap_name") != expected["job"] + "-preflight-cm"
        or plan.get("scored_configmap_name") != expected["job"] + "-scored-cm"
        or plan.get("preflight_configmap_name") == plan.get("scored_configmap_name")
        or plan.get("sfs_root") != expected["job"]
        or plan.get("task_count") != expected["tasks"]
        or plan.get("pass_k") != 4
        or plan.get("total_session_count") != expected["tasks"] * 4
        or plan.get("new_session_count") != expected["cells"]
        or plan.get("legacy_credited_sessions") != 0
        or [int(row["rank"]) for row in tasks] != list(range(1, expected["tasks"] + 1))
        or [int(row["source_rank"]) for row in tasks] != source_ranks
        or cells != expected_cells
        or len(cells) != len(attempts)
        or len(attempts) != expected["cells"]
        or credited_cells != expected_credits
        or [int(row["ordinal"]) for row in attempts] != list(range(1, len(attempts) + 1))
        or len({row["run_id"] for row in attempts}) != len(attempts)
        or source.get("campaign_sha256") != CAMPAIGN_SHA
        or source.get("scientific_mapping_sha256") != MAPPING_SHA
        or source.get("task_identity_sha256") != TASK_IDENTITY_SHA
        or source.get("inventory_expected_sha256") != INVENTORY_EXPECTED_SHA
        or source.get("inventory_terminal_receipt_sha256") != INVENTORY_TERMINAL_SHA
        or source.get("dedicated_parity_receipt_sha256") != PARITY_SHA
        or source.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or source.get("shared_pvc_flock_receipt_sha256") != FLOCK_SHA
        or not _is_sha256(source.get("bulk_controller_compatibility_receipt_sha256"))
        or model.get("repository") != spec["repository"]
        or model.get("revision") != spec["revision"]
        or model.get("served_id") != spec["served_id"]
        or model.get("session_model") != spec["session_model"]
        or harness.get("context_management") != CONTEXT_POLICY
        or harness.get("context_window_size") != 262144
        or harness.get("compaction_headroom_tokens") != 20000
        or harness.get("max_output_tokens") != 32768
        or harness.get("settings_canonical_sha256") != spec["settings_canonical_sha256"]
        or harness.get("settings_file_sha256") != spec["settings_file_sha256"]
        or execution.get("task_partition") != "complete_task_boundary_after_exact_canary_credit_v1"
        or execution.get("same_task_max_inflight") != 1
        or execution.get("score_blind_concurrency_schedule") != SCHEDULE
        or execution.get("required_task_tools") != TOOL_NAMES
        or execution.get("required_task_tool_catalog_sha256") != TOOL_CATALOG_SHA
        or execution.get("required_priority_class") != PRIORITY
        or execution.get("true_non_preemptible_available") is not False
        or execution.get("priority_class_is_not_preemption_immunity") is not True
        or execution.get("launch_authorized") is not False
        or execution.get("global_cell_claim")
        != {
            "claim_root": GLOBAL_CLAIM_ROOT,
            "key_fields": [
                "campaign_sha256",
                "serving_block",
                "model_revision",
                "session_model",
                "task_version_id",
                "attempt",
            ],
            "atomicity": "task_flock_then_o_excl_per_cell_v1",
            "before_model_or_task_instance_side_effect": True,
        }
        or execution.get("endpoint_lease")
        != {
            "lease_root": LEASE_ROOT,
            "endpoint_key": expected["serving_block"],
            "maximum_streams": 2,
        }
        or treatment.get("serving_block") != expected["serving_block"]
        or treatment.get("harness") != harness
        or treatment.get("required_task_tools") != TOOL_NAMES
        or treatment.get("required_task_tool_catalog_sha256") != TOOL_CATALOG_SHA
        or plan.get("privacy")
        != {
            "credentials_included": False,
            "prompts_included": False,
            "scores_included": False,
            "transcripts_included": False,
        }
    ):
        raise ValueError("bulk shard plan drifted")
    all_cells = cells | credited_cells
    wanted = {(rank, attempt) for rank in source_ranks for attempt in range(1, 5)}
    if all_cells != wanted or len(all_cells) != expected["tasks"] * 4:
        raise ValueError("bulk shard Cartesian coverage drifted")
    expected_policy = (
        "immutable_plan_claim_and_endpoint_uid_v1"
        if expected["serving_block"] in DEDICATED
        else "conservative_no_same_model_session_for_task_key_v1"
    )
    if execution.get("inventory_policy") != expected_policy:
        raise ValueError("bulk inventory policy drifted")
    if expected["serving_block"] in DEDICATED:
        runtime = DEDICATED[expected["serving_block"]]
        if (
            treatment.get("kind") != "dedicated_inference_endpoint_v1"
            or any(treatment.get(key) != value for key, value in runtime.items())
            or treatment.get("new_treatment_parity_receipt_sha256") != PARITY_SHA
        ):
            raise ValueError("bulk dedicated treatment drifted")
    elif (
        treatment.get("kind") != "hosted_inference_endpoint_v1"
        or model.get("endpoint_origin") != "https://inference.flt.build"
    ):
        raise ValueError("bulk hosted treatment drifted")


def _global_claim_identity(
    plan: dict[str, Any], task: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    return {
        "campaign_sha256": CAMPAIGN_SHA,
        "serving_block": plan["serving_block"],
        "model_revision": plan["model"]["revision"],
        "session_model": plan["model"]["session_model"],
        "task_version_id": task["task"]["version_id"],
        "attempt": int(item["attempt"]),
    }


def _global_claim_path(
    plan: dict[str, Any],
    task: dict[str, Any],
    item: dict[str, Any],
    claim_root: Path,
) -> Path:
    digest = self_hosted.sha256(
        self_hosted.canonical_json(_global_claim_identity(plan, task, item))
    ).removeprefix("sha256:")
    return claim_root / "cells" / f"{digest}.json"


def _global_task_lock_path(plan: dict[str, Any], task: dict[str, Any], claim_root: Path) -> Path:
    identity = _global_claim_identity(
        plan,
        task,
        {"attempt": 0},
    )
    identity.pop("attempt")
    digest = self_hosted.sha256(self_hosted.canonical_json(identity)).removeprefix("sha256:")
    return claim_root / "task-locks" / f"{digest}.lock"


def _prepare_global_claim_root(claim_root: Path) -> None:
    if claim_root.is_symlink():
        raise RuntimeError("global corrected-treatment claim root is a symlink")
    claim_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("cells", "task-locks"):
        path = claim_root / name
        if path.is_symlink():
            raise RuntimeError("global corrected-treatment claim directory is a symlink")
        path.mkdir(exist_ok=True, mode=0o700)


def _assert_global_cells_absent(
    plan: dict[str, Any], task: dict[str, Any], items: list[dict[str, Any]], claim_root: Path
) -> None:
    if any(_global_claim_path(plan, task, item, claim_root).exists() for item in items):
        raise RuntimeError("global corrected-treatment cell identity already claimed")


def _claim_global_task_cells(
    plan: dict[str, Any], task: dict[str, Any], items: list[dict[str, Any]], claim_root: Path
) -> list[Path]:
    """Atomically reserve all cells of one task before any scored side effect."""

    _prepare_global_claim_root(claim_root)
    lock_path = _global_task_lock_path(plan, task, claim_root)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _assert_global_cells_absent(plan, task, items, claim_root)
        paths: list[Path] = []
        for item in items:
            path = _global_claim_path(plan, task, item, claim_root)
            claim = {
                "schema_version": "fleet-corrected-treatment-global-cell-claim-v1",
                "identity": _global_claim_identity(plan, task, item),
                "plan_sha256": plan["plan_sha256"],
                "partition_id": plan["shard_key"],
                "run_id": item["run_id"],
                "source_rank": int(task["source_rank"]),
                "task_key": task["task"]["key"],
                "retry_allowed": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            claim["claim_sha256"] = digest_without(claim, "claim_sha256")
            self_hosted.write_json_once(path, claim)
            paths.append(path)
        return paths


def _build_shard_release(
    plan: dict[str, Any],
    authorization: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_observer: dict[str, Any],
    module_sha: str,
) -> dict[str, Any]:
    release = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "campaign_sha256": CAMPAIGN_SHA,
        "partition_id": plan["shard_key"],
        "plan_sha256": plan["plan_sha256"],
        "bulk_authorization_receipt_sha256": authorization["receipt_sha256"],
        "preflight_receipt_sha256": preflight_receipt["receipt_sha256"],
        "preflight_observer_receipt_sha256": preflight_observer["receipt_sha256"],
        "preflight_identity": {
            "job_name": plan["preflight_job_name"],
            "job_uid": preflight_observer["job"]["uid"],
            "pod_name": preflight_observer["pod"]["name"],
            "pod_uid": preflight_observer["pod"]["uid"],
            "configmap_name": plan["preflight_configmap_name"],
            "configmap_uid": preflight_observer["configmap"]["uid"],
        },
        "stage_configmaps": {
            "preflight": plan["preflight_configmap_name"],
            "scored": plan["scored_configmap_name"],
            "distinct": True,
            "create_once": True,
        },
        "evidence": copy.deepcopy(authorization["evidence"]),
        "implementation": {
            "bulk_controller_sha256": module_sha,
            "frozen_controller_sha256": FROZEN_CONTROLLER_SHA,
        },
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "must_not_repeat": True,
            "required_priority_class": PRIORITY,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    release["receipt_sha256"] = digest_without(release, "receipt_sha256")
    return release


def validate_release(
    release: dict[str, Any],
    plan: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_observer: dict[str, Any],
    repo: Path,
) -> None:
    validate_plan(plan)
    _validate_preflight(preflight_receipt, plan)
    _validate_preflight_observer(preflight_observer, preflight_receipt, plan)
    _validate_self_digest(release, "receipt_sha256")
    module_sha = _file_sha(repo / "evals/fleet/autocontinue_bulk_controller.py")
    evidence = release.get("evidence") or {}
    source = plan.get("source") or {}
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or release.get("campaign_sha256") != CAMPAIGN_SHA
        or release.get("partition_id") != plan["shard_key"]
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("bulk_authorization_receipt_sha256")
        != plan["source"]["bulk_authorization_receipt_sha256"]
        or release.get("preflight_receipt_sha256") != preflight_receipt["receipt_sha256"]
        or release.get("preflight_observer_receipt_sha256") != preflight_observer["receipt_sha256"]
        or release.get("preflight_identity")
        != {
            "job_name": plan["preflight_job_name"],
            "job_uid": preflight_observer["job"]["uid"],
            "pod_name": preflight_observer["pod"]["name"],
            "pod_uid": preflight_observer["pod"]["uid"],
            "configmap_name": plan["preflight_configmap_name"],
            "configmap_uid": preflight_observer["configmap"]["uid"],
        }
        or release.get("stage_configmaps")
        != {
            "preflight": plan["preflight_configmap_name"],
            "scored": plan["scored_configmap_name"],
            "distinct": True,
            "create_once": True,
        }
        or (release.get("implementation") or {}).get("bulk_controller_sha256") != module_sha
        or (release.get("implementation") or {}).get("frozen_controller_sha256")
        != FROZEN_CONTROLLER_SHA
        or evidence.get("inventory_expected_sha256") != source.get("inventory_expected_sha256")
        or evidence.get("inventory_terminal_receipt_sha256")
        != source.get("inventory_terminal_receipt_sha256")
        or evidence.get("dedicated_parity_receipt_sha256")
        != source.get("dedicated_parity_receipt_sha256")
        or evidence.get("hosted_health_receipt_sha256")
        != source.get("hosted_health_receipt_sha256")
        or evidence.get("shared_pvc_flock_receipt_sha256")
        != source.get("shared_pvc_flock_receipt_sha256")
        or evidence.get("bulk_controller_compatibility_receipt_sha256")
        != source.get("bulk_controller_compatibility_receipt_sha256")
        or evidence.get("qwen_canary_observer_receipt_sha256")
        != source.get("qwen_canary_observer_receipt_sha256")
        or evidence.get("glm_canary_observer_receipt_sha256")
        != source.get("glm_canary_observer_receipt_sha256")
        or release.get("authorization")
        != {
            "launch_authorized": True,
            "create_once": True,
            "must_not_repeat": True,
            "required_priority_class": PRIORITY,
        }
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("bulk shard release drifted")


def materialize(
    package: dict[str, Any],
    authorization: dict[str, Any],
    qwen_observer: dict[str, Any],
    glm_observer: dict[str, Any],
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    validate_package(package, root)
    qwen = validate_canary_observer(qwen_observer, "qwen3.8-27b")
    glm = validate_canary_observer(glm_observer, "glm-5.3")
    validate_authorization(authorization, package, qwen, glm)
    rows = _inventory_rows(package, root)
    rendered: dict[str, dict[str, Any]] = {}
    plan_digests: dict[str, str] = {}
    for partition_id in PARTITIONS:
        plan = build_plan(partition_id, package, rows, qwen, glm, authorization)
        rendered[partition_id] = plan
        plan_digests[partition_id] = plan["plan_sha256"]
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(mode=0o700)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(mode=0o700)
    for partition_id, plan in rendered.items():
        self_hosted.write_json_once(plans_dir / f"{partition_id}.json", plan)
    receipt = {
        "schema_version": "fleet-opencode-autocontinue-bulk-materialization-v1",
        "campaign_sha256": CAMPAIGN_SHA,
        "package_sha256": package["package_sha256"],
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "plan_digests": plan_digests,
        "release_digests": {},
        "scored_releases_created": 0,
        "scored_releases_require_uid_bound_preflight_post_exit_observers": True,
        "partition_count": 8,
        "planned_new_sessions": {"qwen3.8-27b": 199, "glm-5.3": 399},
        "launch_performed": False,
        "privacy": package["privacy"],
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    self_hosted.write_json_once(out_dir / "MATERIALIZED.json", receipt)
    return receipt


def _validate_gate_inputs(
    plan: dict[str, Any],
    package: dict[str, Any],
    authorization: dict[str, Any],
    qwen_observer: dict[str, Any],
    glm_observer: dict[str, Any],
    repo: Path,
) -> None:
    validate_package(package, repo)
    qwen = validate_canary_observer(qwen_observer, "qwen3.8-27b")
    glm = validate_canary_observer(glm_observer, "glm-5.3")
    validate_authorization(authorization, package, qwen, glm)
    rows = _inventory_rows(package, repo)
    expected = build_plan(plan["shard_key"], package, rows, qwen, glm, authorization)
    if plan != expected:
        raise ValueError("bulk materialized plan does not match gated source inputs")


def preflight(
    plan: dict[str, Any],
    package: dict[str, Any],
    authorization: dict[str, Any],
    qwen_observer: dict[str, Any],
    glm_observer: dict[str, Any],
    repo: Path,
    root: Path,
    key: str,
) -> dict[str, Any]:
    _validate_gate_inputs(plan, package, authorization, qwen_observer, glm_observer, repo)
    roots = hosted._validate_plan_identity_absence(plan, root)
    with hosted._client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
    scratch = Path(f"/tmp/{plan['shard_key']}-preflight-empty")
    if scratch.exists():
        raise RuntimeError("bulk preflight scratch exists")
    (scratch / "attempts").mkdir(parents=True)
    counts: list[int] = []
    try:
        for task in plan["tasks"]:
            items = [row for row in plan["attempts"] if int(row["rank"]) == int(task["rank"])]
            _assert_global_cells_absent(
                plan, task, items, Path(plan["execution"]["global_cell_claim"]["claim_root"])
            )
            counts.append(hosted._validate_inventory_for_task(plan, scratch, task, key))
    finally:
        (scratch / "attempts").rmdir()
        scratch.rmdir()
    receipt = {
        "schema_version": "fleet-opencode-autocontinue-bulk-preflight-v2",
        "plan_sha256": plan["plan_sha256"],
        "bulk_authorization_receipt_sha256": authorization["receipt_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": len(counts),
        "exact_treatment_sessions_reconciled": sum(counts),
        "active_source_attempts": 0,
        "sfs_job_roots_reconciled": roots,
        "current_plan_run_and_claim_identities_absent": True,
        "global_corrected_treatment_cells_reconciled": plan["new_session_count"],
        "global_corrected_treatment_cells_absent": True,
        "global_claim_root": plan["execution"]["global_cell_claim"]["claim_root"],
        "output_root_absent": True,
        "preflight_job_name": plan["preflight_job_name"],
        "preflight_configmap_name": plan["preflight_configmap_name"],
        "scored_configmap_name": plan["scored_configmap_name"],
        "configmaps_distinct": True,
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "configmap_uid": os.environ["CONFIGMAP_UID"],
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def _validate_preflight(value: dict[str, Any], plan: dict[str, Any]) -> None:
    _validate_self_digest(value, "receipt_sha256")
    try:
        uuid.UUID(str(value.get("job_uid")))
        uuid.UUID(str(value.get("pod_uid")))
        uuid.UUID(str(value.get("configmap_uid")))
    except ValueError as exc:
        raise ValueError("bulk preflight UID drifted") from exc
    if (
        value.get("schema_version") != "fleet-opencode-autocontinue-bulk-preflight-v2"
        or value.get("plan_sha256") != plan["plan_sha256"]
        or value.get("bulk_authorization_receipt_sha256")
        != plan["source"]["bulk_authorization_receipt_sha256"]
        or value.get("fleet_team_id") != self_hosted.FLEET_TEAM_ID
        or value.get("tasks_reconciled") != plan["task_count"]
        or value.get("exact_treatment_sessions_reconciled") != len(plan["credited_sessions"])
        or value.get("active_source_attempts") != 0
        or value.get("current_plan_run_and_claim_identities_absent") is not True
        or value.get("global_corrected_treatment_cells_reconciled") != plan["new_session_count"]
        or value.get("global_corrected_treatment_cells_absent") is not True
        or value.get("global_claim_root") != GLOBAL_CLAIM_ROOT
        or value.get("output_root_absent") is not True
        or value.get("preflight_job_name") != plan["preflight_job_name"]
        or value.get("preflight_configmap_name") != plan["preflight_configmap_name"]
        or value.get("scored_configmap_name") != plan["scored_configmap_name"]
        or value.get("configmaps_distinct") is not True
        or value.get("scores_read") is not False
        or value.get("prompts_or_traces_read") is not False
    ):
        raise ValueError("bulk preflight receipt drifted")


def _validate_preflight_observer(
    value: dict[str, Any], preflight_receipt: dict[str, Any], plan: dict[str, Any]
) -> None:
    _validate_self_digest(value, "receipt_sha256")
    job = value.get("job") or {}
    pod = value.get("pod") or {}
    configmap = value.get("configmap") or {}
    try:
        uuid.UUID(str(job.get("uid")))
        uuid.UUID(str(pod.get("uid")))
        uuid.UUID(str(configmap.get("uid")))
    except ValueError as exc:
        raise ValueError("bulk preflight observer UID drifted") from exc
    if (
        value.get("schema_version") != "fleet-opencode-autocontinue-bulk-preflight-post-exit-v1"
        or value.get("status") != "PASSED"
        or value.get("plan_sha256") != plan["plan_sha256"]
        or value.get("preflight_receipt_sha256") != preflight_receipt["receipt_sha256"]
        or job
        != {
            "name": plan["preflight_job_name"],
            "uid": preflight_receipt["job_uid"],
            "succeeded": 1,
            "failed": 0,
        }
        or pod.get("uid") != preflight_receipt["pod_uid"]
        or not isinstance(pod.get("name"), str)
        or not pod.get("name")
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("restart_count") != 0
        or configmap
        != {
            "name": plan["preflight_configmap_name"],
            "uid": preflight_receipt["configmap_uid"],
            "immutable": True,
        }
        or value.get("scored_configmap_name") != plan["scored_configmap_name"]
        or value.get("stage_configmaps_distinct") is not True
        or value.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("bulk preflight post-exit observer drifted")


def authorize_shard(
    plan: dict[str, Any],
    package: dict[str, Any],
    authorization: dict[str, Any],
    qwen_observer: dict[str, Any],
    glm_observer: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_observer: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    _validate_gate_inputs(plan, package, authorization, qwen_observer, glm_observer, repo)
    _validate_preflight(preflight_receipt, plan)
    _validate_preflight_observer(preflight_observer, preflight_receipt, plan)
    release = _build_shard_release(
        plan,
        authorization,
        preflight_receipt,
        preflight_observer,
        _file_sha(repo / "evals/fleet/autocontinue_bulk_controller.py"),
    )
    validate_release(release, plan, preflight_receipt, preflight_observer, repo)
    return release


def run(
    plan: dict[str, Any],
    release: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_observer: dict[str, Any],
    repo: Path,
    root: Path,
    proxy: Path,
) -> dict[str, Any]:
    validate_release(release, plan, preflight_receipt, preflight_observer, repo)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease_config = plan["execution"]["endpoint_lease"]
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease_config["lease_root"]),
        endpoint_key=lease_config["endpoint_key"],
        maximum_streams=lease_config["maximum_streams"],
    ):
        roots = hosted._validate_plan_identity_absence(plan, root)
        root.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (root / name).mkdir(mode=0o700)
        self_hosted.write_json_once(root / "PLAN.json", plan)
        self_hosted.write_json_once(root / "SCORING-RELEASE.json", release)
        self_hosted.write_json_once(root / "PREFLIGHT.json", preflight_receipt)
        self_hosted.write_json_once(root / "PREFLIGHT-POST-EXIT.json", preflight_observer)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        for task in plan["tasks"]:
            hosted._validate_inventory_for_task(plan, root, task, key)
        tasks = {int(row["rank"]): row for row in plan["tasks"]}
        groups = [
            (
                tasks[rank],
                sorted(
                    (row for row in plan["attempts"] if int(row["rank"]) == rank),
                    key=lambda row: int(row["attempt"]),
                ),
            )
            for rank in sorted(tasks)
        ]
        pending = iter(groups)
        active: dict[Future[dict[str, Any]], int] = {}
        accepted = complete = fenced = quarantined = 0
        failure: Exception | None = None
        drain: dict[str, Any] | None = None
        with ThreadPoolExecutor(max_workers=1) as pool:

            def submit() -> None:
                if active or failure is not None or drain is not None:
                    return
                try:
                    task, items = next(pending)
                except StopIteration:
                    return
                _claim_global_task_cells(
                    plan,
                    task,
                    items,
                    Path(plan["execution"]["global_cell_claim"]["claim_root"]),
                )
                future = pool.submit(hosted._run_task, plan, task, items, root, proxy, key)
                active[future] = int(task["rank"])

            submit()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    try:
                        result = future.result()
                        accepted += int(result["accepted"])
                        if result["complete"]:
                            complete += 1
                        elif result.get("quarantined"):
                            quarantined += 1
                        else:
                            fenced += 1
                    except hosted.legacy.DrainRequested as exc:
                        drain = exc.request
                    except Exception as exc:  # noqa: BLE001
                        failure = exc
                submit()
        if failure is not None:
            raise failure
        if drain is not None:
            return hosted._write_hosted_terminal_or_drained(
                plan, root, terminal=None, observed_request=drain
            )
        terminal = {
            "schema_version": "fleet-opencode-autocontinue-bulk-terminal-v1",
            "plan_sha256": plan["plan_sha256"],
            "release_receipt_sha256": release["receipt_sha256"],
            "planned_tasks": plan["task_count"],
            "pass4_complete_tasks": complete,
            "fenced_noncreditable_tasks": fenced,
            "quarantined_infrastructure_tasks": quarantined,
            "accepted_new_outcomes": accepted,
            "credited_canary_outcomes": len(plan["credited_sessions"]),
            "sfs_job_roots_reconciled": roots,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        if complete + fenced + quarantined != plan["task_count"]:
            raise RuntimeError("bulk terminal task accounting drifted")
        terminal["receipt_sha256"] = digest_without(terminal, "receipt_sha256")
        return hosted._write_hosted_terminal_or_drained(plan, root, terminal=terminal)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("preview")
    preview.add_argument("--package", type=Path, required=True)
    preview.add_argument("--held-authorization", type=Path, required=True)
    preview.add_argument("--repo", type=Path, required=True)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--package", type=Path, required=True)
    materialize_parser.add_argument("--authorization", type=Path, required=True)
    materialize_parser.add_argument("--qwen-observer", type=Path, required=True)
    materialize_parser.add_argument("--glm-observer", type=Path, required=True)
    materialize_parser.add_argument("--out-dir", type=Path, required=True)
    materialize_parser.add_argument("--repo", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--release", type=Path, required=True)
    validate.add_argument("--preflight", type=Path, required=True)
    validate.add_argument("--preflight-observer", type=Path, required=True)
    validate.add_argument("--repo", type=Path, required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--plan", type=Path, required=True)
    preflight_parser.add_argument("--package", type=Path, required=True)
    preflight_parser.add_argument("--authorization", type=Path, required=True)
    preflight_parser.add_argument("--qwen-observer", type=Path, required=True)
    preflight_parser.add_argument("--glm-observer", type=Path, required=True)
    preflight_parser.add_argument("--out-dir", type=Path, required=True)
    preflight_parser.add_argument("--out", type=Path, required=True)
    preflight_parser.add_argument("--repo", type=Path, required=True)
    authorize_parser = sub.add_parser("authorize-shard")
    authorize_parser.add_argument("--plan", type=Path, required=True)
    authorize_parser.add_argument("--package", type=Path, required=True)
    authorize_parser.add_argument("--authorization", type=Path, required=True)
    authorize_parser.add_argument("--qwen-observer", type=Path, required=True)
    authorize_parser.add_argument("--glm-observer", type=Path, required=True)
    authorize_parser.add_argument("--preflight", type=Path, required=True)
    authorize_parser.add_argument("--preflight-observer", type=Path, required=True)
    authorize_parser.add_argument("--out", type=Path, required=True)
    authorize_parser.add_argument("--repo", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--release", type=Path, required=True)
    run_parser.add_argument("--preflight", type=Path, required=True)
    run_parser.add_argument("--preflight-observer", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--proxy", type=Path, required=True)
    run_parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preview":
        package = load_object(args.package)
        validate_package(package, args.repo)
        validate_held_authorization(load_object(args.held_authorization), package)
        print(json.dumps({"ok": True, "launch_authorized": False}, sort_keys=True))
        return 0
    if args.command == "materialize":
        receipt = materialize(
            load_object(args.package),
            load_object(args.authorization),
            load_object(args.qwen_observer),
            load_object(args.glm_observer),
            args.repo,
            args.out_dir,
        )
        print(json.dumps({"ok": True, "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
        return 0
    plan = load_object(args.plan)
    if args.command == "validate":
        validate_release(
            load_object(args.release),
            plan,
            load_object(args.preflight),
            load_object(args.preflight_observer),
            args.repo,
        )
        return 0
    if args.command == "authorize-shard":
        release = authorize_shard(
            plan,
            load_object(args.package),
            load_object(args.authorization),
            load_object(args.qwen_observer),
            load_object(args.glm_observer),
            load_object(args.preflight),
            load_object(args.preflight_observer),
            args.repo,
        )
        self_hosted.write_json_once(args.out, release)
        return 0
    if args.command == "preflight":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        receipt = preflight(
            plan,
            load_object(args.package),
            load_object(args.authorization),
            load_object(args.qwen_observer),
            load_object(args.glm_observer),
            args.repo,
            args.out_dir,
            key,
        )
        self_hosted.write_json_once(args.out, receipt)
        return 0
    terminal = run(
        plan,
        load_object(args.release),
        load_object(args.preflight),
        load_object(args.preflight_observer),
        args.repo,
        args.out_dir,
        args.proxy,
    )
    print(json.dumps({"ok": True, "receipt_sha256": terminal["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
