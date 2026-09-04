"""Validate the held, zero-credit OpenCode autocontinue evaluation campaign.

This module is deliberately preview-only.  It binds the scientific task
universe and launch partition without creating Kubernetes objects or calling
the Fleet API.  Runtime releases and executable per-shard plans are separate,
append-only products after the two treatment canaries pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.fleet import campaign_supervisor, self_hosted

CAMPAIGN_SCHEMA = "fleet-opencode-autocontinue-primary-campaign-preview-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-primary-release-preview-v1"
MAPPING_DIGEST = "sha256:de30c778e4337c79dea921b7463aa74760da59f972c928927c4359e828f1f009"
RETIREMENT_DIGEST = "sha256:2a904547d5450e46982d33466f8a55c3bbe41652bd6705aa451172dc20656d8a"
CONTEXT_POLICY = "opencode_1.18.27_native_compaction_autocontinue_v1"
HIGH_PRIORITY = "fleet-train-high"
CAMPAIGN_ID = "chris-cyber-q38-glm53-opencode11827-autocontinue-primary-v1"

EXPECTED_MODELS = {
    "qwen3.8-27b": {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "served_id": "qwen3.8-27b",
        "session_model": (
            "fleet-cluster-opencode-1.18.27/qwen3.8-27b-opencode11827-autocontinue-v1"
        ),
        "task_count": 50,
        "cell_count": 200,
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
        "task_count": 100,
        "cell_count": 400,
        "settings_canonical_sha256": (
            "sha256:9614806454c86784e54fde31a8acc3f51adda11593fad3e762142cd04a4010f2"
        ),
        "settings_file_sha256": (
            "sha256:84a1ca763a297bd8badb184391a02c82d57069b9a211ecef4f5e53dd79e20e20"
        ),
    },
}

EXPECTED_COMPONENT_RANKS = {
    "qwen-hosted-retained-source4": {4},
    "qwen-hosted-primary49": {6, 7, 8, 9, *range(11, 51), 52, 53, 54, 55, 56},
    "glm-hosted-primary46": {*range(13, 100, 2), 108, 109},
    "glm-dedicated-a-v5-primary27": {
        8,
        10,
        12,
        *range(16, 53, 2),
        102,
        104,
        110,
        112,
        113,
    },
    "glm-dedicated-b-v5-primary27": {
        *range(58, 101, 2),
        101,
        103,
        105,
        107,
        114,
    },
}

EXPECTED_PARTITIONS = {
    "qwen-canary1": {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "phase": "canary",
        "partial": {4: {1}},
        "full": set(),
        "tasks": 1,
        "cells": 1,
    },
    "qwen-hosted-a99": {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "partial": {4: {2, 3, 4}},
        "full": {6, 7, 8, 9, *range(11, 31)},
        "tasks": 25,
        "cells": 99,
    },
    "qwen-hosted-b100": {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "partial": {},
        "full": {*range(31, 51), 52, 53, 54, 55, 56},
        "tasks": 25,
        "cells": 100,
    },
    "glm-canary1": {
        "model": "glm-5.3",
        "serving_block": "glm-hosted-autocontinue-v1",
        "phase": "canary",
        "partial": {13: {1}},
        "full": set(),
        "tasks": 1,
        "cells": 1,
    },
    "glm-hosted-a91": {
        "model": "glm-5.3",
        "serving_block": "glm-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "partial": {13: {2, 3, 4}},
        "full": {*range(15, 58, 2)},
        "tasks": 23,
        "cells": 91,
    },
    "glm-hosted-b92": {
        "model": "glm-5.3",
        "serving_block": "glm-hosted-autocontinue-v1",
        "phase": "bulk_after_canary",
        "partial": {},
        "full": {*range(59, 100, 2), 108, 109},
        "tasks": 23,
        "cells": 92,
    },
    "glm-dedicated-a52": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-a-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "partial": {},
        "full": {8, 10, 12, *range(16, 35, 2)},
        "tasks": 13,
        "cells": 52,
    },
    "glm-dedicated-a56": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-a-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "partial": {},
        "full": {*range(36, 53, 2), 102, 104, 110, 112, 113},
        "tasks": 14,
        "cells": 56,
    },
    "glm-dedicated-b52": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-b-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "partial": {},
        "full": {*range(58, 83, 2)},
        "tasks": 13,
        "cells": 52,
    },
    "glm-dedicated-b56": {
        "model": "glm-5.3",
        "serving_block": "glm-dedicated-b-v5-autocontinue-v1",
        "phase": "bulk_after_canary_and_parity",
        "partial": {},
        "full": {*range(84, 101, 2), 101, 103, 105, 107, 114},
        "tasks": 14,
        "cells": 56,
    },
}

EXPECTED_DEDICATED_RUNTIME = {
    "glm-dedicated-a-v5-autocontinue-v1": {
        "replica": "A",
        "ray_job_uid": "ef7cb0f2-84d4-4017-ae31-bf34ebb70d0d",
        "ray_cluster_uid": "5107d72e-ae6a-4582-a490-55b349421d06",
        "service_uid": "2c0e64de-c4a0-4f70-ae07-d15b1adad0b3",
        "head_pod_uid": "3f37ab91-4afa-4e4d-8f29-a3eeda70a774",
    },
    "glm-dedicated-b-v5-autocontinue-v1": {
        "replica": "B",
        "ray_job_uid": "2bbfd34a-d74a-4497-ab28-48cada0c9753",
        "ray_cluster_uid": "f7f7ae34-6a79-482a-b771-4639b786f66a",
        "service_uid": "14ae906a-60ef-4a02-b6bf-814a00c2c89f",
        "head_pod_uid": "bfeb4c6a-9e4a-4f14-adb2-3f516dd25fdb",
    },
}

EXPECTED_SCORED_JOB_NAMES = {
    "qwen-canary1": "chris-q38-ac-canary1-v1",
    "qwen-hosted-a99": "chris-q38-ac-hosted-a99-v1",
    "qwen-hosted-b100": "chris-q38-ac-hosted-b100-v1",
    "glm-canary1": "chris-glm53-ac-canary1-v1",
    "glm-hosted-a91": "chris-glm53-ac-hosted-a91-v1",
    "glm-hosted-b92": "chris-glm53-ac-hosted-b92-v1",
    "glm-dedicated-a52": "chris-glm53-ac-ded-a52-v1",
    "glm-dedicated-a56": "chris-glm53-ac-ded-a56-v1",
    "glm-dedicated-b52": "chris-glm53-ac-ded-b52-v1",
    "glm-dedicated-b56": "chris-glm53-ac-ded-b56-v1",
}


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON file: {path}")
    value = json.loads(path.read_text(), object_pairs_hook=_pairs_no_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def _scientific_identities(mapping: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for component in mapping.get("components") or []:
        component_id = component.get("id")
        model = component.get("model")
        ranks: set[int] = set()
        for source in component.get("scientific_task_sources") or []:
            if source.get("kind") == "plan":
                plan = load_object(root / source["repo_plan_path"])
                if plan.get("plan_sha256") != source.get("plan_sha256") or plan.get(
                    "plan_sha256"
                ) != digest_without(plan, "plan_sha256"):
                    raise ValueError("scientific task-source plan digest drifted")
                by_rank = {int(row["source_rank"]): row for row in plan.get("tasks") or []}
                rows = [by_rank[int(rank)] for rank in source.get("source_ranks") or []]
            elif source.get("kind") == "inline":
                rows = source.get("tasks") or []
            else:
                raise ValueError("unsupported scientific task source")
            for row in rows:
                rank = int(row["source_rank"])
                task = row.get("task") or {}
                if rank in ranks or not task.get("key") or not task.get("version_id"):
                    raise ValueError("scientific task identity is absent or duplicated")
                ranks.add(rank)
                identities.append(
                    {
                        "model": model,
                        "component_id": component_id,
                        "source_rank": rank,
                        "task_key": task["key"],
                        "task_version_id": task["version_id"],
                    }
                )
        if ranks != EXPECTED_COMPONENT_RANKS.get(component_id):
            raise ValueError("frozen component rank set drifted")
    identities.sort(key=lambda row: (row["model"], row["component_id"], row["source_rank"]))
    return identities


def _expand_partition(row: dict[str, Any]) -> set[tuple[int, int]]:
    cells = {
        (int(rank), attempt)
        for rank in row.get("full_source_ranks") or []
        for attempt in range(1, 5)
    }
    for partial in row.get("partial_source_rank_attempts") or []:
        rank = int(partial.get("source_rank") or 0)
        attempts = partial.get("attempts") or []
        if any(type(attempt) is not int or attempt not in range(1, 5) for attempt in attempts):
            raise ValueError("partition has an invalid attempt")
        additions = {(rank, attempt) for attempt in attempts}
        if cells & additions:
            raise ValueError("partition repeats a cell")
        cells |= additions
    return cells


def _validate_renderer(model: dict[str, Any]) -> None:
    config = {
        "model": {"served_id": model["served_id"]},
        "harness": {
            "name": "opencode",
            "version": "1.18.27",
            "context_management": CONTEXT_POLICY,
            "context_window_size": 262144,
            "max_output_tokens": 32768,
            "compaction_headroom_tokens": 20000,
        },
    }
    settings = self_hosted.opencode_settings(config)
    canonical = self_hosted.canonical_json(settings)
    if (
        self_hosted.sha256(canonical) != model["settings_canonical_sha256"]
        or self_hosted.sha256(canonical + b"\n") != model["settings_file_sha256"]
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or settings["provider"]["fleet-cluster"]["models"][model["served_id"]]["limit"]
        != {"context": 262144, "output": 32768, "input": 229376}
    ):
        raise ValueError("autocontinue renderer bytes drifted")


def validate_campaign(campaign: dict[str, Any], *, root: Path = Path(".")) -> dict[str, Any]:
    if (
        campaign.get("schema_version") != CAMPAIGN_SCHEMA
        or campaign.get("campaign_sha256") != digest_without(campaign, "campaign_sha256")
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("status") != "HELD"
        or campaign.get("launch_authorized") is not False
    ):
        raise ValueError("autocontinue campaign envelope is invalid")
    mapping_ref = campaign.get("scientific_mapping") or {}
    if mapping_ref != {
        "path": "evals/fleet/configs/q38-glm53-primary-scientific-mapping-v2.json",
        "mapping_sha256": MAPPING_DIGEST,
        "retired_treatment_receipt_sha256": RETIREMENT_DIGEST,
        "legacy_credit": 0,
    }:
        raise ValueError("scientific mapping or zero-credit binding drifted")
    mapping = load_object(root / mapping_ref["path"])
    if mapping.get("mapping_sha256") != MAPPING_DIGEST:
        raise ValueError("scientific mapping digest drifted")
    mapping_summary = campaign_supervisor.validate_scientific_mapping(mapping, root=root)
    if mapping_summary != {
        "tasks": 150,
        "cells": 600,
        "model_cells": {"glm-5.3": 400, "qwen3.8-27b": 200},
        "unresolved_cells": 4,
        "unresolved_components": ["glm-dedicated-b-v5-primary27"],
        "executable_cells": 596,
        "executable_universe_sha256": None,
    }:
        raise ValueError("frozen scientific task universe drifted")
    identities = _scientific_identities(mapping, root)
    identity_digest = self_hosted.sha256(self_hosted.canonical_json(identities))
    if campaign.get("task_identity_sha256") != identity_digest:
        raise ValueError("frozen task identity digest drifted")
    if (
        len(identities) != 150
        or len({(row["model"], row["task_version_id"]) for row in identities}) != 150
    ):
        raise ValueError("frozen task identity uniqueness drifted")

    models = campaign.get("models") or {}
    if set(models) != set(EXPECTED_MODELS):
        raise ValueError("campaign model set drifted")
    for model_name, expected in EXPECTED_MODELS.items():
        model = models[model_name]
        if any(model.get(key) != value for key, value in expected.items()):
            raise ValueError("campaign exact model identity drifted")
        if model.get("context_management") != CONTEXT_POLICY:
            raise ValueError("campaign context treatment drifted")
        _validate_renderer(model)

    partitions = campaign.get("partitions") or []
    if {row.get("id") for row in partitions} != set(EXPECTED_PARTITIONS):
        raise ValueError("campaign partition set drifted")
    all_cells: dict[str, set[tuple[int, int]]] = {
        "qwen3.8-27b": set(),
        "glm-5.3": set(),
    }
    job_names: list[str] = []
    sfs_roots: list[str] = []
    for row in partitions:
        expected = EXPECTED_PARTITIONS[row["id"]]
        if any(row.get(field) != expected[field] for field in ("model", "serving_block", "phase")):
            raise ValueError("campaign partition treatment drifted")
        if set(row.get("full_source_ranks") or []) != expected["full"]:
            raise ValueError("campaign full-task partition drifted")
        partial = {
            int(item["source_rank"]): set(item.get("attempts") or [])
            for item in row.get("partial_source_rank_attempts") or []
        }
        if partial != expected["partial"]:
            raise ValueError("campaign partial-cell partition drifted")
        cells = _expand_partition(row)
        if (
            len({rank for rank, _ in cells}) != expected["tasks"]
            or len(cells) != expected["cells"]
            or row.get("planned_task_count") != expected["tasks"]
            or row.get("planned_cell_count") != expected["cells"]
            or row.get("workers") != 1
            or row.get("priority_class") != HIGH_PRIORITY
            or row.get("launch_authorized") is not False
            or row.get("preflight_required") is not True
            or row.get("create_once") is not True
        ):
            raise ValueError("campaign partition scheduling drifted")
        scored_name = EXPECTED_SCORED_JOB_NAMES[row["id"]]
        if (
            row.get("scored_job_name") != scored_name
            or row.get("preflight_job_name") != scored_name + "-preflight"
            or row.get("sfs_root") != scored_name
        ):
            raise ValueError("campaign exact create-once identity drifted")
        if all_cells[row["model"]] & cells:
            raise ValueError("campaign partition cells overlap")
        all_cells[row["model"]] |= cells
        job_names.extend([row.get("preflight_job_name"), row.get("scored_job_name")])
        sfs_roots.append(row.get("sfs_root"))
    if (
        any(not isinstance(name, str) or not name for name in [*job_names, *sfs_roots])
        or len(job_names) != len(set(job_names))
        or len(sfs_roots) != len(set(sfs_roots))
    ):
        raise ValueError("campaign create-once identities drifted")

    identity_ranks = {
        model: {int(row["source_rank"]) for row in identities if row["model"] == model}
        for model in EXPECTED_MODELS
    }
    for model, expected in EXPECTED_MODELS.items():
        wanted = {(rank, attempt) for rank in identity_ranks[model] for attempt in range(1, 5)}
        if all_cells[model] != wanted or len(wanted) != expected["cell_count"]:
            raise ValueError("campaign does not cover exact model pass@4 universe")

    runtime = campaign.get("dedicated_runtime_gates") or {}
    if set(runtime) != set(EXPECTED_DEDICATED_RUNTIME):
        raise ValueError("dedicated runtime gate set drifted")
    for block, expected in EXPECTED_DEDICATED_RUNTIME.items():
        gate = runtime[block]
        if (
            any(gate.get(key) != value for key, value in expected.items())
            or gate.get("new_treatment_parity_required") is not True
            or gate.get("new_treatment_parity_passed") is not False
            or gate.get("new_treatment_parity_receipt_sha256") is not None
        ):
            raise ValueError("dedicated UID-bound parity gate drifted")

    gates = campaign.get("gates") or {}
    drain = gates.get("drain_protocol") or {}
    if (
        gates.get("zero_legacy_credit") is not True
        or gates.get("one_new_treatment_canary_cell_per_model") is not True
        or gates.get("bulk_release_requires_both_canaries_accepted") is not True
        or gates.get("fresh_api_kubernetes_sfs_inventory_required") is not True
        or gates.get("new_campaign_duplicate_key")
        != ["treatment", "model", "task_version_id", "attempt"]
        or gates.get("maximum_streams_per_endpoint") != 2
        or gates.get("atomic_endpoint_lease_required") is not True
        or drain.get("schema_version") != "fleet-selfhosted-opencode-pass4-drain-request-v1"
        or drain.get("checked_before_every_task_claim") is not True
        or drain.get("checked_before_every_attempt_claim") is not True
        or drain.get("uid_and_plan_bound") is not True
        or drain.get("controller_implementation_gate_passed") is not False
    ):
        raise ValueError("campaign safety gates drifted")
    privacy = campaign.get("privacy") or {}
    if privacy != {
        "scores_read": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }:
        raise ValueError("campaign privacy contract drifted")
    return {
        "tasks": 150,
        "cells": 600,
        "models": dict(Counter(row["model"] for row in identities)),
        "partitions": len(partitions),
        "canary_cells": 2,
        "bulk_cells": 598,
        "task_identity_sha256": identity_digest,
        "launch_authorized": False,
    }


def validate_release_preview(
    release: dict[str, Any], campaign: dict[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    summary = validate_campaign(campaign, root=root)
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("status") != "HELD"
        or release.get("campaign_sha256") != campaign["campaign_sha256"]
        or release.get("task_identity_sha256") != summary["task_identity_sha256"]
        or release.get("canary_launch_authorized") is not False
        or release.get("bulk_launch_authorized") is not False
        or release.get("cluster_objects_created") is not False
        or release.get("remaining_runtime_gates")
        != [
            "legacy_interrupted_environment_cleanup_terminal",
            "r114_immutable_hydration_and_execution_binding",
            "fresh_fleet_task_version_environment_verifier_inventory",
            "hosted_endpoint_health",
            "dedicated_a_and_b_uid_bound_autocontinue_parity",
            "uid_plan_bound_drain_gate_in_hosted_controller",
            "qwen_and_glm_one_cell_canary_authorization_and_acceptance",
            "atomic_endpoint_leases_and_fresh_duplicate_preflights",
            "append_only_bulk_release_authorization",
        ]
        or release.get("privacy") != campaign["privacy"]
    ):
        raise ValueError("autocontinue release preview drifted")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[2]
    summary = validate_release_preview(
        load_object(args.release), load_object(args.campaign), root=root
    )
    print(json.dumps({"ok": True, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
