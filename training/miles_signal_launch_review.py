"""Score-blind static launch authority for the four Miles signal lanes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training import miles96_mechanics_canary as mechanics
from training import miles96_signal_qualification as signal
from training import miles_signal_transition as transition
from training import miles_signal_wave

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "configs/qualification/qwen38-miles-signal-score-blind-launch-review-v1.json"
SCHEMA = "cyber_qwen38_miles_signal_score_blind_launch_review_v1"
PURPOSE = (
    "Bind four zero-update signal qualifications without carrying historical scores into "
    "launch review."
)

EXPECTED_FRESH_GATES = [
    {
        "id": "exact_live_task_get",
        "scope": "wave",
        "maximum_age_seconds": transition.FRESH_TASK_SECONDS,
        "required": True,
        "requirement": (
            "Fleet-team GETs reproduce every exact task, environment, verifier, and "
            "sanitized binding digest."
        ),
    },
    {
        "id": "duplicate_and_destination_absence",
        "scope": "lane",
        "maximum_age_seconds": transition.FRESH_PREVIEW_SECONDS,
        "required": True,
        "requirement": (
            "Fresh Jobs API searches by exact name, title, and run directory; Kubernetes "
            "root and owner-linked child searches; and the exact SFS output path all prove "
            "absence. Ambiguous reads fail closed."
        ),
    },
    {
        "id": "live_mcp_tool_list_and_schema_probe",
        "scope": "lane",
        "maximum_age_seconds": transition.FRESH_PREVIEW_SECONDS,
        "required": True,
        "requirement": (
            "A sanitized, non-rewarding probe opens the exact task session, compares its "
            "single raw tool-list read and pinned OpenAI projection with authority, closes "
            "the instance, and proves release. Legacy task metadata alone is insufficient."
        ),
    },
    {
        "id": "capacity_census",
        "scope": "wave",
        "maximum_age_seconds": 120,
        "required": True,
        "requirement": (
            "A cross-namespace owned GPU census includes the planned wave and remains at or "
            "below 10 nodes and 80 GPUs."
        ),
    },
    {
        "id": "two_identical_server_previews",
        "scope": "lane",
        "maximum_age_seconds": transition.FRESH_PREVIEW_SECONDS,
        "required": True,
        "requirement": (
            "Two authenticated previews normalize identically and bind the exact plan, "
            "request, image, one-node/eight-GPU shape, c1/q1, no requeue, backoffLimit zero, "
            "and root fleet.ai/failure-alerts annotation off."
        ),
    },
    {
        "id": "prearmed_cleanup_observer",
        "scope": "lane",
        "maximum_age_seconds": transition.FRESH_PREVIEW_SECONDS,
        "required": True,
        "requirement": (
            "A live exact-name observer is armed before create and binds run directory, "
            "image, plan, rendered manifest, eight GPUs, and the complete phase-one deadline."
        ),
    },
    {
        "id": "final_precreate_recheck",
        "scope": "lane",
        "maximum_age_seconds": 120,
        "required": True,
        "requirement": (
            "Immediately before the single create, observer liveness, exact-name/title/run-"
            "directory duplicates, SFS destination absence, and capacity are revalidated."
        ),
    },
    {
        "id": "durable_single_post_intent",
        "scope": "lane",
        "maximum_age_seconds": 120,
        "required": True,
        "requirement": (
            "Before the one POST, an fsync-durable intent binds the exact lane, plan, "
            "request, normalized manifest, observer, freshness receipts, and attempt state. "
            "An uncertain POST permits read-only reconciliation only."
        ),
    },
    {
        "id": "independent_lane_approval",
        "scope": "lane",
        "maximum_age_seconds": transition.FRESH_PREVIEW_SECONDS,
        "required": True,
        "requirement": (
            "A reviewed successor binds this artifact plus the exact fresh receipts, "
            "operator commit, plan, request, and rendered manifest."
        ),
    },
]

RUNTIME_GATES = {
    "single_raw_tool_list_read_per_session": True,
    "live_raw_tool_catalog_must_match_authority": True,
    "live_openai_tool_projection_must_match_pinned_transform": True,
    "task_environment_verifier_binding_must_match_at_session_open": True,
    "tool_or_binding_drift_closes_instance_and_excludes_episode": True,
    "session_open_tool_contract_is_enforced_by_lifecycle_validation": True,
    "session_open_contract_is_transitively_bound_by_attempt_receipt_hashes": True,
}


def _binding(wave: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    task = {
        **miles_signal_wave.task_binding(wave, candidate),
        "authority_receipt_sha256": candidate["authority_receipt_sha256"],
    }
    plan = signal.build_plan(
        name=candidate["identity"]["name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
        task_binding=task,
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=candidate["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )
    request = signal.job_request(plan)
    return {
        "name": candidate["identity"]["name"],
        "run_dir": candidate["identity"]["run_dir"],
        "split": candidate["split"],
        "task_family": candidate["lineage"]["task_family"],
        "task": {key: candidate["task"][key] for key in ("key", "version_id")},
        "environment": {
            key: candidate["environment"][key]
            for key in ("id", "version", "version_id", "data_id", "data_version")
        },
        "verifier": {
            key: candidate["verifier"][key] for key in ("id", "version", "version_id", "sha256")
        },
        "authority_receipt_sha256": candidate["authority_receipt_sha256"],
        "live_binding_receipt_sha256": candidate["live_binding_receipt_sha256"],
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "request_sha256": "sha256:" + mechanics.digest(request),
        "runtime_bundle_sha256": request["env"]["CYBER_RUNTIME_BUNDLE_SHA256"],
        "request_binding_sha256": request["env"]["CYBER_REQUEST_BINDING_SHA256"],
    }


def load(path: Path = REVIEW_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("signal launch review must be an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != "sha256:" + mechanics.digest(body):
        raise ValueError("signal launch review self digest changed")
    if (
        set(value)
        != {
            "schema",
            "state",
            "launchable",
            "purpose",
            "authority",
            "adapter",
            "model",
            "execution_contract",
            "score_blind",
            "candidate_bindings",
            "required_fresh_at_launch_evidence",
            "runtime_fail_closed_gates",
            "post_create_invariants",
            "future_minimal_successor_binding",
            "privacy",
            "sha256",
        }
        or value.get("schema") != SCHEMA
    ):
        raise ValueError("signal launch review fields changed")
    if (
        value.get("state") != "static_review_only"
        or value.get("launchable") is not False
        or value.get("purpose") != PURPOSE
    ):
        raise ValueError("static launch review cannot authorize a create")

    wave = miles_signal_wave.load()
    tool = mechanics.tool_contract()
    if value["authority"] != {
        "config_path": "configs/qualification/qwen38-miles-signal-wave-v1.json",
        "config_sha256": wave["sha256"],
        "source_commit": wave["source_commit"],
        "task_set_sha256": wave["authorities"]["task_set"]["self_sha256"],
        "production_split_sha256": wave["authorities"]["production_split"]["self_sha256"],
        "raw_tool_catalog_sha256": tool["raw_tool_catalog_sha256"],
        "openai_tool_catalog_sha256": tool["openai_tool_catalog_sha256"],
        "tool_transform_source_sha256": tool["transform_source_sha256"],
    }:
        raise ValueError("signal launch authority changed")
    if value["model"] != {
        "root": signal.HF_MODEL_ROOT,
        "revision": signal.HF_MODEL_REVISION,
        "binding_sha256": signal.HF_MODEL_BINDING_SHA256,
        "image": mechanics.IMAGE,
    }:
        raise ValueError("signal launch model changed")
    if value["adapter"] != {
        "commit": "978df19a1f6b344e2f88d9502060700a59294681",
        "runtime_source_closure_sha256": (
            "sha256:de52b39f55e92a079bef1c0ea14b9e823e4af5096318dca5cf48cc800225fd15"
        ),
        "focused_test_receipt_file_sha256": (
            "sha256:fa48421664a19bb2ff77bf27eab194cfa51488811676dd69610101b18aa6ae13"
        ),
        "focused_test_receipt_sha256": (
            "sha256:e85cf842d1fd8c46752f6d5c8a13313b572e5876bc3cbc7b84d9adbc7c7d8b40"
        ),
    }:
        raise ValueError("signal launch adapter evidence changed")

    rows = sorted(
        (_binding(wave, candidate) for candidate in wave["candidates"]),
        key=lambda row: row["name"],
    )
    if value["candidate_bindings"] != rows:
        raise ValueError("signal launch candidate binding changed")
    for field in ("name", "run_dir", "task_family"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"signal launch {field} is not unique")
    for field in ("version_id",):
        if len({row["task"][field] for row in rows}) != len(rows):
            raise ValueError("signal launch task version is not unique")

    execution = value["execution_contract"]
    if execution != {
        "cluster_target": "prod",
        "namespace": mechanics.NAMESPACE,
        "nodes_per_lane": 1,
        "gpus_per_node": 8,
        "lane_count": 4,
        "maximum_wave_nodes": 4,
        "maximum_wave_gpus": 32,
        "priority_class": "c1",
        "queue_priority": "q1",
        "jobs_api_failure_alerts": False,
        "root_annotations": {"fleet.ai/failure-alerts": "off"},
        "requeue_if_preempted": False,
        "sample_indexes": list(range(8)),
        "max_concurrent_episodes": 2,
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "episode_timeout_seconds": 2400,
        "outer_episode_replacements": 0,
        "optimizer_steps": 0,
        "checkpoint": False,
    }:
        raise ValueError("signal launch execution contract changed")

    blind = value["score_blind"]
    if blind != {
        "historical_scores_included": False,
        "historical_rewards_included": False,
        "historical_pass_counts_included": False,
        "candidate_order": "lexicographic_by_name",
        "candidate_order_is_not_a_launch_priority": True,
        "fresh_rewards_may_be_read_only_after_all_eight_slots_are_terminal": True,
    }:
        raise ValueError("signal launch score-blind contract changed")
    gates = value["required_fresh_at_launch_evidence"]
    if gates != EXPECTED_FRESH_GATES:
        raise ValueError("signal launch fresh evidence gates changed")
    if value["runtime_fail_closed_gates"] != RUNTIME_GATES:
        raise ValueError("signal launch runtime gates changed")
    if value["privacy"] != {
        "prompt_text_included": False,
        "verifier_code_included": False,
        "environment_values_included": False,
        "credentials_included": False,
        "trajectory_content_included": False,
    }:
        raise ValueError("signal launch privacy contract changed")
    if value["post_create_invariants"] != {
        "one_create_attempt_per_lane": True,
        "uncertain_create_is_read_only_reconciled": True,
        "accepted_episode_is_never_replayed": True,
        "all_eight_slots_are_terminally_accounted": True,
        "minimum_normally_completed_gradeable_episodes": 2,
        "minimum_distinct_finite_rewards": 2,
        "unique_verifier_execution_ids_required": True,
        "all_task_instances_released": True,
        "optimizer_update_allowed": False,
        "checkpoint_allowed": False,
    }:
        raise ValueError("signal launch post-create contract changed")
    if value["future_minimal_successor_binding"] != {
        "operator_commit_required": True,
        "operator_source_closure_required": True,
        "exact_image_preflight_receipt_required": True,
        "session_open_tool_contract_preflight_required": True,
        "reviewed_successor_receipt_required_per_lane": True,
    }:
        raise ValueError("signal launch future successor gate changed")
    return value
