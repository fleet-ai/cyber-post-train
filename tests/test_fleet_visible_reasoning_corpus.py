"""Contracts for the separate private student-visible-reasoning lane."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from training import collection_campaign as campaign
from training import fleet_visible_reasoning_corpus as corpus
from training import task_family_split
from training.io import digest_json, file_sha256


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _write(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _inventory() -> dict:
    return campaign.sealed(
        {
            "schema": campaign.INVENTORY_SCHEMA,
            "task_validity_receipt_sha256": _sha("a"),
            "task_versions": [
                {
                    "task_key": f"task-{index}",
                    "task_version_id": f"00000000-0000-4000-8000-{index:012d}",
                    "lineage": {
                        "application": f"app-{index % 2}",
                        "environment": f"env-{index % 2}",
                        "difficulty": "medium",
                        "vulnerability_family": [f"vuln-{index}"],
                        "task_family": f"family-{index}",
                    },
                }
                for index in range(6)
            ],
        }
    )


def _runtime(inventory: dict) -> dict:
    return campaign.sealed(
        {
            "schema": campaign.RUNTIME_BINDINGS_SCHEMA,
            "metadata_inventory_sha256": inventory["sha256"],
            "task_validity_receipt_sha256": inventory["task_validity_receipt_sha256"],
            "task_versions": [
                {
                    "task_key": row["task_key"],
                    "task_version_id": row["task_version_id"],
                    "env_key": "qualified-env",
                    "env_version": "v1",
                    "environment_version_id": "00000000-0000-4000-8001-000000000001",
                    "data_key": "qualified-data",
                    "data_version": "v1",
                }
                for row in inventory["task_versions"]
            ],
        }
    )


_TOOL_CATALOG = [
    {
        "name": "bash",
        "description": "Run a synthetic bash script.",
        "inputSchema": {
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
        },
    },
    {
        "name": "submit_report",
        "description": "Submit a synthetic report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "flag": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["explanation"],
        },
    },
]
_TEMPLATE_TOOLS = corpus._derive_opencode_template_tools(_TOOL_CATALOG)


class _Tokenizer:
    """A synthetic local tokenizer that rejects any unpinned render treatment."""

    def apply_chat_template(self, messages, **kwargs):
        tools = kwargs.pop("tools")
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": messages[-1]["role"] != "assistant",
            "enable_thinking": True,
        }
        if tools == []:
            assert messages[0] == {
                "role": "user",
                "content": [{"type": "text", "text": "synthetic summary request"}],
            }
            if len(messages) == 1:
                return [10, 11]
            assert len(messages) == 2 and messages[1] == {
                "role": "assistant",
                "content": "synthetic summary",
            }
            return [10, 11, 44, 45]
        assert tools == _TEMPLATE_TOOLS
        if len(messages) == 2:
            return [1, 2]
        if len(messages) == 3:
            assistant = messages[-1]
            if (
                assistant.get("reasoning_content") == "synthetic visible reasoning"
                and assistant.get("content") == ""
                and not assistant.get("tool_calls")
            ):
                return [1, 2, 3, 5]
            return [1, 2, 3, 4, 5]
        raise ValueError("unexpected synthetic conversation")


def _window(
    window_id: str = "window-1",
    *,
    ids: list[int] | None = None,
    sequence_index: int = 0,
    message_indices: list[int] | None = None,
) -> dict:
    ids = ids or [1, 2, 3, 4, 5]
    message_indices = message_indices or [0, 1, 2]
    target_message_index = message_indices[-1]
    spans = [
        {
            "kind": "student_visible_reasoning",
            "source_message_index": target_message_index,
            "token_start": 2,
            "token_end": 3,
            "token_ids_sha256": digest_json(ids[2:3]),
        },
        {
            "kind": "visible_action",
            "source_message_index": target_message_index,
            "token_start": 3,
            "token_end": 5,
            "token_ids_sha256": digest_json(ids[3:5]),
        },
    ]
    return {
        "window_id": window_id,
        "sequence_index": sequence_index,
        "assistant_turn_id": "assistant-1",
        "message_indices": message_indices,
        "target_message_index": target_message_index,
        "input_ids": ids,
        "prompt_token_count": 2,
        "prompt_token_sha256": digest_json(ids[:2]),
        "serialized_token_ids_sha256": digest_json(ids),
        "target_spans": spans,
        "window_payload_sha256": digest_json(
            {
                "sequence_index": sequence_index,
                "message_indices": message_indices,
                "target_message_index": target_message_index,
                "input_ids": ids,
                "prompt_token_count": 2,
                "target_spans": spans,
            }
        ),
    }


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    inventory = _inventory()
    split_v1 = task_family_split.build(
        inventory["task_versions"],
        inventory_sha256=inventory["sha256"],
        seed="visible-reasoning-test-v1",
        ratios={"train": 1 / 3, "dev": 1 / 3, "final_test": 1 / 3},
        max_group_task_version_fraction=0.6,
    )
    role_anchor = task_family_split.freeze_role_anchor(split_v1, inventory["task_versions"])
    monkeypatch.setattr(
        task_family_split,
        "trusted_fleet_collection_root_anchor",
        lambda: copy.deepcopy(role_anchor),
    )
    split = task_family_split.build_anchored(
        inventory["task_versions"],
        inventory_sha256=inventory["sha256"],
        role_anchor=role_anchor,
        seed="visible-reasoning-test-v1",
        ratios={"train": 1 / 3, "dev": 1 / 3, "final_test": 1 / 3},
        max_group_task_version_fraction=0.6,
    )
    runtime = _runtime(inventory)
    roles = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    candidate = next(row for row in split["tasks"] if row["split"] == "train")
    lock = campaign.sealed(
        {
            "schema": "cyber_protected_task_family_lock_v1",
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": sorted(
                {row["group_id"] for row in split["tasks"] if row["split"] != "train"}
            ),
        }
    )
    revision = "f" * 40
    qwen_target = {
        "repository": corpus.QWEN_REPOSITORY,
        "revision": revision,
        "tokenizer_sha256": _sha("d"),
        "tokenizer_backend_sha256": _sha("e"),
        "chat_template_sha256": _sha("f"),
    }
    roundtrip = {
        "schema": corpus.ROUNDTRIP_SCHEMA,
        "qwen_target": qwen_target,
        "cases": [
            {
                "case_id": "synthetic-visible-reasoning",
                "messages": [
                    {"role": "system", "content": "synthetic system"},
                    {"role": "user", "content": "synthetic task"},
                    {
                        "role": "assistant",
                        "student_visible_reasoning": "synthetic visible reasoning",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "bash-1",
                                "type": "function",
                                "function": {
                                    "name": "fleet_bash",
                                    "arguments": {"script": "true"},
                                },
                            }
                        ],
                    },
                ],
                "target_message_index": 2,
                "prompt_token_ids": [1, 2],
                "collection_token_ids": [1, 2, 3, 4, 5],
                "training_token_ids": [1, 2, 3, 4, 5],
                "serving_token_ids": [1, 2, 3, 4, 5],
            }
        ],
    }
    roundtrip["sha256"] = digest_json(roundtrip)
    roundtrip_path = tmp_path / "roundtrip.json"
    _write(roundtrip_path, roundtrip)
    source = {
        "kind": "qwen_self",
        "model_alias": "source",
        "model": {
            "repository": corpus.QWEN_REPOSITORY,
            "revision": revision,
            "session_model": "qwen-source",
        },
    }
    opencode = {
        "harness": corpus.OPENCODE_HARNESS,
        "harness_version": corpus.OPENCODE_VERSION,
        "release_asset_sha256": _sha("1"),
        "tool_catalog_sha256": campaign.canonical_digest(_TOOL_CATALOG),
        "template_tools_sha256": campaign.canonical_digest(_TEMPLATE_TOOLS),
        "mcp_server": corpus.OPENCODE_MCP_SERVER,
        "mcp_tools": corpus.OPENCODE_MCP_TOOLS,
        "context_management": corpus.ONLINE_COMPACTION,
        "context_window_tokens": 262144,
        "context_headroom_tokens": 20000,
        "tools": corpus.OPENCODE_TEMPLATE_TOOL_NAMES,
    }
    thinking = {"enable_thinking": True, "preserve_thinking": True}
    authorization = {
        "schema": corpus.SOURCE_AUTHORIZATION_SCHEMA,
        "authority": {
            "kind": "fleet_artifact_registry_immutable_v1",
            "artifact_key": "cyber/runs/synthetic/visible-reasoning/authorization",
            "version_index": 1,
            "content_sha256": _sha("0"),
        },
        "source": source,
        "qwen_target": qwen_target,
        "opencode": opencode,
        "thinking": thinking,
        "reasoning_visibility": "student_visible",
        "private_or_unknown_reasoning": "reject",
        "purpose": "student_visible_reasoning_plus_visible_actions",
    }
    authorization["registry_payload_sha256"] = corpus._registry_payload_sha256(authorization)
    authorization["authority"]["content_sha256"] = authorization["registry_payload_sha256"]
    authorization["sha256"] = digest_json(authorization)
    profile = {
        "schema": corpus.SOURCE_PROFILE_SCHEMA,
        "source": {**source, "authorization_sha256": authorization["sha256"]},
        "qwen_target": qwen_target,
        "opencode": opencode,
        "thinking": thinking,
        "serialization": {
            "schema": "cyber_qwen_opencode_template_roundtrip_v1",
            "roundtrip_fixture_sha256": file_sha256(roundtrip_path),
            "collection_template_sha256": _sha("f"),
            "training_template_sha256": _sha("f"),
            "serving_template_sha256": _sha("f"),
            "round_trip_verified": True,
        },
        "compaction": {
            "accepted_kind": corpus.EXACT_COMPACTION,
            "opaque_compaction_rejected": True,
        },
    }
    profile["sha256"] = digest_json(profile)
    train_identities = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] == "train"
    }
    task_selection = campaign.sealed(
        {
            "schema": campaign.SELECTION_SCHEMA,
            "inventory_sha256": inventory["sha256"],
            "family_split_sha256": split["sha256"],
            "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
            "family_role_anchor_sha256": role_anchor["sha256"],
            "runtime_bindings_sha256": runtime["sha256"],
            "task_validity_receipt_sha256": inventory["task_validity_receipt_sha256"],
            "tasks": [
                row
                for row in runtime["task_versions"]
                if (row["task_key"], row["task_version_id"]) in train_identities
            ],
            "held_out_task_version_count": len(split["tasks"]) - len(train_identities),
            "family_leakage_check": {
                "exact_identity_overlap": 0,
                "reviewed_family_overlap": 0,
                "held_out_roles": ["dev", "final_test"],
            },
        }
    )
    campaign_name = "synthetic-visible-reasoning-v1"
    selected_tasks = sorted(
        (
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "group_id": roles[(row["task_key"], row["task_version_id"])]["group_id"],
            }
            for row in task_selection["tasks"]
        ),
        key=lambda row: (row["task_key"], row["task_version_id"]),
    )
    waves = []
    universe = []
    for wave_index in range(corpus.CAMPAIGN_WAVES):
        first = wave_index * corpus.CAMPAIGN_ATTEMPTS_PER_WAVE + 1
        last = first + corpus.CAMPAIGN_ATTEMPTS_PER_WAVE - 1
        cells = []
        for task in selected_tasks:
            for attempt in range(first, last + 1):
                cell = {
                    "campaign_name": campaign_name,
                    **task,
                    "attempt": attempt,
                    "seed": corpus.CAMPAIGN_BASE_SEED + attempt - 1,
                }
                cell["cell_id"] = (
                    corpus.CELL_ID_PREFIX + digest_json(cell).removeprefix("sha256:")[:24]
                )
                cells.append(cell)
        universe.extend(cells)
        waves.append(
            {
                "wave": wave_index + 1,
                "attempt_first": first,
                "attempt_last": last,
                "seed_first": corpus.CAMPAIGN_BASE_SEED + first - 1,
                "seed_last": corpus.CAMPAIGN_BASE_SEED + last - 1,
                "task_versions": len(selected_tasks),
                "planned_cells": len(cells),
                "cell_intents_sha256": digest_json(cells),
            }
        )
    cell_universe_sha256 = digest_json(universe)
    planned_cells = len(universe)
    source_treatment = {
        "kind": "qwen_self",
        "model": profile["source"]["model"],
        "tokenizer": {
            "manifest_sha256": profile["qwen_target"]["tokenizer_sha256"],
            "backend_sha256": profile["qwen_target"]["tokenizer_backend_sha256"],
            "chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
        },
        "opencode": {
            **profile["opencode"],
            "provider_adapter": "@ai-sdk/openai-compatible",
            "max_output_tokens": 32_768,
            "max_model_requests": 600,
            "timeout_seconds": 28_800,
        },
        "thinking": {**profile["thinking"], "reasoning_visibility": "student_visible"},
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "base_seed": corpus.CAMPAIGN_BASE_SEED,
            "seed_rule": "base_seed_plus_attempt_minus_one",
        },
        "images": {
            "agent": "sha256:c7d048c98e6b8e52e5b76ab4006a7626b1ccf63a37bfa4b47ecd0fe9028e1f92",
            "proxy": (
                "ghcr.io/astral-sh/uv:python3.12-bookworm@"
                "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
            ),
        },
        "route": {
            "model": "qwen3.8-27b-base",
            "served_id": "qwen3.8-27b",
            "task_selection_sha256": task_selection["sha256"],
            "runtime_bindings_sha256": runtime["sha256"],
            "task_versions_sha256": digest_json([row["task_version_id"] for row in selected_tasks]),
            "task_version_count": len(selected_tasks),
            "catalog": {
                "engine": "sglang",
                "precision": "bf16",
                "tensor_parallel_size": 1,
            },
            "model_info": {
                "model_path": f"/scratch/models/qwen3.8-27b/{revision}",
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            },
            "server_info": {
                "model_path": f"/scratch/models/qwen3.8-27b/{revision}",
                "context_length": 262_144,
                "tp_size": 1,
                "dp_size": 8,
                "load_balance_method": "total_tokens",
                "quantization": None,
                "kv_cache_dtype": "fp8_e4m3",
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
            },
            "endpoint_origin": "https://inference.flt.build",
        },
    }
    campaign_plan = campaign.sealed(
        {
            "schema": corpus.CAMPAIGN_PLAN_SCHEMA,
            "campaign_name": campaign_name,
            "task_boundary": {"task_selection_sha256": task_selection["sha256"]},
            "collection": {
                "attempts_per_task_version": corpus.CAMPAIGN_ATTEMPTS_PER_TASK,
                "minimum_unique_supervised_tokens": corpus.MINIMUM_SUPERVISED_TOKENS,
                "minimum_successful_families": corpus.MINIMUM_SUCCESSFUL_FAMILIES,
                "maximum_family_target_token_fraction": (corpus.MAXIMUM_FAMILY_TOKEN_FRACTION),
            },
            "source_treatment": source_treatment,
            "identity": {
                "cell_identity_universe_sha256": cell_universe_sha256,
                "planned_unique_cell_ids": planned_cells,
            },
        }
    )
    wave_plan = campaign.sealed(
        {
            "schema": corpus.WAVE_PLAN_SCHEMA,
            "campaign_plan_sha256": campaign_plan["sha256"],
            "waves": waves,
        }
    )
    operation_authorization = {
        "schema": corpus.OPERATION_AUTHORIZATION_SCHEMA,
        "authority": {
            "kind": "fleet_artifact_registry_immutable_v1",
            "artifact_key": "cyber/runs/synthetic/visible-reasoning/operation",
            "version_index": 1,
            "content_sha256": _sha("0"),
        },
        "campaign_name": campaign_name,
        "campaign_plan_sha256": campaign_plan["sha256"],
        "wave_plan_sha256": wave_plan["sha256"],
        "cell_identity_universe_sha256": cell_universe_sha256,
        "task_selection_sha256": task_selection["sha256"],
        "operation_root_name": "synthetic-visible-reasoning-v1",
        "dedicated_ledger_id": _sha("3"),
        "policy": {
            "canonical_private_operation_root_required": True,
            "dedicated_empty_ledger_required": True,
            "exclusive_pre_mutation_intent_required": True,
            "ambiguous_external_mutation_replay_allowed": False,
            "same_cell_retry_allowed": False,
        },
    }
    operation_authorization["registry_payload_sha256"] = corpus._registry_payload_sha256(
        operation_authorization
    )
    operation_authorization["authority"]["content_sha256"] = operation_authorization[
        "registry_payload_sha256"
    ]
    operation_authorization["sha256"] = digest_json(operation_authorization)
    packet = {
        "schema": corpus.PACKET_SCHEMA,
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": authorization["sha256"],
        "campaign_name": campaign_name,
        "campaign_plan_sha256": campaign_plan["sha256"],
        "wave_plan_sha256": wave_plan["sha256"],
        "cell_identity_universe_sha256": cell_universe_sha256,
        "operation_authorization_sha256": operation_authorization["sha256"],
        "task_selection_sha256": task_selection["sha256"],
        "attempts_per_task_version": corpus.CAMPAIGN_ATTEMPTS_PER_TASK,
        "base_seed": corpus.CAMPAIGN_BASE_SEED,
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "runtime_bindings_sha256": runtime["sha256"],
        "training_data_eligible": True,
        "objective": "student_visible_reasoning_plus_visible_actions",
        "minimum_unique_supervised_tokens": corpus.MINIMUM_SUPERVISED_TOKENS,
        "minimum_successful_families": corpus.MINIMUM_SUCCESSFUL_FAMILIES,
        "maximum_family_target_token_fraction": corpus.MAXIMUM_FAMILY_TOKEN_FRACTION,
        "matched_action_only_required": True,
        "deduplication_order": [
            "source_session_identity",
            "normalized_trajectory_digest",
            "source_target_digest",
            "packed_window_payload_digest",
        ],
        "rejection_policy": {
            "private_or_unknown_reasoning": "reject",
            "opaque_compaction": "reject",
            "unknown_serialization": "reject",
            "heldout_family": "reject",
        },
    }
    packet["sha256"] = digest_json(packet)
    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic task"},
        {
            "role": "assistant",
            "student_visible_reasoning": "synthetic visible reasoning",
            "content": "",
            "tool_calls": [
                {
                    "id": "bash-1",
                    "type": "function",
                    "function": {
                        "name": "fleet_bash",
                        "arguments": {"script": "true"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "synthetic result", "tool_call_id": "bash-1"},
    ]
    record = {
        "schema": corpus.RECORD_SCHEMA,
        "record_id": "record-a",
        "wave": 1,
        "attempt": 1,
        "seed": corpus.CAMPAIGN_BASE_SEED,
        "source_profile_sha256": profile["sha256"],
        "lineage": {
            "task_key": candidate["task_key"],
            "task_version_id": candidate["task_version_id"],
            "group_id": candidate["group_id"],
        },
        "original_task_digest": _sha("7"),
        "evidence": {
            "source_session_identity_sha256": _sha("4"),
            "normalized_trajectory_sha256": digest_json(messages),
            "transcript_sha256": _sha("5"),
        },
        "reasoning_visibility": "student_visible",
        "messages": messages,
        "windows": [_window()],
        "compaction": {"kind": "none"},
    }
    record["cell_id"] = corpus._planned_cell_id(packet, {**record["lineage"], **record})
    record["content_digest"] = digest_json(record)
    selected = {
        "record_id": record["record_id"],
        "source_session_identity_sha256": record["evidence"]["source_session_identity_sha256"],
        "normalized_record_sha256": digest_json(record),
        "normalized_trajectory_sha256": record["evidence"]["normalized_trajectory_sha256"],
        "transcript_sha256": record["evidence"]["transcript_sha256"],
        "task_key": candidate["task_key"],
        "task_version_id": candidate["task_version_id"],
        "group_id": candidate["group_id"],
        "wave": record["wave"],
        "attempt": record["attempt"],
        "seed": record["seed"],
        "reasoning_visibility": "student_visible",
        "compaction_kind": "none",
    }
    selected["cell_id"] = record["cell_id"]
    evidence = {
        "schema": corpus.SUCCESS_EVIDENCE_SCHEMA,
        "authority": {
            "kind": "fleet_artifact_registry_immutable_v1",
            "artifact_key": "cyber/runs/synthetic/visible-reasoning/evidence",
            "version_index": 1,
            "content_sha256": _sha("6"),
        },
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": authorization["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "campaign_plan_sha256": campaign_plan["sha256"],
        "wave_plan_sha256": wave_plan["sha256"],
        "cell_identity_universe_sha256": cell_universe_sha256,
        "operation_authorization_sha256": operation_authorization["sha256"],
        "task_selection_sha256": task_selection["sha256"],
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "records": [
            {
                **{
                    name: selected[name]
                    for name in (
                        "record_id",
                        "cell_id",
                        "wave",
                        "attempt",
                        "seed",
                        "source_session_identity_sha256",
                        "normalized_record_sha256",
                        "normalized_trajectory_sha256",
                        "transcript_sha256",
                        "task_key",
                        "task_version_id",
                        "group_id",
                    )
                },
                "verifier_execution_sha256": _sha("8"),
                "outcome": "verified_success",
            }
        ],
    }
    evidence["registry_payload_sha256"] = corpus._registry_payload_sha256(evidence)
    evidence["authority"]["content_sha256"] = evidence["registry_payload_sha256"]
    evidence["sha256"] = digest_json(evidence)
    selection = {
        "schema": corpus.SELECTION_SCHEMA,
        "collection_packet_sha256": packet["sha256"],
        "verified_success_evidence_sha256": evidence["sha256"],
        "campaign_plan_sha256": campaign_plan["sha256"],
        "wave_plan_sha256": wave_plan["sha256"],
        "cell_identity_universe_sha256": cell_universe_sha256,
        "operation_authorization_sha256": operation_authorization["sha256"],
        "task_selection_sha256": task_selection["sha256"],
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "max_sessions_per_task_version": 1,
        "selected": [selected],
    }
    selection["sha256"] = digest_json(selection)
    source_census = {
        "schema": "cyber_qwen_opencode_student_visible_reasoning_census_v1",
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": authorization["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "private_selection_sha256": selection["sha256"],
        "success_evidence_sha256": evidence["sha256"],
        "sessions": {"candidates": 1, "verified_successes": 1, "selected": 1},
        "visibility": {"student_visible": 1, "private_or_unknown": 0, "absent": 0},
        "target_tokens": {"student_visible_reasoning": 1, "visible_action": 2},
        "compaction": {"none": 1, "exact_student_generated": 0, "opaque_rejected": 0},
    }
    source_census["sha256"] = digest_json(source_census)
    model_lock = {
        "repo": corpus.QWEN_REPOSITORY,
        "revision": revision,
        "tokenizer": {"manifest_sha256": _sha("d"), "files": []},
    }
    monkeypatch.setattr(
        corpus.dense_corpus,
        "local_tokenizer",
        lambda _lock, _root: (
            _Tokenizer(),
            {
                "repo": corpus.QWEN_REPOSITORY,
                "revision": revision,
                "backend_sha256": "e" * 64,
                "chat_template_sha256": "f" * 64,
            },
        ),
    )
    values = {
        "profile.json": profile,
        "source-authorization.json": authorization,
        "campaign-plan.json": campaign_plan,
        "wave-plan.json": wave_plan,
        "operation-authorization.json": operation_authorization,
        "packet.json": packet,
        "selection.json": selection,
        "success-evidence.json": evidence,
        "reasoning-census.json": source_census,
        "inventory.json": inventory,
        "split.json": split,
        "role-anchor.json": role_anchor,
        "lock.json": lock,
        "runtime.json": runtime,
        "task-selection.json": task_selection,
        "roundtrip.json": roundtrip,
        "tool-catalog.json": _TOOL_CATALOG,
        "model-lock.json": model_lock,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / name
        _write(path, value)
        paths[name] = path
    records = tmp_path / "records.private.jsonl"
    records.write_text(json.dumps(record) + "\n")
    paths["records"] = records
    config = {
        "schema": corpus.REQUEST_SCHEMA,
        "source_profile": _ref(paths["profile.json"]),
        "source_authorization": _ref(paths["source-authorization.json"]),
        "campaign_plan": _ref(paths["campaign-plan.json"]),
        "wave_plan": _ref(paths["wave-plan.json"]),
        "operation_authorization": _ref(paths["operation-authorization.json"]),
        "collection_packet": _ref(paths["packet.json"]),
        "selection": _ref(paths["selection.json"]),
        "success_evidence": _ref(paths["success-evidence.json"]),
        "reasoning_census": _ref(paths["reasoning-census.json"]),
        "inventory": _ref(paths["inventory.json"]),
        "family_split": _ref(paths["split.json"]),
        "role_anchor": _ref(paths["role-anchor.json"]),
        "protected_family_lock": _ref(paths["lock.json"]),
        "runtime_bindings": _ref(paths["runtime.json"]),
        "task_selection": _ref(paths["task-selection.json"]),
        "roundtrip_fixture": _ref(paths["roundtrip.json"]),
        "opencode_tool_catalog": _ref(paths["tool-catalog.json"]),
        "model_lock": _ref(paths["model-lock.json"]),
        "tokenizer_root": str(tmp_path / "tokenizer"),
        "records": _ref(records),
        "output": str(tmp_path / "corpus"),
    }
    return config, {
        "paths": paths,
        "profile": profile,
        "packet": packet,
        "selection": selection,
        "operation_authorization": operation_authorization,
        "campaign_plan": campaign_plan,
        "wave_plan": wave_plan,
        "record": record,
        "split": split,
        "candidate": candidate,
        "roles": roles,
    }


def _rewrite(path: Path, value: dict) -> None:
    _write(path, value)


def _reseal_record(record: dict) -> dict:
    record["content_digest"] = digest_json(
        {key: value for key, value in record.items() if key != "content_digest"}
    )
    return record


def test_materializes_token_only_visible_reasoning_corpus(tmp_path: Path, monkeypatch) -> None:
    config, _ = _fixture(tmp_path, monkeypatch)
    result = corpus.build(config, relative_to=tmp_path)

    assert result["submitted"] is False
    assert result["source_records"] == 1
    assert result["student_visible_reasoning_target_tokens"] == 1
    assert result["visible_action_target_tokens"] == 2
    assert result["sft_ready"] is False
    assert "train.parquet" not in json.dumps(result)
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    assert manifest["schema"] == corpus.CORPUS_SCHEMA
    assert manifest["validation_mode"] == "pending_reasoning_selection"
    assert "synthetic visible reasoning" not in json.dumps(manifest)
    rows = pq.read_table(tmp_path / "corpus" / "train-reasoning-plus-action.parquet").to_pylist()
    action_rows = pq.read_table(
        tmp_path / "corpus" / "train-matched-action-only.parquet"
    ).to_pylist()
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["loss_mask"] == [0, 0, 1, 1, 1]
    assert rows[0]["source_turn_sha256"].startswith("sha256:")
    assert rows[0]["source_target_sha256"].startswith("sha256:")
    assert action_rows[0]["input_ids"] == rows[0]["input_ids"]
    assert action_rows[0]["source_target_sha256"] == rows[0]["source_target_sha256"]
    assert action_rows[0]["paired_window_sha256"] == rows[0]["paired_window_sha256"]
    assert action_rows[0]["loss_mask"] == [0, 0, 0, 1, 1]
    assert manifest["matched_ablation"]["same_selected_windows"] is True
    assert manifest["matched_ablation"]["cross_arm_sha256"].startswith("sha256:")
    for arm_name in ("reasoning_plus_action", "matched_action_only"):
        arm_ref = manifest["arm_manifests"][arm_name]
        arm_manifest = json.loads((tmp_path / "corpus" / arm_ref["path"]).read_text())
        assert file_sha256(tmp_path / "corpus" / arm_ref["path"]) == arm_ref["file_sha256"]
        assert arm_manifest["sha256"] == arm_ref["logical_sha256"]
        assert (
            arm_manifest["paired_window_identity_sha256"]
            == manifest["matched_ablation"]["paired_window_identity_sha256"]
        )
        assert (
            arm_manifest["source_target_identity_sha256"]
            == manifest["matched_ablation"]["source_target_identity_sha256"]
        )
    assert (tmp_path / "corpus").stat().st_mode & 0o777 == 0o700
    for name in (
        "train-reasoning-plus-action.parquet",
        "train-matched-action-only.parquet",
        "reasoning-plus-action.manifest.json",
        "matched-action-only.manifest.json",
        "source-selection.private.jsonl",
        "coverage.private.json",
        "manifest.json",
        "MATERIALIZATION.json",
    ):
        assert (tmp_path / "corpus" / name).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change_assistant_turn_id", [False, True])
def test_rejects_duplicate_source_target_repacked_as_another_window(
    tmp_path: Path, monkeypatch, change_assistant_turn_id: bool
) -> None:
    _config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    duplicate = copy.deepcopy(record["windows"][0])
    duplicate["window_id"] = "repacked-window"
    duplicate["sequence_index"] = 99
    if change_assistant_turn_id:
        duplicate["assistant_turn_id"] = "forged-assistant-id"
    duplicate["window_payload_sha256"] = digest_json(
        {
            "sequence_index": duplicate["sequence_index"],
            "message_indices": duplicate["message_indices"],
            "target_message_index": duplicate["target_message_index"],
            "input_ids": duplicate["input_ids"],
            "prompt_token_count": duplicate["prompt_token_count"],
            "target_spans": duplicate["target_spans"],
        }
    )
    assert duplicate["window_payload_sha256"] != record["windows"][0]["window_payload_sha256"]
    original_identity = corpus._source_target_identity(record, record["windows"][0])
    duplicate_identity = corpus._source_target_identity(record, duplicate)
    assert duplicate_identity["source_turn_sha256"] == original_identity["source_turn_sha256"]
    if not change_assistant_turn_id:
        assert (
            duplicate_identity["source_target_sha256"] == original_identity["source_target_sha256"]
        )
    record["windows"].append(duplicate)
    _reseal_record(record)

    with pytest.raises(ValueError, match="duplicates a source assistant target"):
        corpus._record(record, state["profile"])


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda record: record["messages"][2].update({"thinking": "private"}),
            "private or unknown reasoning fields",
        ),
        (
            lambda record: record["messages"][2].update({"reasoning_content": "provider-private"}),
            "private or unknown reasoning fields",
        ),
        (
            lambda record: record["messages"][2].pop("student_visible_reasoning"),
            "assistant tool actions require explicit student-visible reasoning",
        ),
        (
            lambda record: record.update({"reasoning_visibility": "private_or_unknown"}),
            "private or unknown reasoning cannot be materialized",
        ),
        (
            lambda record: record.update({"compaction": {"kind": "opaque"}}),
            "opaque or unapproved compaction",
        ),
    ],
)
def test_rejects_private_unknown_and_opaque_reasoning(
    tmp_path: Path, monkeypatch, mutator, match: str
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    mutator(record)
    _reseal_record(record)
    _rewrite(state["paths"]["records"], record)
    config["records"] = _ref(state["paths"]["records"])
    with pytest.raises(ValueError, match=match):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_private_reasoning_nested_in_a_tool_argument(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    record["messages"][2]["tool_calls"][0]["function"]["arguments"]["analysis"] = "private"
    _reseal_record(record)
    _rewrite(state["paths"]["records"], record)
    config["records"] = _ref(state["paths"]["records"])
    with pytest.raises(ValueError, match="tool arguments"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_orphaned_opencode_tool_result(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    record["messages"][3]["tool_call_id"] = "unknown-call"
    _reseal_record(record)
    _rewrite(state["paths"]["records"], record)
    config["records"] = _ref(state["paths"]["records"])
    with pytest.raises(ValueError, match="orphaned or duplicated"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_teacher_source_profile() -> None:
    profile = {
        "schema": corpus.SOURCE_PROFILE_SCHEMA,
        "source": {
            "kind": "teacher_visible",
            "model_alias": "teacher",
            "model": {
                "repository": corpus.QWEN_REPOSITORY,
                "revision": "f" * 40,
                "session_model": "teacher",
            },
            "authorization_sha256": _sha("a"),
        },
        "qwen_target": {
            "repository": corpus.QWEN_REPOSITORY,
            "revision": "f" * 40,
            "tokenizer_sha256": _sha("b"),
            "tokenizer_backend_sha256": _sha("c"),
            "chat_template_sha256": _sha("d"),
        },
        "opencode": {
            "harness": corpus.OPENCODE_HARNESS,
            "harness_version": corpus.OPENCODE_VERSION,
            "release_asset_sha256": _sha("e"),
            "tool_catalog_sha256": _sha("f"),
            "template_tools_sha256": _sha("0"),
            "mcp_server": corpus.OPENCODE_MCP_SERVER,
            "mcp_tools": corpus.OPENCODE_MCP_TOOLS,
            "context_management": corpus.ONLINE_COMPACTION,
            "context_window_tokens": 262_144,
            "context_headroom_tokens": 20_000,
            "tools": corpus.OPENCODE_TEMPLATE_TOOL_NAMES,
        },
        "thinking": {"enable_thinking": True, "preserve_thinking": True},
        "serialization": {
            "schema": corpus.ROUNDTRIP_SCHEMA,
            "roundtrip_fixture_sha256": _sha("1"),
            "collection_template_sha256": _sha("d"),
            "training_template_sha256": _sha("d"),
            "serving_template_sha256": _sha("d"),
            "round_trip_verified": True,
        },
        "compaction": {
            "accepted_kind": corpus.EXACT_COMPACTION,
            "opaque_compaction_rejected": True,
        },
    }
    profile["sha256"] = digest_json(profile)
    with pytest.raises(ValueError, match="only the explicit Qwen-self"):
        corpus._profile(profile)


def test_rejects_unbound_registry_payload_before_reading_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    evidence = json.loads(state["paths"]["success-evidence.json"].read_text())
    evidence["authority"]["content_sha256"] = _sha("9")
    evidence["sha256"] = digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["success-evidence.json"], evidence)
    config["success_evidence"] = _ref(state["paths"]["success-evidence.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="immutable Registry payload"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_unbound_source_authorization_before_reading_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    authorization = json.loads(state["paths"]["source-authorization.json"].read_text())
    authorization["authority"]["content_sha256"] = _sha("9")
    authorization["sha256"] = digest_json(
        {key: value for key, value in authorization.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["source-authorization.json"], authorization)
    config["source_authorization"] = _ref(state["paths"]["source-authorization.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="source authorization does not bind"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_operation_authorization_campaign_drift_before_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    authorization = copy.deepcopy(state["operation_authorization"])
    authorization["campaign_plan_sha256"] = _sha("9")
    authorization["registry_payload_sha256"] = corpus._registry_payload_sha256(authorization)
    authorization["authority"]["content_sha256"] = authorization["registry_payload_sha256"]
    authorization["sha256"] = digest_json(
        {key: value for key, value in authorization.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["operation-authorization.json"], authorization)
    config["operation_authorization"] = _ref(state["paths"]["operation-authorization.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="packet-bound immutable artifact"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_selection_with_a_cell_from_the_wrong_wave_before_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    selection = copy.deepcopy(state["selection"])
    selection["selected"][0]["wave"] = 2
    selection["sha256"] = digest_json(
        {key: value for key, value in selection.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["selection.json"], selection)
    config["selection"] = _ref(state["paths"]["selection.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="campaign attempt and seed universe"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (
            {"role": "assistant", "content": "visible", "tool_call_id": "wrong"},
            "assistant message has an unsupported field",
        ),
        (
            {"role": "tool", "content": "result", "tool_calls": []},
            "tool result has an unsupported field",
        ),
        (
            {"role": "user", "content": "task", "tool_call_id": "wrong"},
            "system and user messages have an unsupported field",
        ),
    ],
)
def test_rejects_fields_outside_the_exact_opencode_message_contract(message, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        corpus._message(message)


def test_rejects_template_roundtrip_drift_before_records(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    profile = copy.deepcopy(state["profile"])
    profile["serialization"]["training_template_sha256"] = _sha("9")
    profile["sha256"] = digest_json(
        {key: value for key, value in profile.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["profile.json"], profile)
    config["source_profile"] = _ref(state["paths"]["profile.json"])
    with pytest.raises(ValueError, match="template round-trip"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_roundtrip_continuation_boundary_before_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    fixture = json.loads(state["paths"]["roundtrip.json"].read_text())
    fixture["cases"][0]["prompt_token_ids"] = [99]
    fixture["sha256"] = digest_json(
        {key: value for key, value in fixture.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["roundtrip.json"], fixture)
    config["roundtrip_fixture"] = _ref(state["paths"]["roundtrip.json"])
    profile = copy.deepcopy(state["profile"])
    profile["serialization"]["roundtrip_fixture_sha256"] = config["roundtrip_fixture"]["sha256"]
    profile["sha256"] = digest_json(
        {key: value for key, value in profile.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["profile.json"], profile)
    config["source_profile"] = _ref(state["paths"]["profile.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="round-trip fixture does not prove"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("adversary", ["missing", "schema", "order"])
def test_rejects_missing_different_or_reordered_opencode_tools_before_records(
    tmp_path: Path, monkeypatch, adversary: str
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    catalog = copy.deepcopy(_TOOL_CATALOG)
    if adversary == "missing":
        catalog.pop()
    elif adversary == "schema":
        catalog[0]["inputSchema"]["properties"]["script"]["type"] = "integer"
    else:
        catalog.reverse()
    _write(state["paths"]["tool-catalog.json"], catalog)
    config["opencode_tool_catalog"] = _ref(state["paths"]["tool-catalog.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="tool catalog differs from the source profile"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_a_caller_supplied_renderer_before_records(tmp_path: Path, monkeypatch) -> None:
    config, _state = _fixture(tmp_path, monkeypatch)
    config["renderer_adapter"] = {"path": "untrusted.py", "sha256": _sha("a")}
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="unknown fields"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_bad_protected_lock_before_reading_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    lock = json.loads(state["paths"]["lock.json"].read_text())
    lock["source_split_sha256"] = _sha("f")
    lock["sha256"] = digest_json({key: value for key, value in lock.items() if key != "sha256"})
    _rewrite(state["paths"]["lock.json"], lock)
    config["protected_family_lock"] = _ref(state["paths"]["lock.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="collection packet does not bind"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_heldout_selection_before_reading_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    heldout = next(row for row in state["split"]["tasks"] if row["split"] != "train")
    selection = copy.deepcopy(state["selection"])
    row = selection["selected"][0]
    row.update(
        {
            "task_key": heldout["task_key"],
            "task_version_id": heldout["task_version_id"],
            "group_id": heldout["group_id"],
        }
    )
    selection["sha256"] = digest_json(
        {key: value for key, value in selection.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["selection.json"], selection)
    config["selection"] = _ref(state["paths"]["selection.json"])
    original_file_sha256 = corpus.file_sha256

    def reject_record_hash(path: Path) -> str:
        if Path(path) == state["paths"]["records"]:
            raise AssertionError("held-out selection touched private records")
        return original_file_sha256(path)

    monkeypatch.setattr(corpus, "file_sha256", reject_record_hash)
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="held-out or unbound family"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_missing_success_evidence_before_reading_private_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    selection = copy.deepcopy(state["selection"])
    del selection["verified_success_evidence_sha256"]
    selection["sha256"] = digest_json(
        {key: value for key, value in selection.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["selection.json"], selection)
    config["selection"] = _ref(state["paths"]["selection.json"])
    monkeypatch.setattr(corpus, "iter_jsonl", lambda _path: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="does not bind the immutable success evidence"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_exact_compaction_requires_the_real_next_prompt_and_zero_masked_summary() -> None:
    first = _window("before")
    second = _window(
        "after",
        ids=[1, 2, 6, 7, 8],
        sequence_index=1,
        message_indices=[4, 5],
    )
    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic task"},
        {"role": "assistant", "content": "first target"},
        {"role": "tool", "content": "first result", "tool_call_id": "bash-1"},
        {"role": "assistant", "content": "synthetic summary"},
        {"role": "assistant", "content": "second target"},
    ]
    continuation = [44, 45]
    summary_request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "synthetic summary request"}],
            }
        ],
        "system": [],
        "tools": {},
    }
    summary_request["payload_sha256"] = digest_json(summary_request)
    boundary = {
        "boundary_id": "boundary-1",
        "parent_window_id": "before",
        "original_task_digest": _sha("a"),
        "prior_history_digest": digest_json(messages[:4]),
        "summary_message_index": 4,
        "summary_message_digest": digest_json(messages[4]),
        "continuation_token_sha256": digest_json(continuation),
        "continuation_token_ids": continuation,
        "continuation_tokens": len(continuation),
        "summary_request": summary_request,
        "summary_request_prompt_token_ids": [10, 11],
        "summary_generation_prompt_token_sha256": digest_json([10, 11]),
        "summary_generation_prompt_tokens": 2,
        "pre_compaction_prompt_token_sha256": first["prompt_token_sha256"],
        "pre_compaction_prompt_tokens": first["prompt_token_count"],
        "post_compaction_prompt_token_sha256": second["prompt_token_sha256"],
        "post_compaction_prompt_tokens": second["prompt_token_count"],
        "post_compaction_message_indices": [4],
        "next_target_window_id": "after",
        "next_target_prompt_token_sha256": second["prompt_token_sha256"],
    }
    accepted = {
        "kind": corpus.EXACT_COMPACTION,
        "boundaries": [boundary],
    }
    assert (
        corpus._compaction(
            accepted,
            {"before": first, "after": second},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )
        == accepted
    )
    record = {
        "messages": messages,
        "compaction": accepted,
    }
    corpus._rendered_compaction(_Tokenizer(), record)
    wrong_summary = copy.deepcopy(accepted)
    wrong_summary["boundaries"][0]["continuation_token_ids"] = [99]
    wrong_summary["boundaries"][0]["continuation_tokens"] = 1
    wrong_summary["boundaries"][0]["continuation_token_sha256"] = digest_json([99])
    with pytest.raises(ValueError, match="exact summary message"):
        corpus._rendered_compaction(
            _Tokenizer(), {"messages": messages, "compaction": wrong_summary}
        )
    wrong_prompt = copy.deepcopy(accepted)
    wrong_prompt["boundaries"][0]["next_target_prompt_token_sha256"] = _sha("d")
    with pytest.raises(ValueError, match="true next target prompt"):
        corpus._compaction(
            wrong_prompt,
            {"before": first, "after": second},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )
    wrong_request = copy.deepcopy(accepted)
    wrong_request["boundaries"][0]["summary_request"]["tools"] = _TEMPLATE_TOOLS
    wrong_request["boundaries"][0]["summary_request"]["payload_sha256"] = digest_json(
        {
            name: wrong_request["boundaries"][0]["summary_request"][name]
            for name in ("messages", "system", "tools")
        }
    )
    with pytest.raises(ValueError, match=r"system=\[\] and tools=\{\}"):
        corpus._compaction(
            wrong_request,
            {"before": first, "after": second},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )
    with pytest.raises(ValueError, match="opaque or unapproved compaction"):
        corpus._compaction(
            accepted,
            {"before": first, "after": second},
            messages,
            source_kind="teacher_visible",
            original_task_digest=_sha("a"),
        )


def test_compaction_rejects_preboundary_parent_and_unrelated_later_window() -> None:
    first = _window("before")
    second = _window("after", ids=[1, 2, 6, 7, 8], sequence_index=1, message_indices=[4, 5])
    later = _window("later", ids=[1, 2, 9, 10, 11], sequence_index=2, message_indices=[4, 5, 6, 7])
    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic task"},
        {"role": "assistant", "content": "first target"},
        {"role": "tool", "content": "first result", "tool_call_id": "bash-1"},
        {"role": "assistant", "content": "synthetic summary"},
        {"role": "assistant", "content": "second target"},
        {"role": "user", "content": "later request"},
        {"role": "assistant", "content": "later target"},
    ]
    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "synthetic summary request"}],
            }
        ],
        "system": [],
        "tools": {},
    }
    request["payload_sha256"] = digest_json(request)
    boundary = {
        "boundary_id": "boundary-1",
        "parent_window_id": "before",
        "original_task_digest": _sha("a"),
        "prior_history_digest": digest_json(messages[:4]),
        "summary_message_index": 4,
        "summary_message_digest": digest_json(messages[4]),
        "continuation_token_sha256": digest_json([44, 45]),
        "continuation_token_ids": [44, 45],
        "continuation_tokens": 2,
        "summary_request": request,
        "summary_request_prompt_token_ids": [10, 11],
        "summary_generation_prompt_token_sha256": digest_json([10, 11]),
        "summary_generation_prompt_tokens": 2,
        "pre_compaction_prompt_token_sha256": first["prompt_token_sha256"],
        "pre_compaction_prompt_tokens": first["prompt_token_count"],
        "post_compaction_prompt_token_sha256": second["prompt_token_sha256"],
        "post_compaction_prompt_tokens": second["prompt_token_count"],
        "post_compaction_message_indices": [4],
        "next_target_window_id": "after",
        "next_target_prompt_token_sha256": second["prompt_token_sha256"],
    }
    accepted = {"kind": corpus.EXACT_COMPACTION, "boundaries": [boundary]}
    assert (
        corpus._compaction(
            accepted,
            {"before": first, "after": second, "later": later},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )
        == accepted
    )

    unrelated = copy.deepcopy(later)
    unrelated["message_indices"] = [0, 1, 6, 7]
    with pytest.raises(ValueError, match="does not descend"):
        corpus._compaction(
            accepted,
            {"before": first, "after": second, "later": unrelated},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )

    post_boundary_parent = copy.deepcopy(first)
    post_boundary_parent["message_indices"] = [0, 1, 5]
    post_boundary_parent["target_message_index"] = 5
    with pytest.raises(ValueError, match="lineage is not bound"):
        corpus._compaction(
            accepted,
            {"before": post_boundary_parent, "after": second, "later": later},
            messages,
            source_kind="qwen_self",
            original_task_digest=_sha("a"),
        )


def test_compaction_coverage_counts_all_targets_after_a_boundary() -> None:
    record = {
        "compaction": {
            "kind": corpus.EXACT_COMPACTION,
            "boundaries": [{"next_target_window_id": "after"}],
        },
        "windows": [
            {"window_id": "before", "sequence_index": 0},
            {"window_id": "after", "sequence_index": 1},
            {"window_id": "later", "sequence_index": 2},
        ],
    }
    assert corpus._compacted_target_window_ids(record) == {"after", "later"}


def test_rejects_local_tokenizer_boundary_drift(tmp_path: Path, monkeypatch) -> None:
    _config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    window = record["windows"][0]
    window["input_ids"] = [1, 2, 3, 4, 6]
    window["serialized_token_ids_sha256"] = digest_json(window["input_ids"])
    window["target_spans"][1]["token_ids_sha256"] = digest_json(window["input_ids"][3:5])
    window["window_payload_sha256"] = digest_json(
        {
            "sequence_index": window["sequence_index"],
            "message_indices": window["message_indices"],
            "target_message_index": window["target_message_index"],
            "input_ids": window["input_ids"],
            "prompt_token_count": window["prompt_token_count"],
            "target_spans": window["target_spans"],
        }
    )
    _reseal_record(record)
    checked = corpus._record(record, state["profile"])
    with pytest.raises(ValueError, match="template serialization differs"):
        corpus._rendered_window(_Tokenizer(), checked, checked["windows"][0], _TEMPLATE_TOOLS)


@pytest.mark.parametrize("adversary", ["swapped_labels", "resegmented_boundary"])
def test_rejects_self_consistent_caller_span_adversary(
    tmp_path: Path, monkeypatch, adversary: str
) -> None:
    """The pinned template, not caller labels, owns the component boundary."""

    _config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    window = record["windows"][0]
    if adversary == "swapped_labels":
        window["target_spans"][0]["kind"] = "visible_action"
        window["target_spans"][1]["kind"] = "student_visible_reasoning"
    else:
        window["target_spans"][0]["token_end"] = 4
        window["target_spans"][0]["token_ids_sha256"] = digest_json(window["input_ids"][2:4])
        window["target_spans"][1]["token_start"] = 4
        window["target_spans"][1]["token_ids_sha256"] = digest_json(window["input_ids"][4:5])
    window["window_payload_sha256"] = digest_json(
        {
            "sequence_index": window["sequence_index"],
            "message_indices": window["message_indices"],
            "target_message_index": window["target_message_index"],
            "input_ids": window["input_ids"],
            "prompt_token_count": window["prompt_token_count"],
            "target_spans": window["target_spans"],
        }
    )
    _reseal_record(record)

    # The adversarial record is internally self-consistent. It must still fail
    # once the exact Qwen template independently derives the two components.
    checked = corpus._record(record, state["profile"])
    with pytest.raises(ValueError, match="template-derived reasoning/action components"):
        corpus._rendered_window(_Tokenizer(), checked, checked["windows"][0], _TEMPLATE_TOOLS)
