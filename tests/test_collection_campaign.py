"""Offline contracts for family-safe Qwen trajectory-collection rendering."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from training import collection_campaign as campaign
from training import task_family_split


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _inventory() -> dict:
    rows = []
    for index in range(6):
        rows.append(
            {
                "task_key": f"task-{index}",
                "task_version_id": f"00000000-0000-4000-8000-{index:012d}",
                "lineage": {
                    "application": f"app-{index % 2}",
                    "environment": f"env-{index % 2}",
                    "difficulty": "medium",
                    "vulnerability_family": [f"family-{index}"],
                    "task_family": f"task-family-{index}",
                },
            }
        )
    return campaign.sealed(
        {
            "schema": campaign.INVENTORY_SCHEMA,
            "task_validity_receipt_sha256": _sha("a"),
            "task_versions": rows,
        }
    )


def _split(inventory: dict) -> dict:
    return task_family_split.build(
        inventory["task_versions"],
        inventory_sha256=inventory["sha256"],
        seed="collection-test-v1",
        ratios={"train": 1 / 3, "dev": 1 / 3, "final_test": 1 / 3},
        max_group_task_version_fraction=0.6,
    )


def _bindings(inventory: dict) -> dict:
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


def _request(*, source_kind: str = "self") -> dict:
    request = {
        "schema": campaign.REQUEST_SCHEMA,
        "campaign_name": f"q38-collection-{source_kind}-v1",
        "source_kind": source_kind,
        "source_model": {
            "repository": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "session_model": "qwen/qwen3.8-27b",
        },
        "template_sha256": _sha("8"),
        "source_authorization_receipt_sha256": _sha("b"),
        "route": {
            "name": "source",
            "served_id": "qwen3.8-27b",
            "catalog": {
                "engine": "sglang",
                "precision": "bf16",
                "tensor_parallel_size": 1,
            },
            "model_info": {
                "model_path": "/models/qwen",
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            },
            "server_info": {
                "model_path": "/models/qwen",
                "context_length": 262144,
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
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "release_asset_sha256": _sha("c"),
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": campaign.ONLINE_COMPACTION,
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": _sha("d"),
        },
        "images": {"agent": _sha("e"), "proxy": _sha("f")},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "concurrency": 4,
        "attempts_per_task": 4,
        "target_unique_visible_action_tokens": campaign.MINIMUM_VISIBLE_TARGET_TOKENS,
        "reasoning_policy": campaign.VISIBLE_ACTIONS_ONLY,
        "offline_compaction_policy": campaign.OPAQUE_COMPACTION_REJECT,
    }
    if source_kind == "teacher":
        request["source_model"] = {
            "repository": "example/strong-teacher",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "session_model": "teacher/strong",
        }
        request["route"]["served_id"] = "strong-teacher"
        request["route"]["model_info"]["model_type"] = "teacher"
        request["teacher_strength_receipt_sha256"] = _sha("9")
    return request


def test_render_is_eval_compatible_and_contains_only_train_tasks() -> None:
    inventory = _inventory()
    split = _split(inventory)
    bindings = _bindings(inventory)
    rendered = campaign.render(_request(), inventory, split, bindings)

    selection = rendered["task-selection.json"]
    roles = {(row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]}
    assert selection["schema"] == campaign.SELECTION_SCHEMA
    assert selection["family_leakage_check"] == {
        "exact_identity_overlap": 0,
        "reviewed_family_overlap": 0,
        "held_out_roles": ["dev", "final_test"],
    }
    assert all(
        roles[(row["task_key"], row["task_version_id"])] == "train" for row in selection["tasks"]
    )
    assert len(selection["tasks"]) == split["counts"]["train"]["task_versions"]

    packet = rendered["collection-packet.json"]
    # render() runs the existing compile_eval locally without preflight or a
    # request; the sealed packet keeps the source/admission facts that generic
    # eval plans intentionally do not retain.
    assert packet["training_data_eligible"] is True
    assert packet["source"]["model_alias"] == "source"
    assert packet["source"]["template_sha256"] == _sha("8")
    assert packet["corpus_scope"] == {
        "current": "verified_success_visible_actions_only_v1",
        "visible_reasoning_included": False,
        "future_visible_reasoning_requires": [
            "separate_immutable_campaign_packet",
            "student_visible_source_evidence",
            "explicit_authorization_and_safety_evidence",
        ],
    }
    assert packet["generic_plan_source_job_id"] is None
    assert packet["eval_plan_sha256"].startswith("sha256:")
    assert packet["admission_policy"]["eligible_terminal_outcome"] == "verified_success_v1"
    assert packet["admission_policy"]["minimum_non_submit_tool_responses"] == 1
    assert packet["admission_policy"]["minimum_completed_non_submit_tool_rounds"] == 1
    assert packet["admission_policy"]["maximum_submit_report_response_fraction"] == 0.5
    assert packet["admission_policy"]["maximum_submit_report_target_token_fraction"] == 0.5
    assert packet["admission_policy"]["maximum_family_target_token_fraction"] == 0.25
    assert packet["admission_policy"]["reasoning_policy"] == campaign.VISIBLE_ACTIONS_ONLY
    assert (
        packet["admission_policy"]["offline_compaction_policy"] == campaign.OPAQUE_COMPACTION_REJECT
    )
    assert packet["admission_policy"]["adapter_must_bind"] == [
        "collection_packet_sha256",
        "eval_plan_sha256",
    ]
    assert packet["admission_policy"]["minimum_unique_visible_action_target_tokens"] == 20_000_000


def test_requires_exact_runtime_bindings_and_split_protection() -> None:
    inventory = _inventory()
    split = _split(inventory)
    bindings = _bindings(inventory)

    wrong_bindings = copy.deepcopy(bindings)
    wrong_bindings["metadata_inventory_sha256"] = _sha("0")
    wrong_bindings = campaign.sealed(
        {key: value for key, value in wrong_bindings.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="different metadata inventory"):
        campaign.render(_request(), inventory, split, wrong_bindings)

    mutated_split = copy.deepcopy(split)
    mutated_split["tasks"][0]["split"] = "train"
    mutated_split = task_family_split.canonical_digest(
        {key: value for key, value in mutated_split.items() if key != "sha256"}
    )
    # A recomputed checksum cannot hide an assignment that differs from the
    # deterministic, family-safe split algorithm.
    broken_split = copy.deepcopy(split)
    broken_split["tasks"][0]["split"] = "train"
    broken_split["sha256"] = mutated_split
    with pytest.raises(ValueError, match="split drift"):
        campaign.render(_request(), inventory, broken_split, bindings)


def test_rejects_hidden_reasoning_and_requires_teacher_strength_receipt() -> None:
    inventory = _inventory()
    split = _split(inventory)
    bindings = _bindings(inventory)

    hidden = _request()
    hidden["reasoning_policy"] = "include_hidden_reasoning"
    with pytest.raises(ValueError, match="visible actions only"):
        campaign.render(hidden, inventory, split, bindings)

    teacher = _request(source_kind="teacher")
    teacher.pop("teacher_strength_receipt_sha256")
    with pytest.raises(ValueError, match="teacher-strength"):
        campaign.render(teacher, inventory, split, bindings)

    teacher = campaign.render(_request(source_kind="teacher"), inventory, split, bindings)
    assert teacher["collection-packet.json"]["source"]["kind"] == "teacher"
    assert "teacher_strength_receipt_sha256" in teacher["collection-packet.json"]["source"]


def test_requires_qualified_262k_opencode_actions_only_harness() -> None:
    inventory = _inventory()
    split = _split(inventory)
    bindings = _bindings(inventory)

    short_context = _request()
    short_context["harness"]["context_window_size"] = 98304
    with pytest.raises(ValueError, match="262K OpenCode"):
        campaign.render(short_context, inventory, split, bindings)

    wrong_tools = _request()
    wrong_tools["harness"]["tools"] = ["bash"]
    with pytest.raises(ValueError, match="262K OpenCode"):
        campaign.render(wrong_tools, inventory, split, bindings)


def test_write_and_check_are_create_once_and_no_submit(tmp_path: Path) -> None:
    inventory = _inventory()
    split = _split(inventory)
    bindings = _bindings(inventory)
    rendered = campaign.render(_request(), inventory, split, bindings)
    output = tmp_path / "packet"
    campaign.write_once(output, rendered)
    campaign.check(output, rendered)
    with pytest.raises(FileExistsError, match="already exists"):
        campaign.write_once(output, rendered)

    request, inventory_path, split_path, bindings_path = (
        tmp_path / "request.json",
        tmp_path / "inventory.json",
        tmp_path / "split.json",
        tmp_path / "bindings.json",
    )
    for path, value in (
        (request, _request()),
        (inventory_path, inventory),
        (split_path, split),
        (bindings_path, bindings),
    ):
        path.write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.collection_campaign",
            "--request",
            str(request),
            "--inventory",
            str(inventory_path),
            "--family-split",
            str(split_path),
            "--runtime-bindings",
            str(bindings_path),
            "--output",
            str(output),
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "checked": ["collection-packet.json", "eval-config.json", "task-selection.json"],
        "submitted": False,
    }
