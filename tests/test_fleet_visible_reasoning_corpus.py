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


class _Tokenizer:
    """A synthetic local tokenizer that rejects any unpinned render treatment."""

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": len(messages) in {2, 4},
            "tools": [],
            "enable_thinking": True,
        }
        if len(messages) == 2:
            return [1, 2]
        if len(messages) == 3:
            return [1, 2, 3, 4, 5]
        if len(messages) == 4:
            return [10, 11]
        if len(messages) == 5:
            return [10, 11, 44, 45]
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
                        "content": "synthetic visible reasoning",
                        "tool_calls": [
                            {
                                "id": "bash-1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"script": "true"}},
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
        "tool_catalog_sha256": _sha("2"),
        "context_management": corpus.ONLINE_COMPACTION,
        "context_window_tokens": 262144,
        "context_headroom_tokens": 20000,
        "tools": ["bash", "submit_report"],
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
    packet = {
        "schema": corpus.PACKET_SCHEMA,
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": authorization["sha256"],
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "runtime_bindings_sha256": runtime["sha256"],
        "training_data_eligible": True,
        "objective": "student_visible_reasoning_plus_visible_actions",
        "minimum_unique_supervised_tokens": corpus.MINIMUM_SUPERVISED_TOKENS,
        "maximum_family_target_token_fraction": corpus.MAXIMUM_FAMILY_TOKEN_FRACTION,
        "deduplication_order": [
            "source_session_identity",
            "normalized_trajectory_digest",
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
            "content": "synthetic visible reasoning",
            "tool_calls": [
                {
                    "id": "bash-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "true"}},
                }
            ],
        },
        {"role": "tool", "content": "synthetic result", "tool_call_id": "bash-1"},
    ]
    record = {
        "schema": corpus.RECORD_SCHEMA,
        "record_id": "record-a",
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
        "attempt": 1,
        "reasoning_visibility": "student_visible",
        "compaction_kind": "none",
    }
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
        "packet.json": packet,
        "selection.json": selection,
        "success-evidence.json": evidence,
        "reasoning-census.json": source_census,
        "inventory.json": inventory,
        "split.json": split,
        "role-anchor.json": role_anchor,
        "lock.json": lock,
        "runtime.json": runtime,
        "roundtrip.json": roundtrip,
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
        "collection_packet": _ref(paths["packet.json"]),
        "selection": _ref(paths["selection.json"]),
        "success_evidence": _ref(paths["success-evidence.json"]),
        "reasoning_census": _ref(paths["reasoning-census.json"]),
        "inventory": _ref(paths["inventory.json"]),
        "family_split": _ref(paths["split.json"]),
        "role_anchor": _ref(paths["role-anchor.json"]),
        "protected_family_lock": _ref(paths["lock.json"]),
        "runtime_bindings": _ref(paths["runtime.json"]),
        "roundtrip_fixture": _ref(paths["roundtrip.json"]),
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
    rows = pq.read_table(tmp_path / "corpus" / "train.parquet").to_pylist()
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["loss_mask"] == [0, 0, 1, 1, 1]
    assert (tmp_path / "corpus").stat().st_mode & 0o777 == 0o700
    for name in (
        "train.parquet",
        "source-selection.private.jsonl",
        "coverage.private.json",
        "manifest.json",
        "MATERIALIZATION.json",
    ):
        assert (tmp_path / "corpus" / name).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda record: record["messages"][2].update({"thinking": "private"}),
            "private or unknown reasoning fields",
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
            "context_management": corpus.ONLINE_COMPACTION,
            "context_window_tokens": 262_144,
            "context_headroom_tokens": 20_000,
            "tools": ["bash", "submit_report"],
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
    with pytest.raises(ValueError, match="opaque or unapproved compaction"):
        corpus._compaction(
            accepted,
            {"before": first, "after": second},
            messages,
            source_kind="teacher_visible",
            original_task_digest=_sha("a"),
        )


def test_compaction_coverage_counts_only_targets_after_a_boundary() -> None:
    record = {
        "compaction": {
            "kind": corpus.EXACT_COMPACTION,
            "boundaries": [{"next_target_window_id": "after"}],
        },
        "windows": [{"window_id": "before"}, {"window_id": "after"}],
    }
    assert corpus._compacted_target_window_ids(record) == {"after"}


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
        corpus._rendered_window(_Tokenizer(), checked, checked["windows"][0])
