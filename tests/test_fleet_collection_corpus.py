"""Offline contracts for private, family-safe Fleet corpus materialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import collection_campaign as campaign
from training import fleet_collection_corpus as corpus
from training import task_family_split
from training.io import digest_json, file_sha256


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "sha256"}
    return {**value, "sha256": "sha256:" + digest(value)}


class _Tokenizer:
    def __len__(self) -> int:
        return 256

    def apply_chat_template(self, messages, **kwargs):
        return [1, max(2, len(messages))]


def _helper(messages, tokenizer, **kwargs):
    assistant = messages[0]["role"] == "assistant"
    return [1, 2, 3], [0, int(assistant), int(assistant)], None


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


def _request() -> dict:
    return {
        "schema": campaign.REQUEST_SCHEMA,
        "campaign_name": "q38-collection-teacher-v1",
        "source_kind": "teacher",
        "source_model": {
            "repository": "example/strong-teacher",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "session_model": "teacher/strong",
        },
        "template_sha256": _sha("b"),
        "source_authorization_receipt_sha256": _sha("c"),
        "teacher_strength_receipt_sha256": _sha("d"),
        "route": {
            "name": "source",
            "served_id": "strong-teacher",
            "catalog": {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 1},
            "model_info": {
                "model_path": "/models/teacher",
                "model_type": "teacher",
                "architectures": ["Teacher"],
            },
            "server_info": {
                "model_path": "/models/teacher",
                "context_length": 262144,
                "tp_size": 1,
                "dp_size": 1,
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
            "release_asset_sha256": _sha("e"),
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": campaign.ONLINE_COMPACTION,
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": _sha("f"),
        },
        "images": {"agent": _sha("1"), "proxy": _sha("2")},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "concurrency": 4,
        "attempts_per_task": 4,
        "target_unique_visible_action_tokens": campaign.MINIMUM_VISIBLE_TARGET_TOKENS,
        "reasoning_policy": campaign.VISIBLE_ACTIONS_ONLY,
        "offline_compaction_policy": campaign.OPAQUE_COMPACTION_REJECT,
    }


def _write(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _ref(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bash-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "true"}},
                }
            ],
        },
        {"role": "tool", "content": "synthetic result", "tool_call_id": "bash-1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "report-1",
                    "type": "function",
                    "function": {
                        "name": "submit_report",
                        "arguments": {"flag": "synthetic", "explanation": ""},
                    },
                }
            ],
        },
        {"role": "tool", "content": "accepted", "tool_call_id": "report-1"},
    ]


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    inventory = _inventory()
    split = task_family_split.build(
        inventory["task_versions"],
        inventory_sha256=inventory["sha256"],
        seed="materializer-test-v1",
        ratios={"train": 1 / 3, "dev": 1 / 3, "final_test": 1 / 3},
        max_group_task_version_fraction=0.6,
    )
    runtime = _runtime(inventory)
    rendered = campaign.render(_request(), inventory, split, runtime)
    roles = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    candidate = rendered["task-selection.json"]["tasks"][0]
    role = roles[(candidate["task_key"], candidate["task_version_id"])]
    packet = rendered["collection-packet.json"]
    eval_config = rendered["eval-config.json"]
    messages = _messages()
    harness = {
        "treatment_sha256": "sha256:" + digest(eval_config["harness"]),
        "tool_catalog_sha256": eval_config["harness"]["tool_catalog_sha256"],
    }
    record = {
        "schema": corpus.RECORD_SCHEMA,
        "record_id": "session-a",
        "campaign_plan_sha256": packet["eval_plan_sha256"],
        "cell_sha256": _sha("3"),
        "source": {
            "model_alias": "source",
            "model": _request()["source_model"],
            "harness": harness,
            "template_sha256": _request()["template_sha256"],
        },
        "lineage": {
            "task_key": candidate["task_key"],
            "task_version_id": candidate["task_version_id"],
            "group_id": role["group_id"],
        },
        "content_policy": dict(corpus._POLICY),
        "ingestion": {
            "normalized_trajectory_sha256": digest_json(messages),
            "transcript_sha256": _sha("4"),
        },
        "messages": messages,
    }
    record["content_digest"] = digest_json(record)
    selected = {
        "session_id": record["record_id"],
        "campaign_plan_sha256": packet["eval_plan_sha256"],
        "cell_sha256": record["cell_sha256"],
        "task_key": candidate["task_key"],
        "task_version_id": candidate["task_version_id"],
        "group_id": role["group_id"],
        "attempt": 1,
        "source_kind": "teacher",
        "model": _request()["source_model"],
        "harness": harness,
        "template_sha256": _request()["template_sha256"],
        "normalized_record_sha256": digest_json(record),
        "normalized_trajectory_sha256": record["ingestion"]["normalized_trajectory_sha256"],
        "transcript_sha256": record["ingestion"]["transcript_sha256"],
    }
    heldout_groups = sorted({row["group_id"] for row in split["tasks"] if row["split"] != "train"})
    lock = _seal(
        {
            "schema": "cyber_protected_task_family_lock_v1",
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": heldout_groups,
        }
    )
    selection = _seal(
        {
            "schema": "cyber_fleet_collection_selection_v1",
            "artifact_kind": "metadata_evidence_handoff_only",
            "trainable_corpus_created": False,
            "parquet_created": False,
            "source_text_read": False,
            "next_required_gate": (
                "bind_private_normalized_records_then_run_existing_dense_sft_builder"
            ),
            "campaign_plan_sha256": packet["eval_plan_sha256"],
            "catalog_inventory_sha256": inventory["sha256"],
            "family_split_sha256": split["sha256"],
            "protected_family_lock_sha256": lock["sha256"],
            "source_kind": "teacher",
            "source_model_alias": "source",
            "source_model": _request()["source_model"],
            "harness_treatment_sha256": harness["treatment_sha256"],
            "tool_catalog_sha256": harness["tool_catalog_sha256"],
            "template_sha256": _request()["template_sha256"],
            "max_sessions_per_task_version": 2,
            "selected": [selected],
        }
    )
    receipt = _seal(
        {
            "schema": "cyber_fleet_collection_admission_receipt_v1",
            "artifact_kind": "metadata_evidence_handoff_only",
            "trainable_corpus_created": False,
            "parquet_created": False,
            "source_text_read": False,
            "next_required_gate": (
                "bind_private_normalized_records_then_run_existing_dense_sft_builder"
            ),
            "campaign_plan_sha256": selection["campaign_plan_sha256"],
            "input_file_sha256": {
                name: _sha(character)
                for name, character in zip(
                    ("campaign", "inventory", "family_split", "protected_family_lock", "attempts"),
                    "56789",
                    strict=True,
                )
            },
            "catalog_inventory_sha256": selection["catalog_inventory_sha256"],
            "family_split_sha256": selection["family_split_sha256"],
            "protected_family_lock_sha256": selection["protected_family_lock_sha256"],
            "protected_family_count": len(heldout_groups),
            "source_kind": "teacher",
            "source_model_alias": "source",
            "source_model_identity_sha256": "sha256:" + digest(selection["source_model"]),
            "harness_treatment_sha256": selection["harness_treatment_sha256"],
            "template_sha256": selection["template_sha256"],
            "tool_catalog_sha256": selection["tool_catalog_sha256"],
            "max_sessions_per_task_version": 2,
            "counts": {
                "attempt_metadata_records": 1,
                "planned_cells_for_source_model": 1,
                "train_cells_for_source_model": 1,
                "admitted_sessions": 1,
                "admitted_task_versions": 1,
                "rejections": {reason: 0 for reason in corpus.admission._REJECTION_REASONS},
            },
            "policy": {
                "training_data_eligible_required": True,
                "success_requires_completed_authoritative_verifier": True,
                "visible_actions_only": True,
                "private_or_unknown_reasoning_rejected": True,
                "opaque_or_unapproved_compaction_rejected": True,
                "exact_session_and_trajectory_deduplication": True,
                "per_task_version_cap_is_deterministic": True,
                "heldout_families_admitted": 0,
                "raw_source_payload_read": False,
            },
            "selection_sha256": selection["sha256"],
        }
    )
    values = {
        "selection.json": selection,
        "receipt.json": receipt,
        "packet.json": packet,
        "task-selection.json": rendered["task-selection.json"],
        "eval-config.json": eval_config,
        "inventory.json": inventory,
        "split.json": split,
        "lock.json": lock,
        "runtime.json": runtime,
        "model-lock.json": {},
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / name
        _write(path, value)
        paths[name] = path
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")
    paths["records.jsonl"] = records
    helper = tmp_path / "helper.py"
    helper.write_text("synthetic helper")
    paths["helper.py"] = helper
    config = {
        "schema": corpus.SCHEMA,
        "admission_selection": _ref(paths["selection.json"]),
        "admission_receipt": _ref(paths["receipt.json"]),
        "collection_packet": _ref(paths["packet.json"]),
        "collection_task_selection": _ref(paths["task-selection.json"]),
        "collection_eval_config": _ref(paths["eval-config.json"]),
        "inventory": _ref(paths["inventory.json"]),
        "family_split": _ref(paths["split.json"]),
        "protected_family_lock": _ref(paths["lock.json"]),
        "runtime_bindings": _ref(paths["runtime.json"]),
        "records": _ref(records),
        "model_lock": _ref(paths["model-lock.json"]),
        "tokenizer_root": str(tmp_path),
        "native_helper": _ref(helper),
        "max_length": 64,
        "context_tokens": 16,
        "output": str(tmp_path / "corpus"),
    }
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda lock, root: (_Tokenizer(), {"repo": "Qwen/Qwen3.8-27B", "revision": "f" * 40}),
    )
    monkeypatch.setattr(corpus, "native_helper", lambda path: _helper)
    return config, {"paths": paths, "record": record, "selection": selection, "receipt": receipt}


def _rewrite(path: Path, value: dict) -> None:
    _write(path, value)


def test_materializes_only_bound_visible_action_windows(tmp_path: Path, monkeypatch) -> None:
    config, _ = _fixture(tmp_path, monkeypatch)
    result = corpus.build(config, relative_to=tmp_path)

    assert result["submitted"] is False
    assert result["source_sessions"] == 1
    assert result["visible_action_windows"] == 1
    assert result["sft_ready"] is False
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    assert manifest["validation_mode"] == "collection_pending_target"
    assert manifest["collection_materialization"]["sft_ready"] is False
    assert manifest["catalog_provenance"]["heldout_task_families_excluded_across_all_versions"]
    coverage = json.loads((tmp_path / "corpus" / "coverage.private.json").read_text())
    assert (
        coverage["target_coverage"] == "every fitting visible assistant target appears exactly once"
    )
    rows = pq.read_table(tmp_path / "corpus" / "train.parquet").to_pylist()
    assert len(rows) == 1
    assert rows[0]["source_collection_packet_sha256"] == coverage["collection_packet_sha256"]
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
            lambda record: record["messages"][2].update({"thinking": "synthetic private thought"}),
            "private or unknown message fields",
        ),
        (
            lambda record: record["content_policy"].update({"compaction": "opaque"}),
            "private reasoning or unapproved compaction",
        ),
    ],
)
def test_rejects_private_reasoning_and_opaque_compaction(
    tmp_path: Path, monkeypatch, mutator, match: str
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    mutator(record)
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    _rewrite(state["paths"]["records.jsonl"], record)
    config["records"] = _ref(state["paths"]["records.jsonl"])

    with pytest.raises(ValueError, match=match):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_untagged_assistant_prose_as_unknown_reasoning(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    record["messages"][2]["content"] = "untagged synthetic reasoning"
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    _rewrite(state["paths"]["records.jsonl"], record)
    config["records"] = _ref(state["paths"]["records.jsonl"])

    with pytest.raises(ValueError, match="assistant must not contain prose"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_reasoning_hidden_in_submit_report_arguments(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    report = record["messages"][4]["tool_calls"][0]["function"]["arguments"]
    report["explanation"] = "untagged private reasoning"
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    _rewrite(state["paths"]["records.jsonl"], record)
    config["records"] = _ref(state["paths"]["records.jsonl"])

    with pytest.raises(ValueError, match="submit_report explanation must be empty"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_submit_report_mixed_with_other_tool_work(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    record["messages"][4]["tool_calls"].insert(
        0,
        {
            "id": "bash-2",
            "type": "function",
            "function": {"name": "bash", "arguments": {"script": "true"}},
        },
    )
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    _rewrite(state["paths"]["records.jsonl"], record)
    config["records"] = _ref(state["paths"]["records.jsonl"])

    with pytest.raises(ValueError, match="submit_report must not share"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize(
    ("message_index", "field", "value"),
    [
        (2, "id", "<think>private</think>"),
        (2, "type", "private reasoning"),
        (3, "tool_call_id", "<think>private</think>"),
    ],
)
def test_rejects_nonopaque_tool_identity_fields(
    tmp_path: Path, monkeypatch, message_index: int, field: str, value: str
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(state["record"])
    if message_index == 2:
        record["messages"][message_index]["tool_calls"][0][field] = value
    else:
        record["messages"][message_index][field] = value
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    _rewrite(state["paths"]["records.jsonl"], record)
    config["records"] = _ref(state["paths"]["records.jsonl"])

    with pytest.raises(ValueError, match="identity/type is invalid|opaque tool action"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_weakened_packet_policy_and_unsealed_task_selection(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    packet = json.loads(state["paths"]["packet.json"].read_text())
    packet["admission_policy"]["maximum_submit_report_target_token_fraction"] = 1.0
    packet = campaign.sealed({key: value for key, value in packet.items() if key != "sha256"})
    _rewrite(state["paths"]["packet.json"], packet)
    config["collection_packet"] = _ref(state["paths"]["packet.json"])
    with pytest.raises(ValueError, match="admission policy is insufficient"):
        corpus.build(config, relative_to=tmp_path)

    config, state = _fixture(tmp_path, monkeypatch)
    task_selection = json.loads(state["paths"]["task-selection.json"].read_text())
    task_selection["held_out_task_version_count"] += 1
    _rewrite(state["paths"]["task-selection.json"], task_selection)
    config["collection_task_selection"] = _ref(state["paths"]["task-selection.json"])
    with pytest.raises(ValueError, match="invalid sealed"):
        corpus.build(config, relative_to=tmp_path)


def test_rejects_private_task_selection_before_compilation_or_record_read(
    tmp_path: Path, monkeypatch
) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    task_selection = json.loads(state["paths"]["task-selection.json"].read_text())
    task_selection["trace"] = "synthetic payload that must never reach the compiler"
    task_selection = campaign.sealed(
        {key: value for key, value in task_selection.items() if key != "sha256"}
    )
    _rewrite(state["paths"]["task-selection.json"], task_selection)
    config["collection_task_selection"] = _ref(state["paths"]["task-selection.json"])
    monkeypatch.setattr(
        campaign,
        "_compile_local",
        lambda *_args: pytest.fail("private task-selection reached the evaluator compiler"),
    )
    monkeypatch.setattr(
        corpus,
        "iter_jsonl",
        lambda *_args: pytest.fail("private task-selection reached record ingestion"),
    )

    with pytest.raises(ValueError, match="private content"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_heldout_family_before_reading_private_records(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    split = json.loads(state["paths"]["split.json"].read_text())
    heldout = next(row for row in split["tasks"] if row["split"] != "train")
    selection = copy.deepcopy(state["selection"])
    selection["selected"][0].update(
        {
            "task_key": heldout["task_key"],
            "task_version_id": heldout["task_version_id"],
            "group_id": heldout["group_id"],
        }
    )
    selection = _seal(selection)
    receipt = copy.deepcopy(state["receipt"])
    receipt["selection_sha256"] = selection["sha256"]
    receipt = _seal(receipt)
    _rewrite(state["paths"]["selection.json"], selection)
    _rewrite(state["paths"]["receipt.json"], receipt)
    config["admission_selection"] = _ref(state["paths"]["selection.json"])
    config["admission_receipt"] = _ref(state["paths"]["receipt.json"])

    with pytest.raises(ValueError, match="held-out family"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_rejects_packet_template_binding_drift(tmp_path: Path, monkeypatch) -> None:
    config, state = _fixture(tmp_path, monkeypatch)
    packet = json.loads(state["paths"]["packet.json"].read_text())
    packet["source"]["template_sha256"] = _sha("0")
    packet = campaign.sealed({key: value for key, value in packet.items() if key != "sha256"})
    _rewrite(state["paths"]["packet.json"], packet)
    config["collection_packet"] = _ref(state["paths"]["packet.json"])

    with pytest.raises(ValueError, match="does not match admission selection"):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()
