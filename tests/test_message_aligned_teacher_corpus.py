import copy
import json
from pathlib import Path

import pytest

from training import message_aligned_teacher_corpus as corpus
from training.dense import Excluded
from training.io import digest_json, file_sha256
from training.task_family_split import TRUSTED_FLEET_COLLECTION_ROOT_ID


def _sealed(value: dict) -> dict:
    value["sha256"] = digest_json(value)
    return value


def _contract() -> dict:
    catalog = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
        ).read_text()
    )
    capture = _capture()
    return _sealed(
        {
            "schema": corpus.TOOL_CONTRACT_SCHEMA,
            "source_catalog_sha256": corpus.SOURCE_TOOL_CATALOG_SHA256,
            "target_harness": {
                "name": "opencode",
                "version": corpus.OPENCODE_VERSION,
                "release_asset_sha256": corpus.OPENCODE_RELEASE_SHA256,
                "mcp_server": "fleet",
            },
            "target_model": {"repo": "Qwen/Qwen3.8-27B", "revision": "1" * 40},
            "provider_schema_transform_sha256": "sha256:" + "b" * 64,
            "model_request_capture_sha256": capture["sha256"],
            "source_aliases": {
                "bash": "fleet_bash",
                "submit_report": "fleet_submit_report",
            },
            "target_tools": corpus.derive_opencode_tools(catalog),
        }
    )


def _capture() -> dict:
    catalog = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
        ).read_text()
    )
    return _sealed(
        {
            "schema": corpus.MODEL_REQUEST_CAPTURE_SCHEMA,
            "target_harness": {
                "name": "opencode",
                "version": corpus.OPENCODE_VERSION,
                "release_asset_sha256": corpus.OPENCODE_RELEASE_SHA256,
                "mcp_server": "fleet",
            },
            "target_model": {"repo": "Qwen/Qwen3.8-27B", "revision": "1" * 40},
            "provider_schema_transform_sha256": "sha256:" + "b" * 64,
            "request_envelope_sha256": "sha256:" + "a" * 64,
            "captured_tools": corpus.derive_opencode_tools(catalog),
        }
    )


def _record(task_key: str = "train-task", version: str = "train-version") -> dict:
    row = {
        "schema": corpus.RECORD_SCHEMA,
        "record_id": "session-1",
        "source": {
            "session_id": "session-1",
            "model": "teacher",
            "harness_mode": "tool-use",
            "harness_sha256": "sha256:" + "a" * 64,
        },
        "lineage": {"task_key": task_key, "eval_task_version_id": version},
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
        "messages": [
            {"role": "system", "content": "system contract"},
            {"role": "user", "content": "task anchor"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "bash-call",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"script": "true"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "bash-call", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "report-call",
                        "type": "function",
                        "function": {
                            "name": "submit_report",
                            "arguments": {"flag": "synthetic", "explanation": "done"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "report-call", "content": "accepted"},
        ],
    }
    row["content_digest"] = digest_json(row)
    return row


def _proof(record: dict) -> dict:
    return _sealed(
        {
            "schema": corpus.EVIDENCE_SCHEMA,
            "session_id": record["record_id"],
            "model_id": record["source"]["model"],
            "task_key": record["lineage"]["task_key"],
            "task_version_id": record["lineage"]["eval_task_version_id"],
            "normalized_record_sha256": digest_json(record),
            "transcript_sha256": "sha256:" + "b" * 64,
            "verifier_execution_id": "verifier-1",
            "routes": {
                "summary": "GET /v1/sessions?eval_task_id=<project-task-id>",
                "transcript": corpus.TRANSCRIPT_ROUTE,
            },
            "successful_report_call_id": "report-call",
            "outcome": {
                "status": "completed",
                "verifier_process_success": True,
                "score_at_least_one": True,
            },
        }
    )


def _roster(*, heldout_alias: bool = False) -> dict:
    group = "sha256:" + ("d" if heldout_alias else "c") * 64
    split = "dev" if heldout_alias else "train"
    identities = [
        {
            "task_key": "train-task" if not heldout_alias else "reviewed-holdout-name",
            "task_version_id": "train-version" if not heldout_alias else "reviewed-version",
            "group_id": group,
            "split": split,
        }
    ]
    if heldout_alias:
        identities.append(
            {
                "task_key": "historical-alias",
                "task_version_id": "historical-version",
                "group_id": group,
                "split": split,
            }
        )
    else:
        identities.append(
            {
                "task_key": "heldout-task",
                "task_version_id": "heldout-version",
                "group_id": "sha256:" + "e" * 64,
                "split": "final_test",
            }
        )
    return _sealed(
        {
            "schema": corpus.FAMILY_ROSTER_SCHEMA,
            "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
            "family_role_anchor_sha256": "sha256:" + "f" * 64,
            "heldout_group_ids": sorted(
                {row["group_id"] for row in identities if row["split"] != "train"}
            ),
            "identities": identities,
        }
    )


class Tokenizer:
    def __init__(self):
        self.anchor_tools = None

    def __len__(self):
        return 512

    def apply_chat_template(self, messages, **kwargs):
        self.anchor_tools = kwargs["tools"]
        assert [message["role"] for message in messages[:2]] == ["system", "user"]
        tokens = [400, 401, 402, 403]
        for message in messages[2:]:
            if message["role"] == "assistant":
                name = message["tool_calls"][0]["function"]["name"]
                token = 11 if name == "fleet_bash" else 12
                tokens += [token, 20, 21, token]
            else:
                tokens += [30, 31, 32, 33]
        return tokens


def _helper(messages, tokenizer, **kwargs):
    message = messages[0]
    if message["role"] == "assistant":
        name = message["tool_calls"][0]["function"]["name"]
        token = 11 if name == "fleet_bash" else 12
        return [token, 20, 21, token], [0, 1, 1, 0], None
    return [30, 31, 32, 33], [0, 0, 0, 0], None


def _materialize(record: dict, roster: dict | None = None):
    tokenizer = Tokenizer()
    roles = corpus.family_roster(roster or _roster())
    rows, transform, _ = corpus.materialize_record(
        record,
        _proof(record),
        roles,
        _contract(),
        tokenizer,
        _helper,
        max_length=17,
        context_tokens=4,
        family_role_anchor_sha256="sha256:" + "f" * 64,
    )
    return rows, transform, tokenizer


def test_windows_repeat_full_anchor_and_begin_only_at_message_boundaries() -> None:
    record = _record()
    record["messages"][4:4] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bash-call-2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "false || true"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "bash-call-2", "content": "ok"},
    ]
    rows, transform, tokenizer = _materialize(record)

    assert len(rows) == 2
    assert all(row["input_ids"][:4] == [400, 401, 402, 403] for row in rows)
    assert {row["context_start_message_index"] for row in rows} <= {2, 4, 6}
    assert all(row["window_algorithm"] == corpus.ALGORITHM for row in rows)
    assert all(row["window_schema"] == corpus.WINDOW_SCHEMA for row in rows)
    assert [span["assistant_index"] for row in rows for span in row["target_spans"]] == [
        0,
        1,
        2,
    ]
    assert all(span["source_target_sha256"] for row in rows for span in row["target_spans"])
    assert [tool["function"]["name"] for tool in tokenizer.anchor_tools] == [
        "fleet_bash",
        "fleet_submit_report",
    ]
    assert transform["output_messages_sha256"] == rows[0]["source_messages_sha256"]


def test_mid_message_window_start_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = corpus.segment_record

    def clipped(*args, **kwargs):
        rows = original(*args, **kwargs)
        anchor = args[1]
        rows[0]["input_ids"].pop(len(anchor))
        rows[0]["loss_mask"].pop(len(anchor))
        rows[0]["token_count"] -= 1
        return rows

    monkeypatch.setattr(corpus, "segment_record", clipped)
    with pytest.raises(ValueError, match="complete repeated anchor/message boundary"):
        _materialize(_record())


def test_full_template_exception_cannot_surface_private_source_text() -> None:
    class SecretTokenizer(Tokenizer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("secret-task-anchor-and-token-123")
            return super().apply_chat_template(messages, **kwargs)

    record = _record()
    with pytest.raises(Excluded) as raised:
        corpus.materialize_record(
            record,
            _proof(record),
            corpus.family_roster(_roster()),
            _contract(),
            SecretTokenizer(),
            _helper,
            max_length=17,
            context_tokens=4,
            family_role_anchor_sha256="sha256:" + "f" * 64,
        )
    assert str(raised.value) == "full_template_render_contract"
    assert "secret" not in str(raised.value)


def test_missing_task_anchor_is_never_repaired_or_inferred() -> None:
    record = _record()
    record["messages"][1]["content"] = ""
    with pytest.raises(Excluded, match="missing_original_system_and_task_anchor"):
        _materialize(record)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda call: call["function"].update(name="unknown_shell"), "tool_name_contract"),
        (
            lambda call: call["function"].update(arguments={"command": "true"}),
            "tool_argument_schema_mismatch",
        ),
    ],
)
def test_tool_name_or_argument_schema_mismatch_fails_closed(mutate, reason) -> None:
    record = _record()
    mutate(record["messages"][2]["tool_calls"][0])
    with pytest.raises(Excluded, match=reason):
        _materialize(record)


def test_lifecycle_salvage_without_retained_successful_report_is_rejected() -> None:
    record = _record()
    record["messages"].insert(
        4, {"role": "tool", "tool_call_id": "orphan", "content": "legacy orphan"}
    )
    with pytest.raises(Excluded, match="lifecycle_salvage_lost_successful_report"):
        _materialize(record)


def test_successful_report_is_terminal_and_trailing_bash_is_not_trained() -> None:
    record = _record()
    record["messages"] += [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "trailing-bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "echo late"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "trailing-bash", "content": "late"},
    ]

    rows, transform, _ = _materialize(record)

    assert transform["terminal_report_cut"] == {"kept_messages": 6, "dropped_messages": 2}
    assert [span["assistant_index"] for row in rows for span in row["target_spans"]] == [0, 1]


def test_second_report_is_rejected_even_after_the_successful_report() -> None:
    record = _record()
    record["messages"] += copy.deepcopy(record["messages"][4:6])
    record["messages"][-2]["tool_calls"][0]["id"] = "report-call-2"
    record["messages"][-1]["tool_call_id"] = "report-call-2"

    with pytest.raises(Excluded, match="unbound_or_multiple_report_submission"):
        _materialize(record)


def test_report_only_source_fails_the_preserved_task_rich_gate() -> None:
    record = _record()
    record["messages"] = record["messages"][:2] + record["messages"][4:]

    with pytest.raises(Excluded, match="insufficient_task_rich_targets"):
        _materialize(record)


def test_family_level_holdout_alias_is_rejected_even_under_another_task_key() -> None:
    record = _record("historical-alias", "historical-version")
    roles = corpus.family_roster(_roster(heldout_alias=True))
    with pytest.raises(ValueError, match="held-out task-family alias"):
        corpus.materialize_record(
            record,
            _proof(record),
            roles,
            _contract(),
            Tokenizer(),
            _helper,
            max_length=13,
            context_tokens=4,
            family_role_anchor_sha256="sha256:" + "f" * 64,
        )


def test_successful_report_must_be_the_exact_evidence_bound_call() -> None:
    record = _record()
    proof = _proof(record)
    proof["successful_report_call_id"] = "different-report"
    proof["sha256"] = digest_json({key: value for key, value in proof.items() if key != "sha256"})
    with pytest.raises(Excluded, match="unbound_or_multiple_report_submission"):
        corpus.materialize_record(
            record,
            proof,
            corpus.family_roster(_roster()),
            _contract(),
            Tokenizer(),
            _helper,
            max_length=13,
            context_tokens=4,
            family_role_anchor_sha256="sha256:" + "f" * 64,
        )


def test_tool_contract_cannot_silently_keep_bare_training_names() -> None:
    contract = copy.deepcopy(_contract())
    contract["target_tools"][0]["function"]["name"] = "bash"
    contract["sha256"] = digest_json(
        {key: value for key, value in contract.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="OpenCode Fleet surface"):
        corpus.tool_contract(contract)


def test_captured_model_request_must_contain_the_exact_contract_tools() -> None:
    capture = _capture()
    capture["captured_tools"][0]["function"]["name"] = "bash"
    capture["sha256"] = digest_json(
        {key: value for key, value in capture.items() if key != "sha256"}
    )
    contract = _contract()
    contract["model_request_capture_sha256"] = capture["sha256"]
    contract["sha256"] = digest_json(
        {key: value for key, value in contract.items() if key != "sha256"}
    )

    with pytest.raises(ValueError, match="captured model request tools"):
        corpus.model_request_capture(capture, contract, contract["target_tools"])


def test_opencode_tool_derivation_is_exact_and_schema_complete() -> None:
    catalog = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
        ).read_text()
    )
    tools = corpus.derive_opencode_tools(catalog)

    assert corpus._wire_digest(tools) == corpus.TARGET_TOOLS_SHA256
    assert tools[0]["function"]["parameters"]["properties"]["timeoutMs"]["maximum"] == 300_000
    assert set(tools[1]["function"]["parameters"]["properties"]) == {
        "explanation",
        "flag",
        "flags",
        "verdict",
    }
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in tools)


def test_create_once_builder_emits_new_algorithm_and_preserves_dense_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record()
    inputs = {
        "normalized": tmp_path / "normalized.jsonl",
        "evidence": tmp_path / "evidence.jsonl",
        "family_roster": tmp_path / "family-roster.json",
        "tool_catalog": tmp_path / "tool-catalog.json",
        "tool_contract": tmp_path / "tool-contract.json",
        "model_request_capture": tmp_path / "model-request-capture.json",
        "model_lock": tmp_path / "model-lock.json",
        "native_helper": tmp_path / "native-helper.py",
    }
    inputs["normalized"].write_text(json.dumps(record) + "\n")
    inputs["evidence"].write_text(json.dumps(_proof(record)) + "\n")
    inputs["family_roster"].write_text(json.dumps(_roster()))
    inputs["tool_catalog"].write_text(
        (
            Path(__file__).parents[1]
            / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
        ).read_text()
    )
    inputs["tool_contract"].write_text(json.dumps(_contract()))
    inputs["model_request_capture"].write_text(json.dumps(_capture()))
    inputs["model_lock"].write_text(
        json.dumps(
            {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": "1" * 40,
                "tokenizer": {"files": []},
            }
        )
    )
    inputs["native_helper"].write_text("helper")
    tokenizer = Tokenizer()
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda lock, root: (
            tokenizer,
            {"repo": "Qwen/Qwen3.8-27B", "revision": "1" * 40},
        ),
    )
    monkeypatch.setattr(corpus, "native_helper", lambda path: _helper)
    request = _sealed(
        {
            "schema": corpus.REQUEST_SCHEMA,
            **{
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in inputs.items()
            },
            "tokenizer_root": str(tmp_path),
            "teacher_models": ["teacher"],
            "max_length": 17,
            "context_tokens": 4,
            "output": str(tmp_path / "output"),
        }
    )

    result = corpus.build(request, relative_to=tmp_path)

    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    rows = pq.read_table(tmp_path / "output/train.parquet").to_pylist()
    assert result["manifest"] == manifest["sha256"]
    assert manifest["schema"] == corpus.CORPUS_SCHEMA
    assert manifest["algorithm"] == corpus.ALGORITHM
    assert manifest["files"]["train"]["format"] == "pretokenized_assistant_segments_v1"
    assert rows[0]["input_ids"][:4] == [400, 401, 402, 403]
    assert rows[0]["tool_contract_sha256"] == _contract()["sha256"]
    assert rows[0]["source_trace_sha256"] == _proof(record)["transcript_sha256"]
    assert set(manifest["builder_sha256"]) == {
        "message_aligned_teacher_corpus.py",
        "dense.py",
        "corpus.py",
        "native_helper",
    }
    receipt = json.loads((tmp_path / "output/RECEIPT.json").read_text())
    assert receipt["sha256"] == digest_json(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    selection = json.loads((tmp_path / "output/source-selection.private.jsonl").read_text())
    assert selection["task_version_id"] == "train-version"
    assert selection["non_submit_tool_responses"] == 1
    with pytest.raises(FileExistsError):
        corpus.build(request, relative_to=tmp_path)

    changed = copy.deepcopy(request)
    changed["output"] = str(tmp_path / "output-mutated")
    changed["sha256"] = digest_json(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    original_write = corpus.atomic_write_json

    def mutate_bound_input(path, value, **kwargs):
        original_write(path, value, **kwargs)
        if Path(path).name == "RECEIPT.json":
            inputs["evidence"].write_text(inputs["evidence"].read_text() + "\n")

    monkeypatch.setattr(corpus, "atomic_write_json", mutate_bound_input)
    with pytest.raises(ValueError, match="bound materialization input changed"):
        corpus.build(changed, relative_to=tmp_path)
    assert not (tmp_path / "output-mutated").exists()


def test_rebuild_plan_is_sealed_blocked_and_cannot_launch() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/data/qwen38-teacher3k-message-aligned-32k-rebuild-v1.plan.json"
    )
    plan = json.loads(path.read_text())

    assert plan["sha256"] == digest_json(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    assert plan["status"] == "blocked_pending_exact_source_evidence"
    assert plan["planned_request_schema"] == corpus.REQUEST_SCHEMA
    assert plan["planned_algorithm"] == corpus.ALGORITHM
    assert plan["launch_authorized"] is False
    assert plan["private_data_materialized"] is False
    assert plan["gpu_jobs_submitted"] == 0
    assert {item["id"] for item in plan["blocking_evidence"]} == {
        "exact-family-role-roster",
        "successful-report-call-binding",
        "exact-opencode-model-tool-contract",
        "private-source-readback",
    }
