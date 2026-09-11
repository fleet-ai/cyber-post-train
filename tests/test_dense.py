import copy

import pytest

from training import dense
from training.dense import Excluded, compatible_messages, segment_record
from training.io import file_sha256


def test_native_helper_checks_digest_and_executes_only_selected_functions(tmp_path, monkeypatch):
    path = tmp_path / "native.py"
    path.write_text(
        "raise RuntimeError('top-level code must never execute')\n"
        + "\n".join(f"def {name}(*args, **kwargs): return 7" for name in dense.HELPER_NAMES)
    )
    with pytest.raises(ValueError, match="digest"):
        dense.native_helper(path)
    monkeypatch.setattr(dense, "NATIVE_HELPER_SHA", file_sha256(path))
    assert dense.native_helper(path)() == 7
    path.write_text("def get_generation_prompt_ids(): return 1\n")
    monkeypatch.setattr(dense, "NATIVE_HELPER_SHA", file_sha256(path))
    with pytest.raises(ValueError, match="missing"):
        dense.native_helper(path)


@pytest.mark.parametrize(
    "defect", [None, "length", "empty_mask", "sparse_mask", "tool_mask", "exception"]
)
def test_encoding_keeps_native_masks_and_sanitizes_incompatible_rendering(defect):
    messages = [{"role": role} for role in ("system", "user", "assistant", "tool")]

    class Tokenizer:
        def apply_chat_template(self, rows, **kwargs):
            assert rows == messages[:2]
            assert kwargs["tools"] == [] and kwargs["add_generation_prompt"] is False
            return [9, 9]

    def helper(rows, tokenizer, **kwargs):
        if defect == "exception":
            raise ValueError("private source tokens")
        mask = [0, 1, 1, 0] if rows[0]["role"] == "assistant" else [0] * 4
        if defect == "length":
            mask.pop()
        elif defect == "empty_mask":
            mask = [0] * 4
        elif defect == "sparse_mask":
            mask = [1, 0, 1, 0]
        elif defect == "tool_mask" and rows[0]["role"] == "tool":
            mask = [0, 1, 0, 0]
        return [1, 2, 3, 4], mask, None

    if defect:
        with pytest.raises(Excluded, match="native_template_or_mask_contract") as exc:
            dense.encode_record(messages, Tokenizer(), helper)
        assert "private source tokens" not in str(exc.value)
    else:
        anchor, encoded = dense.encode_record(messages, Tokenizer(), helper)
        assert anchor == [9, 9]
        assert encoded[0]["target"] == (1, 3) and encoded[0]["assistant_index"] == 0
        assert encoded[1]["target"] is None and encoded[1]["assistant_index"] is None


def source():
    return {
        "record_id": "source-a",
        "lineage": {"task_key": "train-task"},
        "source": {"model": "gpt-5.6-sol"},
    }


def chunks(observations=(8, 8, 8, 8)):
    out = []
    for i, size in enumerate(observations):
        out.append(
            {
                "ids": [10, 20 + i, 30 + i, 40],
                "mask": [0, 1, 1, 0],
                "message_index": 2 + 2 * i,
                "assistant_index": i,
                "target": (1, 3),
            }
        )
        out.append(
            {
                "ids": [80] * size,
                "mask": [0] * size,
                "message_index": 3 + 2 * i,
                "assistant_index": None,
                "target": None,
            }
        )
    return out


def test_per_response_exclusion_preserves_later_targets_and_original_ordinals():
    rows = segment_record(source(), [1, 2], chunks((100, 8, 8, 8)), max_tokens=32)
    assert [s["assistant_index"] for r in rows for s in r["target_spans"]] == [0, 2, 3]
    assert sum(r["target_token_count"] for r in rows) == 6
    for row in rows:
        assert row["source_assistant_count"] == 4
        assert row["eligible_assistant_indices"] == [0, 2, 3]
        assert row["excluded_assistant_targets"] == [
            {
                "assistant_index": 1,
                "source_message_index": 4,
                "reason": "overlength_required_previous_round",
            }
        ]
    later = next(r for r in rows if r["target_spans"][0]["assistant_index"] == 2)
    assert 1 in later["copied_context_assistant_indices"]
    expected = [0] * later["token_count"]
    for span in later["target_spans"]:
        expected[span["token_start"] : span["token_end"]] = [1] * (
            span["token_end"] - span["token_start"]
        )
    assert later["loss_mask"] == expected


def test_fit_all_targets_matches_partition_and_multitarget_segments():
    rows = segment_record(source(), [1, 2], chunks(), max_tokens=32)
    assert [s["assistant_index"] for r in rows for s in r["target_spans"]] == list(range(4))
    assert all(r["excluded_assistant_targets"] == [] for r in rows)
    assert any(len(r["target_spans"]) > 1 for r in rows)


def test_overlarge_assistant_excluded_without_token_truncation():
    encoded = chunks()
    encoded[2]["ids"] *= 30
    encoded[2]["mask"] *= 30
    rows = segment_record(source(), [1, 2], encoded, max_tokens=32)
    assert rows[0]["excluded_assistant_targets"][0]["reason"] == "overlength_assistant_target"
    assert rows[0]["excluded_assistant_targets"][0]["assistant_index"] == 1
    assert all(r["token_count"] <= 32 for r in rows)


def test_impossible_anchor_excludes_source_explicitly():
    with pytest.raises(Excluded, match="overlength_original_task_anchor"):
        segment_record(source(), [1] * 32, chunks(), max_tokens=32)


def test_no_fitting_source_does_not_create_empty_row():
    encoded = chunks((8,))
    encoded[0]["ids"] *= 30
    encoded[0]["mask"] *= 30
    with pytest.raises(Excluded, match="no_context_fitting_assistant_targets"):
        segment_record(source(), [1, 2], encoded, max_tokens=32)


def test_dev_targets_keep_complete_immediate_round(monkeypatch):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(4):
        messages.extend(
            [
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "o" * (100 if i == 0 else 2)},
            ]
        )
    monkeypatch.setattr(dense, "compatible_messages", lambda _: (messages, {}))

    class Tokenizer:
        def apply_chat_template(self, current, **kwargs):
            return [1] * sum(len(m["content"]) + 1 for m in current)

    refs = dense.clean_dev_windows(source(), Tokenizer(), max_tokens=32)
    assert [w["source_assistant_index"] for w in refs] == [0, 2, 3]
    assert all(w["token_count"] <= 32 for w in refs)
    assert all(
        w["context_start_message_index"] <= max(2, w["source_message_index"] - 2) for w in refs
    )


def record():
    return {
        "record_id": "s1",
        "lineage": {"task_key": "train-task"},
        "source": {"model": "test-teacher"},
        "messages": [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "",
                "thinking": "never exported",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"script":"true","timeoutMs":1000}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "submit_report",
                            "arguments": {"flag": "synthetic", "explanation": "synthetic"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "received"},
        ],
    }


def test_original_direct_tool_contract_and_reasoning_omission():
    src = record()
    original = copy.deepcopy(src)
    messages, counts = compatible_messages(src)
    assert counts == {"bash": 1, "submit_report": 1}
    assert "thinking" not in messages[2]
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["script"] == "true"
    assert src == original


@pytest.mark.parametrize(
    "tool,reason",
    [
        ("context_compaction", "opaque_compaction"),
        ("search_tool", "unsupported_or_unproven_tool_interface"),
        ("use_tool", "unsupported_or_unproven_tool_interface"),
        ("read_file", "unsupported_or_unproven_tool_interface"),
    ],
)
def test_unproven_tools_reject_whole_trajectory(tool, reason):
    src = record()
    src["messages"][2]["tool_calls"][0]["function"]["name"] = tool
    with pytest.raises(Excluded, match=reason):
        compatible_messages(src)


def test_invalid_argument_and_missing_result_rejected():
    src = record()
    src["messages"][2]["tool_calls"][0]["function"]["arguments"] = {
        "script": "true",
        "unexpected": 1,
    }
    with pytest.raises(Excluded, match="unproven_bash_argument_schema"):
        compatible_messages(src)
    src = record()
    src["messages"].pop(3)
    with pytest.raises(Excluded, match="missing_tool_result_before_next_assistant"):
        compatible_messages(src)


def test_orphan_and_duplicate_results_rejected():
    src = record()
    src["messages"][3]["tool_call_id"] = "not-called"
    with pytest.raises(Excluded, match="orphan_or_duplicate_tool_result"):
        compatible_messages(src)
