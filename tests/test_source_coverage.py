"""Real dense/coverage boundaries, with synthetic private records and tiny tokenizer."""

import copy
import json

import pytest

from training import corpus, dense
from training import source_coverage as sc
from training.io import digest_json, file_sha256
from training.study_data import _coverage, check_seal, seal

SHA = "sha256:" + "1" * 64
PRIVATE = "DO_NOT_EMIT_SYNTHETIC_PRIVATE_CONTENT"


def source():
    return {
        "record_id": "synthetic-session",
        "source": {"model": "synthetic-teacher"},
        "lineage": {
            "task_key": "synthetic-train-task",
            "eval_task_version_id": "exact-version",
            "application": "synthetic-app",
            "task_family": "synthetic-family",
        },
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
        "messages": [
            {"role": "system", "content": PRIVATE},
            {"role": "user", "content": PRIVATE},
            {"role": "assistant", "content": PRIVATE},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "bash-1", "function": {"name": "bash", "arguments": {"script": PRIVATE}}}
                ],
            },
            {"role": "tool", "tool_call_id": "bash-1", "content": PRIVATE},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "report-1",
                        "function": {
                            "name": "submit_report",
                            "arguments": {"flag": PRIVATE, "explanation": PRIVATE},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "report-1", "content": PRIVATE},
            {"role": "assistant", "content": "", "thinking": PRIVATE},
        ],
    }


def reviewed(record):
    return seal(
        {
            "schema": sc.EVIDENCE_SCHEMA,
            "origin": "fleet_cyber",
            "review_receipt_sha256": SHA,
            "episodes": [
                {
                    "episode_id": record["record_id"],
                    "task_key": record["lineage"]["task_key"],
                    "task_version_id": record["lineage"]["eval_task_version_id"],
                    "model_id": record["source"]["model"],
                    "source_kind": "teacher",
                    "validity": "valid",
                    "verified_success": True,
                    "acceptance_sha256": SHA,
                    "trace_sha256": SHA,
                    "normalized_record_sha256": digest_json(record),
                }
            ],
        }
    )


def training_split():
    return seal(
        {
            "schema": "cyber_task_split_v2",
            "tasks": [
                {
                    "task_key": "synthetic-train-task",
                    "task_version_id": "exact-version",
                    "split": "train",
                }
            ],
        }
    )


class Tokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 2 and kwargs["tools"] == []
        assert kwargs["add_generation_prompt"] is False
        return [1, 2]


def helper(messages, tokenizer, **kwargs):
    assert kwargs == {"tokenizer_kwargs": {"tools": []}}
    message = messages[0]
    if message["role"] == "tool":
        n = 50 if message["content"] == "oversized-synthetic-observation" else 2
        return [10] * n, [0] * n, None
    names = {c["function"]["name"] for c in message.get("tool_calls", [])}
    n = 7 if "submit_report" in names else 2 if names else 3 if message["content"] else 1
    return [10] * (n + 2), [0] + [1] * n + [0], None


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / "native-helper.py"
    path.write_text("# synthetic helper fixture\n")
    monkeypatch.setattr(dense, "NATIVE_HELPER_SHA", file_sha256(path))
    return sc.target_policy(
        {"repo": "synthetic", "revision": "a" * 40}, path, max_length=64, context_tokens=16
    )


def test_exact_native_dense_targets_become_coverage_counts_without_private_outputs(policy):
    record = source()
    original = copy.deepcopy(record)
    episodes, receipts, audit = sc.extract(
        [record], reviewed(record), training_split(), policy, Tokenizer(), helper
    )
    assert record == original
    assert audit == {"skip_counts": {}, "excluded_sources": []}
    assert len(episodes) == len(receipts) == 1
    item, receipt = episodes[0], receipts[0]
    check_seal(item, "cyber_sft_episode_metadata_v1")
    check_seal(receipt, sc.COVERAGE_SCHEMA)
    assert item["coverage"]["receipt_sha256"] == receipt["sha256"]
    _coverage(item["coverage"], policy["sha256"])
    assert {k: item["coverage"][k] for k in sc.COUNTS} == {
        "assistant_responses": 4,
        "supervised_tokens": 13,
        "submit_report_responses": 1,
        "submit_report_tokens": 7,
        "non_submit_tool_responses": 1,
        "decision_responses": 1,
        "other_responses": 1,
        "completed_non_submit_tool_rounds": 1,
    }
    serialized = json.dumps([episodes, receipts, audit])
    assert PRIVATE not in serialized
    assert all(
        term not in serialized for term in ('"messages"', '"input_ids"', '"score"', '"thinking"')
    )
    assert receipt["source_assistant_responses"] == 4
    assert receipt["excluded_assistant_responses"] == 0


def test_overlength_eligibility_not_raw_counts_and_copied_context_never_recounted(policy):
    record = source()
    record["messages"][4]["content"] = "oversized-synthetic-observation"
    short = seal({k: v for k, v in {**policy, "max_length": 16}.items() if k != "sha256"})
    items, receipts, _ = sc.extract(
        [record], reviewed(record), training_split(), short, Tokenizer(), helper
    )
    assert items[0]["coverage"]["assistant_responses"] == 3
    assert items[0]["coverage"]["supervised_tokens"] == 6
    assert items[0]["coverage"]["submit_report_responses"] == 0
    assert items[0]["coverage"]["submit_report_tokens"] == 0
    assert receipts[0]["excluded_reasons"] == {"overlength_required_previous_round": 1}
    assert receipts[0]["segments"] > 1
    _coverage(items[0]["coverage"], short["sha256"])


def test_mixed_submit_and_bash_is_conservatively_submission_class(policy):
    record = source()
    record["messages"][3]["tool_calls"].append(record["messages"][5]["tool_calls"][0])
    record["messages"] = record["messages"][:5] + record["messages"][6:]
    items, _, _ = sc.extract(
        [record], reviewed(record), training_split(), policy, Tokenizer(), helper
    )
    c = items[0]["coverage"]
    assert c["submit_report_responses"] == 1
    assert c["non_submit_tool_responses"] == c["completed_non_submit_tool_rounds"] == 0
    assert c["supervised_tokens"] == 11


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("unsupported", "unsupported_or_unproven_tool_interface"),
        ("compaction", "opaque_compaction"),
        ("missing_result", "missing_tool_result_before_next_assistant"),
    ],
)
def test_incompatible_sources_emit_only_explicit_exclusion_metadata(policy, defect, reason):
    record = source()
    if defect == "missing_result":
        record["messages"].pop(4)
    else:
        record["messages"][3]["tool_calls"][0]["function"]["name"] = (
            "context_compaction" if defect == "compaction" else "unproven_tool"
        )
    items, receipts, audit = sc.extract(
        [record], reviewed(record), training_split(), policy, Tokenizer(), helper
    )
    assert not items and not receipts
    assert audit["skip_counts"] == {reason: 1}
    assert audit["excluded_sources"][0]["episode_id"] == record["record_id"]
    assert PRIVATE not in json.dumps(audit)


@pytest.mark.parametrize(
    "change", ["acceptance", "trace", "success", "model", "record", "missing", "duplicate"]
)
def test_reviewed_evidence_is_required_exact_and_complete(policy, change):
    record = source()
    evidence = reviewed(record)
    records = [record]
    if change in {"acceptance", "trace"}:
        evidence["episodes"][0][f"{change}_sha256"] = "unknown"
    elif change == "success":
        evidence["episodes"][0]["verified_success"] = False
    elif change == "model":
        evidence["episodes"][0]["model_id"] = "another-model"
    elif change == "record":
        record["messages"][0]["content"] += "modified"
    elif change == "missing":
        records = []
    else:
        records *= 2
    evidence = seal({k: v for k, v in evidence.items() if k != "sha256"})
    with pytest.raises(sc.CoverageError):
        sc.extract(records, evidence, training_split(), policy, Tokenizer(), helper)


def test_heldout_records_are_skipped_before_payload_outcome_or_model_access(policy):
    class Heldout(dict):
        def __getitem__(self, key):
            assert key == "lineage"
            return {"task_key": "heldout", "eval_task_version_id": "heldout-version"}

    record = source()
    items, _, audit = sc.extract(
        [Heldout(), record], reviewed(record), training_split(), policy, Tokenizer(), helper
    )
    assert len(items) == 1
    assert audit["skip_counts"] == {"outside_training_split": 1}


def test_private_library_output_and_unexpected_exception_details_never_escape(
    policy, monkeypatch, capsys
):
    def broken(_):
        print(PRIVATE)
        raise ValueError(PRIVATE)

    monkeypatch.setattr(dense, "compatible_messages", broken)
    record = source()
    with pytest.raises(sc.CoverageError) as exc:
        sc.extract([record], reviewed(record), training_split(), policy, Tokenizer(), helper)
    assert PRIVATE not in str(exc.value)
    assert capsys.readouterr() == ("", "")


def test_normalized_success_disagreement_blocks_even_with_a_review_claim(policy):
    record = source()
    record["outcome"]["success"] = False
    with pytest.raises(sc.CoverageError, match="corpus eligibility differs"):
        sc.extract([record], reviewed(record), training_split(), policy, Tokenizer(), helper)


def test_completed_tool_round_is_structural_not_a_command_success_guess(policy):
    record = source()
    record["messages"][4]["content"] = "synthetic tool returned a nonzero exit status"
    items, _, _ = sc.extract(
        [record], reviewed(record), training_split(), policy, Tokenizer(), helper
    )
    assert items[0]["coverage"]["completed_non_submit_tool_rounds"] == 1


@pytest.fixture
def config(tmp_path, monkeypatch):
    record = source()
    (tmp_path / "source.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "evidence.json").write_text(json.dumps(reviewed(record)))
    (tmp_path / "split.json").write_text(json.dumps(training_split()))
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer/tokenizer.json").write_text("synthetic-tokenizer")
    files = [
        {"path": "tokenizer.json", "sha256": file_sha256(tmp_path / "tokenizer/tokenizer.json")}
    ]
    (tmp_path / "lock.json").write_text(json.dumps({"tokenizer": {"files": files}}))
    (tmp_path / "helper.py").write_text("# synthetic native helper\n")
    monkeypatch.setattr(dense, "NATIVE_HELPER_SHA", file_sha256(tmp_path / "helper.py"))
    monkeypatch.setattr(dense, "native_helper", lambda _: helper)
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda *_: (
            Tokenizer(),
            {
                "repo": "synthetic",
                "revision": "a" * 40,
                "files": files,
                "chat_template_sha256": SHA,
                "backend_sha256": SHA,
            },
        ),
    )
    return {
        "source": {"path": "source.jsonl", "sha256": file_sha256(tmp_path / "source.jsonl")},
        "reviewed_evidence": {
            "path": "evidence.json",
            "sha256": file_sha256(tmp_path / "evidence.json"),
        },
        "split": "split.json",
        "model_lock": "lock.json",
        "tokenizer_root": "tokenizer",
        "native_helper": "helper.py",
        "max_length": 64,
        "context_tokens": 16,
        "output": "coverage",
    }


def test_real_create_once_files_are_metadata_only_private_and_digest_bound(config, tmp_path):
    result = sc.build(config, relative_to=tmp_path)
    assert result["counts"]["supervised_tokens"] == 13
    root = tmp_path / "coverage"
    assert {p.name for p in root.iterdir()} == {
        "manifest.json",
        "target-policy.json",
        "episodes.jsonl",
        "coverage-receipts.jsonl",
    }
    manifest = json.loads((root / "manifest.json").read_text())
    check_seal(manifest, "cyber_sft_coverage_extraction_v1")
    for p in root.iterdir():
        assert p.stat().st_mode & 0o777 == 0o600
        assert PRIVATE not in p.read_text()
        assert '"score"' not in p.read_text()
    assert all(file_sha256(root / name) == sha for name, sha in manifest["files"].items())
    with pytest.raises(sc.CoverageError, match="already exists"):
        sc.build(config, relative_to=tmp_path)


def test_file_or_compiler_policy_drift_fails_before_publication(config, tmp_path, monkeypatch):
    original = dense.segment_record

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "tokenizer/tokenizer.json").write_text("changed")
        return result

    monkeypatch.setattr(dense, "segment_record", changed)
    with pytest.raises(sc.CoverageError, match="changed during"):
        sc.build(config, relative_to=tmp_path)
    assert not (tmp_path / "coverage").exists()


def test_policy_changes_with_native_bounds_and_template_identity(tmp_path, monkeypatch):
    p = tmp_path / "helper.py"
    p.write_text("synthetic")
    monkeypatch.setattr(dense, "NATIVE_HELPER_SHA", file_sha256(p))
    a = sc.target_policy({"template": "a"}, p, max_length=32, context_tokens=8)
    b = sc.target_policy({"template": "a"}, p, max_length=33, context_tokens=8)
    c = sc.target_policy({"template": "b"}, p, max_length=32, context_tokens=8)
    assert len({a["sha256"], b["sha256"], c["sha256"]}) == 3


def test_cli_failure_does_not_print_source_error_or_traceback(tmp_path, capsys):
    path = tmp_path / "bad-config.json"
    path.write_text(PRIVATE)
    assert sc.main(["--config", str(path)]) == 2
    captured = capsys.readouterr()
    assert PRIVATE not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
