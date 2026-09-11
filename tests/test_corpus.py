"""Real Parquet/manifest boundaries with synthetic records and a tiny renderer."""

import copy
import json
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import corpus
from training.io import digest_json, file_sha256


def record(name, split):
    return {
        "record_id": name,
        "source": {"model": "synthetic-teacher"},
        "split": split,
        "lineage": {
            "task_key": name,
            "eval_task_version_id": f"version-{name}",
            "application": "synthetic-app",
            "task_family": name,
        },
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
        "messages": [
            {"role": "system", "content": "synthetic"},
            {"role": "user", "content": "synthetic"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"script": "true"}},
                    }
                ],
            },
            {"role": "tool", "content": "synthetic", "tool_call_id": "a"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "submit_report",
                            "arguments": {"flag": "synthetic", "explanation": "test"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "synthetic", "tool_call_id": "b"},
        ],
    }


def seal(value):
    value.pop("sha256", None)
    value["sha256"] = "sha256:" + digest(value)
    return value


@pytest.fixture
def source_split():
    records = [
        record("train-a", "train"),
        record("train-b", "train"),
        record("dev-a", "dev"),
        record("test-a", "test"),
    ]
    split = seal(
        {
            "schema": "cyber_task_split_v1",
            "tasks": [
                {
                    "task_key": r["lineage"]["task_key"],
                    "task_version_id": r["lineage"]["eval_task_version_id"],
                    "split": r["split"],
                    "reference_session_id": r["record_id"] if r["split"] == "dev" else None,
                }
                for r in records
            ],
        }
    )
    return records, split


def test_select_sources_is_order_independent_and_never_uses_test(source_split):
    rows, split = source_split
    train, dev, excluded = corpus.select_sources(rows, split, ["synthetic-teacher"])
    assert [r["record_id"] for r in train] == ["train-a", "train-b"]
    assert [r["record_id"] for r in dev] == ["dev-a"]
    assert excluded == {"heldout_or_reserved": 1}
    assert corpus.select_sources(rows[::-1], split, ["synthetic-teacher"]) == (train, dev, excluded)
    rows[0]["outcome"]["success"] = False
    train, _, excluded = corpus.select_sources(rows, split, ["synthetic-teacher"])
    assert len(train) == 1 and excluded["not_verified_success"] == 1


def test_outcome_only_selection_never_reads_dev_reference(source_split):
    rows, split = source_split
    for row in rows:
        row["lineage"].pop("application")
        row["lineage"].pop("task_family")
        if row["split"] != "train":
            # Held-out payloads are neither needed nor inspected by this selector.
            for key in ("messages", "outcome", "eligibility", "source"):
                row.pop(key)
    split["schema"] = "cyber_task_split_v2"
    for task in split["tasks"]:
        task.pop("reference_session_id")
    seal(split)
    train, dev, excluded = corpus.select_sources(
        rows,
        split,
        ["synthetic-teacher"],
        reference_validation=False,
    )
    assert [row["record_id"] for row in train] == ["train-a", "train-b"]
    assert dev == []
    assert excluded == {"heldout_or_reserved": 2}

    split["tasks"][2]["reference_session_id"] = "dev-a"
    seal(split)
    with pytest.raises(ValueError, match="must not bind"):
        corpus.select_sources(
            rows,
            split,
            ["synthetic-teacher"],
            reference_validation=False,
        )


@pytest.mark.parametrize(
    "score", [True, False, None, "1", float("nan"), float("inf"), float("-inf"), 0.99]
)
def test_non_numeric_or_nonfinite_scores_cannot_enter_sft(source_split, score):
    rows, split = source_split
    rows[0]["outcome"]["score"] = score
    train, _, excluded = corpus.select_sources(rows, split, ["synthetic-teacher"])
    assert [r["record_id"] for r in train] == ["train-b"]
    assert excluded["not_verified_success"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "digest",
        "duplicate",
        "missing_reference",
        "bad_reference",
        "reference_version",
        "family_leak",
        "empty_filter",
        "wrong_model",
        "duplicate_task",
    ],
)
def test_selection_rejects_ambiguous_or_leaky_inputs(source_split, defect):
    rows, split = source_split
    models = ["synthetic-teacher"]
    if defect == "digest":
        split["sha256"] = "wrong"
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "missing_reference":
        rows.pop(2)
    elif defect == "bad_reference":
        rows[2]["outcome"]["success"] = False
    elif defect == "reference_version":
        rows[2]["lineage"]["eval_task_version_id"] = "different"
    elif defect == "family_leak":
        rows[2]["lineage"]["task_family"] = "train-a"
    elif defect == "empty_filter":
        models = []
    elif defect == "wrong_model":
        models = ["other-student"]
    else:
        split["tasks"].append(copy.deepcopy(split["tasks"][0]))
        seal(split)
    with pytest.raises(ValueError):
        corpus.select_sources(rows, split, models)


class Tokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3] * len(messages)


@pytest.fixture
def data_config(source_split, tmp_path, monkeypatch):
    rows, split = source_split
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "lock.json").write_text("{}")
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda *args: (
            Tokenizer(),
            {
                "repo": "synthetic",
                "revision": "a" * 40,
            },
        ),
    )

    def helper(messages, tokenizer, **kwargs):
        return [1, 2, 3], [0, int(messages[0]["role"] == "assistant"), 0], None

    monkeypatch.setattr(corpus, "native_helper", lambda _: helper)
    return {
        "source": {"path": "source.jsonl", "sha256": file_sha256(source)},
        "split": "split.json",
        "model_lock": "lock.json",
        "tokenizer_root": "tokenizer",
        "native_helper": "helper.py",
        "train_models": ["synthetic-teacher"],
        "max_length": 64,
        "context_tokens": 16,
        "dev_windows": 1,
        "output": "data",
    }


def test_actual_dense_parquet_and_manifest_are_private_create_once(data_config, tmp_path):
    result = corpus.build(data_config, relative_to=tmp_path)
    assert result["train"] == {"rows": 2, "source_sessions": 2, "supervised_tokens": 4}
    assert result["dev"] == {"rows": 1, "tasks": 1}
    root = tmp_path / "data"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["sha256"] == "sha256:" + digest(
        {k: v for k, v in manifest.items() if k != "sha256"}
    )
    for split in ("train", "dev"):
        path = root / manifest["files"][split]["path"]
        assert path.stat().st_mode & 0o777 == 0o600
        assert file_sha256(path) == manifest["files"][split]["sha256"]
        assert len(pq.read_table(path)) == manifest["files"][split]["rows"]
    with pytest.raises(FileExistsError):
        corpus.build(data_config, relative_to=tmp_path)


def test_outcome_only_corpus_contains_no_teacher_reference_dev_data(data_config, tmp_path):
    split_path = tmp_path / data_config["split"]
    split = json.loads(split_path.read_text())
    split["schema"] = "cyber_task_split_v2"
    for task in split["tasks"]:
        task.pop("reference_session_id")
    split_path.write_text(json.dumps(seal(split)))
    data_config.update(
        validation_mode="task_outcomes_only",
        fleet_dev_protocol_sha256="sha256:" + "d" * 64,
        dev_windows=0,
    )

    result = corpus.build(data_config, relative_to=tmp_path)
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["fleet_dev_protocol_sha256"] == data_config["fleet_dev_protocol_sha256"]
    assert set(manifest["files"]) == {"train"}
    assert not (tmp_path / "data/dev.parquet").exists()
    assert result["dev"] == {"rows": 0, "tasks": 0}


def add_study_selection(data_config, tmp_path, selected=("train-a",)):
    rows = [json.loads(line) for line in (tmp_path / "source.jsonl").read_text().splitlines()]
    training_split = seal(
        {
            "schema": "cyber_task_split_v2",
            "tasks": [
                {
                    "task_key": row["lineage"]["task_key"],
                    "task_version_id": row["lineage"]["eval_task_version_id"],
                    "split": "train",
                }
                for row in rows
                if row["record_id"].startswith("train-")
            ],
        }
    )
    lock = "sha256:" + "1" * 64
    empty_evaluation = {
        name: seal(
            {
                "schema": "cyber_eval_task_selection_v1",
                "selection_role": name,
                "final_test_lock_sha256": lock,
                "tasks": [],
            }
        )
        for name in ("dev", "final_test")
    }
    study = seal(
        {
            "schema": "cyber_study_split_v1",
            "final_test_lock_sha256": lock,
            "tasks": [{**task, "group_id": task["task_key"]} for task in training_split["tasks"]],
            "training_split": training_split,
            "evaluation": empty_evaluation,
        }
    )
    by_id = {row["record_id"]: row for row in rows}
    selection = seal(
        {
            "schema": "cyber_sft_source_selection_v1",
            "status": "ready",
            "study_split_sha256": study["sha256"],
            "training_split_sha256": training_split["sha256"],
            "selected_episode_ids": list(selected),
            "selected_evidence": [
                {
                    "episode_id": sid,
                    "normalized_record_sha256": digest_json(by_id[sid]),
                }
                for sid in selected
            ],
        }
    )
    (tmp_path / "split-v2.json").write_text(json.dumps(training_split))
    (tmp_path / "study.json").write_text(json.dumps(study))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    data_config.update(
        split="split-v2.json",
        study_split="study.json",
        source_selection="selection.json",
        validation_mode="task_outcomes_only",
        fleet_dev_protocol_sha256="sha256:" + "d" * 64,
        dev_windows=0,
    )


def test_outcome_corpus_enforces_exact_certified_source_selection(data_config, tmp_path):
    add_study_selection(data_config, tmp_path)
    result = corpus.build(data_config, relative_to=tmp_path)
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    assert result["train"]["source_sessions"] == 1
    assert manifest["source_selection"] == {
        "study_split_sha256": json.loads((tmp_path / "study.json").read_text())["sha256"],
        "source_selection_sha256": json.loads((tmp_path / "selection.json").read_text())["sha256"],
        "selected_episode_count": 1,
    }
    assert (tmp_path / "data/study-split.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "data/source-selection.json").stat().st_mode & 0o777 == 0o600


def test_outcome_corpus_rejects_selected_private_record_drift(data_config, tmp_path):
    add_study_selection(data_config, tmp_path)
    source = tmp_path / "source.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["outcome"]["score"] = 0
    source.write_text("\n".join(json.dumps(row) for row in rows))
    data_config["source"]["sha256"] = file_sha256(source)
    with pytest.raises(ValueError, match="differs from certified metadata"):
        corpus.build(data_config, relative_to=tmp_path)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("defect", ["source", "unknown", "bounds", "incompatible_dev"])
def test_builder_rejects_before_publication(data_config, tmp_path, defect):
    if defect == "source":
        data_config["source"]["sha256"] = "0" * 64
    elif defect == "unknown":
        data_config["typo"] = True
    elif defect == "bounds":
        data_config["dev_windows"] = 0
    else:
        path = tmp_path / "source.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[2]["messages"][2]["tool_calls"][0]["function"]["name"] = "unproven-tool"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        data_config["source"]["sha256"] = file_sha256(path)
    with pytest.raises(ValueError):
        corpus.build(data_config, relative_to=tmp_path)
    assert not (tmp_path / "data").exists()


def test_tokenizer_loads_only_verified_local_files(tmp_path, monkeypatch):
    import transformers

    path = tmp_path / "tokenizer.json"
    path.write_text("synthetic")
    lock = {
        "repo": "synthetic",
        "revision": "a" * 40,
        "tokenizer": {"files": [{"path": path.name, "sha256": file_sha256(path)}]},
    }
    loaded = SimpleNamespace(
        chat_template="synthetic", backend_tokenizer=SimpleNamespace(to_str=lambda: "backend")
    )

    def load(root, **kwargs):
        assert root == tmp_path
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return loaded

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    assert corpus.local_tokenizer(lock, tmp_path)[0] is loaded
    path.write_text("tampered")
    with pytest.raises(ValueError):
        corpus.local_tokenizer(lock, tmp_path)
