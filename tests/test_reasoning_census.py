import copy
import json
from pathlib import Path

import pytest

from training import reasoning_census as census
from training.io import digest_json


def seal(value: dict) -> dict:
    value.pop("sha256", None)
    value["sha256"] = digest_json(value)
    return value


def corpus_manifest() -> dict:
    return seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "catalog_provenance": {
                "normalized_records_file_sha256": "sha256:" + "a" * 64,
                "success_evidence_file_sha256": "sha256:" + "b" * 64,
            },
            "files": {
                "train": {
                    "source_sessions": 3,
                    "supervised_tokens": 50,
                    "assistant_responses": 10,
                }
            },
        }
    )


def aggregate_census(*, available_thinking_sessions: int = 0) -> dict:
    fields = {
        name: {
            "available_sessions": 0,
            "selected_sessions": 0,
            "selected_target_tokens": 0,
        }
        for name in census.REASONING_FIELDS
    }
    fields["thinking"]["available_sessions"] = available_thinking_sessions
    return seal(
        {
            "schema": census.CENSUS_SCHEMA,
            "source": {
                "normalized_records_sha256": "sha256:" + "a" * 64,
                "success_evidence_sha256": "sha256:" + "b" * 64,
                "candidate_sessions": 12,
                "selected_sessions": 3,
            },
            "selected": {"supervised_tokens": 50, "assistant_targets": 10},
            "fields": fields,
            "reasoning_schema": None,
            "method": census.AGGREGATE_METHOD,
        }
    )


def test_visible_action_selection_uses_only_aggregate_bindings():
    source = aggregate_census(available_thinking_sessions=2)
    manifest = corpus_manifest()

    selection = census.select(source, manifest, objective=census.VISIBLE_ACTIONS_ONLY)

    assert selection["authorized_reasoning_schema"] is None
    assert selection["selected"] == {
        "source_sessions": 3,
        "supervised_tokens": 50,
        "assistant_targets": 10,
    }
    assert selection["source"] == {
        "normalized_records_sha256": "sha256:" + "a" * 64,
        "success_evidence_sha256": "sha256:" + "b" * 64,
    }
    assert "session_ids" not in json.dumps(selection)
    assert (
        census.validate_selection(selection, census=source, corpus_manifest=manifest) == selection
    )


def test_visible_action_selection_rejects_any_selected_reasoning_target():
    source = aggregate_census()
    source["fields"]["thinking"] = {
        "available_sessions": 1,
        "selected_sessions": 1,
        "selected_target_tokens": 3,
    }
    seal(source)

    with pytest.raises(ValueError, match="visible-action selection contains reasoning targets"):
        census.select(source, corpus_manifest(), objective=census.VISIBLE_ACTIONS_ONLY)


def test_unknown_or_unimplemented_reasoning_schema_fails_closed():
    source = aggregate_census()
    source["reasoning_schema"] = "qwen3_student_visible_reasoning_v1"
    source["fields"]["reasoning_content"] = {
        "available_sessions": 1,
        "selected_sessions": 1,
        "selected_target_tokens": 3,
    }
    seal(source)

    with pytest.raises(ValueError, match="not authorized"):
        census.select(source, corpus_manifest(), objective=census.STUDENT_VISIBLE_REASONING)


@pytest.mark.parametrize(
    "path",
    [
        ("source", "session_ids"),
        ("fields", "unexpected"),
        ("selected", "private_reasoning_tokens"),
    ],
)
def test_census_rejects_nonaggregate_or_unknown_fields(path):
    source = aggregate_census()
    source[path[0]][path[1]] = ["not-allowed"]
    seal(source)

    with pytest.raises(ValueError, match="unknown or missing fields"):
        census.validate_census(source)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["catalog_provenance"].update(
            {"normalized_records_file_sha256": "sha256:" + "c" * 64}
        ),
        lambda value: value["files"]["train"].update({"source_sessions": 4}),
        lambda value: value["files"]["train"].update({"supervised_tokens": 51}),
        lambda value: value["files"]["train"].update({"assistant_responses": 11}),
    ],
)
def test_selection_rejects_manifest_binding_drift(mutate):
    manifest = corpus_manifest()
    mutate(manifest)
    seal(manifest)

    with pytest.raises(ValueError, match="does not bind"):
        census.select(aggregate_census(), manifest, objective=census.VISIBLE_ACTIONS_ONLY)


def test_validation_rejects_handwritten_selection_even_if_resealed():
    source = aggregate_census()
    manifest = corpus_manifest()
    selection = census.select(source, manifest, objective=census.VISIBLE_ACTIONS_ONLY)
    forged = copy.deepcopy(selection)
    forged["selected"]["assistant_targets"] = 9
    seal(forged)

    with pytest.raises(ValueError, match="differs from"):
        census.validate_selection(forged, census=source, corpus_manifest=manifest)


def test_selection_rejects_empty_corpus_aggregates():
    manifest = corpus_manifest()
    manifest["files"]["train"]["source_sessions"] = 0
    manifest["files"]["train"]["supervised_tokens"] = 0
    manifest["files"]["train"]["assistant_responses"] = 0
    seal(manifest)

    with pytest.raises(ValueError, match="aggregate totals must be positive"):
        census.select(aggregate_census(), manifest, objective=census.VISIBLE_ACTIONS_ONLY)


def test_cli_is_create_once_and_emits_only_selection_digest(tmp_path: Path, capsys):
    source = tmp_path / "census.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "selection.json"
    source.write_text(json.dumps(aggregate_census()))
    manifest.write_text(json.dumps(corpus_manifest()))

    assert (
        census.main(
            [
                "--census",
                str(source),
                "--corpus-manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema"] == census.SELECTION_SCHEMA
    assert (
        census.main(
            ["--census", str(source), "--corpus-manifest", str(manifest), "--output", str(output)]
        )
        == 2
    )
    assert capsys.readouterr().err.strip() == "error: FileExistsError"
