"""Target-anchored visible-only packing never inherits old-anchor semantics."""

import json
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from training import source
from training import target_dense
from training.dense_bridge import (ALGORITHM as MECHANICS, INPUTS, REQUEST_SCHEMA,
                                   SOURCES, NATIVE_HELPER_SHA, _digest, _file_sha, _legacy_digest,
                                   compose_teacher_ce, TARGET_BUILDER_SHA)
from training.target_dense import METHOD as ALGORITHM, audit, build, verify_method


def _source_fixture():
    fixture_type = runpy.run_path(str(Path(__file__).with_name("test_source.py")))["SourceTests"]
    fixture = fixture_type("test_private_success_and_corpus_interface")
    fixture.setUp()
    fixture.messages[2]["thinking"] = "PRIVATE_REASONING_SENTINEL"
    fixture.selection["trace_sha256"] = source.digest(fixture.envelope)
    return fixture


def test_new_dense_wrapper_retains_old_mechanics_but_is_not_train_ready():
    import pyarrow as pa
    import pyarrow.parquet as pq

    fixture = _source_fixture()
    try:
        with (patch.object(source, "TOOL_DIGEST", source.digest(fixture.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            source.fetch(fixture.request(), get=fixture.get)
            private = fixture.root / "private-output"
            checked_source = audit(private)
            assert checked_source["method"] == ALGORITHM
            assert checked_source["hidden_reasoning_fields_removed"] == 1
            assert checked_source["trainer_ready"] is False
            assert "PRIVATE_REASONING_SENTINEL" not in json.dumps(checked_source)
            bindings = {}
            for name in INPUTS:
                if name in {"normalized", "evidence", "model_request_capture"}:
                    file = private / ({"normalized": "dense-target-anchored.jsonl",
                                       "evidence": "dense-success-evidence.jsonl",
                                       "model_request_capture": "model-request-capture.json"}[name])
                else:
                    file = fixture.root / name
                    file.write_text("{}\n")
                bindings[name] = {"path": str(file), "sha256": _file_sha(file)}
            destination = fixture.root / "target-dense"
            request = {"schema": REQUEST_SCHEMA, **bindings,
                       "tokenizer_root": str(fixture.root / "tokenizer"),
                       "teacher_models": ["teacher"], "max_length": 98304,
                       "context_tokens": 98304, "output": str(destination)}
            request["sha256"] = _digest(request)
            request_file = fixture.root / "request.json"
            request_file.write_text(json.dumps(request))

            def fake_frozen_builder(path, *, target_names):
                assert target_names is True
                generated = json.loads(Path(path).read_text())
                stage = Path(generated["output"])
                stage.mkdir()
                pq.write_table(pa.Table.from_pylist([{
                    "source_session_id": "session-a", "target_spans": [
                        {"assistant_index": 0, "token_start": 1, "token_end": 3}],
                    "target_token_count": 2, "loss_mask": [0, 1, 1, 0],
                    "window_algorithm": MECHANICS}]), stage / "train.parquet")
                (stage / "source-selection.private.jsonl").write_text("{}\n")
                train = {"path": "train.parquet", "sha256": _file_sha(stage / "train.parquet"),
                         "format": "pretokenized_assistant_segments_v1", "rows": 1,
                         "supervised_tokens": 2}
                manifest = {"schema": "cyber_dense_sft_corpus_v1", "algorithm": MECHANICS,
                            "validation_mode": "task_outcomes_only", "files": {"train": train},
                            "materialization": {"request_sha256": generated["sha256"],
                                                "normalized_sha256": generated["normalized"]["sha256"],
                                                "success_evidence_sha256": generated["evidence"]["sha256"],
                                                "model_request_capture_sha256": generated["model_request_capture"]["sha256"]},
                            "builder_sha256": {
                                "message_aligned_teacher_corpus.py": TARGET_BUILDER_SHA,
                                "dense.py": "sha256:" + SOURCES["training/dense.py"],
                                "corpus.py": "sha256:" + SOURCES["training/corpus.py"],
                                "native_helper": NATIVE_HELPER_SHA}, "limitations": []}
                manifest["sha256"] = _legacy_digest(manifest)
                (stage / "manifest.json").write_text(json.dumps(manifest))
                receipt = {"schema": "cyber_message_aligned_teacher_corpus_receipt_v1",
                           "manifest_sha256": manifest["sha256"],
                           "manifest_file_sha256": _file_sha(stage / "manifest.json"),
                           "train_parquet_sha256": train["sha256"],
                           "source_selection_sha256": _file_sha(stage / "source-selection.private.jsonl"),
                           "rows": 1,
                           "supervised_tokens": 2}
                receipt["sha256"] = _legacy_digest(receipt)
                (stage / "RECEIPT.json").write_text(json.dumps(receipt))
                return {"manifest_sha256": manifest["sha256"], "receipt_sha256": receipt["sha256"],
                        "train_sha256": train["sha256"], "rows": 1, "supervised_tokens": 2}

            with patch.object(target_dense, "build_dense", fake_frozen_builder):
                published = build(request_file, private, destination)
            checked = verify_method(destination, private)
            with pytest.raises(ValueError, match="lacks live serving attestation"):
                compose_teacher_ce(destination, fixture.root / "dev", fixture.root / "dev-source",
                                   fixture.root / "roster.json", fixture.root, fixture.root / "mixed.json",
                                   train_source_dir=private)
        assert json.loads((destination / "TARGET-METHOD.json").read_text())["method"] == ALGORITHM
        assert published["trainer_ready"] is False
        assert checked["method_sha256"] == published["method_sha256"]
        assert checked["trainer_ready"] is False
        with (patch.object(source, "TOOL_DIGEST", source.digest(fixture.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              pytest.raises(FileExistsError, match="immutable output already exists")):
            build(request_file, private, destination)
        original = private / "raw-sources.private.jsonl"
        original.write_text(original.read_text() + "\n")
        with (patch.object(source, "TOOL_DIGEST", source.digest(fixture.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              pytest.raises(ValueError, match="bound private source file differs")):
            audit(private)
    finally:
        fixture.temp.cleanup()
