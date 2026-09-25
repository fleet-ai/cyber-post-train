"""The frozen dense packer can be staged and its input gate rejects drift."""

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.dense_bridge import (COMMIT, SOURCES, NATIVE_HELPER_SHA, _digest, _legacy_digest, stage_historical,
                                   validate_request, compose_teacher_ce, _file_sha,
                                   INPUTS, REQUEST_SCHEMA, ALGORITHM, MANIFEST_SCHEMA,
                                   RECEIPT_SCHEMA, TARGET_BUILDER_SHA, TARGET_ROOT_ID)


class DenseBridgeTest(unittest.TestCase):
    def test_target_only_patch_groups_adjacent_tool_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_historical(root, target_names=True)
            tree = ast.parse((root / "training/message_aligned_teacher_corpus.py").read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_record")
            scope = {"Excluded": ValueError}
            exec(compile("from __future__ import annotations\n" + ast.unparse(function), "<target-only>", "exec"), scope)

            class Tokenizer:
                def apply_chat_template(self, messages, **_):
                    rendered = ""
                    for i, m in enumerate(messages):
                        if m["role"] == "tool":
                            rendered += ("U" if i == 0 or messages[i - 1]["role"] != "tool" else "")
                            rendered += "T" + m["content"]
                            rendered += ("E" if i == len(messages) - 1 or messages[i + 1]["role"] != "tool" else "")
                        else:
                            rendered += m["role"][0] + m["content"]
                    return list(rendered.encode())

            base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            scope["encode_messages_subset"] = lambda ms, tok, tokenizer_kwargs: tok.apply_chat_template(
                base + ms)[len(tok.apply_chat_template(base)):]
            exec("def helper(ms, tok, tokenizer_kwargs):\n"
                 " ids = encode_messages_subset(ms, tok, tokenizer_kwargs)\n"
                 " return ids, [int(ms[0]['role']=='assistant')]*len(ids), None", scope)
            messages = [{"role": role, "content": value} for role, value in
                        [("system", "s"), ("user", "u"), ("assistant", "a"),
                         ("tool", "x"), ("tool", "y"), ("assistant", "b")]]
            tokenizer = Tokenizer()
            anchor, chunks = scope["encode_record"](messages, tokenizer, scope["helper"], tools=[])
            self.assertEqual(len(chunks), 3)
            self.assertFalse(any(chunks[1]["mask"]))
            self.assertEqual(anchor + [token for chunk in chunks for token in chunk["ids"]],
                             tokenizer.apply_chat_template(messages))
            dense = ast.parse((root / "training/dense.py").read_text())
            segment = next(n for n in dense.body if isinstance(n, ast.FunctionDef) and n.name == "segment_record")
            scope.update(MAX_TOKENS=1024, CONTEXT_BUDGET=1024, digest_json=_legacy_digest)
            exec(compile("from __future__ import annotations\n" + ast.unparse(segment), "<pinned-segment>", "exec"), scope)
            record = {"record_id": "synthetic", "lineage": {"task_key": "synthetic"},
                      "source": {"model": "synthetic"}}
            rows = scope["segment_record"](record, anchor, chunks, max_tokens=1024, context_budget=1024)
            spans = [span for row in rows for span in row["target_spans"]]
            self.assertEqual([span["assistant_index"] for span in spans], [0, 1])
            for row in rows:
                last = max(span["source_message_index"] for span in row["target_spans"])
                selected = messages[:2] + messages[row["context_start_message_index"]:last + 1]
                self.assertEqual(row["input_ids"], tokenizer.apply_chat_template(selected))

    def test_exact_frozen_directory_needs_no_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_historical(root / "sealed")
            with patch.dict(os.environ, {"CYBER_HISTORICAL_ROOT": str(root / "sealed")}):
                self.assertEqual(stage_historical(root / "copy", repository=root / "not-a-repo",
                                                  target_names=True), TARGET_BUILDER_SHA)
                (root / "sealed/training/dense.py").write_bytes(b"drift")
                with self.assertRaisesRegex(ValueError, "differs"):
                    stage_historical(root / "rejected", repository=root / "not-a-repo")

    def test_historical_dependency_closure_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_historical(root)
            result = subprocess.run(
                [sys.executable, "-m", "training.message_aligned_teacher_corpus", "--help"],
                cwd=root, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, "frozen packer import closure changed")

    def test_frozen_builder_creates_synthetic_dense_parquet(self):
        # Re-run the historical packer's own end-to-end synthetic fixture from
        # the same pinned Git object, with no private data or tokenizer access.
        assets = {
            "tests/test_message_aligned_teacher_corpus.py":
                "62d1e7e384361296b837c63665b8e7a2dcedb10320b8256a528d2143a38cebbe",
            "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json":
                "e4a3c4fb5b5c34cdaf64ec568eb31fcc0d55a63cc0a134808db348c65d7b6858",
        }
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_historical(root)
            for relative, expected in assets.items():
                source = subprocess.run(
                    ["git", "-C", str(repository), "show", f"{COMMIT}:{relative}"],
                    capture_output=True, check=True,
                ).stdout
                self.assertEqual(hashlib.sha256(source).hexdigest(), expected)
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q",
                 "tests/test_message_aligned_teacher_corpus.py::test_create_once_builder_emits_new_algorithm_and_preserves_dense_format"],
                cwd=root, capture_output=True, check=False, env={**os.environ, "PYTHONPATH": str(root)},
            )
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors="replace")[-1000:])

    def test_new_method_mechanics_patch_accepts_exact_target_tool_names(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(stage_historical(root, target_names=True), TARGET_BUILDER_SHA)
            for relative in ("tests/test_message_aligned_teacher_corpus.py",
                             "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"):
                payload = subprocess.run(["git", "-C", str(repository), "show", f"{COMMIT}:{relative}"],
                                         capture_output=True, check=True).stdout
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            script = ("import runpy; from pathlib import Path; "
                      "x=runpy.run_path('tests/test_message_aligned_teacher_corpus.py'); "
                      "r=x['_record'](); "
                      "[c['function'].__setitem__('name', 'fleet_bash' if c['function']['name']=='bash' "
                      "else 'fleet_submit_report') for m in r['messages'] for c in m.get('tool_calls',[])]; "
                      "r['content_digest']=x['digest_json']({k:v for k,v in r.items() if k!='content_digest'}); "
                      f"z=x['_roster'](); z['root_role_anchor_id']='{TARGET_ROOT_ID}'; "
                      "z=x['_sealed']({k:v for k,v in z.items() if k!='sha256'}); "
                      "rows,_,_=x['_materialize'](r,z); "
                      "assert rows and all(s['source_target_sha256'] for row in rows "
                      "for s in row['target_spans'])")
            result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True,
                                    env={**os.environ, "PYTHONPATH": str(root)}, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace")[-1000:])

    def test_request_shape_and_bound_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings = {}
            for name in INPUTS:
                payload = json.dumps({"synthetic": name}).encode()
                (root / name).write_bytes(payload)
                bindings[name] = {"path": name, "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}
            request = {"schema": REQUEST_SCHEMA, **bindings,
                       "tokenizer_root": str(root / "frozen-tokenizer"),
                       "teacher_models": ["synthetic-teacher"], "max_length": 98304,
                       "context_tokens": 98304, "output": "dense-out"}
            request["sha256"] = _digest(request)
            path = root / "request.json"
            path.write_text(json.dumps(request))
            self.assertEqual(validate_request(path), request)
            (root / "normalized").write_text("changed")
            with self.assertRaisesRegex(ValueError, "bound input differs"):
                validate_request(path)

    def test_request_seal_and_output_are_create_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings = {}
            for name in INPUTS:
                (root / name).write_bytes(b"{}")
                bindings[name] = {"path": name, "sha256": "sha256:" + hashlib.sha256(b"{}").hexdigest()}
            request = {"schema": REQUEST_SCHEMA, **bindings, "tokenizer_root": "frozen",
                       "teacher_models": ["teacher"], "max_length": 32, "context_tokens": 16,
                       "output": "dense-out"}
            request["sha256"] = _digest(request)
            path = root / "request.json"
            path.write_text(json.dumps(request))
            request["max_length"] = 33
            path.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "seal differs"):
                validate_request(path)
            request["sha256"] = _digest({k: v for k, v in request.items() if k != "sha256"})
            path.write_text(json.dumps(request))
            (root / "dense-out").mkdir()
            with self.assertRaises(FileExistsError):
                validate_request(path)

    def test_request_and_legacy_receipt_json_seals_are_distinct(self):
        self.assertNotEqual(_digest({"name": "é"}), _legacy_digest({"name": "é"}))

    def test_mixed_manifest_compiles_with_pinned_teacher_ce_runtime(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from training import launch
        from training.runtime import BACKEND_SHA, MODEL, TOKENIZER_FILES, TOKENIZER_SHA

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense_dir, dev_dir, source_dir = (root / name for name in ("dense", "dev", "source"))
            for directory in (dense_dir, dev_dir, source_dir):
                directory.mkdir()
            train_group, dev_group, test_group = ("sha256:" + c * 64 for c in "cde")
            pq.write_table(pa.Table.from_pylist([{"task_key": "train-task",
                                                   "source_task_version_id": "train-version",
                                                   "source_group_id": train_group}]),
                           dense_dir / "train.parquet")
            pq.write_table(pa.Table.from_pylist([{"task_version_id": "dev-version",
                                                   "family_id": dev_group}]),
                           dev_dir / "dev.parquet")
            roster = {"schema": "cyber_exact_task_family_role_roster_v1",
                      "root_role_anchor_id": "fleet-blackbox-current-study-20260914-v2",
                      "family_role_anchor_sha256": "sha256:" + "f" * 64,
                      "heldout_group_ids": sorted([dev_group, test_group]),
                      "identities": [
                          {"task_key": "train-task", "task_version_id": "train-version",
                           "group_id": train_group, "split": "train"},
                          {"task_key": "dev-task", "task_version_id": "dev-version",
                           "group_id": dev_group, "split": "dev"},
                          {"task_key": "test-task", "task_version_id": "test-version",
                           "group_id": test_group, "split": "final_test"}]}
            roster["sha256"] = _digest(roster)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps(roster))
            projected = {item["task_version_id"]: {"family_id": item["group_id"],
                         "split": "test" if item["split"] == "final_test" else item["split"]}
                         for item in roster["identities"]}
            tools_sha = "sha256:" + "a" * 64
            (source_dir / "dev.jsonl").write_text("{}\n")
            dev_proof = {"rows": 1, "inputs_sha256": {"roster": _legacy_digest(projected),
                                                       "tools": tools_sha}}
            dev_proof["sha256"] = _legacy_digest(dev_proof)
            source = {"format": "structured_message_windows_v1", "trainer_ready": False,
                      "partitions": {"train": {}, "dev": {
                          "path": "dev.jsonl", "bytes": (source_dir / "dev.jsonl").stat().st_size,
                          "sha256": _file_sha(source_dir / "dev.jsonl"), "receipt": dev_proof}}}
            source["sha256"] = _legacy_digest(source)
            (source_dir / "RECEIPT.json").write_text(json.dumps(source))
            train = {"path": "train.parquet", "sha256": _file_sha(dense_dir / "train.parquet"),
                     "rows": 1, "task_keys": ["train-task"],
                     "format": "pretokenized_assistant_segments_v1", "supervised_tokens": 10,
                     "assistant_responses": 1, "source_sessions": 1,
                     "source_total_assistant_responses": 1, "excluded_assistant_responses": 0}
            dense = {"schema": MANIFEST_SCHEMA, "algorithm": ALGORITHM,
                     "validation_mode": "task_outcomes_only", "split_sha256": roster["family_role_anchor_sha256"],
                     "materialization": {"family_roster_sha256": roster["sha256"],
                                         "target_tools_sha256": tools_sha},
                     "builder_sha256": {
                         "message_aligned_teacher_corpus.py": "sha256:" + SOURCES["training/message_aligned_teacher_corpus.py"],
                         "dense.py": "sha256:" + SOURCES["training/dense.py"],
                         "corpus.py": "sha256:" + SOURCES["training/corpus.py"],
                         "native_helper": NATIVE_HELPER_SHA},
                     "tokenizer": {"repo": MODEL[0], "revision": MODEL[1],
                                   "files": [{"path": p, "sha256": h} for p, h in TOKENIZER_FILES.items()],
                                   "chat_template_sha256": TOKENIZER_FILES["chat_template.jinja"],
                                   "backend_sha256": BACKEND_SHA},
                     "files": {"train": train}, "dev_windows": 0}
            dense["sha256"] = _legacy_digest(dense)
            dense_path = dense_dir / "manifest.json"
            dense_path.write_text(json.dumps(dense))
            receipt = {"schema": RECEIPT_SCHEMA, "manifest_file_sha256": _file_sha(dense_path),
                       "manifest_sha256": dense["sha256"], "train_parquet_sha256": train["sha256"],
                       "rows": 1, "supervised_tokens": 10}
            receipt["sha256"] = _legacy_digest(receipt)
            (dense_dir / "RECEIPT.json").write_text(json.dumps(receipt))
            dev = {"path": "dev.parquet", "sha256": _file_sha(dev_dir / "dev.parquet"),
                   "rows": 1, "task_keys": ["dev-version"],
                   "format": "chat_messages_last_assistant_v2"}
            native = {"schema": "qwen38_tool_aware_parquet_v1", "trainer_ready": True,
                      "validation_mode": "teacher_cross_entropy",
                      "source_receipt_sha256": source["sha256"],
                      "tokenizer": {"repo": MODEL[0], "revision": MODEL[1],
                                    "sha256": TOKENIZER_SHA},
                      "files": {"train": {"format": "chat_messages_last_assistant_v2"}, "dev": dev}}
            native["sha256"] = _legacy_digest(native)
            (dev_dir / "manifest.json").write_text(json.dumps(native))
            output = root / "mixed.json"
            composed = compose_teacher_ce(dense_dir, dev_dir, source_dir, roster_path, root, output)
            manifest = json.loads(output.read_text())
            self.assertEqual(composed["manifest_sha256"], manifest["sha256"])
            self.assertEqual(manifest["files"]["train"]["path"], "dense/train.parquet")
            self.assertEqual(manifest["files"]["dev"]["path"], "dev/dev.parquet")
            config = json.loads((Path(__file__).parents[1] / "configs/runs/qwen38-96k-full-v1.json").read_text())
            config["name"] = "synthetic-dense-ce"
            config["output_root"] = "/mnt/sfs/jobs/synthetic-dense-ce"
            config["data"] = {"manifest": "../data/corpus.json", "root": "/mnt/sfs/jobs/synthetic-dense-ce"}
            config["recipe"].update(eval_interval=1, checkpoint_interval=1, keep_checkpoints=1)
            compiled = launch._legacy("compile", {"config": {**config,
                "model": {"root": config["model"]["root"],
                          "lock": "../models/qwen38-27b-1d4bf0f2.lock.json",
                          "weights": "../models/qwen38-27b-1d4bf0f2.weights.json"}}},
                manifest=output.read_bytes())
            plan = compiled["plan"]
            self.assertEqual(plan["validation_mode"], "teacher_cross_entropy")
            self.assertEqual(plan["datasets"]["train"]["format"], "pretokenized_assistant_segments_v1")
            self.assertEqual(plan["datasets"]["dev"]["format"], "chat_messages_last_assistant_v2")
            self.assertEqual(plan["recipe"]["max_steps"], 1)
            with self.assertRaises(ValueError):
                compose_teacher_ce(dense_dir, dev_dir, source_dir, roster_path, root, output)
            pq.write_table(pa.Table.from_pylist([{"task_version_id": "dev-version",
                                                   "family_id": train_group}]),
                           dev_dir / "dev.parquet")
            dev["sha256"] = _file_sha(dev_dir / "dev.parquet")
            native["sha256"] = _legacy_digest({k: v for k, v in native.items() if k != "sha256"})
            (dev_dir / "manifest.json").write_text(json.dumps(native))
            with self.assertRaisesRegex(ValueError, "family roles"):
                compose_teacher_ce(dense_dir, dev_dir, source_dir, roster_path, root, root / "bad.json")


if __name__ == "__main__":
    unittest.main()
