"""The frozen dense packer can be staged and its input gate rejects drift."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.dense_bridge import (COMMIT, _digest, _legacy_digest, stage_historical,
                                   validate_request, INPUTS, REQUEST_SCHEMA)


class DenseBridgeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
