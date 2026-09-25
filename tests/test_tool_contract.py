import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import tool_contract as contract


REVISION = "1" * 40
TOOLS = [
    {"type": "function", "function": {"name": "fleet_bash", "description": "Test tool",
                                   "parameters": {"type": "object", "properties": {"script": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "fleet_submit_report", "description": "Test report",
                                   "parameters": {"type": "object", "properties": {"flag": {"type": "string"}}}}},
]


def request(*, tools=TOOLS, model="test-model", message="PRIVATE TEST PROMPT") -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": message}],
                       "tools": tools}, separators=(",", ":")).encode()


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.pinned = patch.object(contract, "TOOLS", contract._digest(TOOLS, ascii=True))
        self.pinned.start()
        self.addCleanup(self.pinned.stop)

    def test_capture_is_sealed_and_contains_no_message(self):
        raw = request()
        result = contract.capture_request(raw, model_id="test-model", revision=REVISION)
        self.assertEqual(result["request_envelope_sha256"],
                         "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["sha256"], contract._digest(
            {key: value for key, value in result.items() if key != "sha256"}))
        self.assertEqual(result["captured_tools"], TOOLS)
        self.assertNotIn("PRIVATE TEST PROMPT", json.dumps(result))
        self.assertEqual(result["provider_schema_transform_sha256"], contract.RELEASE)

    def test_rejects_wrong_or_extra_tools(self):
        for tools in ([], TOOLS[:1], TOOLS + TOOLS[:1], list(reversed(TOOLS)),
                      [{"type": "function", "function": "bad"}, TOOLS[1]]):
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                contract.capture_request(request(tools=tools), model_id="test-model", revision=REVISION)

    def test_rejects_wrong_model_path_and_duplicate_json_key(self):
        with self.assertRaises(ValueError):
            contract.capture_request(request(model="other"), model_id="test-model", revision=REVISION)
        with self.assertRaises(ValueError):
            contract.capture_request(request(), model_id="test-model", revision=REVISION, path="/other")
        with self.assertRaises(ValueError):
            contract.capture_request(b'{"model":"test-model","model":"test-model","tools":[]}',
                                     model_id="test-model", revision=REVISION)

    def test_private_create_once(self):
        receipt = contract.capture_request(request(), model_id="test-model", revision=REVISION)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.json"
            contract.write_once(output, receipt)
            self.assertEqual(stat.S_IMODE(os.stat(output).st_mode), 0o600)
            self.assertEqual(json.loads(output.read_text()), receipt)
            with self.assertRaises(FileExistsError):
                contract.write_once(output, receipt)
            self.assertNotIn("PRIVATE TEST PROMPT", output.read_text())

    def test_anchor_hash_comparison_never_emits_text(self):
        raw = json.dumps({"model": "test-model", "tools": TOOLS,
                          "messages": [{"role": "system", "content": "PRIVATE SYSTEM"},
                                       {"role": "user", "content": "PRIVATE TASK"}]}).encode()
        source_system = "sha256:" + hashlib.sha256(b"PRIVATE SYSTEM").hexdigest()
        source_task = "sha256:" + hashlib.sha256(b"PRIVATE TASK").hexdigest()
        result = contract.anchor_digests(
            raw, task_version_id="version-1", source_system_sha256=source_system,
            source_task_sha256=source_task, model_id="test-model", revision=REVISION)
        self.assertTrue(result["system_equal"])
        self.assertTrue(result["task_equal"])
        self.assertNotIn("PRIVATE", json.dumps(result))
        mismatched = contract.anchor_digests(
            raw, task_version_id="version-1", source_system_sha256=source_system,
            source_task_sha256="sha256:" + "0" * 64, model_id="test-model", revision=REVISION)
        self.assertFalse(mismatched["task_equal"])
        multipart = json.dumps({"model": "test-model", "tools": TOOLS,
                                "messages": [{"role": "system", "content": "PRIVATE SYSTEM"},
                                             {"role": "developer", "content": "EXTRA"},
                                             {"role": "user", "content": [
                                                 {"type": "text", "text": "PRIVATE "},
                                                 {"type": "text", "text": "TASK"}]}]}).encode()
        result = contract.anchor_digests(
            multipart, task_version_id="version-1", source_system_sha256=source_system,
            source_task_sha256=source_task, model_id="test-model", revision=REVISION)
        self.assertFalse(result["system_equal"])
        self.assertTrue(result["task_equal"])
        self.assertEqual(len(result["target_system_message_sha256"]), 2)

    def test_private_target_anchor_requires_observed_cli_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt"
            prompt.write_text("PRIVATE TASK")
            output = Path(directory) / "target-anchor.json"
            raw = json.dumps({"model": "test-model", "tools": TOOLS,
                              "messages": [{"role": "system", "content": "PRIVATE SYSTEM"},
                                           {"role": "user", "content": '"PRIVATE TASK"'}]}).encode()
            safe = contract.write_private_target_anchor(
                raw, model_id="test-model", revision=REVISION,
                task_version_id="version-1", prompt_path=prompt, output=output)
            self.assertTrue(safe["verified_cli_argument_transform"])
            self.assertNotIn("PRIVATE", json.dumps(safe))
            self.assertEqual(stat.S_IMODE(os.stat(output).st_mode), 0o600)
            self.assertEqual(json.loads(output.read_text())["messages"][1]["content"], '"PRIVATE TASK"')
            with self.assertRaises(ValueError):
                contract.write_private_target_anchor(
                    raw.replace(b'\\"PRIVATE TASK\\"', b'PRIVATE TASK'),
                    model_id="test-model", revision=REVISION,
                    task_version_id="version-1", prompt_path=prompt,
                    output=Path(directory) / "rejected.json")

    def test_cli_transform_escapes_embedded_quotes_but_keeps_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt"
            prompt.write_text('Do "this".\nThen act.\n')
            raw = json.dumps({"model": "test-model", "tools": TOOLS,
                              "messages": [{"role": "system", "content": "SYS"},
                                           {"role": "user", "content": '"Do \\"this\\".\nThen act."'}]}).encode()
            safe = contract.write_private_target_anchor(
                raw, model_id="test-model", revision=REVISION,
                task_version_id="version-1", prompt_path=prompt,
                output=Path(directory) / "target.json")
            self.assertTrue(safe["verified_cli_argument_transform"])


if __name__ == "__main__":
    unittest.main()
