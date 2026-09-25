"""The source boundary is tested with synthetic Fleet envelopes only."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import source
from training.runtime import sha256 as corpus_sha256


def seal(value):
    return {**value, "sha256": source.digest(value)}


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.messages = [
            {"role": "system", "content": "Follow the task."},
            {"role": "user", "content": [{"type": "text", "text": "Find the issue."}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-1", "function": {"name": "bash", "arguments": {"script": "true"}}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-2", "function": {"name": "submit_report", "arguments": {"explanation": "done"}}}]},
            {"role": "tool", "tool_call_id": "call-2", "content": "accepted"},
        ]
        self.verifier = {"id": "v-1", "score": 1, "success": True, "execution_time_ms": 42}
        self.harness = {"mode": "tool-use", "tool_names": ["bash", "submit_report"]}
        self.envelope = {"harness": self.harness,
                         "instance": {"team_id": source.TEAM},
                         "task": {"key": "task-a", "eval_task_version_id": "version-a",
                                  "prompt": "Find the issue."},
                         "transcript": self.messages, "verifier_execution": self.verifier}
        self.summary = {"session_id": "session-a", "task_key": "task-a", "model": "teacher",
                        "status": "completed", "verifier_execution": self.verifier}
        self.selection = {"session_id": "session-a", "task_key": "task-a",
                          "task_version_id": "version-a", "model_id": "teacher",
                          "group_id": "task-key:synthetic",
                          "trace_sha256": source.digest(self.envelope),
                          "acceptance_sha256": "sha256:" + "2" * 64,
                          "harness_mode": "tool-use", "harness_sha256": source.digest(self.harness)}
        self.roles = seal({"schema": "fleet_teacher_family_roles_v1", "identities": [
            {"task_key": "task-a", "task_version_id": "version-a",
             "family_id": self.selection["group_id"], "split": "train"}]})
        self.tools = [{"type": "function", "function": {"name": name, "description": "test",
                                                   "parameters": {"type": "object"}}}
                      for name in ("fleet_bash", "fleet_submit_report")]
        self.capture = seal({"schema": "cyber_opencode_model_request_tool_capture_v1",
                             "target_harness": source.HARNESS,
                             "target_model": {"repo": "Qwen/Qwen3.8-27B", "revision": "a" * 40},
                             "provider_schema_transform_sha256": "sha256:" + "3" * 64,
                             "request_envelope_sha256": "sha256:" + "4" * 64,
                             "captured_tools": self.tools})
        self.anchor = seal({"schema": "cyber_opencode_private_target_anchor_v1",
                            "task_version_id": "version-a", "request_envelope_sha256": "sha256:" + "5" * 64,
                            "messages": [{"role": "system", "content": "Target system."},
                                         {"role": "user", "content": '"Find the issue."'}]})
        self.get_calls = []

    def _file(self, name, value, *, jsonl=False):
        path = self.root / name
        path.write_text(("\n".join(json.dumps(row) for row in value) + "\n") if jsonl else json.dumps(value))
        return {"path": str(path), "sha256": source.file_digest(path)}

    def request(self):
        selection = self._file("selection.jsonl", [self.selection], jsonl=True)
        hydration_request = seal({"schema": "fleet_teacher_source_hydration_v1",
                                  "selection": selection,
                                  "output": str(self.root / "source-hydration")})
        source.hydrate(hydration_request, get=self.get)
        return seal({"schema": "fleet_teacher_source_fetch_v1",
                     "selection": selection,
                     "hydration": {"path": str(self.root / "source-hydration" / "HYDRATED.json"),
                                   "sha256": source.file_digest(self.root / "source-hydration" / "HYDRATED.json")},
                     "roles": self._file("roles.json", self.roles),
                     "tool_capture": self._file("capture.json", self.capture),
                     "target_anchors": [self._file("anchor.json", self.anchor)],
                     "model_revision": "a" * 40, "output": str(self.root / "private-output")})

    def get(self, path):
        self.get_calls.append(path)
        if path == "/v1/account":
            return {"team_id": source.TEAM, "team_name": "fleet"}
        if path.startswith("/v1/sessions?"):
            return {"sessions": [self.summary], "has_more": False}
        if path == "/v1/sessions/session-a/transcript":
            return self.envelope
        raise AssertionError("unexpected route")

    def test_private_success_and_corpus_interface(self):
        self.messages[2]["thinking"] = "HIDDEN_REASONING_SENTINEL"
        self.messages[2]["analysis"] = "HIDDEN_ANALYSIS_SENTINEL"
        self.selection["trace_sha256"] = source.digest(self.envelope)
        request = self.request()
        self.get_calls.clear()
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(request, get=self.get)
        self.assertFalse(any(path.endswith("/transcript") for path in self.get_calls))
        directory = self.root / "private-output"
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((directory / "records.jsonl").stat().st_mode), 0o600)
        self.assertEqual(receipt["retained_sessions"], 1)
        self.assertFalse(receipt["training_ready"])
        self.assertEqual(receipt["model_facing_tools_sha256"], source.digest(self.tools, ascii=True))
        record = json.loads((directory / "records.jsonl").read_text().splitlines()[0])
        proof = json.loads((directory / "success-evidence.json").read_text())["session-a"]
        self.assertEqual(record["messages"][0]["content"], "Target system.")
        self.assertEqual(record["messages"][1]["content"], '"Find the issue."')
        self.assertEqual([call["function"]["name"] for message in record["messages"]
                          for call in message.get("tool_calls", [])],
                         ["fleet_bash", "fleet_submit_report"])
        self.assertEqual(record["messages"][2]["tool_calls"][0]["function"]["arguments"],
                         {"script": "true"})
        self.assertEqual(record["messages"][3], self.messages[3])
        self.assertEqual(record["tool_transform"]["target_messages_sha256"],
                         source.digest(record["messages"]))
        self.assertEqual(record["anchor_transform"]["source_user_sha256"],
                         source.text_digest("Find the issue."))
        self.assertEqual(record["anchor_transform"]["target_user_sha256"],
                         source.text_digest('"Find the issue."'))
        self.assertNotIn("HIDDEN_REASONING_SENTINEL", json.dumps(record))
        self.assertNotIn("HIDDEN_ANALYSIS_SENTINEL", json.dumps(record))
        raw = json.loads((directory / "raw-sources.private.jsonl").read_text())
        self.assertEqual(raw["transcript_envelope"], self.envelope)
        self.assertEqual(record["visibility_transform"]["hidden_reasoning_fields_removed"], 2)
        self.assertEqual(proof["source_sha256"], corpus_sha256(record))
        self.assertEqual(proof["report_call_id"], "call-2")
        self.assertEqual(receipt["files"]["records"], source.file_digest(directory / "records.jsonl"))
        dense = json.loads((directory / "dense-target-anchored.jsonl").read_text())
        dense_proof = json.loads((directory / "dense-success-evidence.jsonl").read_text())
        self.assertEqual(dense["schema"], "fleet_cyber_trajectory_v1")
        self.assertEqual(dense["content_digest"], source.digest({k: v for k, v in dense.items()
                                                                 if k != "content_digest"}))
        self.assertEqual(dense_proof["normalized_record_sha256"], source.digest(dense))
        self.assertEqual(dense_proof["successful_report_call_id"], "call-2")
        self.assertEqual([call["function"]["name"] for message in dense["messages"]
                          for call in message.get("tool_calls", [])],
                         ["fleet_bash", "fleet_submit_report"])
        self.assertEqual(dense_proof["sha256"], source.digest({k: v for k, v in dense_proof.items()
                                                               if k != "sha256"}))

    def test_legacy_anchor_retained_only_with_explicit_target_transform(self):
        self.messages[1]["content"][0]["text"] = "Call the bash tool."
        self.envelope["task"]["prompt"] = "Call the bash tool."
        self.selection["trace_sha256"] = source.digest(self.envelope)
        self.anchor = seal({**{k: v for k, v in self.anchor.items() if k != "sha256"},
                            "messages": [{"role": "system", "content": "Target system."},
                                         {"role": "user", "content": '"Call the bash tool."'}]})
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 1)
        self.assertFalse(receipt["training_ready"])
        self.assertEqual(receipt["original_anchor_legacy_tool_name_sessions"], 1)
        record = json.loads((self.root / "private-output" / "records.jsonl").read_text())
        raw = json.loads((self.root / "private-output" / "raw-sources.private.jsonl").read_text())
        self.assertEqual(raw["transcript_envelope"], self.envelope)

    def test_missing_exact_capture_fails_before_any_api_call(self):
        self.capture["captured_tools"][0]["function"]["name"] = "bash"
        self.capture = seal({k: v for k, v in self.capture.items() if k != "sha256"})
        request = self.request()
        self.get_calls.clear()
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            with self.assertRaises(source.SourceError):
                source.fetch(request, get=self.get)
        self.assertFalse(self.get_calls)
        self.assertFalse((self.root / "private-output").exists())

    def test_source_tool_discovery_is_not_silently_deleted(self):
        self.messages[2:2] = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "discovery-1", "function": {"name": "search_tool", "arguments": {"query": "Fleet"}}}]},
            {"role": "tool", "tool_call_id": "discovery-1", "content": "metadata"},
        ]
        self.selection["trace_sha256"] = source.digest(self.envelope)
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 0)
        self.assertEqual(receipt["exact_discovery_elided_sessions"], 0)
        raw = json.loads((self.root / "private-output" / "raw-sources.private.jsonl").read_text())
        self.assertIn("search_tool", json.dumps(raw["transcript_envelope"]))

    def test_changed_authoritative_verifier_has_no_completion_receipt(self):
        self.summary["verifier_execution"] = {**self.verifier, "score": 0}
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            with self.assertRaises(source.SourceError):
                source.fetch(self.request(), get=self.get)
        self.assertFalse((self.root / "private-output" / "RECEIPT.json").exists())

    def test_hydration_resumes_without_refetching_existing_private_source(self):
        selection = self._file("hydration-selection.jsonl", [self.selection], jsonl=True)
        request = seal({"schema": "fleet_teacher_source_hydration_v1", "selection": selection,
                        "output": str(self.root / "private-hydration")})
        first = source.hydrate(request, get=self.get)
        self.assertTrue(first["complete"])
        self.assertEqual(first["new_sessions"], 1)
        self.assertEqual(stat.S_IMODE((self.root / "private-hydration").stat().st_mode), 0o700)
        session_file = next((self.root / "private-hydration" / "sessions").glob("*.json"))
        self.assertEqual(stat.S_IMODE(session_file.stat().st_mode), 0o600)
        self.assertEqual(json.loads(session_file.read_text())["transcript_envelope"], self.envelope)
        self.get_calls.clear()
        second = source.hydrate(request, get=self.get)
        self.assertEqual(second["new_sessions"], 0)
        self.assertEqual(self.get_calls, ["/v1/account"])
        self.assertEqual(second["receipt_sha256"], first["receipt_sha256"])
        attested = source.verify_hydration_cache(
            self.root / "private-hydration", Path(selection["path"]),
            source.file_digest(self.root / "private-hydration" / "HYDRATED.json"))
        self.assertEqual(attested["selected_sessions"], 1)
        self.assertEqual(attested["selection_sha256"], selection["sha256"])
        session_file.chmod(0o644)
        with self.assertRaises(source.SourceError):
            source.verify_hydration_cache(self.root / "private-hydration", Path(selection["path"]),
                                          source.file_digest(self.root / "private-hydration" / "HYDRATED.json"))

    def test_hydration_rejects_changed_success_without_receipt(self):
        selection = self._file("hydration-selection.jsonl", [self.selection], jsonl=True)
        request = seal({"schema": "fleet_teacher_source_hydration_v1", "selection": selection,
                        "output": str(self.root / "private-hydration")})
        self.envelope["verifier_execution"] = {**self.verifier, "score": 0}
        with self.assertRaises(source.SourceError):
            source.hydrate(request, get=self.get)
        self.assertFalse((self.root / "private-hydration" / "HYDRATED.json").exists())

    def test_unproven_tool_result_order_is_quarantined(self):
        self.messages[3], self.messages[2] = self.messages[2], self.messages[3]
        self.selection["trace_sha256"] = source.digest(self.envelope)
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 0)
        self.assertEqual(receipt["excluded_sessions"],
                         {"tool_argument_or_result_contract_mismatch": 1})

    def test_non_text_tool_result_is_not_laundered_into_target(self):
        self.messages[3]["content"] = {"stdout": "ok"}
        self.selection["trace_sha256"] = source.digest(self.envelope)
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 0)
        self.assertEqual(receipt["excluded_sessions"],
                         {"tool_argument_or_result_contract_mismatch": 1})

    def test_source_use_tool_wrapper_is_not_assumed_opencode_text(self):
        self.messages[2]["tool_calls"][0]["function"] = {
            "name": "use_tool", "arguments": {"tool_name": "fleet_environment__bash",
                                             "tool_input": {"script": "true"}}}
        self.messages[3]["content"] = json.dumps({"OkayOutput": "ok"})
        self.selection["trace_sha256"] = source.digest(self.envelope)
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 0)
        self.assertEqual(receipt["excluded_sessions"],
                         {"tool_argument_or_result_contract_mismatch": 1})

    def test_exact_target_tool_names_are_accepted_without_alias_guess(self):
        self.messages[2]["tool_calls"][0]["function"]["name"] = "fleet_bash"
        self.messages[4]["tool_calls"][0]["function"]["name"] = "fleet_submit_report"
        self.selection["trace_sha256"] = source.digest(self.envelope)
        with (patch.object(source, "TOOL_DIGEST", source.digest(self.tools, ascii=True)),
              patch.object(source, "TARGET_SYSTEM_DIGEST", source.text_digest("Target system.")),
              patch.object(source, "MIN_ANCHOR_PROBES", 1)):
            receipt = source.fetch(self.request(), get=self.get)
        self.assertEqual(receipt["retained_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
