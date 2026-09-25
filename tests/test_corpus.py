"""Synthetic contract tests; no Fleet task text or private sessions."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.corpus import build_corpus, materialize_pair, sha256


TOOLS = [
    {"type": "function", "function": {"name": name, "description": name,
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    for name in ("fleet_bash", "fleet_submit_report")
]
ALIASES = {"bash": "fleet_bash", "submit_report": "fleet_submit_report",
           "fleet_bash": "fleet_bash", "fleet_submit_report": "fleet_submit_report"}


def record(session="s1", version="v1"):
    return {"session_id": session, "task_version_id": version, "messages": [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "b1", "function": {"name": "bash", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "b1", "content": "synthetic observation"},
        {"role": "assistant", "content": "synthetic next action"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "r1", "function": {"name": "submit_report", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "r1", "content": "synthetic success"},
        {"role": "assistant", "content": "post-success text must be cut"},
    ]}


def measure(messages, tools, target_index):
    assert tools == TOOLS and messages[target_index]["role"] == "assistant"
    return 10 * len(messages) + len(tools), 10


def run(records=None, *, roster=None, evidence=None, tools=None, aliases=None, **overrides):
    records = records if records is not None else [record()]
    roster = roster if roster is not None else {"v1": {"family_id": "family-a", "split": "train"}}
    evidence = evidence if evidence is not None else {
        row["session_id"]: {"source_sha256": sha256(row), "report_call_id": "r1",
                            "verified_success": True} for row in records}
    tools = tools if tools is not None else TOOLS
    aliases = aliases if aliases is not None else ALIASES
    args = {"evidence": evidence, "roster": roster, "tools": tools, "aliases": aliases,
            "expected_sha256": {"records": sha256(sorted(records, key=lambda r: r["session_id"])),
                                "evidence": sha256(evidence), "roster": sha256(roster),
                                "tools": sha256(tools), "aliases": sha256(aliases)},
            "count_tokens": measure, "tokenizer_sha256": "sha256:" + "a" * 64,
            "max_length": 200, "context_tokens": 100, "max_family_tokens": 100,
            "max_session_tokens": 100}
    args.update(overrides)
    return build_corpus(records, **args)


class CorpusTests(unittest.TestCase):
    def test_complete_rounds_exact_tools_and_once_only_targets(self):
        source = record()
        before = copy.deepcopy(source)
        rows, receipt = run([source])
        self.assertEqual(source, before)
        self.assertEqual(len(rows), 3)
        self.assertEqual(receipt["supervised_tokens"], 30)
        self.assertEqual(len({r["target_id"] for r in rows}), 3)
        self.assertEqual(receipt["sha256"], sha256({k: v for k, v in receipt.items() if k != "sha256"}))
        for row in rows:
            self.assertEqual(row["messages"][:2], source["messages"][:2])
            self.assertEqual(row["tools"], TOOLS)
            self.assertEqual(row["messages"][row["target_message_index"]]["role"], "assistant")
            self.assertNotIn("post-success", str(row["messages"]))
        self.assertEqual([m["role"] for m in rows[-1]["messages"]],
                         ["system", "user", "assistant", "tool", "assistant", "assistant"])
        self.assertEqual(rows[-1]["messages"][2]["tool_calls"][0]["function"]["name"],
                         "fleet_bash")
        self.assertEqual(rows[-1]["messages"][-1]["tool_calls"][0]["function"]["name"],
                         "fleet_submit_report")

    def test_no_raw_clipping_even_at_tight_context_budget(self):
        rows, _ = run(context_tokens=0)
        self.assertTrue(all(len(row["messages"]) == 3 for row in rows))
        with self.assertRaisesRegex(ValueError, "complete anchor"):
            run(max_length=25, context_tokens=0)

    def test_missing_or_mismatched_tool_result_rejected(self):
        for changed in (lambda m: m.pop(3),
                        lambda m: m[3].update(tool_call_id="wrong")):
            row = record()
            changed(row["messages"])
            with self.assertRaisesRegex(ValueError, "result"):
                run([row])
        row = record()
        row["messages"].pop(6)
        with self.assertRaisesRegex(ValueError, "result"):
            run([row])

    def test_multiple_tool_results_may_arrive_in_either_order(self):
        row = record()
        row["messages"][2]["tool_calls"].append(
            {"id": "b2", "function": {"name": "bash", "arguments": {}}})
        row["messages"].insert(4, {"role": "tool", "tool_call_id": "b2",
                                   "content": "second synthetic observation"})
        row["messages"][3], row["messages"][4] = row["messages"][4], row["messages"][3]
        rows, _ = run([row])
        self.assertEqual({m["tool_call_id"] for m in rows[-1]["messages"][3:5]},
                         {"b1", "b2"})

    def test_evidence_tool_contract_and_family_roster_fail_closed(self):
        row = record()
        bad_proof = {"s1": {"source_sha256": "sha256:" + "0" * 64,
                            "report_call_id": "r1", "verified_success": True}}
        with self.assertRaisesRegex(ValueError, "success evidence"):
            run([row], evidence=bad_proof)
        with self.assertRaisesRegex(ValueError, "input digest"):
            run(expected_sha256={name: "sha256:" + "0" * 64 for name in
                                    ("records", "evidence", "roster", "tools", "aliases")})
        bad_tools = copy.deepcopy(TOOLS)
        bad_tools[0]["function"]["name"] = "bash"
        with self.assertRaisesRegex(ValueError, "model-facing"):
            run(tools=bad_tools, count_tokens=lambda *_: (30, 10))
        bad_aliases = dict(ALIASES, bash="unknown")
        with self.assertRaisesRegex(ValueError, "alias"):
            run(aliases=bad_aliases)
        conflicting = {"v1": {"family_id": "same", "split": "train"},
                       "v2": {"family_id": "same", "split": "test"}}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            run(roster=conflicting)

    def test_heldout_versions_and_whole_session_caps(self):
        first, second = record("s1", "v1"), record("s2", "v2")
        roster = {"v1": {"family_id": "family-a", "split": "train"},
                  "v2": {"family_id": "family-b", "split": "dev"}}
        rows, receipt = run([first, second], roster=roster)
        self.assertEqual(len(rows), 3)
        self.assertEqual(receipt["excluded_sessions"]["other_split"], 1)
        self.assertNotIn("family-b", {r["family_id"] for r in rows})
        roster["v2"] = {"family_id": "family-a", "split": "train"}
        rows, receipt = run([second, first], roster=roster, max_family_tokens=30)
        self.assertEqual(receipt["selected_sessions"], 1)
        self.assertEqual(receipt["excluded_sessions"]["family_cap"], 1)
        self.assertEqual({r["task_version_id"] for r in rows}, {"v1"})
        self.assertEqual(receipt, run([first, second], roster=roster,
                                      max_family_tokens=30)[1])
        rows, receipt = run([first], max_session_tokens=20)
        self.assertEqual(rows, [])
        self.assertEqual(receipt["excluded_sessions"]["session_cap"], 1)

    def test_create_once_pair_has_disjoint_roles_and_sanitized_receipt(self):
        train, dev = record("s1", "v1"), record("s2", "v2")
        roster = {"v1": {"family_id": "family-a", "split": "train"},
                  "v2": {"family_id": "family-b", "split": "dev"}}
        records = [train, dev]
        evidence = {row["session_id"]: {"source_sha256": sha256(row),
                    "report_call_id": "r1", "verified_success": True} for row in records}
        expected = {"records": sha256(sorted(records, key=lambda r: r["session_id"])),
                    "evidence": sha256(evidence), "roster": sha256(roster),
                    "tools": sha256(TOOLS), "aliases": sha256(ALIASES)}
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "corpus"
            args = {"evidence": evidence, "roster": roster, "tools": TOOLS,
                    "aliases": ALIASES, "expected_sha256": expected,
                    "count_tokens": measure, "tokenizer_sha256": "sha256:" + "a" * 64,
                    "max_length": 200, "context_tokens": 100,
                    "max_family_tokens": 100, "max_session_tokens": 100}
            receipt = materialize_pair(records, destination=dest, **args)
            self.assertFalse(receipt["trainer_ready"])
            self.assertEqual(receipt["sha256"], sha256({k: v for k, v in receipt.items()
                                                         if k != "sha256"}))
            for split, version in (("train", "v1"), ("dev", "v2")):
                payload = (dest / f"{split}.jsonl").read_bytes()
                rows = [json.loads(line) for line in payload.splitlines()]
                self.assertEqual({row["task_version_id"] for row in rows}, {version})
                self.assertEqual(receipt["partitions"][split]["receipt"]["rows"], 3)
                self.assertEqual(receipt["partitions"][split]["sha256"],
                                 "sha256:" + hashlib.sha256(payload).hexdigest())
            self.assertNotIn("synthetic task", json.dumps(receipt))
            with self.assertRaises(FileExistsError):
                materialize_pair(records, destination=dest, **args)


if __name__ == "__main__":
    unittest.main()
