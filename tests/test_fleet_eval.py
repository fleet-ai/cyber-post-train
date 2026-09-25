"""Synthetic protocol and outcome tests; no Fleet access or private task content."""

import json
import tempfile
import unittest
from pathlib import Path

from evals.fleet import claim_checkpoint_eval, dev_decision, seal_protocol, summarize, validate_protocol


def sha(char):
    return "sha256:" + char * 64


def protocol(count=2, role="dev"):
    return seal_protocol({
        "schema": "fleet_paired_pass4_v1",
        "study_id": "qwen-synthetic-comparison",
        "role": role,
        "final_selection_sha256": sha("f") if role == "final" else None,
        "tasks": [{
            "task_key": f"task-{i}", "task_version_id": f"version-{i}",
            "application": "app", "family_id": f"family-{i}",
            "environment_version_id": f"environment-{i}", "data_version": f"data-{i}",
            "verifier_sha256": sha("e"),
        } for i in range(count)],
        "common": {
            "model_repository": "Qwen/example", "tokenizer_sha256": sha("a"),
            "chat_template_sha256": sha("b"), "serving_image": "serve@" + sha("c"),
            "serving_config_sha256": sha("d"), "harness_image": "agent@" + sha("a"),
            "harness_version": "pinned", "system_prompt_sha256": sha("b"),
            "tools": ["fleet_bash", "fleet_submit_report"], "tool_schema_sha256": sha("c"),
            "context_policy": "native_compaction", "context_window_tokens": 262144,
            "max_output_tokens": 32768, "max_model_requests": 600,
            "timeout_seconds": 28800, "temperature": 0.6, "top_p": 0.95,
            "seeds": [11, 12, 13, 14], "retry_limit": 0,
        },
        "arms": {
            "base": {"model_revision": "base-revision", "weights_sha256": sha("1")},
            "candidate": {"model_revision": "trained-revision", "weights_sha256": sha("2"),
                          "checkpoint_sha256": sha("3")},
        },
    })


def events(plan, *, candidate_wins=()):
    rows = []
    for task in plan["tasks"]:
        version = task["task_version_id"]
        for arm in ("base", "candidate"):
            artifact = plan["arms"][arm]
            for attempt, seed in enumerate(plan["common"]["seeds"], 1):
                rows.append({
                    "protocol_sha256": plan["sha256"], "arm": arm,
                    "task_version_id": version, "attempt": attempt, "seed": seed,
                    "model_revision": artifact["model_revision"],
                    "weights_sha256": artifact["weights_sha256"],
                    "checkpoint_sha256": artifact.get("checkpoint_sha256"),
                    "process_exit_code": 0, "termination": "completed",
                    "verifier_sha256": task["verifier_sha256"],
                    "verifier_status": "completed",
                    "success": arm == "candidate" and version in candidate_wins and attempt == 1,
                })
    return rows


class FleetEvalTests(unittest.TestCase):
    def test_frozen_parity_and_checkpoint_identity(self):
        plan = protocol()
        validate_protocol(plan)
        changed = json.loads(json.dumps(plan))
        changed["common"]["tools"].reverse()
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_protocol(changed)
        changed = {k: v for k, v in plan.items() if k != "sha256"}
        changed["arms"]["base"]["tools"] = ["different"]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            seal_protocol(changed)

    def test_one_exact_version_per_family_and_final_receipt(self):
        raw = {k: v for k, v in protocol().items() if k != "sha256"}
        raw["tasks"][1]["family_id"] = raw["tasks"][0]["family_id"]
        with self.assertRaisesRegex(ValueError, "one exact task version"):
            seal_protocol(raw)
        raw = {k: v for k, v in protocol(role="final").items() if k != "sha256"}
        raw["final_selection_sha256"] = None
        with self.assertRaisesRegex(ValueError, "selection receipt"):
            seal_protocol(raw)
        raw = {k: v for k, v in protocol().items() if k != "sha256"}
        raw["final_selection_sha256"] = sha("f")
        with self.assertRaisesRegex(ValueError, "selection receipt"):
            seal_protocol(raw)

    def test_images_must_be_digest_pinned(self):
        raw = {k: v for k, v in protocol().items() if k != "sha256"}
        raw["common"]["serving_image"] = "serve:latest"
        with self.assertRaisesRegex(ValueError, "pinned by digest"):
            seal_protocol(raw)

    def test_valid_zero_and_paired_pass4(self):
        plan = protocol()
        result = summarize(plan, events(plan, candidate_wins={"version-0"}))
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["base_pass4"], result["candidate_pass4"]), (0, 1))
        self.assertEqual(result["candidate_minus_base_pass4"], 0.5)
        self.assertEqual(dev_decision(result), (0.5, 0.5))

    def test_invalid_output_or_process_error_never_becomes_zero(self):
        plan = protocol()
        for field, value in (("termination", "output_limit"), ("process_exit_code", 1)):
            rows = events(plan)
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, "cannot carry a capability result"):
                summarize(plan, rows)
            rows[0]["success"] = None
            result = summarize(plan, rows)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["candidate_minus_base_pass4"])
            self.assertEqual(result["infrastructure_invalid_attempts"], 1)
            self.assertEqual(result["invalid_cells"][0]["cell"], ["base", "version-0", 1])

    def test_missing_duplicate_seed_and_checkpoint_mismatch(self):
        plan = protocol()
        rows = events(plan)
        missing = summarize(plan, rows[:-1])
        self.assertEqual(missing["missing_attempts"], 1)
        self.assertEqual(missing["missing_cells"], [["candidate", "version-1", 4]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(plan, rows + [rows[0]])
        changed = json.loads(json.dumps(rows))
        changed[0]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            summarize(plan, changed)
        changed = json.loads(json.dumps(rows))
        changed[-1]["checkpoint_sha256"] = sha("9")
        with self.assertRaisesRegex(ValueError, "checkpoint binding mismatch"):
            summarize(plan, changed)
        changed = json.loads(json.dumps(rows))
        changed[-1]["model_revision"] = "unverified-route"
        with self.assertRaisesRegex(ValueError, "checkpoint binding mismatch"):
            summarize(plan, changed)

    def test_final_results_cannot_select_checkpoint(self):
        plan = protocol(role="final")
        result = summarize(plan, events(plan))
        self.assertEqual(result["role"], "final")
        with self.assertRaisesRegex(ValueError, "development"):
            dev_decision(result)

    def test_exact_discordance_interval(self):
        plan = protocol(count=6)
        result = summarize(plan, events(plan, candidate_wins={f"version-{i}" for i in range(6)}))
        self.assertAlmostEqual(result["one_sided_exact_discordance_p"], 1 / 64)
        low, high = result["conditional_95pct_interval"]
        self.assertLess(low, result["candidate_minus_base_pass4"])
        self.assertAlmostEqual(high, 1.0)

    def test_atomic_checkpoint_claim_only_not_execution(self):
        plan = protocol()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = claim_checkpoint_eval(plan, root, "fleet_pass4")
            self.assertIsNotNone(claim)
            self.assertEqual(claim_checkpoint_eval(plan, root, "fleet_pass4"), None)
            self.assertEqual(json.loads(claim.read_text())["checkpoint_sha256"], sha("3"))
            self.assertEqual(list(root.iterdir()), [claim])
            with self.assertRaisesRegex(ValueError, "only protocol-bound"):
                claim_checkpoint_eval(plan, root, "teacher_ce")


if __name__ == "__main__":
    unittest.main()
