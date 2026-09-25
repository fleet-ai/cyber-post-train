"""Synthetic protocol and outcome tests; no Fleet access or private task content."""

import json
import unittest

from evals.fleet import dev_decision, seal_protocol, summarize, validate_protocol


def sha(char):
    return "sha256:" + char * 64


def protocol(count=2, role="dev", seed_mode="fixed"):
    return seal_protocol({
        "schema": "fleet_paired_pass4_v2", "study_id": "qwen-synthetic-comparison", "role": role,
        "final_selection_sha256": sha("f") if role == "final" else None,
        "tasks": [{
            "task_key": f"task-{i}", "task_version_id": f"version-{i}",
            "application": "app", "family_id": f"family-{i}", "verifier_sha256": sha("e"),
            "environment_version_id": f"environment-{i}", "data_version": f"data-{i}",
        } for i in range(count)],
        "common": {
            "model_repository": "Qwen/example", "tokenizer_sha256": sha("a"),
            "chat_template_sha256": sha("b"), "serving_image": "serve@" + sha("c"),
            "serving_config_sha256": sha("d"), "harness_image": "agent@" + sha("a"),
            "harness_version": "pinned", "system_prompt_sha256": sha("b"),
            "tools": ["fleet_bash", "fleet_submit_report"], "tool_schema_sha256": sha("c"),
            "context_policy": "native_compaction", "context_window_tokens": 262144,
            "max_output_tokens": 32768, "max_steps": 600, "max_duration_minutes": 480,
            "temperature": 0.6 if seed_mode == "fixed" else None,
            "top_p": 0.95 if seed_mode == "fixed" else None,
            "seed_policy": ({"mode": "fixed", "seeds": [11, 12, 13, 14]}
                            if seed_mode == "fixed" else {"mode": seed_mode}),
            "retry_limit": 0, "scoring_mode": "partial", "pass_criterion": "cyber_ctf_full_solve_v1",
        },
        "arms": {
            "base": {"model_revision": "base-revision", "weights_sha256": sha("1")},
            "candidate": {"model_revision": "trained-revision", "weights_sha256": sha("2"), "checkpoint_sha256": sha("3")},
        },
    })


def events(plan, *, candidate_wins=()):
    policy = plan["common"]["seed_policy"]
    return [{
        "protocol_sha256": plan["sha256"], "arm": arm, "task_version_id": task["task_version_id"],
        "attempt": attempt, "seed": policy["seeds"][attempt - 1] if policy["mode"] == "fixed" else None,
        "model_revision": plan["arms"][arm]["model_revision"],
        "weights_sha256": plan["arms"][arm]["weights_sha256"],
        "checkpoint_sha256": plan["arms"][arm].get("checkpoint_sha256"),
        "process_exit_code": 0, "termination": "completed", "budget_evidence_sha256": None,
        "verifier_sha256": task["verifier_sha256"], "verifier_status": "completed",
        "verifier_execution_id": f"execution-{task['task_version_id']}-{arm}-{attempt}",
        "verifier_result_schema": "cyber_verification_result_v3", "verifier_result_sha256": sha("8"),
        "verifier_result_task_version_id": task["task_version_id"],
        "ctf_score": float(arm == "candidate" and task["task_version_id"] in candidate_wins and attempt == 1),
        "success": arm == "candidate" and task["task_version_id"] in candidate_wins and attempt == 1,
    } for task in plan["tasks"] for arm in ("base", "candidate") for attempt in range(1, 5)]


class FleetEvalTests(unittest.TestCase):
    def assert_protocol_rejects(self, path, value, message, **options):
        raw = protocol(**options)
        del raw["sha256"]
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with self.assertRaisesRegex(ValueError, message):
            seal_protocol(raw)

    def test_frozen_parity_and_checkpoint_identity(self):
        plan = protocol()
        validate_protocol(plan)
        changed = json.loads(json.dumps(plan))
        changed["common"]["tools"].reverse()
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_protocol(changed)

    def test_protocol_rejects_leakage_and_unmatched_treatments(self):
        cases = (
            (("arms", "base", "tools"), ["different"], "unknown fields", {}),
            (("tasks", 1, "family_id"), "family-0", "one exact task version", {}),
            (("final_selection_sha256",), None, "selection receipt", {"role": "final"}),
            (("final_selection_sha256",), sha("f"), "selection receipt", {}),
            (("final_selection_sha256",), "unsealed", "selection receipt", {}),
            (("common", "serving_image"), "serve:latest", "pinned by digest", {}),
            (("common", "seed_policy", "seeds"), [11, 11, 12, 13], "distinct fixed seeds", {}),
            (("common", "seed_policy", "seeds"), [11, 12, 13, 14], "unknown fields", {"seed_mode": "server_assigned_unobserved"}),
            (("common", "temperature"), 0.6, "both be explicit or both be unavailable", {"seed_mode": "server_assigned_unobserved"}),
            (("common", "pass_criterion"), "reward_positive", "full-CTF scoring contract", {}),
            (("common", "scoring_mode"), "binary", "full-CTF scoring contract", {}),
        )
        for path, value, message, options in cases:
            with self.subTest(path=path):
                self.assert_protocol_rejects(path, value, message, **options)

    def test_valid_zero_and_paired_pass4(self):
        plan = protocol()
        result = summarize(plan, events(plan, candidate_wins={"version-0"}))
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["base_pass4"], result["candidate_pass4"]), (0, 1))
        self.assertEqual(result["candidate_minus_base_pass4"], 0.5)
        self.assertEqual(dev_decision(result), (0.5, 0.5))
        self.assertEqual(result["seed_policy"], "fixed")
        self.assertIn("matching fixed seeds", result["pairing_interpretation"])

    def test_native_unseeded_pass4_is_task_paired_not_seed_paired(self):
        plan = protocol(seed_mode="server_assigned_unobserved")
        rows = events(plan, candidate_wins={"version-0"})
        self.assertTrue(all(row["seed"] is None for row in rows))
        result = summarize(plan, rows)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidate_minus_base_pass4"], 0.5)
        self.assertEqual(result["seed_policy"], "server_assigned_unobserved")
        self.assertIn("not seed-paired", result["pairing_interpretation"])
        self.assertIn("not asserted", result["sampling_interpretation"])
        rows[0]["seed"] = 11
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            summarize(plan, rows)

    def test_partial_reward_is_not_a_full_ctf_win(self):
        plan = protocol()
        rows = events(plan)
        rows[0]["ctf_score"] = 0.5
        rows[0]["success"] = True  # weighted reward > 0 is not a full CTF solve
        with self.assertRaisesRegex(ValueError, "full-CTF score"):
            summarize(plan, rows)
        rows[0]["success"] = False
        self.assertEqual(summarize(plan, rows)["status"], "complete")

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

    def test_planned_budget_is_valid_only_with_authoritative_receipt_and_verifier(self):
        plan = protocol()
        for termination in ("planned_max_steps", "planned_wall_deadline"):
            rows = events(plan)
            rows[0]["termination"] = termination
            rows[0]["budget_evidence_sha256"] = sha("9")
            result = summarize(plan, rows)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["infrastructure_invalid_attempts"], 0)

            rows[0]["budget_evidence_sha256"] = None
            with self.assertRaisesRegex(ValueError, "cannot carry a capability result"):
                summarize(plan, rows)
            rows[0]["success"] = None
            self.assertEqual(summarize(plan, rows)["invalid_reasons"],
                             {"unproven_budget_exhaustion": 1})

            rows[0]["budget_evidence_sha256"] = sha("9")
            rows[0]["verifier_status"] = "missing"
            self.assertEqual(summarize(plan, rows)["invalid_reasons"],
                             {"verifier_incomplete": 1})

            rows[0]["verifier_status"] = "completed"
            rows[0]["process_exit_code"] = 1
            self.assertEqual(summarize(plan, rows)["invalid_reasons"],
                             {"process_error": 1})

    def test_provisional_complete_family_subset_is_not_a_final_claim(self):
        plan = protocol(count=3)
        rows = events(plan, candidate_wins={"version-0", "version-2"})
        rows = [row for row in rows if row["task_version_id"] != "version-1"]
        rows[-1]["termination"] = "output_limit"
        rows[-1]["success"] = None
        result = summarize(plan, rows)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["complete_families"], 1)
        self.assertEqual(result["provisional_family_results"][0]["task_version_id"], "version-0")
        self.assertEqual(result["provisional_candidate_minus_base_pass4"], 1.0)
        self.assertIsNone(result["candidate_minus_base_pass4"])
        self.assertIsNone(result["conditional_95pct_interval"])
        with self.assertRaisesRegex(ValueError, "complete development"):
            dev_decision(result)

    def test_missing_duplicate_seed_and_checkpoint_mismatch(self):
        plan = protocol()
        rows = events(plan)
        missing = summarize(plan, rows[:-1])
        self.assertEqual(missing["missing_attempts"], 1)
        self.assertEqual(missing["missing_cells"], [["candidate", "version-1", 4]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(plan, rows + [rows[0]])
        for index, field, value, error in (
            (0, "seed", 999, "seed mismatch"),
            (-1, "checkpoint_sha256", sha("9"), "checkpoint binding mismatch"),
            (-1, "model_revision", "unverified-route", "checkpoint binding mismatch"),
        ):
            changed = events(plan)
            changed[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
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


if __name__ == "__main__":
    unittest.main()
