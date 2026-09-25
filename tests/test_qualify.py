"""Metadata-only qualification must not turn old successes into live task proof."""

import hashlib
import unittest

from training.qualify import components, report


def digest(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def task(key, version, status="not_analyzed", *, live=True):
    return {"task_key": key, "task_version_id": version, "task_shape": "blackbox",
            "qa_status": status, "lifecycle_status": "production" if live else "retired",
            "verifier_attached": live}


def lineage(key, version, *atoms):
    return {"task_key": key, "task_version_id": version,
            "atom_artifact_keys": [f"cyber/atoms/app/{name}" for name in atoms]}


def session(key, version):
    row = {"task_key": key, "task_version_id": version, "session_sha256": digest("session"),
           "receipt_sha256": digest("receipt"), "trace_sha256": digest("trace")}
    row.update({field: True for field in ("sft_eligible", "infrastructure_valid",
                                          "authoritative_success", "finite_outcome",
                                          "verifier_completed", "trace_intact",
                                          "retained_successful_report", "tool_contract_proven")})
    return row


def runtime(key, version):
    row = {"task_key": key, "task_version_id": version, "receipt_sha256": digest("runtime")}
    row.update({field: True for field in ("environment_started", "tools_reachable",
                                          "verifier_executed", "finite_outcome",
                                          "cleanup_complete")})
    return row


class QualificationTests(unittest.TestCase):
    def test_historical_success_survives_current_breakage_but_is_not_live(self):
        key = "example_blackbox_a"
        result = report({"tasks": [task(key, "v1", "broken_task", live=False)]},
                        [{"task_versions": [lineage(key, "v1", "a")]}],
                        {"tasks": [{"task_key": key, "task_version_id": "v1", "split": "train"}]},
                        sessions={"sessions": [session(key, "v1")]},
                        runtime={"task_versions": [runtime(key, "v1")]})
        self.assertEqual(result["counts"]["historical_success_receipt_complete_sessions"], 1)
        self.assertEqual(result["counts"]["historical_sft_ready_train_sessions"], 1)
        self.assertEqual(result["counts"]["current_runtime_receipt_complete_versions"], 0)
        self.assertEqual(result["next_qualification"]["known_broken"], 1)

    def test_success_requires_exact_receipt_retained_report_and_train_role(self):
        key = "example_blackbox_a"
        good = session(key, "old")
        bad = dict(good, session_sha256=digest("second"), receipt_sha256=digest("second-receipt"),
                   trace_sha256=digest("second-trace"), verifier_completed=False)
        missing_report = dict(good, session_sha256=digest("third"),
                              receipt_sha256=digest("third-receipt"),
                              trace_sha256=digest("third-trace"), retained_successful_report=False)
        result = report({"tasks": []}, [{"task_versions": [lineage(key, "old", "a")]}],
                        {"tasks": [{"task_key": key, "task_version_id": "old", "split": "dev"}]},
                        sessions={"sessions": [good, bad, missing_report]})
        self.assertEqual(result["counts"]["historical_success_receipt_complete_sessions"], 2)
        self.assertEqual(result["counts"]["historical_sft_ready_train_sessions"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate opaque session"):
            report({"tasks": []}, [{"task_versions": [lineage(key, "old", "a")]}],
                   sessions={"sessions": [good, good]})
        teacher_val = report({"tasks": []}, [{"task_versions": [lineage(key, "old", "a")]}],
                             {"tasks": [{"task_key": key, "task_version_id": "old",
                                         "split": "teacher_validation"}]},
                             sessions={"sessions": [good]})
        self.assertEqual(teacher_val["counts"]["historical_success_receipt_complete_sessions"], 1)
        self.assertEqual(teacher_val["counts"]["historical_sft_ready_train_sessions"], 0)

    def test_shared_atom_alias_exposes_heldout_across_task_keys(self):
        held = "held_blackbox_task"
        source = "source_blackbox_alias"
        lineages = {"task_versions": [lineage(held, "v1", "same"),
                                      lineage(source, "v2", "same")]}
        teacher = {"training_task_keys": [{"task_key": source,
                   "versions": [{"task_version_id": "v2",
                                 "atom_lineages": ["cyber/atoms/app/same@3:atom_source"]}]}]}
        result = report({"tasks": [task(held, "v1")]}, [lineages],
                        {"tasks": [{"task_key": held, "task_version_id": "v1",
                                    "split": "final_test"}]}, teacher,
                        runtime={"task_versions": [runtime(held, "v1")]})
        self.assertEqual(result["counts"]["current_runtime_receipt_complete_versions"], 1)
        self.assertEqual(result["counts"]["teacher_exposed_split_heldout_versions"], 1)
        self.assertEqual(result["counts"]["lineage_clean_live_heldout_versions"], 0)

    def test_transitive_composites_and_key_versions_cannot_cross_roles(self):
        rows = {"task_versions": [lineage("a_blackbox", "v1", "a"),
                                 lineage("a_blackbox", "v2", "b"),
                                 lineage("c_blackbox", "v1", "b", "c"),
                                 lineage("d_blackbox", "v1", "c")]}
        labels, _ = components(rows)
        self.assertEqual(len(set(labels.values())), 1)
        result = report({"tasks": [task("a_blackbox", "v1"), task("d_blackbox", "v1")]},
                        [rows], {"tasks": [
                            {"task_key": "a_blackbox", "task_version_id": "v1", "split": "train"},
                            {"task_key": "d_blackbox", "task_version_id": "v1", "split": "dev"}]},
                        runtime={"task_versions": [runtime("d_blackbox", "v1")]})
        self.assertEqual(result["counts"]["split_component_role_conflicts"], 1)
        self.assertFalse(result["split_safe"])

    def test_unreviewed_roster_and_missing_runtime_are_not_proof(self):
        rows = [task("unknown_blackbox", "v1"), task("known_blackbox", "v2", "clean")]
        result = report({"tasks": rows}, [{"task_versions": [lineage("known_blackbox", "v2", "a")]}],
                        include_rosters=True)
        self.assertEqual(result["counts"]["unreviewed_not_analyzed_versions"], 1)
        self.assertEqual(result["counts"]["not_analyzed_without_prior_exact_receipt"], 1)
        self.assertEqual(result["counts"]["current_runtime_receipt_complete_versions"], 0)
        self.assertEqual(result["next_qualification"]["review_atom_lineage"], 1)
        self.assertEqual(result["next_qualification"]["run_model_free_runtime_qualification"], 2)
        self.assertEqual(result["rosters"]["qualify_unreviewed_pool"],
                         [{"task_key": "unknown_blackbox", "task_version_id": "v1"}])
        rows[0]["prompt"] = "private sentinel not for output"
        with self.assertRaisesRegex(ValueError, "private task or trace"):
            report({"tasks": rows}, [{"task_versions": [lineage("known_blackbox", "v2", "a")]}])
        with self.assertRaisesRegex(ValueError, "declared task count"):
            report({"task_count": 3, "tasks": rows[1:]},
                   [{"task_versions": [lineage("known_blackbox", "v2", "a")]}])

    def test_prior_receipt_is_historical_not_current_runtime(self):
        key = "example_blackbox_a"
        source = lineage(key, "v1", "a")
        source["provenance"] = {"certification": {"status": "accepted",
                                                   "receipt_sha256": digest("prior")}}
        result = report({"tasks": [task(key, "v1")]}, [{"task_versions": [source]}])
        self.assertEqual(result["counts"]["prior_exact_receipt_current_versions"], 1)
        self.assertEqual(result["counts"]["not_analyzed_without_prior_exact_receipt"], 0)
        self.assertEqual(result["counts"]["current_runtime_receipt_complete_versions"], 0)


if __name__ == "__main__":
    unittest.main()
