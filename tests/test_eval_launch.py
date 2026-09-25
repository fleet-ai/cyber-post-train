"""Native Fleet pass@4 launch-boundary tests; no paid sessions are created."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evals.fleet import seal_protocol
from evals.launch import (
    LaunchError, _identity, _readiness_expected, digest, job_payload,
    launch_once, preview, validate_plan,
)


def sha(char):
    return "sha256:" + char * 64


def fixture():
    task = {
        "task_key": "synthetic-blackbox-task", "task_version_id": "version-1",
        "application": "app", "family_id": "family-dev",
        "environment_version_id": "env-version-1", "data_version": "data-1",
        "verifier_sha256": sha("e"),
    }
    protocol = seal_protocol({
        "schema": "fleet_paired_pass4_v2", "study_id": "synthetic",
        "role": "dev", "final_selection_sha256": None, "tasks": [task],
        "common": {
            "model_repository": "Qwen/synthetic", "tokenizer_sha256": sha("a"),
            "chat_template_sha256": sha("b"), "serving_image": "serve@" + sha("c"),
            "serving_config_sha256": sha("d"), "harness_image": "runner@" + sha("f"),
            "harness_version": "pinned-opencode", "system_prompt_sha256": sha("a"),
            "tools": ["fleet_bash", "fleet_submit_report"],
            "tool_schema_sha256": sha("b"), "context_policy": "native_compaction",
            "context_window_tokens": 262144, "max_output_tokens": 32768,
            "max_steps": 600, "max_duration_minutes": 480,
            "temperature": None, "top_p": None,
            "seed_policy": {"mode": "server_assigned_unobserved"},
            "retry_limit": 0,
        },
        "arms": {
            "base": {"model_revision": "base-rev", "weights_sha256": sha("1")},
            "candidate": {"model_revision": "trained-rev", "weights_sha256": sha("2"),
                          "checkpoint_sha256": sha("3")},
        },
    })
    live = {
        "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
        "key": task["task_key"], "eval_task_version_id": task["task_version_id"],
        "environment_version_id": task["environment_version_id"],
        "data_version": task["data_version"],
        "verifier": {"sha256": "e" * 64, "verifier_version_id": "verifier-v1"},
        "metadata": {"projection_id": "blackbox_ctf_v1",
                     "runtime_seed_manifest": {"content_sha256": "f" * 64}},
        "seed_config": {"non_private_test_seed": True},
        "task_lifecycle_status": "production",
    }
    plan = {
        "schema": "fleet_native_paired_launch_v1", "protocol": protocol,
        "task_group_id": "11111111-1111-4111-8111-111111111111",
        "family_roles": {"train": [["app", "family-train"]],
                         "dev": [["app", "family-dev"]],
                         "final": [["app", "family-final"]]},
        "task_response_sha256": {task["task_version_id"]: digest(live)},
        "task_qualification_sha256": {task["task_version_id"]: sha("9")},
        "readiness_sha256": {
            "candidate_export": sha("1"), "candidate_reload": sha("2"),
            "base_registration": sha("3"), "candidate_registration": sha("4"),
            "live_parity": sha("5"),
        },
        "routes": {
            "base": "fleet-qwen/baseline-synthetic",
            "candidate": "fleet-qwen/checkpoint-synthetic",
        },
    }
    group = {
        "id": plan["task_group_id"],
        "team_id": live["team_id"],
        "members": [{"eval_task_version_id": task["task_version_id"]}],
    }
    return plan, live, group


def gates(plan, live, group):
    return {
        "account_get": lambda: {"team_id": live["team_id"], "team_name": "fleet"},
        "group_get": lambda group_id: copy.deepcopy(group),
        "task_get": lambda key, version: copy.deepcopy(live),
        "qualification_get": lambda: copy.deepcopy(plan["task_qualification_sha256"]),
        "readiness_get": lambda: copy.deepcopy(_readiness_expected(plan)),
        "budget_check": lambda sessions: sessions <= 500,
    }


class LaunchTests(unittest.TestCase):
    def test_exact_native_payload_has_no_invented_sampling_or_local_bash(self):
        plan, live, group = fixture()
        self.assertTrue(validate_plan(plan).startswith("sha256:"))
        packet = preview(plan, "base", **gates(plan, live, group))
        payload = packet["payload"]
        self.assertEqual(set(payload), {
            "name", "models", "pass_k", "task_group_id", "agent_runtime",
            "harness", "mode", "tools", "max_steps", "max_duration_minutes",
        })
        self.assertEqual(payload["models"], [plan["routes"]["base"]])
        self.assertEqual(payload["pass_k"], 4)
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["max_steps"], 600)
        self.assertEqual(payload["max_duration_minutes"], 480)
        self.assertEqual(packet["preview_kind"], "local_read_only_no_server_preview")
        self.assertIn("server defaults", packet["sampling"])
        self.assertIn("not caller-configurable", packet["server_retry_policy"])

    def test_matched_arms_and_renamed_study_are_same_scientific_identity(self):
        plan, live, group = fixture()
        base = job_payload(plan, "base")
        candidate = job_payload(plan, "candidate")
        self.assertEqual({k: v for k, v in base.items() if k not in {"name", "models"}},
                         {k: v for k, v in candidate.items() if k not in {"name", "models"}})
        renamed = copy.deepcopy(plan)
        renamed["protocol"]["study_id"] = "renamed"
        renamed["protocol"] = seal_protocol({k: v for k, v in renamed["protocol"].items() if k != "sha256"})
        self.assertEqual(_identity(plan["protocol"]), _identity(renamed["protocol"]))
        self.assertEqual(job_payload(plan, "base")["name"], job_payload(renamed, "base")["name"])

    def test_no_fixed_seed_or_claimed_sampling_on_native_api(self):
        for change in (
            lambda p: p["protocol"]["common"].update(seed_policy={"mode": "fixed",
                                                                    "seeds": [1, 2, 3, 4]}),
            lambda p: p["protocol"]["common"].update(temperature=0.8, top_p=0.9),
            lambda p: p["protocol"]["common"].update(retry_limit=1),
        ):
            plan, _, _ = fixture()
            change(plan)
            plan["protocol"] = seal_protocol({k: v for k, v in plan["protocol"].items() if k != "sha256"})
            with self.assertRaisesRegex(LaunchError, "native"):
                validate_plan(plan)

    def test_family_leak_and_wrong_group_fail_before_submission(self):
        plan, live, group = fixture()
        plan["family_roles"]["train"].append(["app", "family-dev"])
        with self.assertRaisesRegex(LaunchError, "overlap"):
            validate_plan(plan)
        plan, live, group = fixture()
        group["members"].append({"eval_task_version_id": "unplanned"})
        with self.assertRaisesRegex(LaunchError, "task group differs"):
            preview(plan, "base", **gates(plan, live, group))
        plan, live, group = fixture()
        group["team_id"] = "other"
        with self.assertRaisesRegex(LaunchError, "task group differs"):
            preview(plan, "base", **gates(plan, live, group))

    def test_exact_live_task_and_qualification_are_required(self):
        plan, live, group = fixture()
        live["verifier"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(LaunchError, "binding changed"):
            preview(plan, "base", **gates(plan, live, group))
        plan, live, group = fixture()
        checks = gates(plan, live, group)
        checks["qualification_get"] = lambda: {}
        with self.assertRaisesRegex(LaunchError, "qualification proof"):
            preview(plan, "base", **checks)

    def test_unproven_checkpoint_route_or_tools_cannot_pass(self):
        plan, live, group = fixture()
        checks = gates(plan, live, group)
        proof = _readiness_expected(plan)
        proof["routes"]["candidate"]["gateway_routed"] = False
        checks["readiness_get"] = lambda: proof
        with self.assertRaisesRegex(LaunchError, "checkpoint/route/harness proof"):
            preview(plan, "candidate", **checks)
        plan, live, group = fixture()
        plan["routes"]["candidate"] = "fleet/unrouted-checkpoint"
        with self.assertRaisesRegex(LaunchError, "Fleet Qwen routes"):
            validate_plan(plan)

    def test_wrong_account_and_budget_fail(self):
        plan, live, group = fixture()
        checks = gates(plan, live, group)
        checks["account_get"] = lambda: {"team_id": "other", "team_name": "fleet"}
        with self.assertRaisesRegex(LaunchError, "Fleet-team"):
            preview(plan, "base", **checks)
        checks = gates(plan, live, group)
        checks["budget_check"] = lambda sessions: False
        with self.assertRaisesRegex(LaunchError, "budget gate"):
            preview(plan, "base", **checks)

    def test_create_once_journal_prevents_duplicate_post(self):
        plan, live, group = fixture()
        calls = []

        def post(payload):
            calls.append(copy.deepcopy(payload))
            return {"job_id": "synthetic-id", "name": payload["name"], "status": "pending"}

        with tempfile.TemporaryDirectory() as directory:
            result = launch_once(plan, "base", post_job=post,
                                 journal_dir=Path(directory), **gates(plan, live, group))
            self.assertEqual(result["job_id"], "synthetic-id")
            self.assertEqual(len(calls), 1)
            journal = next(Path(directory).glob("*.jsonl"))
            self.assertEqual([json.loads(line)["state"] for line in journal.read_text().splitlines()],
                             ["POST_INTENT_DO_NOT_RETRY", "CREATED"])
            with self.assertRaisesRegex(LaunchError, "create intent already exists"):
                launch_once(plan, "base", post_job=post,
                            journal_dir=Path(directory), **gates(plan, live, group))
            self.assertEqual(len(calls), 1)

    def test_uncertain_post_must_be_reconciled_not_retried(self):
        plan, live, group = fixture()
        calls = []

        def post(payload):
            calls.append(payload)
            raise TimeoutError("response lost")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LaunchError, "outcome uncertain"):
                launch_once(plan, "candidate", post_job=post,
                            journal_dir=Path(directory), **gates(plan, live, group))
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(next(Path(directory).glob("*.jsonl")).read_text().splitlines()[0])["state"],
                             "POST_INTENT_DO_NOT_RETRY")
            with self.assertRaisesRegex(LaunchError, "create intent already exists"):
                launch_once(plan, "candidate", post_job=post,
                            journal_dir=Path(directory), **gates(plan, live, group))

    def test_preflight_drift_never_posts(self):
        plan, live, group = fixture()
        checks = gates(plan, live, group)
        counter = {"n": 0}

        def varying():
            counter["n"] += 1
            proof = _readiness_expected(plan)
            if counter["n"] == 2:
                proof["routes"]["base"]["gateway_routed"] = False
            return proof

        checks["readiness_get"] = varying
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LaunchError):
                launch_once(plan, "base", post_job=lambda payload: self.fail("POST must not run"),
                            journal_dir=Path(directory), **checks)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
