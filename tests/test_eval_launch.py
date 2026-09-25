"""Synthetic launch-boundary checks. These tests create no Fleet sessions or Jobs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evals.fleet import seal_protocol
from evals.launch import LaunchError, _identity, digest, launch_once, preview, validate_plan


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
        "schema": "fleet_paired_pass4_v1", "study_id": "synthetic",
        "role": "dev", "final_selection_sha256": None, "tasks": [task],
        "common": {
            "model_repository": "Qwen/synthetic", "tokenizer_sha256": sha("a"),
            "chat_template_sha256": sha("b"), "serving_image": "serve@" + sha("c"),
            "serving_config_sha256": sha("d"), "harness_image": "runner@" + sha("f"),
            "harness_version": "pinned-opencode", "system_prompt_sha256": sha("a"),
            "tools": ["fleet_bash", "fleet_submit_report"],
            "tool_schema_sha256": sha("b"), "context_policy": "native_compaction",
            "context_window_tokens": 262144, "max_output_tokens": 32768,
            "max_model_requests": 600, "timeout_seconds": 28800,
            "temperature": 0.6, "top_p": 0.95, "seeds": [41, 42, 43, 44],
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
    identity = _identity(protocol)
    roots = {arm: f"/mnt/sfs/jobs/chris-synthetic-{arm}" for arm in ("base", "candidate")}
    jobs = {}
    for arm in ("base", "candidate"):
        artifact = protocol["arms"][arm]
        env_values = {
            "EVAL_HARNESS": "opencode", "EVAL_PROTOCOL_SHA256": protocol["sha256"],
            "EVAL_ARM": arm, "EVAL_MODEL_REVISION": artifact["model_revision"],
            "EVAL_WEIGHTS_SHA256": artifact["weights_sha256"],
            "EVAL_CHECKPOINT_SHA256": artifact.get("checkpoint_sha256", ""),
            "EVAL_TASK_ROSTER_SHA256": digest(protocol["tasks"]),
            "EVAL_OUTPUT": roots[arm], "EVAL_SERVED_ID": f"served-{arm}",
        }
        jobs[arm] = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {
                "name": f"chris-q38-fleet-{identity[7:19]}-{arm}",
                "namespace": "fleet-train-jobs",
                "labels": {"cyber-post-train.fleet.ai/identity": identity[7:55]},
                "annotations": {
                    "fleet.ai/failure-alerts": "off",
                    "cyber-post-train.fleet.ai/protocol-sha256": protocol["sha256"],
                    "cyber-post-train.fleet.ai/identity-sha256": identity,
                    "cyber-post-train.fleet.ai/arm": arm,
                },
            },
            "spec": {
                "backoffLimit": 0, "activeDeadlineSeconds": 36000,
                "template": {"spec": {
                    "priorityClassName": "c1", "restartPolicy": "Never",
                    "containers": [{"name": "evaluator", "image": "runner@" + sha("f"),
                                    "command": ["python"], "args": ["-m", "evals.runner"],
                                    "env": [{"name": k, "value": v} for k, v in env_values.items()],
                                    "resources": {"requests": {"cpu": "2", "memory": "4Gi"}}}],
                }},
            },
        }
    plan = {
        "schema": "fleet_paired_launch_v1", "harness": "opencode",
        "protocol": protocol,
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
        "jobs": jobs, "output_roots": roots,
    }
    return plan, live


class FakeCluster:
    def __init__(self):
        self.jobs = []
        self.creates = []
        self.render_mutation = None

    def jobs_for_identity(self, identity):
        return copy.deepcopy(self.jobs)

    def server_preview(self, job):
        result = copy.deepcopy(job)
        result["metadata"]["uid"] = "a" * 32
        result["spec"]["template"]["spec"]["schedulerName"] = "default-scheduler"
        for row in result["spec"]["template"]["spec"]["containers"][0]["env"]:
            if row.get("value") == "":
                row.pop("value")
        if self.render_mutation:
            self.render_mutation(result)
        return result

    def create_once(self, job):
        self.creates.append(copy.deepcopy(job))
        result = copy.deepcopy(job)
        result["metadata"]["uid"] = "synthetic-uid"
        self.jobs.append(result)
        return result


def gates(plan, live, cluster):
    return {
        "cluster": cluster,
        "account_get": lambda: {"team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
                                "team_name": "fleet"},
        "task_get": lambda key, version: copy.deepcopy(live),
        "readiness_check": lambda protocol, expected: expected,
        "qualification_check": lambda protocol, expected: expected,
        "output_exists": lambda root: False,
        "capacity_check": lambda: True,
    }


def launch_gates(plan, live, cluster):
    return {**gates(plan, live, cluster),
            "capacity_reserve": lambda identity, arm: "synthetic-shared-lease",
            "capacity_release": lambda lease: None}


class LaunchTests(unittest.TestCase):
    def test_preview_is_read_only_and_binds_both_arms(self):
        plan, live = fixture()
        cluster = FakeCluster()
        self.assertTrue(validate_plan(plan).startswith("sha256:"))
        self.assertTrue(preview(plan, "base", **gates(plan, live, cluster)).startswith("sha256:"))
        self.assertTrue(preview(plan, "candidate", **gates(plan, live, cluster)).startswith("sha256:"))
        self.assertEqual(cluster.creates, [])

    def test_fail_closed_on_family_leak_or_task_drift(self):
        plan, live = fixture()
        plan["family_roles"]["train"].append(["app", "family-dev"])
        with self.assertRaisesRegex(LaunchError, "overlap"):
            validate_plan(plan)
        plan, live = fixture()
        live["verifier"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(LaunchError, "binding changed"):
            preview(plan, "base", **gates(plan, live, FakeCluster()))

    def test_root_alert_priority_and_paired_runtime_are_mandatory(self):
        for mutation, reason in (
            (lambda p: p["jobs"]["base"]["metadata"]["annotations"].pop("fleet.ai/failure-alerts"), "root Job annotations"),
            (lambda p: p["jobs"]["base"]["spec"]["template"]["spec"].update(priorityClassName="c0"), "c1"),
            (lambda p: p["jobs"]["base"]["spec"]["template"]["spec"]["containers"][0].update(image="runner:latest"), "immutable"),
            (lambda p: p["jobs"]["base"]["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"].update({"nvidia.com/gpu": "1"}), "zero GPUs"),
            (lambda p: p["jobs"]["base"]["spec"]["template"]["spec"]["containers"][0].update(command=["different"]), "differ beyond"),
        ):
            with self.subTest(reason=reason):
                plan, _ = fixture()
                mutation(plan)
                with self.assertRaisesRegex(LaunchError, reason):
                    validate_plan(plan)

    def test_server_rendered_root_and_independent_gates(self):
        plan, live = fixture()
        cluster = FakeCluster()
        cluster.render_mutation = lambda job: job["metadata"]["annotations"].update({"fleet.ai/failure-alerts": "on"})
        with self.assertRaisesRegex(LaunchError, "server-rendered"):
            preview(plan, "base", **gates(plan, live, cluster))
        cluster.render_mutation = None
        checks = gates(plan, live, cluster)
        checks["readiness_check"] = lambda protocol, expected: {}
        with self.assertRaisesRegex(LaunchError, "proof did not match"):
            preview(plan, "base", **checks)
        checks = gates(plan, live, cluster)
        checks["qualification_check"] = lambda protocol, expected: {}
        with self.assertRaisesRegex(LaunchError, "qualification proof did not match"):
            preview(plan, "base", **checks)
        checks = gates(plan, live, cluster)
        checks["account_get"] = lambda: {"team_id": "other", "team_name": "fleet"}
        with self.assertRaisesRegex(LaunchError, "Fleet-team"):
            preview(plan, "base", **checks)

    def test_duplicate_output_and_capacity_are_fail_closed(self):
        plan, live = fixture()
        cluster = FakeCluster()
        cluster.jobs.append(copy.deepcopy(plan["jobs"]["base"]))
        with self.assertRaisesRegex(LaunchError, "already exists"):
            preview(plan, "base", **gates(plan, live, cluster))
        cluster.jobs.clear()
        checks = gates(plan, live, cluster)
        checks["output_exists"] = lambda root: True
        with self.assertRaisesRegex(LaunchError, "output already exists"):
            preview(plan, "base", **checks)
        checks = gates(plan, live, cluster)
        checks["capacity_check"] = lambda: False
        with self.assertRaisesRegex(LaunchError, "capacity gate"):
            preview(plan, "base", **checks)

    def test_single_create_intent_never_replays(self):
        plan, live = fixture()
        cluster = FakeCluster()
        with tempfile.TemporaryDirectory() as directory:
            result = launch_once(plan, "base", **launch_gates(plan, live, cluster),
                                 journal_dir=Path(directory))
            self.assertEqual(result["job_uid"], "synthetic-uid")
            self.assertEqual(len(cluster.creates), 1)
            journal = next(Path(directory).glob("*.jsonl"))
            self.assertEqual([json.loads(x)["state"] for x in journal.read_text().splitlines()],
                             ["CREATE_INTENT_DO_NOT_RETRY", "CREATED"])
            with self.assertRaisesRegex(LaunchError, "create intent already exists"):
                launch_once(plan, "base", **launch_gates(plan, live, cluster),
                            journal_dir=Path(directory))
            self.assertEqual(len(cluster.creates), 1)

    def test_atomic_capacity_reservation_is_required_before_create(self):
        plan, live = fixture()
        cluster = FakeCluster()
        with tempfile.TemporaryDirectory() as directory:
            checks = launch_gates(plan, live, cluster)
            checks["capacity_reserve"] = lambda identity, arm: ""
            with self.assertRaisesRegex(LaunchError, "reservation was not acquired"):
                launch_once(plan, "base", **checks, journal_dir=Path(directory))
            self.assertEqual(cluster.creates, [])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
