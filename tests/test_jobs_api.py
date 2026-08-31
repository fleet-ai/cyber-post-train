import json
import tempfile
import unittest
from pathlib import Path

import httpx

from training.io import digest_json
from training.jobs_api import (
    RL_TOOL_EVIDENCE_SCHEMA,
    JobsAPIError,
    TrainingJobsClient,
    concise_status,
    load_run_config,
    rl_paid_launch_blockers,
    run_kind,
)


def _rl_config(*, cap: str | None = "130") -> dict:
    args = [] if cap is None else [f"trainer.max_training_steps={cap}"]
    return {
        "kind": "rl",
        "title": "Chris successor RL",
        "grpo": {"max_steps": 130},
        "trainer": {"args": args},
        "tasks": {"task_versions": [{"task_version_id": "train-version"}]},
        "eval": {"task_versions": [{"task_version_id": "dev-version"}]},
    }


def _tool_evidence() -> dict:
    bindings = [
        {"task_version_id": "train-version", "tools": ["bash", "submit_report"]},
        {"task_version_id": "dev-version", "tools": ["bash", "submit_report"]},
    ]
    return {
        "schema": RL_TOOL_EVIDENCE_SCHEMA,
        "source": "authoritative_task_version_metadata",
        "source_field": "metadata.tools",
        "bindings": bindings,
        "bindings_sha256": digest_json(bindings),
    }


def _rl_preview(*, cap: str | None = "130", tools: bool = True) -> dict:
    words = ["python", "-m", "rl_rollout.entrypoint", "trainer.epochs=130"]
    if cap is not None:
        words.append(f"trainer.max_training_steps={cap}")
    preview = {
        "name": "ft-run-preview",
        "manifest_yaml": "spec:\n  entrypoint: " + " ".join(words) + "\n",
        "warnings": [],
    }
    if tools:
        preview["task_tool_allowlist_evidence"] = _tool_evidence()
    return preview


class JobsAPIClientTests(unittest.TestCase):
    def test_load_config_requires_typed_kind_and_title(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            path.write_text(json.dumps({"kind": "rl", "title": "Chris gate"}))
            self.assertEqual(load_run_config(path)["kind"], "rl")
            path.write_text(json.dumps({"kind": "custom", "title": "Chris gate"}))
            with self.assertRaisesRegex(JobsAPIError, "kind"):
                load_run_config(path)

    def test_rl_kind_is_inferred_from_typed_schema_without_mutating_payload(self):
        config = {"grpo": {}, "tasks": {}, "title": "Chris RL gate"}
        self.assertEqual(run_kind(config), "rl")
        self.assertNotIn("kind", config)

    def test_submit_previews_then_submits_without_serializing_bearer(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "GET":
                return httpx.Response(200, json=[])
            if request.url.path.endswith("/preview"):
                return httpx.Response(200, json={"errors": [], "warnings": []})
            return httpx.Response(202, json={"name": "ft-run-abcd1234"})

        with TrainingJobsClient(
            "secret-token",
            base_url="https://jobs.invalid",
            transport=httpx.MockTransport(handler),
        ) as client:
            receipt = client.submit({"kind": "sft", "title": "Chris gate"})
        self.assertEqual(receipt["name"], "ft-run-abcd1234")
        self.assertEqual([request.method for request in seen], ["GET", "POST", "POST"])
        self.assertNotIn("secret-token", "".join(request.content.decode() for request in seen))

    def test_submit_refuses_duplicate_title_before_preview(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            return httpx.Response(200, json=[{"name": "ft-run-old", "title": "Chris gate"}])

        with TrainingJobsClient(
            "token",
            base_url="https://jobs.invalid",
            transport=httpx.MockTransport(handler),
        ) as client, self.assertRaisesRegex(JobsAPIError, "duplicate"):
            client.submit({"kind": "rl", "title": "Chris gate"})

    def test_paid_rl_refuses_epoch_mapping_without_a_true_step_cap(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=_rl_preview(cap=None))

        with TrainingJobsClient(
            "token", base_url="https://jobs.invalid", transport=httpx.MockTransport(handler)
        ) as client, self.assertRaisesRegex(JobsAPIError, "trainer.max_training_steps=130"):
            client.submit(_rl_config(cap=None))
        self.assertEqual(seen, ["GET", "POST"], "the paid POST must never be reached")

    def test_paid_rl_refuses_when_preview_has_no_authoritative_tool_evidence(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=_rl_preview(tools=False))

        with TrainingJobsClient(
            "token", base_url="https://jobs.invalid", transport=httpx.MockTransport(handler)
        ) as client, self.assertRaisesRegex(JobsAPIError, "authoritative task_tool_allowlist"):
            client.submit(_rl_config())
        self.assertEqual(seen, ["GET", "POST"], "the paid POST must never be reached")

    def test_paid_rl_accepts_current_epoch_mapping_only_with_true_cap_and_exact_tools(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "GET":
                return httpx.Response(200, json=[])
            if request.url.path.endswith("/preview"):
                return httpx.Response(200, json=_rl_preview())
            return httpx.Response(202, json={"name": "ft-run-successor"})

        config = _rl_config()
        with TrainingJobsClient(
            "token", base_url="https://jobs.invalid", transport=httpx.MockTransport(handler)
        ) as client:
            receipt = client.submit(config)
        self.assertEqual(receipt["name"], "ft-run-successor")
        self.assertEqual([request.method for request in seen], ["GET", "POST", "POST"])
        self.assertEqual(json.loads(seen[-1].content), config)

    def test_tool_evidence_rejects_empty_tools_digest_drift_and_identity_drift(self):
        preview = _rl_preview()
        evidence = preview["task_tool_allowlist_evidence"]
        evidence["bindings"][0]["tools"] = []
        evidence["bindings"][1]["task_version_id"] = "wrong-version"
        blockers = rl_paid_launch_blockers(_rl_config(), preview)
        self.assertTrue(any("bindings_sha256 mismatch" in item for item in blockers))
        self.assertTrue(any("non-empty" in item for item in blockers))
        self.assertTrue(any("does not exactly match" in item for item in blockers))

    def test_paid_cyber_rl_rejects_any_tool_surface_beyond_bash_and_submit_report(self):
        preview = _rl_preview()
        preview["task_tool_allowlist_evidence"]["bindings"][0]["tools"].append("text_editor")
        bindings = preview["task_tool_allowlist_evidence"]["bindings"]
        preview["task_tool_allowlist_evidence"]["bindings_sha256"] = digest_json(bindings)
        blockers = rl_paid_launch_blockers(_rl_config(), preview)
        self.assertEqual(len(blockers), 1)
        self.assertIn("exactly ordered", blockers[0])

    def test_paid_cyber_rl_rejects_reversed_cyber_tool_order(self):
        preview = _rl_preview()
        preview["task_tool_allowlist_evidence"]["bindings"][0]["tools"] = [
            "submit_report",
            "bash",
        ]
        bindings = preview["task_tool_allowlist_evidence"]["bindings"]
        preview["task_tool_allowlist_evidence"]["bindings_sha256"] = digest_json(bindings)

        blockers = rl_paid_launch_blockers(_rl_config(), preview)

        self.assertEqual(len(blockers), 1)
        self.assertIn("exactly ordered", blockers[0])

    def test_no_eval_arm_requires_tools_only_for_exact_training_versions(self):
        config = _rl_config()
        config["eval"]["task_versions"] = []
        preview = _rl_preview()
        bindings = preview["task_tool_allowlist_evidence"]["bindings"][:1]
        preview["task_tool_allowlist_evidence"]["bindings"] = bindings
        preview["task_tool_allowlist_evidence"]["bindings_sha256"] = digest_json(bindings)
        self.assertEqual(rl_paid_launch_blockers(config, preview), [])

    def test_concise_status_drops_manifest(self):
        result = concise_status(
            {
                "name": "ft-run-a",
                "kind": "rl",
                "status": "Running",
                "manifest": {"secret": "large"},
                "steps": [{"step": 1}],
                "checkpoints": [{"step": 1}],
                "sessions": {"training": [{"id": "s"}], "eval": []},
            }
        )
        self.assertNotIn("manifest", result)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["checkpoints"], 1)
        self.assertEqual(result["training_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
