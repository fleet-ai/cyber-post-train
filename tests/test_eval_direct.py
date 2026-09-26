"""Synthetic direct-eval lifecycle; no Fleet request is created."""

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from evals.direct import CAPABILITY, LaunchError, _expected_readiness, digest, preview, run_once
from evals.fleet import seal_protocol, summarize
from tests.eval_fixture import fixture, sha

RUN = {"exit_code": 0, "trace_sha256": sha("a"), "steps": 2}
runner = lambda *args: RUN


def direct_fixture():
    native, live, _ = fixture()
    raw = {k: copy.deepcopy(v) for k, v in native["protocol"].items() if k != "sha256"}
    raw["common"].update(harness_version="1.18.27", context_window_tokens=98304, max_output_tokens=16384,
                         context_policy="native_compaction_reserved_32768")
    tools = [{"name": name, "inputSchema": {}} for name in ("bash", "submit_report")]
    raw["common"]["tool_schema_sha256"] = digest(tools)
    live.update(environment_id="synthetic-env", version="synthetic-v1", data_id="synthetic-data", prompt="Synthetic challenge.")
    plan = {"schema": "fleet_direct_opencode_v1", "protocol": seal_protocol(raw), "family_roles": native["family_roles"],
            "routes": {"base": "chris-q38-base-pass4-v1", "candidate": "chris-q38-study-step-16"},
            "task_response_sha256": {"version-1": digest(live)}, "task_qualification_sha256": {"version-1": sha("9")},
            "readiness_sha256": {key: sha(str(n)) for n, key in enumerate(
                ("candidate_export", "candidate_reload", "base_route", "candidate_route", "tool_parity"), 1)}}
    return plan, live, tools


class FakeAPI:
    def __init__(self, plan, live, *, capabilities=CAPABILITY, score=1.0, uncertain=False, runtime_bound=True):
        self.live, self.capabilities, self.score, self.uncertain, self.runtime_bound = live, capabilities, score, uncertain, runtime_bound
        self.posts, self.deleted = 0, False

    def account_get(self):
        return {"team_id": self.live["team_id"], "team_name": "fleet"}

    def task_get(self, key, version):
        return copy.deepcopy(self.live)

    def _request(self, method, path, *, body=None):
        if path == "/v1/rollout-rewards/capabilities":
            return self.capabilities
        if method == "POST" and path.endswith("/instances"):
            self.posts += 1
            if self.uncertain:
                raise TimeoutError("uncertain instance create")
            return {"task_key": self.live["key"], "task_version_id": self.live["eval_task_version_id"],
                    "instance_id": "owned-synthetic-instance", "evidence_run_id": "synthetic-evidence", "create_request_id": body["create_request_id"]}
        if method == "GET" and path.endswith("/owned-synthetic-instance"):
            return {"instance_id": "owned-synthetic-instance", "team_id": self.live["team_id"], "status": "running",
                    "terminated_at": None, "env_key": self.live["environment_id"], "version": self.live["version"],
                    "data_key": self.live["data_id"], "data_version": self.live["data_version"],
                    "eval_task_version_id": self.live["eval_task_version_id"] if self.runtime_bound else None,
                    "environment_version_id": self.live["environment_version_id"] if self.runtime_bound else None,
                    "seed_config": self.live["seed_config"] if self.runtime_bound else None,
                    "multi_app_seed_bindings": self.live["multi_app_seed_bindings"] if self.runtime_bound else None,
                    "multi_app_seed_versions": self.live["multi_app_seed_versions"] if self.runtime_bound else None,
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                    "urls": {"root": "https://synthetic.invalid"}}
        if path == "/v1/runner-auth/token":
            return {"header": "X-Test-Runner", "token": "synthetic-token-long"}
        if method == "POST" and path.endswith("/versions/version-1"):
            execution = "synthetic-verifier-execution"
            return {"task_key": self.live["key"], "task_version_id": "version-1", "instance_id": body["instance_id"],
                    "verifier_execution_id": execution, "reward": self.score,
                    "cyber_verification_result": {"schema_version": "cyber_verification_result_v3", "reward": self.score,
                        "bindings": {"task_version_id": "version-1"}, "components": {"ctf": {"score": self.score}}},
                    "cyber_evidence": {"mode": "authoritative", "status": "authoritative", "match": True,
                        "production_execution_id": execution, "direct_verifier": {"status": "authoritative", "match": True,
                        "execution_id": execution, "verifier_contract_version": "3.0.0", "context_schema_version": "cyber_verification_context_v1"}}}
        if method == "DELETE" and path.endswith("/owned-synthetic-instance"):
            self.deleted = True
            return {"terminated_at": "2026-09-25T00:00:00Z"}
        raise AssertionError((method, path))


def gates(plan):
    return {"qualification_get": lambda: copy.deepcopy(plan["task_qualification_sha256"]),
            "readiness_get": lambda: copy.deepcopy(_expected_readiness(plan)), "budget_check": lambda n: n == 1,
            "model_get": lambda served: {"served_model_name": served, "model_type": "qwen3_5", "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder", "context_window_tokens": 98304, "max_output_tokens": 16384},
            "harness_get": lambda image: "1.18.27"}


class DirectTests(unittest.TestCase):
    def test_capability_and_route_gate_before_any_post(self):
        plan, live, _ = direct_fixture()
        api = FakeAPI(plan, live, capabilities={key: value for key, value in CAPABILITY.items() if key != "exact_instance_runtime_readback"})
        with self.assertRaisesRegex(LaunchError, "not deployed"):
            preview(plan, "base", "version-1", 1, api=api, **gates(plan))
        self.assertEqual(api.posts, 0)
        api.capabilities = CAPABILITY
        checks = gates(plan)
        checks["model_get"] = lambda served: {}
        with self.assertRaisesRegex(LaunchError, "profile changed"):
            preview(plan, "base", "version-1", 1, api=api, **checks)
        self.assertEqual(api.posts, 0)
        api.capabilities = {**CAPABILITY, "unrelated_future_capability": "v1"}
        with self.assertRaisesRegex(LaunchError, "profile changed"):
            preview(plan, "base", "version-1", 1, api=api, **checks)
        self.assertEqual(api.posts, 0)

    def test_one_claim_one_score_full_ctf_only_and_release(self):
        plan, live, tools = direct_fixture()
        api = FakeAPI(plan, live, score=0.4)
        with tempfile.TemporaryDirectory() as directory, patch("evals.direct._mcp_tools", return_value=tools):
            event = run_once(plan, "base", "version-1", 1, Path(directory), api=api, runner=runner, **gates(plan))
            self.assertFalse(event["success"])
            self.assertEqual(event["ctf_score"], 0.4)
            self.assertTrue(api.deleted)
            self.assertEqual(api.posts, 1)
            with self.assertRaisesRegex(LaunchError, "already claimed"):
                run_once(plan, "base", "version-1", 1, Path(directory), api=api, runner=runner, **gates(plan))
            self.assertEqual(api.posts, 1)
            self.assertEqual(summarize(plan["protocol"], [event])["status"], "incomplete")
        prefixed = [{"name": "fleet_" + tool["name"], "inputSchema": {}} for tool in tools]
        with tempfile.TemporaryDirectory() as directory, patch("evals.direct._mcp_tools", return_value=prefixed):
            with self.assertRaisesRegex(LaunchError, "MCP tool schema"):
                run_once(plan, "base", "version-1", 1, Path(directory), api=api, runner=runner, **gates(plan))
        self.assertTrue(api.deleted)

    def test_uncertain_create_never_retries(self):
        plan, live, _ = direct_fixture()
        api = FakeAPI(plan, live, uncertain=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                run_once(plan, "candidate", "version-1", 2, Path(directory), api=api, runner=runner, **gates(plan))
            self.assertEqual(api.posts, 1)
            with self.assertRaisesRegex(LaunchError, "already claimed"):
                run_once(plan, "candidate", "version-1", 2, Path(directory), api=api, runner=runner, **gates(plan))
            self.assertEqual(api.posts, 1)

    def test_unbound_runtime_is_released_without_scoring(self):
        plan, live, _ = direct_fixture()
        api = FakeAPI(plan, live, runtime_bound=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LaunchError, "pinned task/runtime/seed"):
                run_once(plan, "base", "version-1", 1, Path(directory), api=api, runner=runner, **gates(plan))
        self.assertEqual(api.posts, 1)
        self.assertTrue(api.deleted)


if __name__ == "__main__":
    unittest.main()
