"""The model-free Fleet probe must bind exact identities and always clean up."""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx

from training import runtime_qualify as q


ROW = {"task_key": "example_blackbox_ctf_v1", "task_version_id": "version-1",
       "task_id": "task-1", "environment_version_id": "env-version-1",
       "atom_artifact_keys": ["cyber/atoms/app/a"]}
BINDING = {"environment_id": "app", "environment_version": "v1"}


def response(code, value=None, *, headers=None):
    return httpx.Response(code, json=value, headers=headers)


class RuntimeQualificationTests(unittest.TestCase):
    def test_dev_first_cell_is_frozen_and_later_cells_stay_preview_only(self):
        path = Path(__file__).parents[1] / "configs/data/fleet-blackbox-development-qualification-preview16-20260925-v1.json"
        wave, row, _sha, _request_id = q.load_cell(path, 0)
        self.assertEqual((row["reserved_role"], row["qualification_rank"]), ("dev", 25))
        self.assertEqual(q.load_cell(path, 1)[1]["qualification_rank"], 26)
        self.assertFalse(wave["launch_authorized"])
        altered = json.loads(path.read_text())
        altered["wave"][0]["reserved_role"] = "final_test"
        altered = q.seal({key: value for key, value in altered.items() if key != "sha256"})
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "wave.json"
            copy.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "frozen qualification wave"):
                q.load_cell(copy, 0)

    def test_only_first_two_final_cells_are_authorized(self):
        root = Path(__file__).parents[1] / "configs/data"
        old = root / "fleet-blackbox-qualification-rank16-20260925-v1.json"
        dev = root / "fleet-blackbox-development-qualification-preview16-20260925-v1.json"
        self.assertFalse(q.authorized(q.load_cell(old, 2)[0], 2))
        self.assertTrue(q.authorized(q.load_cell(dev, 2)[0], 2))
        with self.assertRaisesRegex(ValueError, "not authorized"):
            q.run_cell(old, 2, Path("not-created"))
        self.assertEqual(q.RUN_AUTHORIZED, {0, 1})

    def test_preflight_reads_only_and_binds_exact_version(self):
        methods = []

        def handle(request):
            methods.append(request.method)
            path = request.url.path
            if path == "/v1/account":
                return response(200, {"team_id": q.TEAM, "team_name": "fleet",
                                      "instance_count": 1, "instance_limit": 10})
            if path == "/openapi.json":
                return response(200, {"paths": {
                    "/v1/env/instances/create-requests/{request_id}":
                        {"get": {}, "delete": {}},
                    "/v1/env/instances/{instance_id}": {"get": {}, "delete": {}},
                    "/v1/env/instances/{instance_id}/ttl": {"post": {"requestBody": {
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/SetTTLRequest"}}}}}}}})
            if path == "/v1/rollout-rewards/capabilities":
                return response(200, {"version_scoped_durable_create_claim": "v1",
                                      "create_request_field": "create_request_id",
                                      "claim_route": "/v1/env/instances/create-requests/{request_id}",
                                      "ttl_seconds_range": [60, 3600]})
            if path.startswith("/v1/rollout-rewards/"):
                return response(405)
            if path.startswith("/v1/env/instances/create-requests/"):
                return response(404, {"detail": {"error": "durable_create_request_not_found"}})
            if path == "/v1/pipeline/tasks/task-1/status":
                return response(200, {"eval_task_id": "task-1", "current_version_id": "version-1",
                                      "lifecycle_status": "production", "verifier_attached": True})
            if path.endswith("/task-quality"):
                return response(200, {"tasks": [{"task_key": ROW["task_key"],
                                                  "eval_task_version_id": "version-1",
                                                  "status": "not_analyzed"}]})
            if path == "/v1/tasks/example_blackbox_ctf_v1":
                return response(200, {"key": ROW["task_key"],
                                      "eval_task_version_id": "version-1",
                                      "environment_id": "app", "version": "v1",
                                      "environment_version_id": "env-version-1",
                                      "verifier_id": "verifier-1",
                                      "verifier": {"verifier_version_id": "verifier-version-1"},
                                      "metadata": {"projection_id": "blackbox_ctf_v1",
                                                   "cyber_contract": q.CONTRACT,
                                                   "cyber_subject": {"atom_sources": [{
                                                       "artifact_key": "cyber/atoms/app/a",
                                                       "version_index": 2,
                                                       "locator": "cyber/atoms/app/a@2:atom_source"}]}}})
            raise AssertionError(path)

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            binding = q.preflight(client, ROW, "request-1")
        self.assertEqual(binding["environment_version_id"], "env-version-1")
        self.assertEqual(binding["verifier_version_id"], "verifier-version-1")
        self.assertTrue(all(method == "GET" for method in methods))

    def test_create_uses_durable_id_and_rejects_missing_echo(self):
        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(json.loads(request.content),
                             {"create_request_id": "request-1", "ttl_seconds": 900})
            return response(200, {"task_key": ROW["task_key"],
                                  "task_version_id": ROW["task_version_id"],
                                  "instance_id": "instance-1", "evidence_run_id": "run-1"})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaisesRegex(ValueError, "provision response identity mismatch"):
                q.create_instance(client, ROW, "request-1")

    def test_cleanup_deletes_only_bound_instance(self):
        methods = []

        def handle(request):
            methods.append((request.method, request.url.path))
            if request.url.path.endswith("/create-requests/request-1"):
                return response(200, {"request_id": "request-1", "team_id": q.TEAM,
                                      "state": "materialized", "instance_id": "instance-1"})
            if request.url.path.endswith("/instances/instance-1"):
                if request.method == "GET":
                    return response(200, {"instance_id": "instance-1", "team_id": q.TEAM,
                                          "env_key": "app", "version": "v1"})
                return response(200, {"terminated_at": "now"})
            raise AssertionError(request.url.path)

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            result = q.cleanup(client, {"request_id": "request-1"}, BINDING, "instance-1")
        self.assertTrue(result["cleanup_complete"])
        self.assertEqual(methods[-1], ("DELETE", "/v1/env/instances/instance-1"))

    def test_wave_digest_and_duplicate_gate(self):
        source = q.ORDER_PATH.with_name("fleet-blackbox-qualification-rank16-20260925-v1.json")
        wave = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.json"
            path.write_text(json.dumps(wave))
            _, row, _, first = q.load_cell(path, 0)
            self.assertEqual(row, wave["wave"][0])
            self.assertEqual(first, q.load_cell(path, 0)[3])
            changed = {**wave, "wave": [{**wave["wave"][0], "task_version_id": "other"},
                                       *wave["wave"][1:]]}
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "wave is invalid"):
                q.load_cell(path, 0)

    def test_clamp_ttl_requires_persisted_short_expiry(self):
        expiry = (datetime.now(UTC) + timedelta(seconds=900)).isoformat()
        def handle(req):
            self.assertEqual(req.url.path, "/v1/env/instances/instance-1" +
                             ("/ttl" if req.method == "POST" else ""))
            return response(200, {"instance_id": "instance-1", "status": "running",
                                  "expires_at": expiry})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            self.assertTrue(q.clamp_ttl(client, "instance-1")["ttl_clamped"])
        expiry = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaisesRegex(ValueError, "not bounded"):
                q.clamp_ttl(client, "instance-1")

    def test_mcp_session_header_is_used_and_closed(self):
        calls = []

        def handle(request):
            payload = json.loads(request.content) if request.method == "POST" else {}
            calls.append((request.method, payload.get("method"),
                          request.headers.get("Mcp-Session-Id")))
            if payload.get("method") == "initialize":
                return response(200, {"id": 1, "result": {}},
                                headers={"mcp-session-id": "session-1"})
            if payload.get("method") == "tools/list":
                return response(200, {"id": 2, "result": {"tools": [
                    {"name": "bash"}, {"name": "submit_report"}]}})
            if payload.get("method") == "tools/call":
                return response(200, {"id": payload["id"], "result": {"isError": False}})
            return response(200, {})

        original = httpx.Client
        with patch.object(q.httpx, "Client", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            digest = q.probe_tools("https://example.test", "X-Runner", "secret")
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(calls[-1], ("DELETE", None, "session-1"))
        self.assertTrue(all(session == "session-1" for _, _, session in calls[1:]))

    def test_mcp_failure_records_only_fixed_tool_stage(self):
        def handle(request):
            payload = json.loads(request.content) if request.method == "POST" else {}
            method = payload.get("method")
            if method == "initialize":
                return response(200, {"id": 1, "result": {}},
                                headers={"mcp-session-id": "session-1"})
            if method == "tools/list":
                return response(200, {"id": 2, "result": {"tools": [
                    {"name": "bash"}, {"name": "submit_report"}]}})
            if method == "tools/call":
                return response(200, {"id": payload["id"],
                                      "result": {"isError": True}})
            return response(200, {})

        original = httpx.Client
        with patch.object(q.httpx, "Client", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            stage = ["runner_auth"]
            with self.assertRaisesRegex(RuntimeError, "rejected model-free probe"):
                q.probe_tools("https://example.test", "X-Runner", "dummy", stage)
        self.assertEqual(stage, ["mcp_bash"])
        self.assertIn(stage[0], q.FAILURE_STAGES)


if __name__ == "__main__":
    unittest.main()
