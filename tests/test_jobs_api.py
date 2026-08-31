import json
import tempfile
import unittest
from pathlib import Path

import httpx

from training.jobs_api import JobsAPIError, TrainingJobsClient, concise_status, load_run_config


class JobsAPIClientTests(unittest.TestCase):
    def test_load_config_requires_typed_kind_and_title(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            path.write_text(json.dumps({"kind": "rl", "title": "Chris gate"}))
            self.assertEqual(load_run_config(path)["kind"], "rl")
            path.write_text(json.dumps({"kind": "custom", "title": "Chris gate"}))
            with self.assertRaisesRegex(JobsAPIError, "kind"):
                load_run_config(path)

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
