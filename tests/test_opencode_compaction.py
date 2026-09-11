"""Growth budgeting and actual pinned OpenCode compaction/continuation, no Fleet."""

import json
import os
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evals.fleet import opencode_self_hosted as harness


def config(policy, context=262144, output=65536, headroom=20000):
    return {
        "model": {"served_id": "synthetic"},
        "harness": {
            "name": "opencode",
            "version": "1.18.27",
            "context_management": policy,
            "context_window_size": context,
            "max_output_tokens": output,
            "compaction_headroom_tokens": headroom,
        },
    }


@pytest.mark.parametrize("output", [8192, 32768, 65536])
def test_reserves_next_response_and_compaction_output(output):
    settings = harness.opencode_settings(
        config(harness.OPENCODE_GROWTH_AWARE_CONTEXT_MANAGEMENT, output=output)
    )
    limits = settings["provider"]["fleet-cluster"]["models"]["synthetic"]["limit"]
    threshold = limits["input"] - settings["compaction"]["reserved"]
    assert threshold == 262144 - 2 * output - 20000
    # Largest preceding non-compacting step + full next response + overhead
    # + the compaction response still fits the declared context.
    assert threshold - 1 + output + 20000 + output < limits["context"]


def test_legacy_policy_unchanged_and_new_policy_rejects_no_room():
    old = harness.opencode_settings(config(harness.OPENCODE_CONTEXT_MANAGEMENT))
    assert old["compaction"] == {"auto": True, "reserved": 20000}
    with pytest.raises(ValueError, match="leave room"):
        harness.opencode_settings(
            config(harness.OPENCODE_GROWTH_AWARE_CONTEXT_MANAGEMENT, context=151072)
        )
    # This shape was valid under v1. Do not rewrite historical experiments.
    harness.opencode_settings(config(harness.OPENCODE_CONTEXT_MANAGEMENT, context=151072))


@pytest.mark.parametrize(
    "policy,expected_compactions",
    [
        (harness.OPENCODE_CONTEXT_MANAGEMENT, 0),
        (harness.OPENCODE_GROWTH_AWARE_CONTEXT_MANAGEMENT, 1),
    ],
)
def test_native_binary_compacts_and_continues(tmp_path, policy, expected_compactions):
    image = os.environ.get("CYBER_TEST_OPENCODE_IMAGE")
    if not image:
        pytest.skip("requires the already-staged pinned OpenCode Docker image")
    assert image.startswith("sha256:") or "@sha256:" in image
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            agent = self.headers.get("x-synthetic-agent")
            requests.append((agent, body))
            first = agent == "build" and sum(a == "build" for a, _ in requests) == 1
            usage = {"prompt_tokens": 5500 if first else 100, "completion_tokens": 200}
            usage["total_tokens"] = sum(usage.values())
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Synthetic fixture."},
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            response = (
                "".join(
                    "data: "
                    + json.dumps(
                        {
                            "id": "synthetic",
                            "object": "chat.completion.chunk",
                            "model": "synthetic",
                            **c,
                        }
                    )
                    + "\n\n"
                    for c in chunks
                )
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(response.encode())))
            self.end_headers()
            self.wfile.write(response.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = harness.opencode_settings(config(policy, context=10000, output=2048, headroom=512))
    settings["provider"]["fleet-cluster"]["options"]["baseURL"] = (
        f"http://host.docker.internal:{server.server_port}/v1"
    )
    settings["mcp"] = {}
    settings["plugin"] = ["file:///home/node/label-agent.mjs"]
    settings["share"] = "disabled"
    directory = tmp_path / ".config/opencode"
    directory.mkdir(parents=True)
    (directory / "opencode.json").write_text(json.dumps(settings))
    (tmp_path / "label-agent.mjs").write_text(
        'export const Label = async () => ({"chat.headers": async (i,o) => '
        '{o.headers["x-synthetic-agent"] = i.agent;}});'
    )
    if os.geteuid() == 0:
        for path in [tmp_path, *tmp_path.rglob("*")]:
            os.chown(path, 1000, 1000)
    name = "cpt-synthetic-compaction-" + uuid.uuid4().hex[:12]
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--platform",
                "linux/amd64",
                "--name",
                name,
                "--add-host",
                "host.docker.internal:host-gateway",
                *harness.agent_container_user_args(),
                "-v",
                f"{tmp_path}:/home/node",
                "-e",
                "OPENCODE_DISABLE_MODELS_FETCH=true",
                "-e",
                "OPENCODE_DISABLE_AUTOUPDATE=true",
                image,
                "opencode",
                "run",
                "--format",
                "json",
                "--model",
                "fleet-cluster/synthetic",
                "--dir",
                "/workspace",
                "--auto",
                "--",
                "Synthetic compaction test. Return a short answer.",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert (
            harness.opencode_termination(events, malformed_lines=0, exit_code=0, timed_out=False)
            == "completed"
        )
        assert sum(a == "compaction" for a, _ in requests) == expected_compactions
        assert sum(a == "build" for a, _ in requests) == 1 + expected_compactions
        for agent, body in requests:
            if agent == "compaction":
                assert not body.get("tools")
        if expected_compactions:
            last = next(body for agent, body in reversed(requests) if agent == "build")
            assert "Continue if you have next steps" in json.dumps(last["messages"])
    finally:
        subprocess.run(["docker", "stop", "--time", "1", name], capture_output=True, timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
