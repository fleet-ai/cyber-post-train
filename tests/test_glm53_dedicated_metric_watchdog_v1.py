from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evals.fleet import glm53_dedicated_metric_watchdog_canary_v1 as cluster_canary


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = ROOT / "evals/fleet/scripts/glm53_dedicated_metric_lifecycle_v1.sh"

FAKE_SERVER = r"""
import http.server
import pathlib
import sys

counter = pathlib.Path(sys.argv[1])
counter.write_text("0")

class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"ok\n"
        elif self.path == "/metrics":
            value = int(counter.read_text())
            body = (
                f"sglang:num_requests_total{{model_name=\"glm-5.3\"}} {value}\n"
                f"sglang:prompt_tokens_total{{model_name=\"glm-5.3\"}} {value * 2}\n"
                f"sglang:generation_tokens_total{{model_name=\"glm-5.3\"}} {value * 3}\n"
            ).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        counter.write_text(str(int(counter.read_text()) + 1))
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return

Server(("127.0.0.1", 8000), Handler).serve_forever()
"""


def _start(tmp_path: Path, name: str) -> tuple[subprocess.Popen[bytes], Path]:
    run_dir = tmp_path / name
    server = tmp_path / f"{name}.py"
    counter = tmp_path / f"{name}.count"
    server.write_text(FAKE_SERVER)
    env = {
        **os.environ,
        "GLM53_RUN_DIR": str(run_dir),
        "GLM53_REPLICA": "A",
        "GLM53_IDLE_SECONDS": "2",
        "GLM53_POLL_SECONDS": "1",
    }
    process = subprocess.Popen(
        ["bash", str(LIFECYCLE), sys.executable, str(server), str(counter)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 8
    while time.time() < deadline and not (run_dir / "lifecycle/READY").is_file():
        if process.poll() is not None:
            raise AssertionError("watchdog test server exited before readiness")
        time.sleep(0.05)
    assert (run_dir / "lifecycle/READY").is_file()
    return process, run_dir


def test_health_only_releases_but_inference_counter_growth_keeps_server(
    tmp_path: Path,
) -> None:
    idle, idle_dir = _start(tmp_path, "idle")
    for _ in range(3):
        assert urllib.request.urlopen("http://127.0.0.1:8000/health").status == 200
        time.sleep(0.2)
    assert idle.wait(timeout=6) == 0
    assert (idle_dir / "lifecycle/IDLE-TIMEOUT").is_file()
    assert not (idle_dir / "lifecycle/MODEL-TRAFFIC.json").exists()

    active, active_dir = _start(tmp_path, "active")
    try:
        for _ in range(5):
            request = urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions", data=b"{}", method="POST"
            )
            assert urllib.request.urlopen(request).status == 200
            time.sleep(0.7)
            assert active.poll() is None
        assert (active_dir / "lifecycle/MODEL-TRAFFIC.json").is_file()
        (active_dir / "lifecycle/DRAIN").write_text("test\n")
        assert active.wait(timeout=5) == 0
        assert not (active_dir / "lifecycle/IDLE-TIMEOUT").exists()
    finally:
        if active.poll() is None:
            active.terminate()
            active.wait(timeout=5)


def test_production_watchdog_defaults_to_600_seconds_and_cannot_be_extended() -> None:
    script = LIFECYCLE.read_text()
    assert "IDLE_SECONDS=${GLM53_IDLE_SECONDS:-600}" in script
    assert "(( IDLE_SECONDS <= 600 ))" in script
    assert "local_sglang_inference_metrics" in script
    assert "traffic-stream-1" not in script


def test_cluster_canary_is_cpu_only_non_scored_and_exact_source() -> None:
    value = cluster_canary.render(ROOT)
    source, job = value["objects"]["items"]
    pod = job["spec"]["template"]["spec"]
    assert value["status"] == "READY_NON_SCORED"
    assert value["launch_authorized"] is False
    assert source["data"]["lifecycle.sh"] == LIFECYCLE.read_text()
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    assert "gpu" not in str(pod["containers"][0]["resources"]).lower()
    assert "task_instance_session_verifier_scoring_calls" in source["data"]["probe.py"]
    assert "FLEET_API_KEY" not in source["data"]["probe.py"]
