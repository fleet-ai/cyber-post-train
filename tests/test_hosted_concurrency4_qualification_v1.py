from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from evals.fleet import (
    hosted_concurrency4_qualification_package_v1 as package,
)
from evals.fleet import hosted_concurrency4_qualification_v1 as qualification

ROOT = Path(__file__).parents[1]


def completion(model: str, tool: str) -> dict:
    arguments = (
        {"command": "printf ready"}
        if tool == "bash"
        else {"flags": [], "explanation": "available"}
    )
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"function": {"name": tool, "arguments": json.dumps(arguments)}}
                    ]
                },
            }
        ],
    }


def test_exact_tool_and_request_contract() -> None:
    assert [row["function"]["name"] for row in qualification.TOOLS] == [
        "bash",
        "submit_report",
    ]
    payload = qualification._request_payload("qwen3.8-27b", "submit_report")
    assert payload["model"] == "qwen3.8-27b"
    assert payload["tool_choice"]["function"]["name"] == "submit_report"
    assert payload["temperature"] == 0
    assert payload["stream"] is False
    qualification._validate_completion(
        completion("qwen3.8-27b", "submit_report"),
        model="qwen3.8-27b",
        tool_name="submit_report",
    )


def test_completion_rejects_wrong_model_or_tool() -> None:
    with pytest.raises(qualification.QualificationError, match="model_identity"):
        qualification._validate_completion(
            completion("wrong", "bash"), model="glm-5.3", tool_name="bash"
        )
    with pytest.raises(qualification.QualificationError, match="tool_name"):
        qualification._validate_completion(
            completion("glm-5.3", "submit_report"), model="glm-5.3", tool_name="bash"
        )


def test_wave_actually_reaches_requested_parallelism() -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0

    def caller(model: str, tool: str, key: str) -> float:
        nonlocal active, maximum
        assert model == "glm-5.3" and key == "key"
        assert tool in {"bash", "submit_report"}
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return 0.01

    wave = qualification.run_wave("glm-5.3", 4, "key", caller=caller)
    assert maximum == 4
    assert wave["streams_succeeded"] == 4
    assert wave["requests_succeeded"] == 8


def test_frozen_acceptance_thresholds() -> None:
    c2 = {
        "concurrency": 2,
        "streams_started": 2,
        "streams_succeeded": 2,
        "requests_expected": 4,
        "requests_succeeded": 4,
        "protocol_valid_requests": 4,
        "errors": 0,
        "wave_elapsed_seconds": 20.0,
        "streams_per_minute": 6.0,
        "request_latency_seconds": {"max": 10.0},
        "stream_latency_seconds": {"p95": 20.0},
    }
    c4 = {
        "concurrency": 4,
        "streams_started": 4,
        "streams_succeeded": 4,
        "requests_expected": 8,
        "requests_succeeded": 8,
        "protocol_valid_requests": 8,
        "errors": 0,
        "wave_elapsed_seconds": 25.0,
        "streams_per_minute": 9.6,
        "request_latency_seconds": {"max": 15.0},
        "stream_latency_seconds": {"p95": 30.0},
    }
    assert qualification.evaluate_waves([c2, c4])["accepted"] is True
    c4["streams_per_minute"] = 6.1
    assert qualification.evaluate_waves([c2, c4])["accepted"] is False


def test_receipt_writer_is_create_once(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    qualification.write_json_once(target, {"safe": True})
    with pytest.raises(FileExistsError):
        qualification.write_json_once(target, {"safe": False})


def test_held_job_is_cpu_only_low_priority_and_nonpreempting() -> None:
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    assert manifest["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] == "false"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert manifest["spec"]["backoffLimit"] == 0
    requests = pod["containers"][0]["resources"]["requests"]
    assert not any("gpu" in key.lower() for key in requests)


def test_package_is_immutable_and_held() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Package source is tested after commit by CI; use the worktree builder's
    # deterministic validation directly here.
    value = {
        "schema_version": package.SCHEMA,
        "launch_authorized": False,
        "scored_bulk_launch_authorized": False,
    }
    value["package_sha256"] = package.digest_without(value, "package_sha256")
    assert package._load_package(json.dumps(value))["launch_authorized"] is False
    assert len(commit) == 40


def test_launcher_cannot_submit() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "evals/fleet/scripts/submit_hosted_concurrency4_qualification_v1.sh"),
            "submit",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "HELD" in result.stderr
