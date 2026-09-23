"""Run one sealed external-CTF cell inside an isolated Linux sandbox."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

RESULT = Path("/workspace/external-ctf-result.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write(value: object) -> None:
    RESULT.write_bytes(_canonical(value) + b"\n")
    os.chmod(RESULT, 0o600)


def _protocol() -> dict:
    raw = base64.b64decode(os.environ.pop("EXTERNAL_CTF_PROTOCOL_B64"), validate=True)
    value = json.loads(raw)
    unsigned = {key: item for key, item in value.items() if key != "protocol_sha256"}
    if value.get("protocol_sha256") != _sha(_canonical(unsigned)):
        raise ValueError("protocol_digest_mismatch")
    return value


def _run_cvebench(protocol: dict, task_id: str, arm: str, checkout: Path) -> bool:
    benchmark = protocol["benchmarks"]["cvebench_zero_day"]
    bindings = benchmark["bindings"]
    bound = {
        "src/cvebench/prompts.py": bindings["system_prompt_sha256"],
        "src/cvebench/cvebench.py": bindings["tool_schema_sha256"],
        "src/cvebench/agent.py": bindings["runner_sha256"],
        "src/cvebench/utils.py": bindings["grader_sha256"],
    }
    if any(_sha((checkout / path).read_bytes()) != expected for path, expected in bound.items()):
        raise ValueError("official_source_binding_mismatch")
    subprocess.run(["uv", "sync", "--frozen"], cwd=checkout, check=True)
    log_dir = checkout / "sealed-logs"
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.pop("FLEET_API_KEY")
    command = [
        "./run",
        "eval",
        "-T",
        f"challenges={task_id}",
        "-T",
        "variants=zero_day",
        "-T",
        "max_messages=30",
        "--model",
        "openai/" + protocol["arms"][arm]["served_model"],
        "--model-base-url",
        "https://inference.flt.build/v1",
        "--max-connections",
        "1",
        "--max-retries",
        "0",
        "--max-samples",
        "1",
        "--message-limit",
        "30",
        "--time-limit",
        str(benchmark["budget"]["wall_seconds"]),
        "--max-tokens",
        str(benchmark["budget"]["max_output_tokens"]),
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--no-fail-on-error",
        "--no-log-realtime",
        "--display",
        "none",
        "--log-dir",
        str(log_dir),
    ]
    subprocess.run(command, cwd=checkout, env=env, check=True)
    script = """
from inspect_ai.log import list_eval_logs, read_eval_log
logs = list_eval_logs('sealed-logs')
assert len(logs) == 1
log = read_eval_log(logs[0])
assert log.status == 'success' and log.samples and len(log.samples) == 1
scores = log.samples[0].scores
assert scores and len(scores) == 1
value = next(iter(scores.values())).value
print('1' if float(value) > 0 else '0')
"""
    result = subprocess.run(
        [str(checkout / ".venv/bin/python"), "-c", script],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "1"


def main() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("linux_amd64_required")
    protocol = _protocol()
    benchmark_name = os.environ.pop("EXTERNAL_CTF_BENCHMARK")
    task_id = os.environ.pop("EXTERNAL_CTF_TASK_ID")
    arm = os.environ.pop("EXTERNAL_CTF_ARM")
    benchmark = protocol["benchmarks"][benchmark_name]
    if task_id not in benchmark["task_ids"] or arm not in protocol["arms"]:
        raise ValueError("cell_identity_invalid")
    if task_id in benchmark.get("source_unavailable_task_ids", []):
        raise ValueError("source_unavailable_cell_is_not_launchable")
    if task_id in benchmark.get("execution_unavailable_task_ids", []):
        raise ValueError("execution_unavailable_cell_is_not_launchable")
    checkout = Path("/workspace/external-ctf-source")
    source = benchmark["source"]
    subprocess.run(["git", "clone", "--quiet", source["repository"], str(checkout)], check=True)
    subprocess.run(["git", "checkout", "--quiet", source["commit"]], cwd=checkout, check=True)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if observed != source["commit"]:
        raise ValueError("source_commit_mismatch")
    if benchmark_name != "cvebench_zero_day":
        raise RuntimeError("opencode_adapter_not_yet_qualified")
    passed = _run_cvebench(protocol, task_id, arm, checkout)
    _write(
        {
            "schema": "external_ctf_cell_result_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "benchmark": benchmark_name,
            "task_id": task_id,
            "arm": arm,
            "status": "scored",
            "score": int(passed),
            "grader": benchmark["scoring"],
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _write(
            {
                "schema": "external_ctf_cell_result_v1",
                "status": "infrastructure_invalid",
                "error_class": type(error).__name__,
            }
        )
        raise
