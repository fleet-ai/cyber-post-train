"""Run the model-free CVE-Bench runtime and grader qualification contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def bind_source(protocol: dict, checkout: Path) -> dict:
    benchmark = protocol["benchmarks"]["cvebench_zero_day"]
    harness = benchmark.get("harness", {})
    run_binding = harness.get("official_run_script", {})
    if harness.get("kali_size") != "large":
        raise ValueError("cvebench_kali_size_binding_mismatch")
    if (
        set(run_binding) != {"path", "sha256", "copy_solution_argv"}
        or run_binding.get("path") != "run"
        or run_binding.get("copy_solution_argv") != ["./run", "cp-solution", "{task_id}"]
    ):
        raise ValueError("cvebench_official_run_binding_mismatch")
    bindings = benchmark["bindings"]
    bound = {
        run_binding["path"]: run_binding["sha256"],
        "src/cvebench/prompts.py": bindings["system_prompt_sha256"],
        "src/cvebench/cvebench.py": bindings["tool_schema_sha256"],
        "src/cvebench/agent.py": bindings["runner_sha256"],
        "src/cvebench/utils.py": bindings["grader_sha256"],
    }
    if any(_sha((checkout / path).read_bytes()) != expected for path, expected in bound.items()):
        raise ValueError("official_source_binding_mismatch")
    return benchmark


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CVEBENCH_KALI_SIZE"] = "large"
    return environment


def _compose_images(checkout: Path, task_id: str) -> list[str]:
    result = subprocess.run(
        ["./run", "compose", task_id, "config", "--images"],
        cwd=checkout,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    images = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if not images:
        raise RuntimeError("cvebench_compose_image_set_empty")
    return images


def _image_lock(reference: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("cvebench_image_inspect_invalid")
    image_id = rows[0].get("Id")
    digests = rows[0].get("RepoDigests")
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or not isinstance(digests, list)
        or not digests
        or any(not isinstance(item, str) or "@sha256:" not in item for item in digests)
    ):
        raise RuntimeError("cvebench_image_digest_unavailable")
    return {
        "reference": reference,
        "image_id": image_id,
        "repository_digest": sorted(digests)[0],
    }


def _image_locks(checkout: Path, task_id: str) -> list[dict[str, str]]:
    return [_image_lock(reference) for reference in _compose_images(checkout, task_id)]


def validate_images(images: object) -> list[dict[str, str]]:
    if (
        not isinstance(images, list)
        or not images
        or any(not isinstance(row, dict) for row in images)
        or images != sorted(images, key=lambda row: str(row.get("reference")))
        or len({row.get("reference") for row in images}) != len(images)
        or any(
            set(row) != {"reference", "image_id", "repository_digest"}
            or not isinstance(row.get("reference"), str)
            or not row["reference"]
            or not isinstance(row.get("image_id"), str)
            or not row["image_id"].startswith("sha256:")
            or len(row["image_id"]) != 71
            or not isinstance(row.get("repository_digest"), str)
            or "@sha256:" not in row["repository_digest"]
            for row in images
        )
    ):
        raise ValueError("runtime_qualification_image_manifest_invalid")
    return images


def _checker_status(checkout: Path, task_id: str) -> bool:
    result = subprocess.run(
        [
            "./run",
            "compose",
            task_id,
            "exec",
            "-T",
            "target",
            "sh",
            "-c",
            "/evaluator/done.sh",
        ],
        cwd=checkout,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    start = result.stdout.find("{")
    if start < 0:
        raise RuntimeError("cvebench_checker_json_missing")
    value = json.loads(result.stdout[start:])
    if type(value.get("status")) is not bool:
        raise RuntimeError("cvebench_checker_status_invalid")
    return value["status"]


def qualify_task(
    checkout: Path,
    task_id: str,
    *,
    run_official_solution: bool,
) -> dict[str, object]:
    environment = _environment()
    subprocess.run(["./run", "pull", task_id], cwd=checkout, env=environment, check=True)
    before = _image_locks(checkout, task_id)
    primary_error: Exception | None = None
    checker_before: bool | None = None
    checker_after: bool | None = None
    try:
        subprocess.run(
            ["./run", "up", task_id, "--no-build"],
            cwd=checkout,
            env=environment,
            check=True,
        )
        checker_before = _checker_status(checkout, task_id)
        if checker_before:
            raise RuntimeError("cvebench_checker_false_positive_before_solution")
        if run_official_solution:
            subprocess.run(
                ["./run", "cp-solution", task_id],
                cwd=checkout,
                env=environment,
                check=True,
            )
            subprocess.run(
                ["./run", "compose", task_id, "exec", "-T", "agent", "/app/solution.sh"],
                cwd=checkout,
                env=environment,
                check=True,
                timeout=600,
            )
            checker_after = _checker_status(checkout, task_id)
            if not checker_after:
                raise RuntimeError("cvebench_official_solution_not_accepted")
    except Exception as error:
        primary_error = error
    cleanup = subprocess.run(
        ["./run", "down", task_id],
        cwd=checkout,
        env=environment,
        check=False,
    )
    if cleanup.returncode != 0:
        raise RuntimeError("cvebench_qualification_cleanup_failed") from primary_error
    if primary_error is not None:
        raise primary_error
    after = _image_locks(checkout, task_id)
    if after != before:
        raise RuntimeError("cvebench_image_binding_changed_during_qualification")
    return {
        "task_id": task_id,
        "startup_healthy": True,
        "checker_before_solution": checker_before,
        "official_solution_executed": run_official_solution,
        "checker_after_solution": checker_after,
        "images": before,
        "image_set_sha256": _sha(_canonical(before)),
    }


def qualify(protocol: dict, task_id: str, checkout: Path) -> dict[str, object]:
    benchmark = bind_source(protocol, checkout)
    if task_id not in benchmark["task_ids"]:
        raise ValueError("runtime_qualification_task_identity_invalid")
    if "FLEET_API_KEY" in os.environ or "OPENAI_API_KEY" in os.environ:
        raise ValueError("runtime_qualification_model_credential_present")
    subprocess.run(["uv", "sync", "--frozen"], cwd=checkout, check=True)
    inspect_runtime = json.loads(
        subprocess.check_output(
            [
                str(checkout / ".venv/bin/python"),
                "-c",
                (
                    "import hashlib,importlib.metadata,json,pathlib,inspect_ai;"
                    "p=pathlib.Path(inspect_ai.__file__).parent/'_eval/task/run.py';"
                    "print(json.dumps({'inspect_ai_version':"
                    "importlib.metadata.version('inspect-ai'),'inspect_limit_behavior_sha256':"
                    "'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()},sort_keys=True))"
                ),
            ],
            cwd=checkout,
            text=True,
        )
    )
    inspect_binding = benchmark["harness"]["inspect_ai"]
    if inspect_runtime != {
        "inspect_ai_version": inspect_binding["version"],
        "inspect_limit_behavior_sha256": inspect_binding["limit_behavior_source_sha256"],
    }:
        raise RuntimeError("cvebench_inspect_runtime_binding_mismatch")
    docker_versions = (
        subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split("|")
    )
    compose_version = subprocess.run(
        ["docker", "compose", "version", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        len(docker_versions) != 2
        or any(not item for item in docker_versions)
        or not compose_version
    ):
        raise RuntimeError("cvebench_toolchain_version_invalid")
    run_official_solution = task_id == protocol["operational_canary"]["task_id"]
    task = qualify_task(checkout, task_id, run_official_solution=run_official_solution)
    return {
        "schema": "external_ctf_runtime_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "qualification_contract_sha256": benchmark["runtime_qualification"]["contract_sha256"],
        "benchmark": "cvebench_zero_day",
        "task_id": task_id,
        "arm": "qualification",
        "status": "runtime_preflight_passed",
        "source_commit": benchmark["source"]["commit"],
        "kali_size": "large",
        "model_requests": 0,
        "docker_available": True,
        "compose_available": True,
        "toolchain": {
            "docker_client_version": docker_versions[0],
            "docker_server_version": docker_versions[1],
            "docker_compose_version": compose_version,
            **inspect_runtime,
        },
        "positive_grader_control": "passed" if run_official_solution else "not_run",
        "task": task,
        "task_sha256": _sha(_canonical(task)),
    }
