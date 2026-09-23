"""Validate the frozen CVE-Bench, NYU CTF and Cybench comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs/evaluation/qwen38-external-ctf-paired-v1.json"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": None, "seed": None}
RETRY = {
    "automatic_model_retry": False,
    "automatic_process_retry": False,
    "selective_retry": False,
}
BINDINGS = {
    "cvebench_zero_day": {
        "system_prompt_sha256": (
            "sha256:05d0fdccc00339bc89aa0d0d00d89172e1d600578ed3691723cac5af900cb2e2"
        ),
        "tool_schema_sha256": (
            "sha256:f78d8195e4b7ae009a16254e956966163f9e96a1bb96bc64a72bb0c649c74bf7"
        ),
        "runner_sha256": "sha256:bd210dea91d3d9c8fdfe89007c0ca7a4e5179ef419f14a62f3d8d1fa290196f0",
        "grader_sha256": "sha256:d07f1721a60f7ba09e9951a17d95f21a521081ac7bce2a49da82fa6ffc87da8c",
        "adapter_sha256": "sha256:f78d8195e4b7ae009a16254e956966163f9e96a1bb96bc64a72bb0c649c74bf7",
    },
    "nyu_ctf_web_test": {
        "system_prompt_sha256": (
            "sha256:8512560b0706f8859774bfe8e93c2c86ecf6a6a4603fe59db004fd04fcc738d1"
        ),
        "tool_schema_sha256": (
            "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
        ),
        "runner_sha256": "sha256:149a977a452608e4a14e55463b4c987f70341fd0cb13b79d35183a9b6068c59f",
        "grader_sha256": "sha256:149a977a452608e4a14e55463b4c987f70341fd0cb13b79d35183a9b6068c59f",
        "adapter_sha256": "sha256:7976411d3b5b8eacbb887b8e6148fc14680513b0d35f202fbd7a6205080a329c",
    },
    "cybench_web": {
        "system_prompt_sha256": (
            "sha256:8512560b0706f8859774bfe8e93c2c86ecf6a6a4603fe59db004fd04fcc738d1"
        ),
        "tool_schema_sha256": (
            "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
        ),
        "runner_sha256": "sha256:858e50807bfa6bbdbe653e9e07ba92ffbfd6decaa045dd11842693461713998b",
        "grader_sha256": "sha256:7bff10a348d99a0938f6e386af264a60add0fa008814e71b013af8771ea569ec",
        "adapter_sha256": "sha256:7976411d3b5b8eacbb887b8e6148fc14680513b0d35f202fbd7a6205080a329c",
    },
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(value)
    return value


def validate_protocol(value: dict[str, Any]) -> None:
    if value.get("schema") != "qwen38_external_ctf_paired_v1":
        raise ValueError("unsupported protocol schema")
    if value.get("data_policy") != "evaluation_only_never_training_or_tuning":
        raise ValueError("external benchmark data boundary is not closed")
    if value.get("protocol_sha256") != digest(
        {key: item for key, item in value.items() if key != "protocol_sha256"}
    ):
        raise ValueError("protocol digest mismatch")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"base", "step_1000"}:
        raise ValueError("protocol must bind exactly two model arms")
    expected_common = {
        "endpoint_origin_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "serving_runtime_sha256",
        "max_context_size",
        "inference_precision",
        "quantization",
    }
    for arm in arms.values():
        if not isinstance(arm, dict) or set(arm) != {
            "served_model",
            "model_revision",
            "model_artifact_sha256",
            *expected_common,
        }:
            raise ValueError("model arm fields drifted")
    for field in expected_common:
        if arms["base"][field] != arms["step_1000"][field]:
            raise ValueError(f"non-weight model setting differs across arms: {field}")
    if arms["base"]["model_artifact_sha256"] == arms["step_1000"]["model_artifact_sha256"]:
        raise ValueError("model arms do not bind different weights")
    execution = value.get("execution")
    if execution != {
        "required_host_os": "linux",
        "required_host_arch": "x86_64",
        "provider": "tensorlake_sandbox",
        "shared_capacity_limit": 100,
        "shared_create_lock": "state/tensorlake-create.lock",
        "worker_sha256": "sha256:c55832d7e69deb92b017f6c76807b722e7f6a03f67d5722ea7ca5abca08ad1f0",
        "coordinator_sha256": (
            "sha256:259b83fdc27d8ca5c0cec25411e13baba3c0964f07713c7d2a19adfb3d59332a"
        ),
        "sampling": SAMPLING,
        "retry": RETRY,
        "max_parallel_cells": 1,
    }:
        raise ValueError("execution controls drifted")
    for field, source in (
        ("worker_sha256", ROOT / "evals/external_ctf/worker.py"),
        ("coordinator_sha256", ROOT / "evals/external_ctf/tensorlake.py"),
    ):
        if execution[field] != file_digest(source.read_bytes()):
            raise ValueError(f"{field} source drifted")
    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != {
        "cvebench_zero_day",
        "nyu_ctf_web_test",
        "cybench_web",
    }:
        raise ValueError("benchmark set drifted")
    for name, benchmark in benchmarks.items():
        tasks = benchmark.get("task_ids")
        if not isinstance(tasks, list) or not tasks or len(tasks) != len(set(tasks)):
            raise ValueError(f"{name} task roster is empty or duplicated")
        if benchmark.get("task_count") != len(tasks):
            raise ValueError(f"{name} task count mismatch")
        if benchmark.get("task_ids_sha256") != file_digest(("\n".join(tasks) + "\n").encode()):
            raise ValueError(f"{name} task roster digest mismatch")
        source_unavailable = benchmark.get("source_unavailable_task_ids")
        execution_unavailable = benchmark.get("execution_unavailable_task_ids")
        if not all(
            isinstance(items, list) for items in (source_unavailable, execution_unavailable)
        ):
            raise ValueError(f"{name} availability roster is invalid")
        if (
            len(source_unavailable) != len(set(source_unavailable))
            or len(execution_unavailable) != len(set(execution_unavailable))
            or set(source_unavailable) & set(execution_unavailable)
            or not (set(source_unavailable) | set(execution_unavailable)) <= set(tasks)
        ):
            raise ValueError(f"{name} availability roster is invalid")
        if benchmark.get("pass_k") != 1:
            raise ValueError(f"{name} must retain its native one-attempt report")
        harness = benchmark.get("harness")
        if not isinstance(harness, dict) or harness.get("platform") != "linux/amd64":
            raise ValueError(f"{name} must fail closed outside linux/amd64")
        bindings = benchmark.get("bindings")
        required_bindings = {
            "system_prompt_sha256",
            "tool_schema_sha256",
            "runner_sha256",
            "grader_sha256",
            "adapter_sha256",
        }
        if (
            not isinstance(bindings, dict)
            or set(bindings) != required_bindings
            or bindings != BINDINGS[name]
        ):
            raise ValueError(f"{name} execution bindings drifted")
        if any(
            not isinstance(item, str) or not item.startswith("sha256:") or len(item) != 71
            for item in bindings.values()
        ):
            raise ValueError(f"{name} execution binding digest is invalid")
        if benchmark.get("scoring") not in {
            "official_deterministic_exploit_grader",
            "official_exact_flag",
        }:
            raise ValueError(f"{name} does not use an official deterministic grader")
        if benchmark.get("infrastructure_failures") != "reported_separately_never_model_failure":
            raise ValueError(f"{name} infrastructure-failure policy drifted")


def _git(checkout: Path, *args: str, raw: bool = False) -> bytes | str:
    result = subprocess.check_output(["git", "-C", str(checkout), *args])
    return result if raw else result.decode().strip()


def observed_source(protocol: dict[str, Any], name: str, checkout: Path) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][name]
    source = benchmark["source"]
    if _git(checkout, "rev-parse", "HEAD") != source["commit"]:
        raise ValueError(f"{name} checkout commit mismatch")
    remote = _git(checkout, "remote", "get-url", "origin")
    if remote.rstrip("/").removesuffix(".git") != source["repository"].rstrip("/").removesuffix(
        ".git"
    ):
        raise ValueError(f"{name} repository mismatch")
    license_bytes = _git(checkout, "show", f"{source['commit']}:LICENSE", raw=True)
    if file_digest(license_bytes) != source["license_sha256"]:
        raise ValueError(f"{name} license digest mismatch")

    if name == "cvebench_zero_day":
        text = _git(
            checkout,
            "ls-tree",
            "-d",
            "--name-only",
            f"{source['commit']}:src/critical/challenges",
        )
        tasks = str(text).splitlines()
        unavailable: list[str] = []
    elif name == "nyu_ctf_web_test":
        raw = _git(checkout, "show", f"{source['commit']}:test_dataset.json", raw=True)
        if file_digest(raw) != benchmark["source_manifest_sha256"]:
            raise ValueError("NYU test manifest digest mismatch")
        dataset = json.loads(raw)
        tasks = [task_id for task_id, row in dataset.items() if row.get("category") == "web"]
        unavailable = []
        execution_unavailable = []
        for task_id in tasks:
            task = dataset[task_id]
            challenge = json.loads((checkout / task["path"] / "challenge.json").read_bytes())
            if (
                not challenge.get("compose")
                or not (checkout / task["path"] / "docker-compose.yml").is_file()
            ):
                execution_unavailable.append(task_id)
    else:
        raw = _git(checkout, "show", f"{source['commit']}:task_list.txt", raw=True)
        if file_digest(raw) != benchmark["source_manifest_sha256"]:
            raise ValueError("Cybench task manifest digest mismatch")
        tasks = [line for line in raw.decode().splitlines() if "/web/" in line]
        unavailable = []
        for task in tasks:
            exists = (
                subprocess.run(
                    ["git", "-C", str(checkout), "cat-file", "-e", f"{source['commit']}:{task}"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if not exists:
                unavailable.append(task)
        execution_unavailable = []
    if name == "cvebench_zero_day":
        execution_unavailable = []
    if tasks != benchmark["task_ids"]:
        raise ValueError(f"{name} official task roster drifted")
    if unavailable != benchmark.get("source_unavailable_task_ids", []):
        raise ValueError(f"{name} source availability drifted")
    if execution_unavailable != benchmark.get("execution_unavailable_task_ids", []):
        raise ValueError(f"{name} execution availability drifted")
    return {
        "benchmark": name,
        "commit": source["commit"],
        "license": source["license"],
        "task_count": len(tasks),
        "source_unavailable_task_count": len(unavailable),
        "execution_unavailable_task_count": len(execution_unavailable),
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "verified": True,
    }


def build_plan(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][name]
    source_unavailable = set(benchmark.get("source_unavailable_task_ids", []))
    execution_unavailable = set(benchmark.get("execution_unavailable_task_ids", []))
    unavailable = source_unavailable | execution_unavailable
    cells = []
    for index, task_id in enumerate(benchmark["task_ids"]):
        first = "base" if index % 2 == 0 else "step_1000"
        for arm in (first, "step_1000" if first == "base" else "base"):
            cells.append(
                {
                    "task_id": task_id,
                    "attempt": 0,
                    "arm": arm,
                    "launchable": task_id not in unavailable,
                    "preflight_state": (
                        "ready"
                        if task_id not in unavailable
                        else (
                            "infra_invalid_source_missing"
                            if task_id in source_unavailable
                            else "infra_invalid_no_reproducible_runtime"
                        )
                    ),
                }
            )
    plan = {
        "schema": "qwen38_external_ctf_execution_plan_v1",
        "study_id": protocol["study_id"],
        "benchmark": name,
        "protocol_sha256": protocol["protocol_sha256"],
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "pass_k": benchmark["pass_k"],
        "scoring": benchmark["scoring"],
        "harness": benchmark["harness"],
        "budget": benchmark["budget"],
        "bindings": benchmark["bindings"],
        "execution": protocol["execution"],
        "arms": protocol["arms"],
        "cells": cells,
        "official_task_count": len(benchmark["task_ids"]),
        "executable_task_count": len(benchmark["task_ids"]) - len(unavailable),
        "infrastructure_invalid_task_count": len(unavailable),
        "infrastructure_invalid_task_ids": sorted(unavailable),
        "data_policy": protocol["data_policy"],
        "result_policy": "append_only_one_terminal_receipt_per_cell",
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def observe_models(protocol: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    if not key or key.strip() != key or any(character.isspace() for character in key):
        raise ValueError("FLEET_API_KEY is unavailable or malformed")

    def get(url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + key, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    account = get("https://orchestrator.fleetai.com/v1/account")
    if account.get("team_id") != FLEET_TEAM_ID or account.get("team_name") != "fleet":
        raise ValueError("Fleet account identity mismatch")
    models = get("https://inference.flt.build/v1/models")
    available = {row.get("id") for row in models.get("data", []) if isinstance(row, dict)}
    expected = {arm["served_model"] for arm in protocol["arms"].values()}
    if not expected <= available:
        raise ValueError("one or more frozen model routes are unavailable")
    return {"fleet_team_verified": True, "model_routes_verified": sorted(expected)}


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    verify = sub.add_parser("verify-source")
    verify.add_argument("--benchmark", required=True)
    verify.add_argument("--checkout", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--benchmark", required=True)
    plan.add_argument("--output", type=Path, required=True)
    sub.add_parser("observe-models")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "validate":
        value = {"protocol_sha256": protocol["protocol_sha256"], "verified": True}
    elif args.command == "verify-source":
        value = observed_source(protocol, args.benchmark, args.checkout)
    elif args.command == "plan":
        value = build_plan(protocol, args.benchmark)
        _write_once(args.output, value)
    else:
        value = observe_models(protocol)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
