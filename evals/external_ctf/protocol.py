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
        if benchmark.get("pass_k") != 1:
            raise ValueError(f"{name} must retain its native one-attempt report")
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
    if tasks != benchmark["task_ids"]:
        raise ValueError(f"{name} official task roster drifted")
    if unavailable != benchmark.get("source_unavailable_task_ids", []):
        raise ValueError(f"{name} source availability drifted")
    return {
        "benchmark": name,
        "commit": source["commit"],
        "license": source["license"],
        "task_count": len(tasks),
        "source_unavailable_task_count": len(unavailable),
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "verified": True,
    }


def build_plan(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][name]
    unavailable = set(benchmark.get("source_unavailable_task_ids", []))
    cells = []
    for index, task_id in enumerate(benchmark["task_ids"]):
        if task_id in unavailable:
            continue
        first = "base" if index % 2 == 0 else "step_1000"
        for arm in (first, "step_1000" if first == "base" else "base"):
            cells.append({"task_id": task_id, "attempt": 0, "arm": arm})
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
        "arms": protocol["arms"],
        "cells": cells,
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
