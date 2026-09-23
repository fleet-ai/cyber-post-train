"""Seal score-blind coverage and paired-effect metadata for external CTF cells."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from . import execution_packet, tensorlake
from .protocol import build_plan, canonical, digest, load_protocol
from .tensorlake import _read_signed, cell_name


class AnalysisError(RuntimeError):
    """A terminal receipt or paired-analysis binding is invalid."""


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _terminal(
    state: Path,
    *,
    protocol: dict[str, Any],
    benchmark: str,
    task_index: int,
    task_id: str,
    arm: str,
    execution_packet_receipt_sha256: str,
) -> dict[str, Any] | None:
    name = cell_name(benchmark, task_index, arm)
    path = state / f"{name}.terminal.json"
    if not path.exists():
        return None
    try:
        value = _read_signed(path, "terminal")
    except RuntimeError as exc:
        raise AnalysisError("terminal_receipt_invalid") from exc
    if (
        value.get("schema") != "external_ctf_cell_terminal_v1"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or value.get("benchmark") != benchmark
        or value.get("task_index") != task_index
        or value.get("task_id") != task_id
        or value.get("arm") != arm
        or value.get("name") != name
        or value.get("outcome") not in {"accepted_model_outcome", "infrastructure_invalid"}
    ):
        raise AnalysisError("terminal_receipt_binding_mismatch")
    result = value.get("result")
    if value["outcome"] == "accepted_model_outcome":
        if (
            not isinstance(result, dict)
            or result.get("status") != "scored"
            or type(result.get("score")) is not int
            or result["score"] not in {0, 1}
            or value.get("infrastructure_error_class") is not None
        ):
            raise AnalysisError("accepted_terminal_result_invalid")
    elif result is not None and result.get("status") == "scored":
        raise AnalysisError("infrastructure_terminal_contains_score")
    return value


def _runtime_preflight_terminal(
    state: Path,
    *,
    protocol: dict[str, Any],
    task_index: int,
    task_id: str,
    execution_packet_receipt_sha256: str,
) -> dict[str, Any] | None:
    name = cell_name("cvebench_zero_day", task_index, "qualification")
    path = state / f"{name}.terminal.json"
    if not path.exists():
        return None
    try:
        value = _read_signed(path, "runtime_preflight_terminal")
    except RuntimeError as exc:
        raise AnalysisError("runtime_preflight_terminal_invalid") from exc
    if (
        value.get("schema") != "external_ctf_cell_terminal_v1"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or value.get("benchmark") != "cvebench_zero_day"
        or value.get("task_index") != task_index
        or value.get("task_id") != task_id
        or value.get("arm") != "qualification"
        or value.get("name") != name
        or value.get("outcome") not in {"runtime_preflight_passed", "infrastructure_invalid"}
    ):
        raise AnalysisError("runtime_preflight_terminal_binding_mismatch")
    result = value.get("result")
    if value["outcome"] == "runtime_preflight_passed":
        if (
            not isinstance(result, dict)
            or result.get("status") != "runtime_preflight_passed"
            or value.get("infrastructure_error_class") is not None
        ):
            raise AnalysisError("runtime_preflight_passed_result_invalid")
    elif isinstance(result, dict) and result.get("status") in {
        "scored",
        "runtime_preflight_passed",
    }:
        raise AnalysisError("runtime_preflight_infrastructure_terminal_contains_result")
    return value


def _bootstrap_interval(deltas: list[int], protocol: dict[str, Any]) -> dict[str, Any] | None:
    if not deltas:
        return None
    settings = protocol["analysis"]["confidence_interval"]
    generator = random.Random(settings["seed"])
    count = settings["resamples"]
    size = len(deltas)
    means = sorted(
        sum(deltas[generator.randrange(size)] for _ in range(size)) / size for _ in range(count)
    )
    tail = (1.0 - settings["level"]) / 2.0
    lower = means[int(tail * (count - 1))]
    upper = means[int((1.0 - tail) * (count - 1))]
    return {
        "method": settings["method"],
        "resamples": count,
        "seed": settings["seed"],
        "level": settings["level"],
        "lower": lower,
        "upper": upper,
    }


def analyze(
    protocol: dict[str, Any],
    state: Path,
    benchmark: str,
    *,
    scope: str,
    execution_packet_receipt_sha256: str,
) -> dict[str, Any]:
    if _DIGEST.fullmatch(execution_packet_receipt_sha256) is None:
        raise AnalysisError("execution_packet_receipt_invalid")
    expected_namespace = execution_packet_receipt_sha256.removeprefix("sha256:")
    if (
        state.is_symlink()
        or not state.is_dir()
        or state.name != expected_namespace
        or state.parent.name != "packets"
    ):
        raise AnalysisError("execution_packet_state_namespace_mismatch")
    plan = build_plan(protocol, benchmark)
    task_ids = protocol["benchmarks"][benchmark]["task_ids"]
    unavailable = set(plan["infrastructure_invalid_task_ids"])
    executable = [
        (index, task_id) for index, task_id in enumerate(task_ids) if task_id not in unavailable
    ]
    if scope == "operational_canary":
        canary = protocol["operational_canary"]
        if benchmark != canary["benchmark"]:
            raise AnalysisError("operational_canary_benchmark_mismatch")
        selected = [(canary["task_index"], canary["task_id"])]
        aggregate_eligible = False
    elif scope == "full_frozen_roster":
        selected = executable
        aggregate_eligible = plan["adapter_qualified"] is True
    else:
        raise AnalysisError("analysis_scope_invalid")

    tasks: list[dict[str, Any]] = []
    deltas: list[int] = []
    if scope == "full_frozen_roster":
        source_unavailable = set(protocol["benchmarks"][benchmark]["source_unavailable_task_ids"])
        for task_index, task_id in enumerate(task_ids):
            if task_id not in unavailable:
                continue
            tasks.append(
                {
                    "task_index": task_index,
                    "task_id": task_id,
                    "pair_status": (
                        "preflight_infrastructure_invalid_source_missing"
                        if task_id in source_unavailable
                        else "preflight_infrastructure_invalid_runtime_unavailable"
                    ),
                    "paired_delta": None,
                    "runtime_preflight_terminal_receipt_sha256": None,
                    "base_terminal_receipt_sha256": None,
                    "step_1000_terminal_receipt_sha256": None,
                }
            )
    for task_index, task_id in selected:
        runtime_preflight = (
            _runtime_preflight_terminal(
                state,
                protocol=protocol,
                task_index=task_index,
                task_id=task_id,
                execution_packet_receipt_sha256=execution_packet_receipt_sha256,
            )
            if benchmark == "cvebench_zero_day"
            else None
        )
        terminals = {
            arm: _terminal(
                state,
                protocol=protocol,
                benchmark=benchmark,
                task_index=task_index,
                task_id=task_id,
                arm=arm,
                execution_packet_receipt_sha256=execution_packet_receipt_sha256,
            )
            for arm in ("base", "step_1000")
        }
        if (
            benchmark == "cvebench_zero_day"
            and runtime_preflight is not None
            and runtime_preflight["outcome"] == "infrastructure_invalid"
        ):
            if any(value is not None for value in terminals.values()):
                raise AnalysisError("scored_terminal_exists_after_invalid_runtime_preflight")
            pair_status = "excluded_runtime_preflight_infrastructure_invalid"
            delta = None
        elif benchmark == "cvebench_zero_day" and runtime_preflight is None:
            if any(value is not None for value in terminals.values()):
                raise AnalysisError("scored_terminal_missing_runtime_preflight")
            pair_status = "pending"
            delta = None
        elif any(value is None for value in terminals.values()):
            pair_status = "pending"
            delta = None
        elif any(
            value["outcome"] == "infrastructure_invalid"  # type: ignore[index]
            for value in terminals.values()
        ):
            pair_status = "excluded_infrastructure_invalid"
            delta = None
        else:
            pair_status = "valid_paired_task"
            delta = (
                terminals["step_1000"]["result"]["score"]  # type: ignore[index]
                - terminals["base"]["result"]["score"]  # type: ignore[index]
            )
            deltas.append(delta)
        tasks.append(
            {
                "task_index": task_index,
                "task_id": task_id,
                "pair_status": pair_status,
                "paired_delta": delta,
                "runtime_preflight_terminal_receipt_sha256": (
                    None if runtime_preflight is None else runtime_preflight["receipt_sha256"]
                ),
                "base_terminal_receipt_sha256": (
                    None if terminals["base"] is None else terminals["base"]["receipt_sha256"]
                ),
                "step_1000_terminal_receipt_sha256": (
                    None
                    if terminals["step_1000"] is None
                    else terminals["step_1000"]["receipt_sha256"]
                ),
            }
        )

    pending = sum(row["pair_status"] == "pending" for row in tasks)
    invalid = sum(row["pair_status"] == "excluded_infrastructure_invalid" for row in tasks)
    runtime_preflight_invalid = sum(
        row["pair_status"] == "excluded_runtime_preflight_infrastructure_invalid" for row in tasks
    )
    interval = (
        _bootstrap_interval(deltas, protocol)
        if scope == "full_frozen_roster" and len(deltas) >= 2
        else None
    )
    unsigned = {
        "schema": "external_ctf_paired_analysis_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_packet_receipt_sha256": execution_packet_receipt_sha256,
        "plan_sha256": plan["plan_sha256"],
        "benchmark": benchmark,
        "scope": scope,
        "interpretation": (
            "operational_canary_not_full_aggregate"
            if scope == "operational_canary"
            else "full_frozen_roster"
        ),
        "aggregate_eligible": aggregate_eligible and not pending,
        "coverage": {
            "official_tasks": plan["official_task_count"],
            "benchmark_executable_tasks": plan["executable_task_count"],
            "selected_tasks": len(selected),
            "preflight_infrastructure_invalid_tasks": len(unavailable) + runtime_preflight_invalid,
            "runtime_preflight_infrastructure_invalid_tasks": runtime_preflight_invalid,
            "valid_paired_tasks": len(deltas),
            "infrastructure_invalid_paired_tasks": invalid,
            "pending_paired_tasks": pending,
        },
        "primary_effect": (None if not deltas else sum(deltas) / len(deltas)),
        "confidence_interval": interval,
        "infrastructure_invalid_counted_as_zero": False,
        "tasks": tasks,
    }
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    raw = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise AnalysisError("analysis_output_collision")
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--web-retry-execution", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--execution-packet", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("operational_canary", "full_frozen_roster"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    try:
        authority = tensorlake.capacity_authority(
            protocol,
            args.web_retry_execution,
            require_live_owner=False,
        )
        packet, state = execution_packet.load(
            args.execution_packet,
            protocol=protocol,
            authority=authority,
            require_current_source=False,
        )
    except (execution_packet.ExecutionPacketError, tensorlake.ExternalCtfError) as exc:
        raise AnalysisError(str(exc)) from exc
    result = analyze(
        protocol,
        state,
        args.benchmark,
        scope=args.scope,
        execution_packet_receipt_sha256=packet["receipt_sha256"],
    )
    _write_once(args.output, result)
    print(
        json.dumps(
            {
                "status": "sealed",
                "receipt_sha256": result["receipt_sha256"],
                "scope": result["scope"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
