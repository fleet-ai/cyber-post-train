"""Seal the immutable, no-launch authority for one external CTF campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.webexploitbench.tensorlake import collection_replica_set as replica_set
from evals.webexploitbench.tensorlake import shared_capacity_exclusion
from training import checkpoint_serving_route

from . import live_parity
from .protocol import (
    ROOT,
    canonical,
    cve_execution_schedule,
    file_digest,
    git_no_replace_env,
    load_protocol,
    observed_source,
)

SCHEMA = "external_ctf_execution_packet_v1"
BOUND_SCHEMA = "external_ctf_execution_packet_bound_v1"
QUIESCENCE_SCHEMA = "external_ctf_unmigrated_creator_quiescence_v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SOURCE_FILES = {
    "protocol": ROOT / "evals/external_ctf/protocol.py",
    "coordinator": ROOT / "evals/external_ctf/tensorlake.py",
    "worker": ROOT / "evals/external_ctf/worker.py",
    "analyzer": ROOT / "evals/external_ctf/analyze.py",
    "live_parity": ROOT / "evals/external_ctf/live_parity.py",
    "execution_packet": ROOT / "evals/external_ctf/execution_packet.py",
    "checkpoint_serving_route": ROOT / "training/checkpoint_serving_route.py",
    "base_clone_intent": (
        ROOT / "configs/evaluation/qwen38-external-ctf-base-c1-clone-intent-v1.json"
    ),
    "web_collection_launcher": (ROOT / "evals/webexploitbench/tensorlake/collection_launcher.py"),
    "web_collection_replica_set": (
        ROOT / "evals/webexploitbench/tensorlake/collection_replica_set.py"
    ),
    "web_collection_replica_retry": (
        ROOT / "evals/webexploitbench/tensorlake/collection_replica_retry.py"
    ),
    "web_collection_replica_pump": (
        ROOT / "evals/webexploitbench/tensorlake/collection_replica_pump.py"
    ),
    "web_collection_snapshot_export": (
        ROOT / "evals/webexploitbench/tensorlake/collection_snapshot_export.py"
    ),
    "web_shared_capacity_exclusion": (
        ROOT / "evals/webexploitbench/tensorlake/shared_capacity_exclusion.py"
    ),
}
UNMIGRATED_CREATORS = {
    "legacy_controller": ROOT / "evals/webexploitbench/tensorlake/controller.py",
    "source_snapshot_refresh": (
        ROOT / "evals/webexploitbench/tensorlake/source_snapshot_refresh.py"
    ),
    "collection_snapshot_qualification": (
        ROOT / "evals/webexploitbench/tensorlake/collection_snapshot_qualification.py"
    ),
}


class ExecutionPacketError(RuntimeError):
    """A launch packet or one of its immutable prerequisites is invalid."""


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return {**unsigned, "receipt_sha256": _digest(unsigned)}


def _write_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    signed = _signed(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(signed) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return signed


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ExecutionPacketError(label + "_file_invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionPacketError(label + "_file_invalid") from exc
    if (
        not isinstance(value, dict)
        or raw != canonical(value) + b"\n"
        or value.get("receipt_sha256")
        != _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
    ):
        raise ExecutionPacketError(label + "_receipt_invalid")
    return value, raw


def _stable_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ExecutionPacketError(label + "_file_invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionPacketError(label + "_file_invalid") from exc
    if not isinstance(value, dict) or path.read_bytes() != raw:
        raise ExecutionPacketError(label + "_file_invalid")
    return value, raw


def _clone_serving_evidence(
    *,
    protocol: dict[str, Any],
    clone_intent_path: Path,
    clone_plan_path: Path,
    clone_result_path: Path,
    clone_resume_preflight_path: Path,
    clone_lifecycle_result_path: Path,
) -> dict[str, Any]:
    try:
        intent = checkpoint_serving_route.load_external_ctf_clone_intent(clone_intent_path)
    except checkpoint_serving_route.RouteError as exc:
        raise ExecutionPacketError(str(exc)) from exc
    plan, plan_raw = _stable_json(clone_plan_path, "base_clone_plan")
    result, result_raw = _stable_json(clone_result_path, "base_clone_result")
    lifecycle, lifecycle_raw = _stable_json(clone_lifecycle_result_path, "base_clone_lifecycle")
    try:
        checkpoint_serving_route._validate_plan(plan)  # noqa: SLF001
        resume_raw, _resume_bytes = _stable_json(
            clone_resume_preflight_path, "base_clone_resume_preflight"
        )
        resume = checkpoint_serving_route.load_external_ctf_resume_preflight(
            clone_resume_preflight_path,
            clone_intent_path=clone_intent_path,
            plan_path=clone_plan_path,
            result_path=clone_result_path,
            now=checkpoint_serving_route._timestamp(resume_raw.get("observed_at")),  # noqa: SLF001
        )
    except checkpoint_serving_route.RouteError as exc:
        raise ExecutionPacketError(str(exc)) from exc
    registration = plan.get("registration")
    registration_spec = registration.get("spec") if isinstance(registration, dict) else None
    model = registration_spec.get("model") if isinstance(registration_spec, dict) else None
    result_unsigned = {key: item for key, item in result.items() if key != "receipt_sha256"}
    lifecycle_unsigned = {key: item for key, item in lifecycle.items() if key != "receipt_sha256"}
    immediate_capacity = lifecycle.get("immediate_capacity")
    base = protocol["arms"]["base"]
    candidate = protocol["arms"]["step_1000"]
    if (
        plan.get("external_ctf_clone_intent_sha256") != intent["intent_sha256"]
        or not isinstance(registration, dict)
        or registration.get("id") != base["served_model"]
        or not isinstance(model, dict)
        or model.get("revision") != base["model_revision"]
        or model.get("sourcePath") != base["source_path"]
        or plan.get("normalized_contract_sha256") != base["serving_runtime_sha256"]
        or result.get("receipt_sha256") != checkpoint_serving_route._digest(result_unsigned)  # noqa: SLF001
        or result.get("plan_sha256") != plan.get("plan_sha256")
        or result.get("registration_sha256") != plan.get("registration_sha256")
        or result.get("model_id") != base["served_model"]
        or result.get("phase") != "paused"
        or result.get("post_attempts") != 1
        or resume.get("target", {}).get("id") != base["served_model"]
        or resume.get("candidate", {}).get("id") != candidate["served_model"]
        or lifecycle.get("receipt_sha256") != checkpoint_serving_route._digest(lifecycle_unsigned)  # noqa: SLF001
        or lifecycle.get("schema") != "external_ctf_base_clone_lifecycle_v1"
        or lifecycle.get("model_id") != base["served_model"]
        or lifecycle.get("action") != "resume"
        or lifecycle.get("external_ctf_resume_preflight_receipt_sha256") != resume["receipt_sha256"]
        or lifecycle.get("before_phase") != "paused"
        or lifecycle.get("before_resource_version")
        != resume.get("target", {}).get("resource_version")
        or lifecycle.get("accepted_phase") not in {"resuming", "ready"}
        or not isinstance(lifecycle.get("accepted_resource_version"), str)
        or not lifecycle.get("accepted_resource_version")
        or lifecycle.get("target_readback") != resume.get("target")
        or lifecycle.get("candidate_readback") != resume.get("candidate")
        or not isinstance(immediate_capacity, dict)
        or immediate_capacity.get("sha256")
        != checkpoint_serving_route.gpu_capacity._digest(  # noqa: SLF001
            {key: item for key, item in immediate_capacity.items() if key != "sha256"}
        )
        or lifecycle.get("immediate_capacity_sha256")
        != "sha256:" + str(immediate_capacity.get("sha256"))
        or immediate_capacity.get("qualified") is not True
        or immediate_capacity.get("limits")
        != {
            "nodes": checkpoint_serving_route.EXTERNAL_CTF_CAPACITY_MAX_NODES,
            "gpus": checkpoint_serving_route.EXTERNAL_CTF_CAPACITY_MAX_GPUS,
        }
        or immediate_capacity.get("planned") != {"nodes": 1, "gpus": 8}
    ):
        raise ExecutionPacketError("base_clone_serving_evidence_binding_invalid")
    return {
        "intent": {
            "path": str(clone_intent_path.resolve()),
            "file_sha256": file_digest(clone_intent_path.read_bytes()),
            "intent_sha256": intent["intent_sha256"],
        },
        "plan": {
            "path": str(clone_plan_path.resolve()),
            "file_sha256": file_digest(plan_raw),
            "plan_sha256": plan["plan_sha256"],
            "registration_sha256": plan["registration_sha256"],
        },
        "create_result": {
            "path": str(clone_result_path.resolve()),
            "file_sha256": file_digest(result_raw),
            "receipt_sha256": result["receipt_sha256"],
        },
        "resume_preflight": {
            "path": str(clone_resume_preflight_path.resolve()),
            "file_sha256": file_digest(clone_resume_preflight_path.read_bytes()),
            "receipt_sha256": resume["receipt_sha256"],
        },
        "resume_result": {
            "path": str(clone_lifecycle_result_path.resolve()),
            "file_sha256": file_digest(lifecycle_raw),
            "receipt_sha256": lifecycle["receipt_sha256"],
        },
    }


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
        env=git_no_replace_env(),
    ).strip()


def _source_identity(*, require_clean: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise ExecutionPacketError("repository_not_clean")
    origin = _git("remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") != "https://github.com/fleet-ai/cyber-post-train":
        raise ExecutionPacketError("repository_origin_invalid")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "origin": origin,
        "source_sha256": {
            name: file_digest(path.read_bytes()) for name, path in sorted(SOURCE_FILES.items())
        },
    }


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ExecutionPacketError("timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExecutionPacketError("timestamp_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExecutionPacketError("timestamp_invalid")
    return parsed.astimezone(UTC)


def _process_matches() -> list[int]:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except subprocess.SubprocessError as exc:
        raise ExecutionPacketError("process_inventory_failed") from exc
    script_tokens: set[str] = set()
    module_tokens: set[str] = set()
    for path in UNMIGRATED_CREATORS.values():
        script_tokens.update({str(path), str(path.relative_to(ROOT)), path.name})
        module_tokens.add(".".join(path.relative_to(ROOT).with_suffix("").parts))
    matches: list[int] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            command = shlex.split(fields[1])
        except ValueError:
            command = fields[1].split()
        if any(token in script_tokens or token in module_tokens for token in command):
            matches.append(int(fields[0]))
    return sorted(matches)


def seal_quiescence(
    *,
    output: Path,
    capacity_successor_receipt_sha256: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    if _DIGEST.fullmatch(capacity_successor_receipt_sha256) is None:
        raise ExecutionPacketError("quiescence_binding_invalid")
    observed_host = socket.gethostname()
    if not observed_host or observed_host.strip() != observed_host:
        raise ExecutionPacketError("quiescence_host_invalid")
    matches = _process_matches()
    if matches:
        raise ExecutionPacketError("unmigrated_tensorlake_creator_is_running")
    now = datetime.now(UTC) if observed_at is None else observed_at.astimezone(UTC)
    return _write_once(
        output,
        {
            "schema": QUIESCENCE_SCHEMA,
            "status": "reviewed_prohibited_and_absent",
            "observed_at": now.isoformat().replace("+00:00", "Z"),
            "capacity_successor_receipt_sha256": capacity_successor_receipt_sha256,
            "reviewed_host_scope": [observed_host],
            "active_process_matches": [],
            "prohibition": "entrypoints_must_remain_absent_until_campaign_terminal",
            "unmigrated_creator_source_sha256": {
                name: file_digest(path.read_bytes())
                for name, path in sorted(UNMIGRATED_CREATORS.items())
            },
        },
    )


def _load_quiescence(
    path: Path,
    *,
    capacity_successor_receipt_sha256: str,
    now: datetime,
) -> dict[str, Any]:
    value, _raw = _read(path, "creator_quiescence")
    expected_sources = {
        name: file_digest(source.read_bytes())
        for name, source in sorted(UNMIGRATED_CREATORS.items())
    }
    age = (now - _timestamp(value.get("observed_at"))).total_seconds()
    if (
        value.get("schema") != QUIESCENCE_SCHEMA
        or value.get("status") != "reviewed_prohibited_and_absent"
        or value.get("capacity_successor_receipt_sha256") != capacity_successor_receipt_sha256
        or value.get("reviewed_host_scope") != [socket.gethostname()]
        or value.get("active_process_matches") != []
        or value.get("prohibition") != "entrypoints_must_remain_absent_until_campaign_terminal"
        or value.get("unmigrated_creator_source_sha256") != expected_sources
        or age < -60
        or age > 300
        or _process_matches()
    ):
        raise ExecutionPacketError("creator_quiescence_invalid_or_stale")
    return value


def _protocol_root(authority: dict[str, Any]) -> Path:
    root = Path(authority["external_state"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _packet_state(root: Path, receipt_sha256: str) -> Path:
    if _DIGEST.fullmatch(receipt_sha256) is None:
        raise ExecutionPacketError("packet_receipt_invalid")
    state = root / "packets" / receipt_sha256.removeprefix("sha256:")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.is_symlink() or state.resolve() != state or state.stat().st_mode & 0o077:
        raise ExecutionPacketError("packet_state_invalid")
    return state


def _initial_state_absent(root: Path, external_names: set[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.startswith(name + ".") for name in external_names):
            raise ExecutionPacketError("external_campaign_state_not_initial")
    if (root / "SET_EXTERNAL_CTF_EXECUTION_PACKET_BOUND.json").exists():
        raise ExecutionPacketError("external_execution_packet_already_bound")


def _stable_authority(original: dict[str, Any], refreshed: dict[str, Any]) -> None:
    fields = (
        "state",
        "external_state",
        "snapshot_id",
        "owned_names",
        "retry_execution_receipt_sha256",
        "capacity_successor_receipt_sha256",
        "capacity_successor_state_receipt_sha256",
        "shared_capacity_roster_receipt_sha256",
        "live_owner_receipt_sha256",
    )
    if any(original.get(field) != refreshed.get(field) for field in fields):
        raise ExecutionPacketError("shared_capacity_authority_changed_during_packet_seal")


def _seal_under_shared_lock(
    *,
    protocol: dict[str, Any],
    authority: dict[str, Any],
    retry_execution_path: Path,
    tensorlake: Any,
    source: dict[str, Any],
    official_sources: dict[str, Any],
    base_clone_serving: dict[str, Any],
    parity: dict[str, Any],
    parity_bytes: bytes,
    quiescence: dict[str, Any],
    quiescence_bytes: bytes,
    retry_bytes: bytes,
    protocol_bytes: bytes,
    schedule: list[dict[str, Any]],
    current: datetime,
    output: Path,
) -> dict[str, Any]:
    refreshed = tensorlake.capacity_authority(
        protocol, retry_execution_path, require_live_owner=True
    )
    _stable_authority(authority, refreshed)
    root = _protocol_root(refreshed)
    try:
        prohibition = shared_capacity_exclusion.load_prohibition(
            state=refreshed["state"],
            capacity_successor_receipt_sha256=refreshed["capacity_successor_receipt_sha256"],
        )
    except shared_capacity_exclusion.SharedCapacityExclusionError as exc:
        raise ExecutionPacketError(str(exc)) from exc
    expected_prohibition_sources = {
        name: file_digest(path.read_bytes()) for name, path in sorted(UNMIGRATED_CREATORS.items())
    }
    if prohibition.get("source_sha256") != expected_prohibition_sources:
        raise ExecutionPacketError("unmigrated_creator_prohibition_source_drifted")
    if _process_matches():
        raise ExecutionPacketError("unmigrated_tensorlake_creator_is_running")
    external_names = tensorlake.external_names(protocol)
    _initial_state_absent(root, external_names)
    client = tensorlake._client()  # noqa: SLF001
    rows = client.inventory()
    active_rows = [row for row in rows if row.get("status") != "terminated"]
    if any(row.get("name") in external_names for row in active_rows):
        raise ExecutionPacketError("external_provider_name_already_active")
    try:
        active = replica_set.shared_project_capacity_count(
            rows, refreshed["owned_names"], refreshed["state"]
        )
    except replica_set.CollectionReplicaSetError as exc:
        raise ExecutionPacketError(str(exc)) from exc
    if active >= replica_set.PROJECT_ACTIVE_SANDBOX_LIMIT:
        raise ExecutionPacketError("shared_project_active_sandbox_limit")
    packet = _signed(
        {
            "schema": SCHEMA,
            "status": "sealed_no_launch",
            "sealed_at": current.isoformat().replace("+00:00", "Z"),
            "protocol": {
                "receipt_sha256": protocol["protocol_sha256"],
                "file_sha256": file_digest(protocol_bytes),
                "execution_benchmark": protocol["execution_benchmark"],
            },
            "repository": source,
            "official_sources": official_sources,
            "models": protocol["arms"],
            "base_clone_serving": base_clone_serving,
            "live_parity": {
                "receipt": parity,
                "file_sha256": file_digest(parity_bytes),
            },
            "shared_capacity": {
                "limit": replica_set.PROJECT_ACTIVE_SANDBOX_LIMIT,
                "active_at_seal": active,
                "snapshot_id": refreshed["snapshot_id"],
                "retry_execution_file_sha256": file_digest(retry_bytes),
                "retry_execution_receipt_sha256": refreshed["retry_execution_receipt_sha256"],
                "capacity_successor_receipt_sha256": refreshed["capacity_successor_receipt_sha256"],
                "capacity_successor_state_receipt_sha256": refreshed[
                    "capacity_successor_state_receipt_sha256"
                ],
                "shared_capacity_roster_receipt_sha256": refreshed[
                    "shared_capacity_roster_receipt_sha256"
                ],
                "live_owner_receipt_sha256": refreshed["live_owner_receipt_sha256"],
                "pending_reservation_schema": replica_set.SHARED_CAPACITY_RESERVATION_SCHEMA,
                "owned_names_sha256": _digest(sorted(refreshed["owned_names"])),
            },
            "creator_quiescence": {
                "receipt": quiescence,
                "file_sha256": file_digest(quiescence_bytes),
                "prohibition_receipt": prohibition,
            },
            "execution": {
                "state_namespace_policy": "protocol_root/packets/packet_receipt_sha256",
                "max_parallel_cells": 1,
                "task_scoped_runtime_preflight_before_each_pair": True,
                "task5_preflight_requires_official_solution_positive_control": True,
                "scored_starts_require_fresh_packet_bound_route_and_fleet_team_preflight": True,
                "ordered_cells": schedule,
                "ordered_cells_sha256": protocol["execution_schedule"]["ordered_cells_sha256"],
                "pause_after_cell_count": protocol["execution_schedule"]["pause_after_cell_count"],
                "score_blind_until_terminal_analysis": True,
            },
            "initial_state": {
                "provider_external_active_names": [],
                "local_external_cell_receipts": [],
            },
            "external_mutations_performed": 0,
        }
    )
    if output.exists():
        raise ExecutionPacketError("execution_packet_output_exists")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(packet) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    bound = _write_once(
        root / "SET_EXTERNAL_CTF_EXECUTION_PACKET_BOUND.json",
        {
            "schema": BOUND_SCHEMA,
            "status": "bound_create_once",
            "protocol_sha256": protocol["protocol_sha256"],
            "packet_path": str(output.resolve()),
            "packet_file_sha256": file_digest(output.read_bytes()),
            "packet_receipt_sha256": packet["receipt_sha256"],
            "capacity_successor_receipt_sha256": refreshed["capacity_successor_receipt_sha256"],
            "unmigrated_creator_prohibition_receipt_sha256": prohibition["receipt_sha256"],
        },
    )
    _packet_state(root, packet["receipt_sha256"])
    return {**packet, "bound_receipt_sha256": bound["receipt_sha256"]}


def seal(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    live_parity_path: Path,
    quiescence_path: Path,
    clone_intent_path: Path,
    clone_plan_path: Path,
    clone_result_path: Path,
    clone_resume_preflight_path: Path,
    clone_lifecycle_result_path: Path,
    source_checkouts: dict[str, Path],
    output: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    from . import tensorlake

    protocol = load_protocol(protocol_path)
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    source = _source_identity(require_clean=True)
    parity = live_parity.load(live_parity_path, protocol=protocol, now=current)
    authority = tensorlake.capacity_authority(
        protocol, retry_execution_path, require_live_owner=True
    )
    quiescence = _load_quiescence(
        quiescence_path,
        capacity_successor_receipt_sha256=authority["capacity_successor_receipt_sha256"],
        now=current,
    )
    expected_checkouts = {"cvebench_zero_day", "nyu_ctf_web_test", "cybench_web"}
    if set(source_checkouts) != expected_checkouts:
        raise ExecutionPacketError("official_source_checkout_set_invalid")
    official_sources = {
        name: observed_source(protocol, name, source_checkouts[name])
        for name in sorted(expected_checkouts)
    }
    base_clone_serving = _clone_serving_evidence(
        protocol=protocol,
        clone_intent_path=clone_intent_path,
        clone_plan_path=clone_plan_path,
        clone_result_path=clone_result_path,
        clone_resume_preflight_path=clone_resume_preflight_path,
        clone_lifecycle_result_path=clone_lifecycle_result_path,
    )
    protocol_bytes = protocol_path.read_bytes()
    retry_bytes = retry_execution_path.read_bytes()
    parity_bytes = live_parity_path.read_bytes()
    quiescence_bytes = quiescence_path.read_bytes()
    schedule = cve_execution_schedule(protocol)
    with replica_set._shared_tensorlake_create_lock(authority["state"]):  # noqa: SLF001
        return _seal_under_shared_lock(
            protocol=protocol,
            authority=authority,
            retry_execution_path=retry_execution_path,
            tensorlake=tensorlake,
            source=source,
            official_sources=official_sources,
            base_clone_serving=base_clone_serving,
            parity=parity,
            parity_bytes=parity_bytes,
            quiescence=quiescence,
            quiescence_bytes=quiescence_bytes,
            retry_bytes=retry_bytes,
            protocol_bytes=protocol_bytes,
            schedule=schedule,
            current=current,
            output=output,
        )


def load(
    path: Path,
    *,
    protocol: dict[str, Any],
    authority: dict[str, Any],
    require_current_source: bool = True,
) -> tuple[dict[str, Any], Path]:
    packet, raw = _read(path, "execution_packet")
    if (
        packet.get("schema") != SCHEMA
        or packet.get("status") != "sealed_no_launch"
        or packet.get("external_mutations_performed") != 0
        or packet.get("protocol", {}).get("receipt_sha256") != protocol["protocol_sha256"]
        or packet.get("protocol", {}).get("execution_benchmark") != protocol["execution_benchmark"]
        or packet.get("models") != protocol["arms"]
        or packet.get("execution", {}).get("ordered_cells") != cve_execution_schedule(protocol)
        or packet.get("execution", {}).get("ordered_cells_sha256")
        != protocol["execution_schedule"]["ordered_cells_sha256"]
        or packet.get("execution", {}).get("max_parallel_cells") != 1
    ):
        raise ExecutionPacketError("execution_packet_binding_invalid")
    clone = packet.get("base_clone_serving")
    base_provenance = protocol["arms"]["base"]["provenance"]
    if (
        not isinstance(clone, dict)
        or clone.get("intent", {}).get("intent_sha256") != base_provenance["clone_intent_sha256"]
        or clone.get("plan", {}).get("registration_sha256") is None
        or clone.get("create_result", {}).get("receipt_sha256") is None
        or clone.get("resume_preflight", {}).get("receipt_sha256") is None
        or clone.get("resume_result", {}).get("receipt_sha256") is None
    ):
        raise ExecutionPacketError("execution_packet_base_clone_binding_invalid")
    capacity = packet.get("shared_capacity")
    expected_capacity = {
        "snapshot_id": authority["snapshot_id"],
        "retry_execution_receipt_sha256": authority["retry_execution_receipt_sha256"],
        "capacity_successor_receipt_sha256": authority["capacity_successor_receipt_sha256"],
        "capacity_successor_state_receipt_sha256": authority[
            "capacity_successor_state_receipt_sha256"
        ],
        "shared_capacity_roster_receipt_sha256": authority["shared_capacity_roster_receipt_sha256"],
    }
    if not isinstance(capacity, dict) or any(
        capacity.get(key) != value for key, value in expected_capacity.items()
    ):
        raise ExecutionPacketError("execution_packet_capacity_binding_invalid")
    creator_binding = packet.get("creator_quiescence")
    try:
        prohibition = shared_capacity_exclusion.load_prohibition(
            state=authority["state"],
            capacity_successor_receipt_sha256=authority["capacity_successor_receipt_sha256"],
        )
    except shared_capacity_exclusion.SharedCapacityExclusionError as exc:
        raise ExecutionPacketError(str(exc)) from exc
    if (
        not isinstance(creator_binding, dict)
        or creator_binding.get("prohibition_receipt") != prohibition
    ):
        raise ExecutionPacketError("execution_packet_creator_prohibition_invalid")
    if require_current_source:
        source = _source_identity(require_clean=True)
        if packet.get("repository") != source or _process_matches():
            raise ExecutionPacketError("execution_packet_source_or_creator_drifted")
    parity_binding = packet.get("live_parity")
    if not isinstance(parity_binding, dict) or not isinstance(parity_binding.get("receipt"), dict):
        raise ExecutionPacketError("execution_packet_live_parity_invalid")
    observed = _timestamp(parity_binding["receipt"].get("observed_at"))
    live_parity.validate(parity_binding["receipt"], protocol=protocol, now=observed)
    root = _protocol_root(authority)
    bound, _bound_raw = _read(
        root / "SET_EXTERNAL_CTF_EXECUTION_PACKET_BOUND.json", "execution_packet_bound"
    )
    if (
        bound.get("schema") != BOUND_SCHEMA
        or bound.get("status") != "bound_create_once"
        or bound.get("protocol_sha256") != protocol["protocol_sha256"]
        or bound.get("packet_path") != str(path.resolve())
        or bound.get("packet_file_sha256") != file_digest(raw)
        or bound.get("packet_receipt_sha256") != packet["receipt_sha256"]
        or bound.get("capacity_successor_receipt_sha256")
        != authority["capacity_successor_receipt_sha256"]
        or bound.get("unmigrated_creator_prohibition_receipt_sha256")
        != prohibition["receipt_sha256"]
    ):
        raise ExecutionPacketError("execution_packet_bound_receipt_invalid")
    return packet, _packet_state(root, packet["receipt_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quiescence = sub.add_parser("seal-quiescence")
    quiescence.add_argument("--output", type=Path, required=True)
    quiescence.add_argument("--capacity-successor-receipt-sha256", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--protocol", type=Path, required=True)
    seal_parser.add_argument("--web-retry-execution", type=Path, required=True)
    seal_parser.add_argument("--live-parity", type=Path, required=True)
    seal_parser.add_argument("--creator-quiescence", type=Path, required=True)
    seal_parser.add_argument("--base-clone-intent", type=Path, required=True)
    seal_parser.add_argument("--base-clone-plan", type=Path, required=True)
    seal_parser.add_argument("--base-clone-result", type=Path, required=True)
    seal_parser.add_argument("--base-clone-resume-preflight", type=Path, required=True)
    seal_parser.add_argument("--base-clone-lifecycle-result", type=Path, required=True)
    seal_parser.add_argument("--cve-checkout", type=Path, required=True)
    seal_parser.add_argument("--nyu-checkout", type=Path, required=True)
    seal_parser.add_argument("--cybench-checkout", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal-quiescence":
        result = seal_quiescence(
            output=args.output,
            capacity_successor_receipt_sha256=args.capacity_successor_receipt_sha256,
        )
    else:
        result = seal(
            protocol_path=args.protocol,
            retry_execution_path=args.web_retry_execution,
            live_parity_path=args.live_parity,
            quiescence_path=args.creator_quiescence,
            clone_intent_path=args.base_clone_intent,
            clone_plan_path=args.base_clone_plan,
            clone_result_path=args.base_clone_result,
            clone_resume_preflight_path=args.base_clone_resume_preflight,
            clone_lifecycle_result_path=args.base_clone_lifecycle_result,
            source_checkouts={
                "cvebench_zero_day": args.cve_checkout,
                "nyu_ctf_web_test": args.nyu_checkout,
                "cybench_web": args.cybench_checkout,
            },
            output=args.output,
        )
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
