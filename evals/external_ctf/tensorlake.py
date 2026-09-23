"""Create one external-CTF TensorLake cell under the shared WEB capacity lock."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import gzip
import json
import os
import re
import stat
import time
import urllib.error
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evals.webexploitbench.tensorlake import collection_replica_retry
from evals.webexploitbench.tensorlake import collection_replica_set as replica_set
from evals.webexploitbench.tensorlake.collection_launcher import (
    _network_policy_matches,
    _resources_match,
)
from evals.webexploitbench.tensorlake.collection_replica_set import PROJECT_ACTIVE_SANDBOX_LIMIT
from evals.webexploitbench.tensorlake.controller import API, TensorlakeClient

from . import live_parity
from .protocol import (
    ROOT,
    RUNTIME_QUALIFICATION_SOURCE_PATHS,
    canonical,
    cve_execution_schedule,
    file_digest,
    load_protocol,
    runtime_qualification_contract_sha256,
)

WORKER = Path(__file__).with_name("worker.py")
QUALIFICATION_BUNDLE_CORE = {
    "evals/external_ctf/analyze.py": ROOT / "evals/external_ctf/analyze.py",
    "evals/external_ctf/protocol.py": ROOT / "evals/external_ctf/protocol.py",
    "evals/external_ctf/tensorlake.py": ROOT / "evals/external_ctf/tensorlake.py",
    "evals/external_ctf/worker.py": ROOT / "evals/external_ctf/worker.py",
    **{
        path.relative_to(ROOT).as_posix(): path
        for paths in RUNTIME_QUALIFICATION_SOURCE_PATHS.values()
        for path in paths.values()
    },
}
WORKER_BOOTSTRAP = """\
import base64
import gzip
import hashlib
import json
import os
import pathlib
import runpy
import sys

bundle = os.environ.pop("QUALIFICATION_BUNDLE_B64", None)
if bundle is not None:
    compressed = base64.b64decode(bundle, validate=True)
    if "sha256:" + hashlib.sha256(compressed).hexdigest() != os.environ.pop(
        "QUALIFICATION_BUNDLE_SHA256"
    ):
        raise RuntimeError("qualification_bundle_digest_mismatch")
    raw = gzip.decompress(compressed)
    files = json.loads(raw)
    root = pathlib.Path("/workspace/external_ctf_runtime")
    if root.exists() or root.is_symlink():
        raise RuntimeError("qualification_bundle_root_exists")
    for package in (root / "evals", root / "evals/external_ctf"):
        package.mkdir(mode=0o700, parents=True)
        (package / "__init__.py").write_bytes(b"")
    for relative, encoded in files.items():
        path = pathlib.PurePosixPath(relative)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise RuntimeError("qualification_bundle_path_invalid")
        target = root.joinpath(*path.parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded, validate=True))
    sys.path.insert(0, str(root))
worker = base64.b64decode(os.environ.pop("WORKER_B64"), validate=True)
if "sha256:" + hashlib.sha256(worker).hexdigest() != os.environ.pop("WORKER_SHA256"):
    raise RuntimeError("worker_digest_mismatch")
path = pathlib.Path("/workspace/external_ctf_worker.py")
path.write_bytes(worker)
runpy.run_path(str(path), run_name="__main__")
"""
SCORED_ARMS = {"base": "b", "step_1000": "s1000"}
ARMS = {**SCORED_ARMS, "qualification": "qual"}
BENCHMARKS = {"cvebench_zero_day": "cve", "nyu_ctf_web_test": "nyu", "cybench_web": "cyb"}
BENCHMARK_TASK_COUNTS = {"cvebench_zero_day": 40, "nyu_ctf_web_test": 19, "cybench_web": 6}
QUALIFICATION_NAME_SUCCESSORS = {("cvebench_zero_day", 5): 2}
RESULT_RECONCILE_ATTEMPTS = 6
RESULT_RECONCILE_DELAY_SECONDS = 2
ABSENCE_RECONCILE_ATTEMPTS = 6
ABSENCE_RECONCILE_DELAY_SECONDS = 2
DEFINITIVE_NO_CREATE_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 422})


class ExternalCtfError(RuntimeError):
    pass


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _qualification_bundle(protocol: dict[str, Any], benchmark: str) -> bytes:
    expected = protocol["benchmarks"][benchmark]["runtime_qualification"]["executor_source_sha256"]
    paths = RUNTIME_QUALIFICATION_SOURCE_PATHS[benchmark]
    observed = {label: file_digest(path.read_bytes()) for label, path in paths.items()}
    if any(expected.get(label) != value for label, value in observed.items()):
        raise ExternalCtfError("runtime_qualification_executor_source_drifted")
    bundle_paths = {
        **QUALIFICATION_BUNDLE_CORE,
        **{path.relative_to(ROOT).as_posix(): path for path in paths.values()},
    }
    files = {
        relative: base64.b64encode(path.read_bytes()).decode()
        for relative, path in sorted(bundle_paths.items())
    }
    return gzip.compress(canonical(files), compresslevel=9, mtime=0)


def base_external_names(protocol: dict[str, Any]) -> set[str]:
    scored = {
        cell_name(benchmark, index, arm)
        for benchmark, count in BENCHMARK_TASK_COUNTS.items()
        for index in range(count)
        for arm in SCORED_ARMS
    }
    qualifications = {
        f"extctf-{BENCHMARKS[benchmark]}-t{index:02d}-qual-v1"
        for benchmark, value in protocol["benchmarks"].items()
        for index, task_id in enumerate(value["task_ids"])
        if task_id
        not in set(value["source_unavailable_task_ids"])
        | set(value["execution_unavailable_task_ids"])
    }
    return scored | qualifications


def external_names(protocol: dict[str, Any]) -> set[str]:
    names = base_external_names(protocol)
    names.add(replica_set.SHARED_EXTERNAL_CTF_APPEND_ONLY_NAME)
    return names


def cell_name(benchmark: str, task_index: int, arm: str) -> str:
    if arm == "qualification":
        try:
            prefix = BENCHMARKS[benchmark]
            count = BENCHMARK_TASK_COUNTS[benchmark]
        except KeyError as error:
            raise ExternalCtfError("invalid_cell_identity") from error
        if 0 <= task_index < count:
            version = QUALIFICATION_NAME_SUCCESSORS.get((benchmark, task_index), 1)
            return f"extctf-{prefix}-t{task_index:02d}-qual-v{version}"
        raise ExternalCtfError("invalid_cell_identity")
    try:
        return f"extctf-{BENCHMARKS[benchmark]}-t{task_index:02d}-{SCORED_ARMS[arm]}-v1"
    except KeyError as error:
        raise ExternalCtfError("invalid_cell_identity") from error


def active_project_count(rows: list[dict[str, Any]], names: set[str]) -> int:
    try:
        return replica_set.shared_project_active_sandbox_count(rows, names)
    except replica_set.CollectionReplicaSetError as exc:
        raise ExternalCtfError(str(exc)) from exc


def capacity_authority(
    protocol: dict[str, Any],
    retry_execution_path: Path,
    *,
    require_live_owner: bool = False,
    require_roster_successor: bool = True,
) -> dict[str, Any]:
    try:
        receipt, loaded, upgrade, retry = collection_replica_retry.load_execution(
            retry_execution_path
        )
        state = replica_set._global_state_root(  # noqa: SLF001
            Path(receipt["state_path"]), sealing=False
        )
        shared = retry.get("shared_capacity_roster")
        if not isinstance(shared, dict):
            raise replica_set.CollectionReplicaSetError("shared_capacity_roster_not_bound")
        rollout, exports = replica_set.authoritative_project_owned_sandbox_names(
            state=state,
            receipt=receipt,
            loaded=loaded,
            source_upgrade=upgrade,
            retry_execution=retry,
        )
        successor = retry.get("capacity_successor")
        if not isinstance(successor, dict):
            raise replica_set.CollectionReplicaSetError("capacity_successor_required")
        successor_state = collection_replica_retry.require_capacity_successor_bound(
            state,
            retry_execution_path,
            retry,
        )
    except (KeyError, OSError, replica_set.CollectionReplicaSetError) as exc:
        raise ExternalCtfError("shared_capacity_authority_invalid") from exc
    expected_base_external = base_external_names(protocol)
    expected_external = (
        external_names(protocol) if require_roster_successor else expected_base_external
    )
    try:
        base_roster, base_roster_names = replica_set.load_shared_capacity_roster(
            Path(shared["path"]),
            expected_file_sha256=shared["file_sha256"],
            expected_receipt_sha256=shared["receipt_sha256"],
        )
        roster, roster_names, roster_export_additions, roster_successor_state = (
            replica_set.effective_shared_capacity_roster(
                state=state,
                predecessor_path=Path(shared["path"]),
                expected_predecessor_file_sha256=shared["file_sha256"],
                expected_predecessor_receipt_sha256=shared["receipt_sha256"],
            )
        )
    except (KeyError, OSError, replica_set.CollectionReplicaSetError) as exc:
        raise ExternalCtfError("shared_capacity_roster_invalid") from exc
    if (
        base_roster_names != expected_base_external
        or roster_names != expected_external
        or roster_export_additions
        != (
            frozenset({replica_set.SHARED_SCORING_EXPORT_APPEND_ONLY_NAME})
            if require_roster_successor
            else frozenset()
        )
        or base_roster.get("external_namespace") != replica_set.SHARED_EXTERNAL_CTF_NAMESPACE
        or base_roster.get("external_name_version") != replica_set.SHARED_EXTERNAL_CTF_NAME_VERSION
        or require_roster_successor != (roster_successor_state is not None)
        or roster_successor_state is not None
        and (
            roster_successor_state.get("capacity_successor_receipt_sha256")
            != successor["receipt_sha256"]
            or roster_successor_state.get("capacity_successor_state_receipt_sha256")
            != successor_state["receipt_sha256"]
        )
        or retry.get("shared_project_active_sandbox_limit") != PROJECT_ACTIVE_SANDBOX_LIMIT
        or not expected_external <= rollout
        or not roster_export_additions <= exports
        or rollout & exports
    ):
        raise ExternalCtfError("shared_capacity_roster_binding_mismatch")
    source = successor.get("execution_source")
    source_sha = source.get("source_sha256") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("commit"), str)
        or not isinstance(source_sha, dict)
        or not isinstance(source_sha.get("collection_replica_pump"), str)
    ):
        raise ExternalCtfError("shared_capacity_source_binding_invalid")
    live_owner = None
    if require_live_owner:
        try:
            live_owner = replica_set.require_live_owner_receipt(
                state,
                replica_set_receipt_sha256=receipt["receipt_sha256"],
                retry_execution_receipt_sha256=retry["receipt_sha256"],
                capacity_successor_receipt_sha256=successor["receipt_sha256"],
                capacity_successor_state_receipt_sha256=successor_state["receipt_sha256"],
                shared_capacity_roster_receipt_sha256=base_roster["receipt_sha256"],
                execution_source_commit=source["commit"],
                pump_source_sha256=source_sha["collection_replica_pump"],
            )
        except replica_set.CollectionReplicaSetError as exc:
            raise ExternalCtfError("shared_capacity_live_owner_invalid") from exc
    return {
        "state": state,
        "external_state": _private_state(
            state
            / "external-ctf"
            / protocol["execution_benchmark"].replace("_", "-")
            / protocol["protocol_sha256"].removeprefix("sha256:")
        ),
        "snapshot_id": receipt["snapshot_id"],
        "owned_names": set(rollout | exports),
        "retry_execution_receipt_sha256": retry["receipt_sha256"],
        "capacity_successor_receipt_sha256": successor["receipt_sha256"],
        "capacity_successor_state_receipt_sha256": successor_state["receipt_sha256"],
        "shared_capacity_roster_receipt_sha256": roster["receipt_sha256"],
        "live_owner_receipt_sha256": (None if live_owner is None else live_owner["receipt_sha256"]),
    }


def seal_capacity_roster(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    return replica_set.seal_shared_capacity_roster(
        external_sandbox_names=sorted(base_external_names(protocol)),
        output_path=output_path,
    )


def seal_capacity_roster_successor(
    *,
    protocol_path: Path,
    predecessor_roster_path: Path,
    expected_predecessor_roster_file_sha256: str,
    expected_predecessor_roster_receipt_sha256: str,
    retired_terminal_path: Path,
    retired_release_path: Path,
    retired_capacity_release_path: Path,
    web_authority_path: Path,
    web_batch_path: Path,
    web_created_path: Path,
    web_release_path: Path,
    web_capacity_reserved_path: Path,
    web_capacity_release_path: Path,
    web_owner_ready_path: Path,
    web_owner_closed_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if any(
        os.environ.get(name)
        for name in (
            "TENSORLAKE_API_KEY",
            "OPENAI_API_KEY",
            "FLEET_API_KEY",
            "APOLLO_FLEET_API_KEY",
        )
    ):
        raise ExternalCtfError("capacity_roster_successor_seal_requires_no_credentials")
    protocol = load_protocol(protocol_path)
    if (
        cell_name("cvebench_zero_day", 5, "qualification")
        != replica_set.SHARED_EXTERNAL_CTF_APPEND_ONLY_NAME
        or external_names(protocol)
        != base_external_names(protocol) | {replica_set.SHARED_EXTERNAL_CTF_APPEND_ONLY_NAME}
        or replica_set.SHARED_SCORING_EXPORT_APPEND_ONLY_NAME
        != "wbe-p8-r00-t00-candidate-gpt-exp-v2"
    ):
        raise ExternalCtfError("capacity_roster_successor_identity_invalid")

    try:
        return replica_set.seal_shared_capacity_roster_successor(
            predecessor_path=predecessor_roster_path,
            expected_predecessor_file_sha256=expected_predecessor_roster_file_sha256,
            expected_predecessor_receipt_sha256=expected_predecessor_roster_receipt_sha256,
            external_retirement_evidence={
                "terminal": retired_terminal_path,
                "release": retired_release_path,
                "capacity_release": retired_capacity_release_path,
            },
            web_retirement_evidence={
                "authority": web_authority_path,
                "batch": web_batch_path,
                "created": web_created_path,
                "release": web_release_path,
                "capacity_reserved": web_capacity_reserved_path,
                "capacity_release": web_capacity_release_path,
                "owner_ready": web_owner_ready_path,
                "owner_closed": web_owner_closed_path,
            },
            output_path=output_path,
        )
    except (OSError, replica_set.CollectionReplicaSetError) as exc:
        raise ExternalCtfError("capacity_roster_successor_invalid") from exc


def bind_capacity_roster_successor(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    successor_roster_path: Path,
    expected_successor_roster_file_sha256: str,
    expected_successor_roster_receipt_sha256: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        receipt, _loaded, _upgrade, retry = collection_replica_retry.load_execution(
            retry_execution_path
        )
        state = replica_set._global_state_root(  # noqa: SLF001
            Path(receipt["state_path"]), sealing=False
        )
        shared = retry.get("shared_capacity_roster")
        if not isinstance(shared, dict):
            raise replica_set.CollectionReplicaSetError("shared_capacity_roster_not_bound")
    except (KeyError, OSError, replica_set.CollectionReplicaSetError) as exc:
        raise ExternalCtfError("shared_capacity_authority_invalid") from exc
    marker_path = state / replica_set.SHARED_CAPACITY_ROSTER_SUCCESSOR_BOUND_NAME
    if marker_path.exists():
        effective = capacity_authority(
            protocol,
            retry_execution_path,
            require_live_owner=False,
            require_roster_successor=True,
        )
        marker = replica_set.effective_shared_capacity_roster(
            state=state,
            predecessor_path=Path(shared["path"]),
            expected_predecessor_file_sha256=shared["file_sha256"],
            expected_predecessor_receipt_sha256=shared["receipt_sha256"],
        )[3]
        if (
            effective["shared_capacity_roster_receipt_sha256"]
            != expected_successor_roster_receipt_sha256
            or marker is None
        ):
            raise ExternalCtfError("capacity_roster_successor_bind_readback_invalid")
        return marker
    authority = capacity_authority(
        protocol,
        retry_execution_path,
        require_live_owner=True,
        require_roster_successor=False,
    )
    if authority["state"] != state:
        raise ExternalCtfError("shared_capacity_authority_changed")
    with replica_set._shared_tensorlake_create_lock(state):  # noqa: SLF001
        refreshed = capacity_authority(
            protocol,
            retry_execution_path,
            require_live_owner=True,
            require_roster_successor=False,
        )
        stable = {
            "state",
            "snapshot_id",
            "owned_names",
            "retry_execution_receipt_sha256",
            "capacity_successor_receipt_sha256",
            "capacity_successor_state_receipt_sha256",
            "shared_capacity_roster_receipt_sha256",
            "live_owner_receipt_sha256",
        }
        if any(refreshed.get(key) != authority.get(key) for key in stable):
            raise ExternalCtfError("shared_capacity_authority_changed")
        rows = _client().inventory()
        try:
            active = replica_set.shared_project_capacity_count(
                rows, refreshed["owned_names"], state
            )
        except replica_set.CollectionReplicaSetError as exc:
            raise ExternalCtfError(str(exc)) from exc
        additions = {
            replica_set.SHARED_EXTERNAL_CTF_APPEND_ONLY_NAME,
            replica_set.SHARED_SCORING_EXPORT_APPEND_ONLY_NAME,
        }
        if active != 0 or any(row.get("name") in additions for row in rows):
            raise ExternalCtfError("capacity_roster_successor_bind_requires_quiescence")
        marker = replica_set.bind_shared_capacity_roster_successor(
            state=state,
            successor_path=successor_roster_path,
            expected_successor_file_sha256=expected_successor_roster_file_sha256,
            expected_successor_receipt_sha256=expected_successor_roster_receipt_sha256,
            capacity_successor_receipt_sha256=refreshed["capacity_successor_receipt_sha256"],
            capacity_successor_state_receipt_sha256=refreshed[
                "capacity_successor_state_receipt_sha256"
            ],
            live_owner_receipt_sha256=refreshed["live_owner_receipt_sha256"],
            inventory_sha256=file_digest(canonical(rows)),
            inventory_count=len(rows),
            active_project_sandboxes=0,
        )
    effective = capacity_authority(
        protocol,
        retry_execution_path,
        require_live_owner=True,
        require_roster_successor=True,
    )
    if (
        effective["shared_capacity_roster_receipt_sha256"]
        != expected_successor_roster_receipt_sha256
    ):
        raise ExternalCtfError("capacity_roster_successor_bind_readback_invalid")
    return marker


def _load_execution_packet(
    protocol: dict[str, Any],
    authority: dict[str, Any],
    execution_packet_path: Path,
    *,
    require_current_source: bool,
) -> tuple[dict[str, Any], Path]:
    from . import execution_packet

    try:
        return execution_packet.load(
            execution_packet_path,
            protocol=protocol,
            authority=authority,
            require_current_source=require_current_source,
        )
    except execution_packet.ExecutionPacketError as exc:
        raise ExternalCtfError(str(exc)) from exc


def _execution_context(
    protocol: dict[str, Any],
    retry_execution_path: Path,
    execution_packet_path: Path,
    *,
    require_live_owner: bool,
    require_current_source: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = capacity_authority(
        protocol,
        retry_execution_path,
        require_live_owner=require_live_owner,
    )
    packet, state = _load_execution_packet(
        protocol,
        authority,
        execution_packet_path,
        require_current_source=require_current_source,
    )
    return (
        {
            **authority,
            "external_state": state,
            "execution_packet_receipt_sha256": packet["receipt_sha256"],
        },
        packet,
    )


def _load_start_route_preflight(
    *,
    path: Path,
    protocol: dict[str, Any],
    packet: dict[str, Any],
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    frozen = packet.get("live_parity", {}).get("receipt")
    if not isinstance(frozen, dict) or not _is_digest(frozen.get("receipt_sha256")):
        raise ExternalCtfError("execution_packet_live_parity_invalid")
    try:
        return live_parity.load_start_preflight(
            path,
            protocol=protocol,
            execution_packet_receipt_sha256=packet["receipt_sha256"],
            benchmark=benchmark,
            task_index=task_index,
            arm=arm,
            frozen_live_parity=frozen,
        )
    except (KeyError, live_parity.LiveParityError) as exc:
        raise ExternalCtfError("scored_start_route_preflight_invalid") from exc


def seal_start_route_preflight(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
    kubernetes_context: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    if arm not in SCORED_ARMS:
        raise ExternalCtfError("route_preflight_only_for_scored_cell")
    _authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=True,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    key = os.environ.pop("FLEET_API_KEY", "")
    if not key or key.strip() != key or any(character.isspace() for character in key):
        raise ExternalCtfError("fleet_credential_missing_or_invalid")
    try:
        value = live_parity.capture_start_preflight(
            protocol=protocol,
            execution_packet=packet,
            benchmark=benchmark,
            task_index=task_index,
            arm=arm,
            key=key,
            context=kubernetes_context,
        )
    except live_parity.LiveParityError as exc:
        raise ExternalCtfError(str(exc)) from exc
    finally:
        key = ""
    if output_path.exists():
        raise ExternalCtfError("route_preflight_output_exists")
    _write_once(output_path, value)
    return {
        "schema": value["schema"],
        "status": value["status"],
        "receipt_sha256": value["receipt_sha256"],
    }


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return {**unsigned, "receipt_sha256": file_digest(canonical(unsigned))}


def _write_signed_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    expected = _signed(value)
    if path.exists():
        observed = _read_canonical(path, "signed")
        if observed != expected:
            raise ExternalCtfError("signed_receipt_collision")
        return observed
    _write_once(path, expected)
    return expected


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalCtfError(f"invalid_{label}_receipt") from exc
    if not isinstance(value, dict) or raw != canonical(value) + b"\n" or path.read_bytes() != raw:
        raise ExternalCtfError(f"invalid_{label}_receipt")
    return value


def _read_signed(path: Path, label: str) -> dict[str, Any]:
    value = _read_canonical(path, label)
    if value != _signed(value):
        raise ExternalCtfError(f"invalid_{label}_receipt")
    return value


def _private_state(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute.is_symlink() or absolute.resolve() != absolute:
        raise ExternalCtfError("external_state_path_not_exact")
    absolute.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not absolute.is_dir() or stat.S_IMODE(absolute.stat().st_mode) & 0o077:
        raise ExternalCtfError("external_state_not_private")
    return absolute


@contextlib.contextmanager
def _state_lock(state: Path) -> Iterator[None]:
    exact = _private_state(state)
    path = exact / "owner.lock"
    if path.is_symlink():
        raise ExternalCtfError("external_state_lock_invalid")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ExternalCtfError("external_state_lock_invalid")
        if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) & 0o077:
            raise ExternalCtfError("external_state_lock_not_private")
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


def _open_claims(state: Path, suffix: str) -> set[str]:
    names: set[str] = set()
    for path in state.glob(f"*{suffix}"):
        name = path.name.removesuffix(suffix)
        if not name or (state / f"{name}.released.json").exists():
            continue
        names.add(name)
    return names


def _enforce_single_open_cell(state: Path, *, target: str) -> None:
    open_creates = _open_claims(state, ".create-claim.json")
    if open_creates - {target}:
        raise ExternalCtfError("max_parallel_cells_exceeded")
    open_processes = _open_claims(state, ".process-claim.json")
    if open_processes - {target}:
        raise ExternalCtfError("max_parallel_cells_exceeded")


def _ordered_launch_cells(protocol: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = cve_execution_schedule(protocol)
    return [
        (
            cell_name(row["benchmark"], row["task_index"], row["arm"]),
            dict(row),
        )
        for row in rows
    ]


def _enforce_counterbalanced_order(
    state: Path,
    *,
    protocol: dict[str, Any],
    target: str,
    execution_packet_receipt_sha256: str,
) -> None:
    ordered = _ordered_launch_cells(protocol)
    released = [(state / f"{name}.released.json").exists() for name, _cell in ordered]
    first_open = next((index for index, value in enumerate(released) if not value), len(ordered))
    if any(released[first_open + 1 :]):
        raise ExternalCtfError("ordered_execution_state_has_gap")
    for name, cell in ordered[:first_open]:
        value = _read_signed(state / f"{name}.released.json", "release")
        if (
            value.get("schema") != "external_ctf_sandbox_release_v1"
            or value.get("name") != name
            or value.get("status")
            not in {
                "terminated",
                "absent",
                "skipped_runtime_preflight_infrastructure_invalid",
            }
            or value.get("protocol_sha256") != protocol["protocol_sha256"]
            or value.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
            or not _is_digest(value.get("terminal_receipt_sha256"))
            or any(value.get(key) != item for key, item in cell.items())
        ):
            raise ExternalCtfError("ordered_execution_release_binding_mismatch")
        if value["status"] == "skipped_runtime_preflight_infrastructure_invalid":
            qualification_name = cell_name("cvebench_zero_day", cell["task_index"], "qualification")
            qualification = _read_signed(
                state / f"{qualification_name}.terminal.json",
                "qualification_terminal",
            )
            if (
                qualification.get("name") != qualification_name
                or qualification.get("task_index") != cell["task_index"]
                or qualification.get("task_id") != cell["task_id"]
                or qualification.get("arm") != "qualification"
                or qualification.get("outcome") != "infrastructure_invalid"
                or qualification.get("execution_packet_receipt_sha256")
                != execution_packet_receipt_sha256
                or value.get("terminal_receipt_sha256") != qualification.get("receipt_sha256")
            ):
                raise ExternalCtfError("ordered_execution_release_binding_mismatch")
        else:
            terminal = _read_signed(state / f"{name}.terminal.json", "terminal")
            if (
                terminal.get("name") != name
                or any(terminal.get(key) != item for key, item in cell.items())
                or terminal.get("execution_packet_receipt_sha256")
                != execution_packet_receipt_sha256
                or terminal.get("outcome")
                not in {"accepted_model_outcome", "infrastructure_invalid"}
                or value.get("terminal_receipt_sha256") != terminal.get("receipt_sha256")
            ):
                raise ExternalCtfError("ordered_execution_release_binding_mismatch")
    if first_open == len(ordered):
        raise ExternalCtfError("ordered_execution_complete")
    if first_open >= protocol["execution_schedule"]["pause_after_cell_count"]:
        _require_cve_continuation(
            state,
            protocol=protocol,
            ordered=ordered,
            execution_packet_receipt_sha256=execution_packet_receipt_sha256,
        )
    if target != ordered[first_open][0]:
        raise ExternalCtfError("cell_out_of_counterbalanced_order")


def _canary_release_bindings(
    state: Path,
    *,
    protocol: dict[str, Any],
    ordered: list[tuple[str, dict[str, Any]]],
    execution_packet_receipt_sha256: str,
) -> list[dict[str, str]]:
    count = protocol["execution_schedule"]["pause_after_cell_count"]
    bindings: list[dict[str, str]] = []
    for name, cell in ordered[:count]:
        value = _read_signed(state / f"{name}.released.json", "canary_release")
        if (
            value.get("schema") != "external_ctf_sandbox_release_v1"
            or value.get("name") != name
            or value.get("status") != "terminated"
            or value.get("protocol_sha256") != protocol["protocol_sha256"]
            or value.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
            or not _is_digest(value.get("terminal_receipt_sha256"))
            or any(value.get(key) != item for key, item in cell.items())
        ):
            raise ExternalCtfError("canary_release_binding_mismatch")
        bindings.append({"name": name, "release_receipt_sha256": value["receipt_sha256"]})
    return bindings


def _require_cve_continuation(
    state: Path,
    *,
    protocol: dict[str, Any],
    ordered: list[tuple[str, dict[str, Any]]],
    execution_packet_receipt_sha256: str,
) -> dict[str, Any]:
    value = _read_signed(state / "CVE_FULL_CONTINUATION.json", "cve_continuation")
    expected = {
        "schema": "external_ctf_cve_full_continuation_v1",
        "status": "reviewed_continue_frozen_roster",
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_packet_receipt_sha256": execution_packet_receipt_sha256,
        "ordered_cells_sha256": protocol["execution_schedule"]["ordered_cells_sha256"],
        "canary_release_bindings": _canary_release_bindings(
            state,
            protocol=protocol,
            ordered=ordered,
            execution_packet_receipt_sha256=execution_packet_receipt_sha256,
        ),
        "remaining_cell_count": protocol["execution_schedule"]["remaining_cell_count_after_canary"],
        "score_reads": 0,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ExternalCtfError("cve_continuation_binding_mismatch")
    return value


def seal_cve_continuation(
    *,
    protocol: dict[str, Any],
    state: Path,
    execution_packet_receipt_sha256: str,
) -> dict[str, Any]:
    ordered = _ordered_launch_cells(protocol)
    value = {
        "schema": "external_ctf_cve_full_continuation_v1",
        "status": "reviewed_continue_frozen_roster",
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_packet_receipt_sha256": execution_packet_receipt_sha256,
        "ordered_cells_sha256": protocol["execution_schedule"]["ordered_cells_sha256"],
        "canary_release_bindings": _canary_release_bindings(
            state,
            protocol=protocol,
            ordered=ordered,
            execution_packet_receipt_sha256=execution_packet_receipt_sha256,
        ),
        "remaining_cell_count": protocol["execution_schedule"]["remaining_cell_count_after_canary"],
        "score_reads": 0,
    }
    return _write_signed_once(state / "CVE_FULL_CONTINUATION.json", value)


def _require_runtime_qualification(
    state: Path,
    *,
    protocol: dict[str, Any],
    task_index: int,
    execution_packet_receipt_sha256: str,
) -> None:
    cell = _cell(protocol, "cvebench_zero_day", task_index, "qualification")
    name = cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
    terminal = _read_signed(state / f"{name}.terminal.json", "qualification_terminal")
    release = _read_signed(state / f"{name}.released.json", "qualification_release")
    if (
        terminal.get("schema") != "external_ctf_cell_terminal_v1"
        or any(terminal.get(key) != value for key, value in cell.items())
        or terminal.get("name") != name
        or terminal.get("protocol_sha256") != protocol["protocol_sha256"]
        or terminal.get("qualification_contract_sha256")
        != runtime_qualification_contract_sha256(protocol, "cvebench_zero_day")
        or terminal.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or terminal.get("outcome") != "runtime_preflight_passed"
        or terminal.get("infrastructure_error_class") is not None
        or release.get("schema") != "external_ctf_sandbox_release_v1"
        or any(release.get(key) != value for key, value in cell.items())
        or release.get("name") != name
        or release.get("status") != "terminated"
        or release.get("protocol_sha256") != protocol["protocol_sha256"]
        or release.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or release.get("terminal_receipt_sha256") != terminal["receipt_sha256"]
    ):
        raise ExternalCtfError("runtime_qualification_not_accepted_and_released")


def _require_qualification_for_next_pair(
    state: Path,
    *,
    protocol: dict[str, Any],
    task_index: int,
    execution_packet_receipt_sha256: str,
) -> None:
    ordered = _ordered_launch_cells(protocol)
    released = [(state / f"{name}.released.json").exists() for name, _cell in ordered]
    first_open = next((index for index, value in enumerate(released) if not value), len(ordered))
    if first_open == len(ordered):
        raise ExternalCtfError("ordered_execution_complete")
    next_cell = ordered[first_open][1]
    if next_cell["task_index"] != task_index:
        raise ExternalCtfError("runtime_preflight_not_for_next_pair")
    qualification_name = cell_name("cvebench_zero_day", task_index, "qualification")
    if any(
        (state / f"{qualification_name}{suffix}").exists()
        for suffix in (".create-claim.json", ".created.json", ".terminal.json", ".released.json")
    ):
        raise ExternalCtfError("runtime_preflight_already_claimed")
    # A completed first arm means this pair has already started; a preflight
    # must be sealed before either scored sandbox is created.
    task_names = [cell_name("cvebench_zero_day", task_index, arm) for arm in SCORED_ARMS]
    if any(
        (state / f"{name}{suffix}").exists()
        for name in task_names
        for suffix in (".create-claim.json", ".created.json")
    ):
        raise ExternalCtfError("runtime_preflight_must_precede_task_pair")
    if first_open >= protocol["execution_schedule"]["pause_after_cell_count"]:
        _require_cve_continuation(
            state,
            protocol=protocol,
            ordered=ordered,
            execution_packet_receipt_sha256=execution_packet_receipt_sha256,
        )


def _post_failure_value(
    *,
    operation: str,
    cell: dict[str, Any],
    name: str,
    protocol_sha256: str,
    execution_packet_receipt_sha256: str,
    claim_path: Path,
    error: Exception,
) -> dict[str, Any]:
    http_status = error.code if isinstance(error, urllib.error.HTTPError) else None
    return {
        "schema": "external_ctf_provider_post_failure_v1",
        **cell,
        "name": name,
        "operation": operation,
        "protocol_sha256": protocol_sha256,
        "execution_packet_receipt_sha256": execution_packet_receipt_sha256,
        "claim_file_sha256": file_digest(claim_path.read_bytes()),
        "error_type": type(error).__name__,
        "http_status": http_status,
        "classification": (
            "definitive_no_object"
            if http_status in DEFINITIVE_NO_CREATE_HTTP_STATUSES
            else "ambiguous_provider_outcome"
        ),
    }


def _read_definitive_post_failure(
    path: Path,
    *,
    operation: str,
    cell: dict[str, Any],
    name: str,
    protocol_sha256: str,
    execution_packet_receipt_sha256: str,
    claim_path: Path,
) -> dict[str, Any]:
    failure = _read_signed(path, "provider_post_failure")
    if (
        failure.get("schema") != "external_ctf_provider_post_failure_v1"
        or any(failure.get(key) != value for key, value in cell.items())
        or failure.get("name") != name
        or failure.get("operation") != operation
        or failure.get("protocol_sha256") != protocol_sha256
        or failure.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or failure.get("claim_file_sha256") != file_digest(claim_path.read_bytes())
        or failure.get("classification") != "definitive_no_object"
        or failure.get("http_status") not in DEFINITIVE_NO_CREATE_HTTP_STATUSES
        or failure.get("error_type") != "HTTPError"
    ):
        raise ExternalCtfError("provider_post_failure_not_definitively_absent")
    return failure


def _seal_qualification_pair_skip(
    state: Path,
    *,
    protocol: dict[str, Any],
    task_index: int,
    qualification_terminal: dict[str, Any],
    execution_packet_receipt_sha256: str,
) -> None:
    """Advance a non-canary task only after its model-free preflight is infra-invalid."""

    if (
        qualification_terminal.get("outcome") != "infrastructure_invalid"
        or qualification_terminal.get("arm") != "qualification"
        or qualification_terminal.get("task_index") != task_index
        or qualification_terminal.get("qualification_contract_sha256")
        != runtime_qualification_contract_sha256(protocol, "cvebench_zero_day")
        or qualification_terminal.get("execution_packet_receipt_sha256")
        != execution_packet_receipt_sha256
    ):
        raise ExternalCtfError("runtime_preflight_skip_terminal_invalid")
    for arm in ("step_1000", "base"):
        cell = _cell(protocol, "cvebench_zero_day", task_index, arm)
        name = cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
        if any(
            (state / f"{name}{suffix}").exists()
            for suffix in (".create-claim.json", ".created.json", ".terminal.json")
        ):
            raise ExternalCtfError("runtime_preflight_skip_after_scored_cell_claimed")
        _write_signed_once(
            state / f"{name}.released.json",
            {
                "schema": "external_ctf_sandbox_release_v1",
                **cell,
                "name": name,
                "sandbox_id": None,
                "status": "skipped_runtime_preflight_infrastructure_invalid",
                "protocol_sha256": protocol["protocol_sha256"],
                "execution_packet_receipt_sha256": execution_packet_receipt_sha256,
                "terminal_receipt_sha256": qualification_terminal["receipt_sha256"],
            },
        )


def _client() -> TensorlakeClient:
    key = os.environ.get("TENSORLAKE_API_KEY", "")
    if not key or key.strip() != key or any(character.isspace() for character in key):
        raise ExternalCtfError("tensorlake_credential_missing_or_invalid")
    return TensorlakeClient(key)


def _cell(protocol: dict[str, Any], benchmark: str, task_index: int, arm: str) -> dict[str, Any]:
    value = protocol["benchmarks"].get(benchmark)
    if value is None or arm not in ARMS or not 0 <= task_index < len(value["task_ids"]):
        raise ExternalCtfError("invalid_cell_identity")
    task_id = value["task_ids"][task_index]
    if arm == "qualification":
        if task_index not in value["runtime_qualification"]["task_indices"]:
            raise ExternalCtfError("runtime_qualification_task_is_not_launchable")
        return {
            "benchmark": benchmark,
            "task_index": task_index,
            "task_id": task_id,
            "arm": arm,
        }
    if benchmark != protocol["execution_benchmark"]:
        raise ExternalCtfError("invalid_cell_identity")
    if value.get("adapter_qualified") is not True:
        raise ExternalCtfError("benchmark_adapter_not_qualified")
    if task_id in value.get("source_unavailable_task_ids", []):
        raise ExternalCtfError("source_unavailable_cell_is_not_launchable")
    if task_id in value.get("execution_unavailable_task_ids", []):
        raise ExternalCtfError("execution_unavailable_cell_is_not_launchable")
    return {"benchmark": benchmark, "task_index": task_index, "task_id": task_id, "arm": arm}


def _qualification_contract_binding(protocol: dict[str, Any], cell: dict[str, Any]) -> str | None:
    return (
        runtime_qualification_contract_sha256(protocol, cell["benchmark"])
        if cell["arm"] == "qualification"
        else None
    )


def _require_packet_allows_cell(
    packet: dict[str, Any], protocol: dict[str, Any], cell: dict[str, Any]
) -> None:
    mode = packet.get("packet_mode")
    if mode == "scored":
        return
    expected = {
        **cell,
        "qualification_contract_sha256": _qualification_contract_binding(protocol, cell),
    }
    if (
        mode != "qualification_only"
        or cell["arm"] != "qualification"
        or expected not in packet.get("execution", {}).get("qualification_cells", [])
    ):
        raise ExternalCtfError("execution_packet_forbids_cell")


def _accepted_qualification_release(
    state: Path,
    *,
    protocol: dict[str, Any],
    row: dict[str, Any],
    packet_receipt_sha256: str,
    protocol_sha256: str,
) -> bool:
    cell = {key: row[key] for key in ("benchmark", "task_index", "task_id", "arm")}
    name = cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
    release_path = state / f"{name}.released.json"
    if not release_path.exists():
        return False
    terminal = _read_signed(state / f"{name}.terminal.json", "qualification_terminal")
    release = _read_signed(release_path, "qualification_release")
    result = terminal.get("result")
    if (
        terminal.get("schema") != "external_ctf_cell_terminal_v1"
        or any(terminal.get(key) != value for key, value in cell.items())
        or terminal.get("name") != name
        or terminal.get("protocol_sha256") != protocol_sha256
        or terminal.get("qualification_contract_sha256") != row["qualification_contract_sha256"]
        or terminal.get("execution_packet_receipt_sha256") != packet_receipt_sha256
        or terminal.get("outcome") != "runtime_preflight_passed"
        or terminal.get("infrastructure_error_class") is not None
        or not isinstance(result, dict)
        or _result_value(canonical(result) + b"\n", protocol=protocol, cell=cell) != result
        or release.get("schema") != "external_ctf_sandbox_release_v1"
        or any(release.get(key) != value for key, value in cell.items())
        or release.get("name") != name
        or release.get("status") != "terminated"
        or release.get("protocol_sha256") != protocol_sha256
        or release.get("execution_packet_receipt_sha256") != packet_receipt_sha256
        or release.get("terminal_receipt_sha256") != terminal.get("receipt_sha256")
    ):
        raise ExternalCtfError("qualification_canary_not_accepted_and_released")
    return True


def _qualification_open_names(
    state: Path,
    rows: list[dict[str, Any]],
    *,
    capacity_state: Path,
    owned_names: set[str] | frozenset[str],
) -> set[str]:
    qualification_names = {
        cell_name(row["benchmark"], row["task_index"], row["arm"]) for row in rows
    }
    claimed = {
        name
        for name in qualification_names
        if (state / f"{name}.create-claim.json").exists()
        and not (state / f"{name}.released.json").exists()
    }
    try:
        pending = set(
            replica_set._shared_capacity_reservations(capacity_state, owned_names)  # noqa: SLF001
        )
    except replica_set.CollectionReplicaSetError as exc:
        raise ExternalCtfError(str(exc)) from exc
    return claimed | (pending & qualification_names)


def _enforce_qualification_stage(
    state: Path,
    *,
    protocol: dict[str, Any],
    packet: dict[str, Any],
    target: str,
    capacity_state: Path,
    owned_names: set[str] | frozenset[str],
) -> None:
    execution = packet.get("execution", {})
    rows = execution.get("qualification_cells")
    stages = execution.get("staged_concurrency")
    if not isinstance(rows, list) or not rows or not isinstance(stages, dict):
        raise ExternalCtfError("qualification_packet_schedule_invalid")
    canary = stages.get("canary")
    batch = stages.get("batch")
    if not isinstance(canary, dict) or not isinstance(batch, dict):
        raise ExternalCtfError("qualification_packet_schedule_invalid")
    canary_rows = canary.get("ordered_cells")
    batch_rows = batch.get("cells")
    if not isinstance(canary_rows, list) or not isinstance(batch_rows, list):
        raise ExternalCtfError("qualification_packet_schedule_invalid")
    for row in canary_rows:
        if _accepted_qualification_release(
            state,
            protocol=protocol,
            row=row,
            packet_receipt_sha256=packet["receipt_sha256"],
            protocol_sha256=packet.get("protocol", {}).get("receipt_sha256"),
        ):
            continue
        expected = cell_name(row["benchmark"], row["task_index"], row["arm"])
        if target != expected:
            raise ExternalCtfError("qualification_canary_order_violation")
        open_names = _qualification_open_names(
            state,
            rows,
            capacity_state=capacity_state,
            owned_names=owned_names,
        )
        if len(open_names - {target}) >= canary["max_parallel_cells"]:
            raise ExternalCtfError("qualification_stage_parallel_limit")
        return
    batch_names = {
        cell_name(row["benchmark"], row["task_index"], row["arm"]): row for row in batch_rows
    }
    if target not in batch_names:
        raise ExternalCtfError("qualification_schedule_complete")
    open_names = _qualification_open_names(
        state,
        rows,
        capacity_state=capacity_state,
        owned_names=owned_names,
    )
    if len(open_names - {target}) >= batch["max_parallel_cells"]:
        raise ExternalCtfError("qualification_stage_parallel_limit")


def _enforce_packet_open_cell_limit(
    state: Path,
    *,
    protocol: dict[str, Any],
    packet: dict[str, Any],
    cell: dict[str, Any],
    target: str,
    capacity_state: Path,
    owned_names: set[str] | frozenset[str],
) -> None:
    if cell["arm"] == "qualification" and packet.get("packet_mode") == "qualification_only":
        _enforce_qualification_stage(
            state,
            protocol=protocol,
            packet=packet,
            target=target,
            capacity_state=capacity_state,
            owned_names=owned_names,
        )
        return
    _enforce_single_open_cell(state, target=target)


def _sandbox_spec(name: str, snapshot_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "snapshot_id": snapshot_id,
        # The bound snapshot already carries its root disk. TensorLake only
        # accepts disk_mb here as a growth override, so inherit it unchanged.
        "resources": {"cpus": 8, "memory_mb": 32768},
        "timeout_secs": 21600,
        "allow_unauthenticated_access": False,
        "network": {"allow_internet_access": True, "allow_out": []},
    }


def _created_value(
    *,
    cell: dict[str, Any],
    name: str,
    sandbox_id: str,
    status: object,
    protocol_sha256: str,
    spec: dict[str, Any],
    claim: Path,
    authority: dict[str, Any],
    reconciled: bool,
) -> dict[str, Any]:
    claim_value = _read_canonical(claim, "create_claim")
    return {
        "schema": "external_ctf_sandbox_created_v1",
        **cell,
        "name": name,
        "sandbox_id": sandbox_id,
        "status": status,
        "protocol_sha256": protocol_sha256,
        "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
        "spec_sha256": file_digest(canonical(spec)),
        "create_claim_file_sha256": file_digest(claim.read_bytes()),
        "retry_execution_receipt_sha256": authority["retry_execution_receipt_sha256"],
        "capacity_successor_receipt_sha256": authority["capacity_successor_receipt_sha256"],
        "capacity_successor_state_receipt_sha256": authority[
            "capacity_successor_state_receipt_sha256"
        ],
        "shared_capacity_roster_receipt_sha256": authority["shared_capacity_roster_receipt_sha256"],
        "live_owner_receipt_sha256": authority["live_owner_receipt_sha256"],
        "capacity_reservation_receipt_sha256": claim_value["capacity_reservation_receipt_sha256"],
        "create_route_preflight_receipt_sha256": claim_value[
            "create_route_preflight_receipt_sha256"
        ],
        "reconciled_after_ambiguous_create": reconciled,
    }


def create(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
    route_preflight_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=True,
        require_current_source=True,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    claim = state / f"{name}.create-claim.json"
    created = state / f"{name}.created.json"
    if claim.exists() or created.exists():
        raise ExternalCtfError("cell_already_claimed")
    spec = _sandbox_spec(name, authority["snapshot_id"])
    with replica_set._shared_tensorlake_create_lock(authority["state"]):  # noqa: SLF001
        refreshed, refreshed_packet = _execution_context(
            protocol,
            retry_execution_path,
            execution_packet_path,
            require_live_owner=True,
            require_current_source=True,
        )
        _require_packet_allows_cell(refreshed_packet, protocol, cell)
        if (
            refreshed["state"] != authority["state"]
            or refreshed["snapshot_id"] != authority["snapshot_id"]
            or refreshed["retry_execution_receipt_sha256"]
            != authority["retry_execution_receipt_sha256"]
            or refreshed["capacity_successor_receipt_sha256"]
            != authority["capacity_successor_receipt_sha256"]
            or refreshed["capacity_successor_state_receipt_sha256"]
            != authority["capacity_successor_state_receipt_sha256"]
            or refreshed["shared_capacity_roster_receipt_sha256"]
            != authority["shared_capacity_roster_receipt_sha256"]
            or refreshed["live_owner_receipt_sha256"] != authority["live_owner_receipt_sha256"]
            or refreshed["external_state"] != authority["external_state"]
            or refreshed_packet["receipt_sha256"] != authority["execution_packet_receipt_sha256"]
        ):
            raise ExternalCtfError("shared_capacity_authority_changed")
        if arm == "qualification":
            if route_preflight_path is not None:
                raise ExternalCtfError("qualification_must_not_use_model_route_preflight")
            create_route_preflight_receipt_sha256 = None
        else:
            if route_preflight_path is None:
                raise ExternalCtfError("scored_create_route_preflight_required")
            create_route_preflight_receipt_sha256 = _load_start_route_preflight(
                path=route_preflight_path,
                protocol=protocol,
                packet=refreshed_packet,
                benchmark=benchmark,
                task_index=task_index,
                arm=arm,
            )["receipt_sha256"]
        client = _client()
        _enforce_packet_open_cell_limit(
            state,
            protocol=protocol,
            packet=refreshed_packet,
            cell=cell,
            target=name,
            capacity_state=refreshed["state"],
            owned_names=refreshed["owned_names"],
        )
        if arm == "qualification":
            if (
                refreshed_packet.get("packet_mode") != "qualification_only"
                and benchmark == "cvebench_zero_day"
            ):
                _require_qualification_for_next_pair(
                    state,
                    protocol=protocol,
                    task_index=task_index,
                    execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
                )
        else:
            _require_runtime_qualification(
                state,
                protocol=protocol,
                task_index=task_index,
                execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            )
            _enforce_counterbalanced_order(
                state,
                protocol=protocol,
                target=name,
                execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            )
        rows = client.inventory()
        if any(row.get("name") == name for row in rows):
            raise ExternalCtfError("duplicate_sandbox_name")
        try:
            reservation, active = replica_set.reserve_shared_capacity_slot(
                state=authority["state"],
                rows=rows,
                owned_names=refreshed["owned_names"],
                sandbox_name=name,
                creator="external_ctf",
                authority_receipt_sha256=authority["capacity_successor_receipt_sha256"],
                spec_sha256=file_digest(canonical(spec)),
            )
        except replica_set.CollectionReplicaSetError as exc:
            raise ExternalCtfError(str(exc)) from exc
        claim_value = {
            "schema": "external_ctf_sandbox_create_claim_v1",
            **cell,
            "name": name,
            "protocol_sha256": protocol["protocol_sha256"],
            "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
            "spec_sha256": file_digest(canonical(spec)),
            "active_before_create": active,
            "shared_limit": PROJECT_ACTIVE_SANDBOX_LIMIT,
            "retry_execution_receipt_sha256": authority["retry_execution_receipt_sha256"],
            "capacity_successor_receipt_sha256": authority["capacity_successor_receipt_sha256"],
            "capacity_successor_state_receipt_sha256": authority[
                "capacity_successor_state_receipt_sha256"
            ],
            "shared_capacity_roster_receipt_sha256": authority[
                "shared_capacity_roster_receipt_sha256"
            ],
            "live_owner_receipt_sha256": authority["live_owner_receipt_sha256"],
            "capacity_reservation_receipt_sha256": reservation["receipt_sha256"],
            "create_route_preflight_receipt_sha256": (create_route_preflight_receipt_sha256),
        }
        _write_once(claim, claim_value)
        try:
            response = client.request("POST", API + "/sandboxes", spec)
        except Exception as exc:
            _write_signed_once(
                state / f"{name}.create-post-failure.json",
                _post_failure_value(
                    operation="sandbox_create",
                    cell=cell,
                    name=name,
                    protocol_sha256=protocol["protocol_sha256"],
                    execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
                    claim_path=claim,
                    error=exc,
                ),
            )
            raise
    sandbox_id = response.get("sandbox_id")
    if not isinstance(sandbox_id, str) or re.fullmatch(r"[A-Za-z0-9._:-]+", sandbox_id) is None:
        raise ExternalCtfError("sandbox_create_response_invalid")
    receipt = _created_value(
        cell=cell,
        name=name,
        sandbox_id=sandbox_id,
        status=response.get("status"),
        protocol_sha256=protocol["protocol_sha256"],
        spec=spec,
        claim=claim,
        authority=authority,
        reconciled=False,
    )
    _write_once(created, receipt)
    return receipt


def reconcile_create(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    """Bind exactly one provider object after an ambiguous create without another POST."""

    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    claim_path = state / f"{name}.create-claim.json"
    created_path = state / f"{name}.created.json"
    with _state_lock(state):
        if created_path.exists():
            return _read_canonical(created_path, "created")
        claim = _read_canonical(claim_path, "create_claim")
        if (
            claim.get("schema") != "external_ctf_sandbox_create_claim_v1"
            or any(claim.get(key) != value for key, value in cell.items())
            or claim.get("name") != name
            or claim.get("protocol_sha256") != protocol["protocol_sha256"]
            or claim.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or claim.get("capacity_successor_receipt_sha256")
            != authority["capacity_successor_receipt_sha256"]
            or claim.get("capacity_successor_state_receipt_sha256")
            != authority["capacity_successor_state_receipt_sha256"]
            or claim.get("shared_capacity_roster_receipt_sha256")
            != authority["shared_capacity_roster_receipt_sha256"]
            or not _is_digest(claim.get("live_owner_receipt_sha256"))
            or not _is_digest(claim.get("capacity_reservation_receipt_sha256"))
            or (
                claim.get("create_route_preflight_receipt_sha256") is not None
                if arm == "qualification"
                else not _is_digest(claim.get("create_route_preflight_receipt_sha256"))
            )
        ):
            raise ExternalCtfError("create_claim_binding_mismatch")
        client = _client()
        rows = [row for row in client.inventory() if row.get("name") == name]
        if len(rows) != 1 or not isinstance(rows[0].get("id"), str):
            raise ExternalCtfError("sandbox_create_not_uniquely_reconciled")
        sandbox_id = rows[0]["id"]
        detail = client.request("GET", API + "/sandboxes/" + sandbox_id)
        spec = _sandbox_spec(name, authority["snapshot_id"])
        if (
            detail.get("id") != sandbox_id
            or detail.get("name") != name
            or detail.get("allow_unauthenticated_access") is not False
            or detail.get("snapshot_id") != spec["snapshot_id"]
            or detail.get("timeout_secs") != spec["timeout_secs"]
            or not _resources_match(spec["resources"], detail.get("resources"))
            or not _network_policy_matches(spec["network"], detail.get("network_policy"))
            or claim.get("spec_sha256") != file_digest(canonical(spec))
        ):
            raise ExternalCtfError("sandbox_create_reconciliation_mismatch")
        receipt = _created_value(
            cell=cell,
            name=name,
            sandbox_id=sandbox_id,
            status=detail.get("status"),
            protocol_sha256=protocol["protocol_sha256"],
            spec=spec,
            claim=claim_path,
            authority={
                **authority,
                "live_owner_receipt_sha256": claim["live_owner_receipt_sha256"],
            },
            reconciled=True,
        )
        _write_once(created_path, receipt)
        return receipt


def abort_create_absent(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    """Seal a definitive failed create as absent after bounded read-only reconciliation."""

    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    claim_path = state / f"{name}.create-claim.json"
    failure_path = state / f"{name}.create-post-failure.json"
    created_path = state / f"{name}.created.json"
    terminal_path = state / f"{name}.terminal.json"
    release_path = state / f"{name}.released.json"
    with _state_lock(state):
        if created_path.exists():
            raise ExternalCtfError("sandbox_create_exists_and_cannot_be_aborted_absent")
        claim = _read_canonical(claim_path, "create_claim")
        if (
            claim.get("schema") != "external_ctf_sandbox_create_claim_v1"
            or any(claim.get(key) != value for key, value in cell.items())
            or claim.get("name") != name
            or claim.get("protocol_sha256") != protocol["protocol_sha256"]
            or claim.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or claim.get("capacity_successor_receipt_sha256")
            != authority["capacity_successor_receipt_sha256"]
            or claim.get("capacity_successor_state_receipt_sha256")
            != authority["capacity_successor_state_receipt_sha256"]
            or claim.get("shared_capacity_roster_receipt_sha256")
            != authority["shared_capacity_roster_receipt_sha256"]
            or not _is_digest(claim.get("capacity_reservation_receipt_sha256"))
            or (
                claim.get("create_route_preflight_receipt_sha256") is not None
                if arm == "qualification"
                else not _is_digest(claim.get("create_route_preflight_receipt_sha256"))
            )
        ):
            raise ExternalCtfError("create_claim_binding_mismatch")
        failure = _read_definitive_post_failure(
            failure_path,
            operation="sandbox_create",
            cell=cell,
            name=name,
            protocol_sha256=protocol["protocol_sha256"],
            execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            claim_path=claim_path,
        )
        client = _client()
        observations = []
        for attempt in range(1, ABSENCE_RECONCILE_ATTEMPTS + 1):
            rows = client.inventory()
            active_project_count(rows, authority["owned_names"])
            matches = [row for row in rows if row.get("name") == name]
            if matches:
                raise ExternalCtfError("sandbox_create_absence_not_confirmed")
            observations.append(
                {
                    "attempt": attempt,
                    "inventory_count": len(rows),
                    "inventory_sha256": file_digest(canonical(rows)),
                }
            )
            if attempt < ABSENCE_RECONCILE_ATTEMPTS:
                time.sleep(ABSENCE_RECONCILE_DELAY_SECONDS)
        with replica_set._shared_tensorlake_create_lock(authority["state"]):  # noqa: SLF001
            final_rows = client.inventory()
            active_project_count(final_rows, authority["owned_names"])
            if any(row.get("name") == name for row in final_rows):
                raise ExternalCtfError("sandbox_create_absence_not_confirmed")
            terminal = _write_signed_once(
                terminal_path,
                {
                    "schema": "external_ctf_cell_terminal_v1",
                    **cell,
                    "name": name,
                    "sandbox_id": None,
                    "pid": None,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "qualification_contract_sha256": (
                        runtime_qualification_contract_sha256(protocol, benchmark)
                        if arm == "qualification"
                        else None
                    ),
                    "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                    "provider_post_failure_receipt_sha256": failure["receipt_sha256"],
                    "absence_reconciliation": observations
                    + [
                        {
                            "attempt": ABSENCE_RECONCILE_ATTEMPTS + 1,
                            "inventory_count": len(final_rows),
                            "inventory_sha256": file_digest(canonical(final_rows)),
                            "under_shared_create_lock": True,
                        }
                    ],
                    "result": None,
                    "outcome": "infrastructure_invalid",
                    "infrastructure_error_class": "sandbox_create_definitive_failure_absent",
                },
            )
            release = _write_signed_once(
                release_path,
                {
                    "schema": "external_ctf_sandbox_release_v1",
                    **cell,
                    "name": name,
                    "sandbox_id": None,
                    "status": "absent",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                    "terminal_receipt_sha256": terminal["receipt_sha256"],
                },
            )
            try:
                replica_set.release_shared_capacity_slot(
                    state=authority["state"],
                    owned_names=authority["owned_names"],
                    sandbox_name=name,
                    reservation_receipt_sha256=claim["capacity_reservation_receipt_sha256"],
                    provider_state="absent",
                    terminal_receipt_sha256=release["receipt_sha256"],
                )
            except replica_set.CollectionReplicaSetError as exc:
                raise ExternalCtfError(str(exc)) from exc
        if (
            arm == "qualification"
            and benchmark == "cvebench_zero_day"
            and packet.get("packet_mode") == "scored"
        ):
            _seal_qualification_pair_skip(
                state,
                protocol=protocol,
                task_index=task_index,
                qualification_terminal=terminal,
                execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            )
        return release


def start(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
    route_preflight_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    name = cell_name(benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=True,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    process_claim = state / f"{name}.process-claim.json"
    process_receipt = state / f"{name}.process.json"
    with _state_lock(state):
        if process_claim.exists() or process_receipt.exists():
            raise ExternalCtfError("cell_start_already_claimed")
        _enforce_packet_open_cell_limit(
            state,
            protocol=protocol,
            packet=packet,
            cell=cell,
            target=name,
            capacity_state=authority["state"],
            owned_names=authority["owned_names"],
        )
        created = _read_canonical(state / f"{name}.created.json", "created")
        if (
            created.get("schema") != "external_ctf_sandbox_created_v1"
            or any(created.get(key) != value for key, value in cell.items())
            or created.get("name") != name
            or created.get("protocol_sha256") != protocol["protocol_sha256"]
            or created.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or not isinstance(created.get("sandbox_id"), str)
            or created.get("capacity_successor_receipt_sha256")
            != authority["capacity_successor_receipt_sha256"]
            or created.get("capacity_successor_state_receipt_sha256")
            != authority["capacity_successor_state_receipt_sha256"]
            or created.get("shared_capacity_roster_receipt_sha256")
            != authority["shared_capacity_roster_receipt_sha256"]
            or not _is_digest(created.get("live_owner_receipt_sha256"))
            or not _is_digest(created.get("capacity_reservation_receipt_sha256"))
            or (
                created.get("create_route_preflight_receipt_sha256") is not None
                if arm == "qualification"
                else not _is_digest(created.get("create_route_preflight_receipt_sha256"))
            )
        ):
            raise ExternalCtfError("created_receipt_binding_mismatch")
        if arm == "qualification":
            if route_preflight_path is not None:
                raise ExternalCtfError("qualification_must_not_use_model_route_preflight")
            route_preflight_receipt_sha256 = None
        else:
            if route_preflight_path is None:
                raise ExternalCtfError("scored_start_route_preflight_required")
            route_preflight_receipt_sha256 = _load_start_route_preflight(
                path=route_preflight_path,
                protocol=protocol,
                packet=packet,
                benchmark=benchmark,
                task_index=task_index,
                arm=arm,
            )["receipt_sha256"]
            if route_preflight_receipt_sha256 != created["create_route_preflight_receipt_sha256"]:
                raise ExternalCtfError("scored_start_route_preflight_changed_after_create")
        client = _client()
        detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
        origin = detail.get("sandbox_url")
        if detail.get("status") != "running" or not isinstance(origin, str):
            raise ExternalCtfError("sandbox_not_runnable")
        worker = WORKER.read_bytes()
        worker_sha256 = file_digest(worker)
        env = {
            "EXTERNAL_CTF_PROTOCOL_B64": base64.b64encode(protocol_path.read_bytes()).decode(),
            "EXTERNAL_CTF_BENCHMARK": benchmark,
            "EXTERNAL_CTF_TASK_ID": cell["task_id"],
            "EXTERNAL_CTF_ARM": arm,
            "WORKER_B64": base64.b64encode(worker).decode(),
            "WORKER_SHA256": worker_sha256,
        }
        if arm == "qualification":
            env["EXTERNAL_CTF_MODE"] = "runtime_qualification"
            bundle = _qualification_bundle(protocol, benchmark)
            if bundle is not None:
                env.update(
                    {
                        "QUALIFICATION_BUNDLE_B64": base64.b64encode(bundle).decode(),
                        "QUALIFICATION_BUNDLE_SHA256": file_digest(bundle),
                    }
                )
            fleet_account_sha256 = None
        else:
            _require_runtime_qualification(
                state,
                protocol=protocol,
                task_index=task_index,
                execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            )
            fleet_key = os.environ.get("FLEET_API_KEY", "")
            if (
                not fleet_key
                or fleet_key.strip() != fleet_key
                or any(character.isspace() for character in fleet_key)
            ):
                raise ExternalCtfError("fleet_credential_missing_or_invalid")
            try:
                fleet_account_sha256 = file_digest(
                    canonical(live_parity._fleet_account(fleet_key))  # noqa: SLF001
                )
            except live_parity.LiveParityError as exc:
                raise ExternalCtfError("fleet_account_identity_mismatch") from exc
            qualification_name = cell_name("cvebench_zero_day", task_index, "qualification")
            qualification_path = state / f"{qualification_name}.terminal.json"
            qualification_raw = qualification_path.read_bytes()
            env.update(
                {
                    "EXTERNAL_CTF_MODE": "scored",
                    "EXTERNAL_CTF_QUALIFICATION_B64": base64.b64encode(qualification_raw).decode(),
                    "EXTERNAL_CTF_QUALIFICATION_SHA256": file_digest(qualification_raw),
                    "FLEET_API_KEY": fleet_key,
                }
            )
        process_spec = {
            "command": "/usr/bin/python3",
            "args": ["-c", WORKER_BOOTSTRAP],
            "user": "root",
            "env": env,
            "stdin_mode": "closed",
            "stdout_mode": "discard",
            "stderr_mode": "discard",
        }
        process_spec_binding = {
            **process_spec,
            "env": {
                **env,
                **({"FLEET_API_KEY": "present_redacted"} if "FLEET_API_KEY" in env else {}),
            },
        }
        _write_once(
            process_claim,
            {
                "schema": "external_ctf_process_claim_v1",
                **cell,
                "name": name,
                "sandbox_id": created["sandbox_id"],
                "protocol_sha256": protocol["protocol_sha256"],
                "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                "start_route_preflight_receipt_sha256": route_preflight_receipt_sha256,
                "fleet_account_sha256": fleet_account_sha256,
                "worker_sha256": worker_sha256,
                "created_receipt_file_sha256": file_digest(
                    (state / f"{name}.created.json").read_bytes()
                ),
                "process_spec_sha256": file_digest(canonical(process_spec_binding)),
            },
        )
        try:
            response = client.request(
                "POST",
                origin + "/api/v1/processes",
                process_spec,
            )
        except Exception as exc:
            _write_signed_once(
                state / f"{name}.process-post-failure.json",
                _post_failure_value(
                    operation="process_start",
                    cell=cell,
                    name=name,
                    protocol_sha256=protocol["protocol_sha256"],
                    execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
                    claim_path=process_claim,
                    error=exc,
                ),
            )
            raise
        pid = response.get("pid")
        if type(pid) is not int or pid < 1:
            raise ExternalCtfError("process_create_response_invalid")
        receipt = {
            **created,
            "pid": pid,
            "worker_sha256": worker_sha256,
            "start_route_preflight_receipt_sha256": route_preflight_receipt_sha256,
            "fleet_account_sha256": fleet_account_sha256,
            "process_claim_file_sha256": file_digest(process_claim.read_bytes()),
        }
        _write_once(process_receipt, receipt)
        return receipt


def reconcile_start(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    """Bind one unique worker process after an ambiguous POST without redispatch."""

    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    claim_path = state / f"{name}.process-claim.json"
    receipt_path = state / f"{name}.process.json"
    with _state_lock(state):
        if receipt_path.exists():
            return _read_canonical(receipt_path, "process")
        claim = _read_canonical(claim_path, "process_claim")
        created_path = state / f"{name}.created.json"
        created = _read_canonical(created_path, "created")
        if (
            claim.get("schema") != "external_ctf_process_claim_v1"
            or any(claim.get(key) != value for key, value in cell.items())
            or claim.get("name") != name
            or claim.get("sandbox_id") != created.get("sandbox_id")
            or claim.get("protocol_sha256") != protocol["protocol_sha256"]
            or claim.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or created.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or claim.get("created_receipt_file_sha256") != file_digest(created_path.read_bytes())
            or not isinstance(claim.get("worker_sha256"), str)
            or not isinstance(claim.get("process_spec_sha256"), str)
            or (
                claim.get("start_route_preflight_receipt_sha256") is not None
                if arm == "qualification"
                else not _is_digest(claim.get("start_route_preflight_receipt_sha256"))
            )
            or (
                claim.get("fleet_account_sha256") is not None
                if arm == "qualification"
                else not _is_digest(claim.get("fleet_account_sha256"))
            )
        ):
            raise ExternalCtfError("process_claim_binding_mismatch")
        client = _client()
        detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
        origin = detail.get("sandbox_url")
        if not isinstance(origin, str):
            raise ExternalCtfError("sandbox_not_runnable")
        response = client.request("GET", origin + "/api/v1/processes")
        rows = response.get("processes") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            raise ExternalCtfError("process_inventory_invalid")
        candidates = [
            row
            for row in rows
            if isinstance(row, dict)
            and type(row.get("pid")) is int
            and row["pid"] > 0
            and row.get("status") in {"running", "exited"}
        ]
        if len(candidates) != 1:
            raise ExternalCtfError("process_start_not_uniquely_reconciled")
        receipt = {
            **created,
            "pid": candidates[0]["pid"],
            "worker_sha256": claim["worker_sha256"],
            "start_route_preflight_receipt_sha256": claim["start_route_preflight_receipt_sha256"],
            "fleet_account_sha256": claim["fleet_account_sha256"],
            "process_claim_file_sha256": file_digest(claim_path.read_bytes()),
            "reconciled_after_ambiguous_start": True,
        }
        _write_once(receipt_path, receipt)
        return receipt


def abort_start_absent(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    """Seal a definitive failed process POST after repeated zero-process inventories."""

    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    claim_path = state / f"{name}.process-claim.json"
    failure_path = state / f"{name}.process-post-failure.json"
    receipt_path = state / f"{name}.process.json"
    terminal_path = state / f"{name}.terminal.json"
    with _state_lock(state):
        if receipt_path.exists():
            raise ExternalCtfError("process_exists_and_cannot_be_aborted_absent")
        claim = _read_canonical(claim_path, "process_claim")
        created_path = state / f"{name}.created.json"
        created = _read_canonical(created_path, "created")
        if (
            claim.get("schema") != "external_ctf_process_claim_v1"
            or any(claim.get(key) != value for key, value in cell.items())
            or claim.get("name") != name
            or claim.get("sandbox_id") != created.get("sandbox_id")
            or claim.get("protocol_sha256") != protocol["protocol_sha256"]
            or claim.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or created.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or claim.get("created_receipt_file_sha256") != file_digest(created_path.read_bytes())
        ):
            raise ExternalCtfError("process_claim_binding_mismatch")
        failure = _read_definitive_post_failure(
            failure_path,
            operation="process_start",
            cell=cell,
            name=name,
            protocol_sha256=protocol["protocol_sha256"],
            execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
            claim_path=claim_path,
        )
        client = _client()
        observations = []
        for attempt in range(1, ABSENCE_RECONCILE_ATTEMPTS + 1):
            detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
            origin = detail.get("sandbox_url")
            if not isinstance(origin, str):
                raise ExternalCtfError("sandbox_not_runnable")
            response = client.request("GET", origin + "/api/v1/processes")
            rows = response.get("processes") if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ExternalCtfError("process_inventory_invalid")
            candidates = [
                row
                for row in rows
                if isinstance(row, dict)
                and type(row.get("pid")) is int
                and row["pid"] > 0
                and row.get("status") in {"running", "exited"}
            ]
            if candidates:
                raise ExternalCtfError("process_start_absence_not_confirmed")
            observations.append(
                {
                    "attempt": attempt,
                    "process_inventory_count": len(rows),
                    "process_inventory_sha256": file_digest(canonical(rows)),
                }
            )
            if attempt < ABSENCE_RECONCILE_ATTEMPTS:
                time.sleep(ABSENCE_RECONCILE_DELAY_SECONDS)
        terminal = _write_signed_once(
            terminal_path,
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "sandbox_id": created["sandbox_id"],
                "pid": None,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": (
                    runtime_qualification_contract_sha256(protocol, benchmark)
                    if arm == "qualification"
                    else None
                ),
                "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                "provider_post_failure_receipt_sha256": failure["receipt_sha256"],
                "absence_reconciliation": observations,
                "result": None,
                "outcome": "infrastructure_invalid",
                "infrastructure_error_class": "process_start_definitive_failure_absent",
            },
        )
        return _terminal_summary(terminal)


def abort_unstarted(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    """Seal a created sandbox with no dispatched worker as infrastructure-invalid."""

    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    name = cell_name(benchmark, task_index, arm)
    created_path = state / f"{name}.created.json"
    process_claim = state / f"{name}.process-claim.json"
    process_receipt = state / f"{name}.process.json"
    terminal_path = state / f"{name}.terminal.json"
    with _state_lock(state):
        if process_claim.exists() or process_receipt.exists():
            raise ExternalCtfError("unstarted_abort_requires_no_process_claim")
        created = _read_canonical(created_path, "created")
        if (
            created.get("schema") != "external_ctf_sandbox_created_v1"
            or any(created.get(key) != value for key, value in cell.items())
            or created.get("name") != name
            or not isinstance(created.get("sandbox_id"), str)
            or created.get("protocol_sha256") != protocol["protocol_sha256"]
            or created.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
        ):
            raise ExternalCtfError("created_receipt_binding_mismatch")
        client = _client()
        observations = []
        for attempt in range(1, ABSENCE_RECONCILE_ATTEMPTS + 1):
            detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
            origin = detail.get("sandbox_url")
            if not isinstance(origin, str):
                raise ExternalCtfError("sandbox_not_runnable")
            response = client.request("GET", origin + "/api/v1/processes")
            rows = response.get("processes") if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ExternalCtfError("process_inventory_invalid")
            if any(
                isinstance(row, dict)
                and type(row.get("pid")) is int
                and row["pid"] > 0
                and row.get("status") in {"running", "exited"}
                for row in rows
            ):
                raise ExternalCtfError("unstarted_abort_process_exists")
            observations.append(
                {
                    "attempt": attempt,
                    "process_inventory_count": len(rows),
                    "process_inventory_sha256": file_digest(canonical(rows)),
                }
            )
            if attempt < ABSENCE_RECONCILE_ATTEMPTS:
                time.sleep(ABSENCE_RECONCILE_DELAY_SECONDS)
        terminal = _write_signed_once(
            terminal_path,
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "sandbox_id": created["sandbox_id"],
                "pid": None,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": (
                    runtime_qualification_contract_sha256(protocol, benchmark)
                    if arm == "qualification"
                    else None
                ),
                "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                "absence_reconciliation": observations,
                "result": None,
                "outcome": "infrastructure_invalid",
                "infrastructure_error_class": "sandbox_unstarted_zero_process_reconciled",
            },
        )
        return _terminal_summary(terminal)


def _bound_process(
    *,
    protocol: dict[str, Any],
    authority: dict[str, Any],
    state: Path,
    cell: dict[str, Any],
    name: str,
) -> tuple[dict[str, Any], Path]:
    path = state / f"{name}.process.json"
    process = _read_canonical(path, "process")
    if (
        process.get("name") != name
        or any(process.get(key) != value for key, value in cell.items())
        or not isinstance(process.get("sandbox_id"), str)
        or type(process.get("pid")) is not int
        or process["pid"] < 1
        or process.get("protocol_sha256") != protocol["protocol_sha256"]
        or process.get("execution_packet_receipt_sha256")
        != authority["execution_packet_receipt_sha256"]
        or process.get("worker_sha256") != protocol["execution"]["worker_sha256"]
        or (
            process.get("start_route_preflight_receipt_sha256") is not None
            if cell["arm"] == "qualification"
            else not _is_digest(process.get("start_route_preflight_receipt_sha256"))
        )
        or (
            process.get("fleet_account_sha256") is not None
            if cell["arm"] == "qualification"
            else not _is_digest(process.get("fleet_account_sha256"))
        )
        or process.get("capacity_successor_receipt_sha256")
        != authority["capacity_successor_receipt_sha256"]
        or process.get("capacity_successor_state_receipt_sha256")
        != authority["capacity_successor_state_receipt_sha256"]
        or process.get("shared_capacity_roster_receipt_sha256")
        != authority["shared_capacity_roster_receipt_sha256"]
        or not _is_digest(process.get("live_owner_receipt_sha256"))
    ):
        raise ExternalCtfError("process_receipt_binding_mismatch")
    claim_path = state / f"{name}.process-claim.json"
    if process.get("process_claim_file_sha256") != file_digest(claim_path.read_bytes()):
        raise ExternalCtfError("process_receipt_binding_mismatch")
    return process, path


def _object_digest_matches(value: object, observed: object) -> bool:
    return observed is None if value is None else observed == file_digest(canonical(value))


def _external_qualification_result(
    value: dict[str, Any], *, protocol: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any] | None:
    benchmark = protocol["benchmarks"][cell["benchmark"]]
    task_index = cell["task_index"]
    expected_name = cell_name(cell["benchmark"], task_index, "qualification")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    status = value.get("status")
    error_code = value.get("error_code")
    admitted = value.get("qualification_name") == expected_name
    expected_schema = {
        "nyu_ctf_web_test": (
            "external_ctf_nyu_runtime_qualification_execution_v1"
            if admitted
            else "external_ctf_nyu_runtime_qualification_precondition_v1"
        ),
        "cybench_web": (
            "cybench_remote_runtime_qualification_task_execution_v1"
            if admitted
            else "cybench_remote_runtime_qualification_precondition_v1"
        ),
    }[cell["benchmark"]]
    common_keys = {
        "schema",
        "protocol_sha256",
        "qualification_contract_sha256",
        "benchmark",
        "qualification_name",
        "task_index",
        "task_id_sha256",
        "source_commit",
        "task_source",
        "task_source_sha256",
        "published_port_entries_removed",
        "container_names_removed",
        "harness_image_id",
        "harness_image",
        "harness_image_sha256",
        "platform",
        "status",
        "error_code",
        "provider_calls",
        "model_requests",
        "scored_attempts",
        "contains_prompts_flags_solutions_traces_or_scores",
        "receipt_sha256",
    }
    benchmark_keys = {
        "nyu_ctf_web_test": {
            "challenge_runtime_images",
            "challenge_runtime_images_sha256",
            "target_address_binding",
            "qualification",
            "official_solution_solvability_claimed",
        },
        "cybench_web": {
            "controller_image_reference",
            "controller_image",
            "controller_image_sha256",
            "challenge_images",
            "challenge_image_set_sha256",
            "official_solution_manifest_sha256",
            "official_solution_helper_sha256",
            "task_evidence",
        },
    }[cell["benchmark"]]
    if (
        set(value) != common_keys | benchmark_keys
        or value.get("receipt_sha256") != file_digest(canonical(unsigned))
        or value.get("schema") != expected_schema
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("qualification_contract_sha256")
        != runtime_qualification_contract_sha256(protocol, cell["benchmark"])
        or value.get("benchmark") != cell["benchmark"]
        or value.get("task_index") != task_index
        or value.get("task_id_sha256") != file_digest(cell["task_id"].encode())
        or value.get("source_commit") != benchmark["source"]["commit"]
        or not _object_digest_matches(value.get("task_source"), value.get("task_source_sha256"))
        or value.get("harness_image_id") != benchmark["harness"]["image_id"]
        or not _object_digest_matches(value.get("harness_image"), value.get("harness_image_sha256"))
        or value.get("platform") != "linux/amd64"
        or value.get("provider_calls") != 0
        or value.get("model_requests") != 0
        or value.get("scored_attempts") != 0
        or value.get("contains_prompts_flags_solutions_traces_or_scores") is not False
        or status not in {"runtime_qualified", "infrastructure_invalid", "precondition_failed"}
        or (
            error_code is not None
            and (not isinstance(error_code, str) or re.fullmatch(r"[a-z0-9_]+", error_code) is None)
        )
        or (status == "runtime_qualified") is (error_code is not None)
        or (status == "precondition_failed") is admitted
        or (status != "precondition_failed") is not admitted
    ):
        return None
    harness = value.get("harness_image")
    if status == "runtime_qualified" and (
        not isinstance(harness, dict) or harness.get("image_id") != benchmark["harness"]["image_id"]
    ):
        return None
    if cell["benchmark"] == "nyu_ctf_web_test":
        images = value.get("challenge_runtime_images")
        qualification = value.get("qualification")
        if (
            not _object_digest_matches(images, value.get("challenge_runtime_images_sha256"))
            or value.get("official_solution_solvability_claimed") is not False
        ):
            return None
        if status == "runtime_qualified":
            task_source = value.get("task_source")
            if (
                not isinstance(qualification, dict)
                or qualification.get("receipt_sha256")
                != file_digest(
                    canonical(
                        {
                            key: item
                            for key, item in qualification.items()
                            if key != "receipt_sha256"
                        }
                    )
                )
                or qualification.get("schema") != "external_ctf_nyu_runtime_qualification_v1"
                or not _is_digest(qualification.get("protocol_sha256"))
                or qualification.get("qualification_contract_sha256")
                != value["qualification_contract_sha256"]
                or qualification.get("benchmark") != cell["benchmark"]
                or qualification.get("task_id") != cell["task_id"]
                or qualification.get("source_commit") != benchmark["source"]["commit"]
                or not isinstance(task_source, dict)
                or qualification.get("challenge_sha256") != task_source.get("challenge_sha256")
                or qualification.get("compose_sha256") != task_source.get("compose_sha256")
                or qualification.get("attachment_manifest_sha256")
                != task_source.get("attachment_manifest_sha256")
                or qualification.get("status") != "runtime_preflight_passed"
                or qualification.get("model_requests") != 0
                or qualification.get("startup_healthy") is not True
                or qualification.get("target_reachable") is not True
                or qualification.get("grader_negative_control") is not False
                or qualification.get("grader_positive_control") is not True
                or qualification.get("agent_image_id") != benchmark["harness"]["image_id"]
                or qualification.get("opencode_version") != benchmark["harness"]["version"]
                or qualification.get("platform") != "linux/amd64"
                or qualification.get("isolated_task_network") is not True
                or qualification.get("images") != images
                or not isinstance(value.get("target_address_binding"), str)
                or not value["target_address_binding"]
            ):
                return None
        elif qualification is not None:
            return None
    else:
        images = value.get("challenge_images")
        expected_controller = benchmark["runtime_qualification"]["controller_image"]
        if (
            value.get("controller_image_reference") != expected_controller
            or not _object_digest_matches(
                value.get("controller_image"), value.get("controller_image_sha256")
            )
            or not _object_digest_matches(images, value.get("challenge_image_set_sha256"))
        ):
            return None
        if status == "runtime_qualified":
            from . import cybench_qualification

            expected = cybench_qualification.expected_remote_runtime_evidence(protocol)["tasks"][
                task_index
            ]
            if value.get("task_evidence") != expected:
                return None
        elif value.get("task_evidence") is not None:
            return None
    return value


def _result_value(
    raw: bytes,
    *,
    protocol: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    common = {
        "schema": (
            "external_ctf_runtime_qualification_v1"
            if cell["arm"] == "qualification"
            else "external_ctf_cell_result_v1"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": cell["benchmark"],
        "task_id": cell["task_id"],
        "arm": cell["arm"],
    }
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        return None
    if cell["arm"] == "qualification":
        if cell["benchmark"] != "cvebench_zero_day":
            return _external_qualification_result(value, protocol=protocol, cell=cell)
        task = value.get("task")
        toolchain = value.get("toolchain")
        inspect_binding = protocol["benchmarks"]["cvebench_zero_day"]["harness"]["inspect_ai"]
        canary_task_id = protocol["operational_canary"]["task_id"]
        is_canary = cell["task_id"] == canary_task_id
        if (
            set(value)
            != {
                *common,
                "qualification_contract_sha256",
                "status",
                "source_commit",
                "kali_size",
                "model_requests",
                "docker_available",
                "compose_available",
                "toolchain",
                "positive_grader_control",
                "task",
                "task_sha256",
            }
            or any(value.get(key) != item for key, item in common.items())
            or value.get("qualification_contract_sha256")
            != runtime_qualification_contract_sha256(protocol, cell["benchmark"])
            or value.get("status") != "runtime_preflight_passed"
            or value.get("source_commit")
            != protocol["benchmarks"]["cvebench_zero_day"]["source"]["commit"]
            or value.get("kali_size") != "large"
            or value.get("model_requests") != 0
            or value.get("docker_available") is not True
            or value.get("compose_available") is not True
            or not isinstance(toolchain, dict)
            or set(toolchain)
            != {
                "docker_client_version",
                "docker_server_version",
                "docker_compose_version",
                "inspect_ai_version",
                "inspect_limit_behavior_sha256",
            }
            or any(
                not isinstance(toolchain.get(field), str)
                or not toolchain[field]
                or len(toolchain[field]) > 64
                or any(character.isspace() for character in toolchain[field])
                for field in (
                    "docker_client_version",
                    "docker_server_version",
                    "docker_compose_version",
                )
            )
            or toolchain.get("inspect_ai_version") != inspect_binding["version"]
            or toolchain.get("inspect_limit_behavior_sha256")
            != inspect_binding["limit_behavior_source_sha256"]
            or value.get("positive_grader_control") != ("passed" if is_canary else "not_run")
            or not isinstance(task, dict)
            or set(task)
            != {
                "task_id",
                "startup_healthy",
                "checker_before_solution",
                "official_solution_executed",
                "checker_after_solution",
                "images",
                "image_set_sha256",
            }
            or task.get("task_id") != cell["task_id"]
            or task.get("startup_healthy") is not True
            or task.get("checker_before_solution") is not False
            or task.get("official_solution_executed") is not is_canary
            or task.get("checker_after_solution") is not (True if is_canary else None)
            or not isinstance(task.get("images"), list)
            or not task["images"]
            or any(not isinstance(image, dict) for image in task["images"])
            or task["images"]
            != sorted(task["images"], key=lambda image: str(image.get("reference")))
            or len({image.get("reference") for image in task["images"] if isinstance(image, dict)})
            != len(task["images"])
            or any(
                not isinstance(image, dict)
                or set(image) != {"reference", "image_id", "repository_digest"}
                or not isinstance(image["reference"], str)
                or not image["reference"]
                or not _is_digest(image["image_id"])
                or not isinstance(image["repository_digest"], str)
                or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image["repository_digest"]) is None
                for image in task["images"]
            )
            or task.get("image_set_sha256") != file_digest(canonical(task["images"]))
            or value.get("task_sha256") != file_digest(canonical(task))
        ):
            return None
        return value
    if value.get("status") == "scored":
        expected = {
            **common,
            "status": "scored",
            "score": value.get("score"),
            "grader": protocol["benchmarks"][cell["benchmark"]]["scoring"],
        }
        if (
            set(value) != set(expected)
            or type(value.get("score")) is not int
            or value["score"] not in {0, 1}
        ):
            return None
        return value if value == expected else None
    expected = {
        **common,
        "status": "infrastructure_invalid",
        "error_class": value.get("error_class"),
    }
    if (
        set(value) != set(expected)
        or not isinstance(value.get("error_class"), str)
        or not value["error_class"]
    ):
        return None
    return value if value == expected else None


def _read_result_with_reconciliation(
    client: TensorlakeClient,
    origin: str,
    *,
    protocol: dict[str, Any],
    cell: dict[str, Any],
) -> tuple[bytes | None, dict[str, Any] | None, int]:
    """Bound eventual file visibility before sealing an irreversible terminal."""

    url = (
        origin
        + "/api/v1/files?"
        + urllib.parse.urlencode({"path": "/workspace/external-ctf-result.json"})
    )
    last_raw: bytes | None = None
    for attempt in range(1, RESULT_RECONCILE_ATTEMPTS + 1):
        try:
            raw = client.request("GET", url, raw=True, max_response_bytes=1048576)
            if isinstance(raw, bytes):
                last_raw = raw
                result = _result_value(raw, protocol=protocol, cell=cell)
                if result is not None:
                    return raw, result, attempt
        except Exception:
            pass
        if attempt < RESULT_RECONCILE_ATTEMPTS:
            time.sleep(RESULT_RECONCILE_DELAY_SECONDS)
    return last_raw, None, RESULT_RECONCILE_ATTEMPTS


def _terminal_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": value["name"],
        "status": "terminal_receipt_sealed",
        "outcome": value["outcome"],
        "receipt_sha256": value["receipt_sha256"],
    }


def status(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    name = cell_name(benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    terminal_path = state / f"{name}.terminal.json"
    with _state_lock(state):
        if terminal_path.exists():
            terminal = _read_signed(terminal_path, "terminal")
            if (
                terminal.get("schema") != "external_ctf_cell_terminal_v1"
                or terminal.get("name") != name
                or any(terminal.get(key) != value for key, value in cell.items())
                or terminal.get("protocol_sha256") != protocol["protocol_sha256"]
                or terminal.get("qualification_contract_sha256")
                != _qualification_contract_binding(protocol, cell)
                or terminal.get("execution_packet_receipt_sha256")
                != authority["execution_packet_receipt_sha256"]
            ):
                raise ExternalCtfError("terminal_receipt_binding_mismatch")
            return _terminal_summary(terminal)
        process, process_path = _bound_process(
            protocol=protocol,
            authority=authority,
            state=state,
            cell=cell,
            name=name,
        )
        client = _client()
        detail = client.request("GET", API + "/sandboxes/" + process["sandbox_id"])
        origin = detail.get("sandbox_url")
        if not isinstance(origin, str):
            raise ExternalCtfError("sandbox_process_status_ambiguous")
        response = client.request("GET", origin + "/api/v1/processes")
        rows = response.get("processes") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            raise ExternalCtfError("process_inventory_invalid")
        matches = [
            row for row in rows if isinstance(row, dict) and row.get("pid") == process["pid"]
        ]
        if len(matches) != 1:
            raise ExternalCtfError("sandbox_process_status_ambiguous")
        row = matches[0]
        if row.get("status") == "running":
            return {"name": name, "status": "running"}
        if row.get("status") != "exited" or type(row.get("exit_code")) is not int:
            raise ExternalCtfError("sandbox_process_status_ambiguous")
        result_raw, result, result_read_attempts = _read_result_with_reconciliation(
            client,
            origin,
            protocol=protocol,
            cell=cell,
        )
        preflight_passed = (
            cell["arm"] == "qualification"
            and row["exit_code"] == 0
            and result is not None
            and result["status"] in {"runtime_preflight_passed", "runtime_qualified"}
        )
        accepted = (
            cell["arm"] != "qualification"
            and row["exit_code"] == 0
            and result is not None
            and result["status"] == "scored"
        )
        terminal = _write_signed_once(
            terminal_path,
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "sandbox_id": process["sandbox_id"],
                "pid": process["pid"],
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": (
                    runtime_qualification_contract_sha256(protocol, cell["benchmark"])
                    if cell["arm"] == "qualification"
                    else None
                ),
                "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                "worker_sha256": process["worker_sha256"],
                "start_route_preflight_receipt_sha256": process[
                    "start_route_preflight_receipt_sha256"
                ],
                "fleet_account_sha256": process["fleet_account_sha256"],
                "process_receipt_file_sha256": file_digest(process_path.read_bytes()),
                "process_status": {
                    key: row.get(key) for key in ("pid", "status", "exit_code", "signal")
                },
                "result_read_attempts": result_read_attempts,
                "result_reconcile_exhausted": result is None,
                "result_file_sha256": (None if result_raw is None else file_digest(result_raw)),
                "result": result,
                "outcome": (
                    "runtime_preflight_passed"
                    if preflight_passed
                    else ("accepted_model_outcome" if accepted else "infrastructure_invalid")
                ),
                "infrastructure_error_class": (
                    None
                    if preflight_passed or accepted
                    else (
                        result.get("error_code")
                        if cell["arm"] == "qualification"
                        and isinstance(result, dict)
                        and isinstance(result.get("error_code"), str)
                        else "worker_or_result_invalid"
                    )
                ),
            },
        )
        return _terminal_summary(terminal)


def release(
    *,
    protocol_path: Path,
    retry_execution_path: Path,
    execution_packet_path: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    name = cell_name(benchmark, task_index, arm)
    authority, packet = _execution_context(
        protocol,
        retry_execution_path,
        execution_packet_path,
        require_live_owner=False,
        require_current_source=False,
    )
    _require_packet_allows_cell(packet, protocol, cell)
    state = authority["external_state"]
    created = _read_canonical(state / f"{name}.created.json", "created")
    if (
        created.get("name") != name
        or not isinstance(created.get("sandbox_id"), str)
        or created.get("execution_packet_receipt_sha256")
        != authority["execution_packet_receipt_sha256"]
    ):
        raise ExternalCtfError("created_receipt_binding_mismatch")
    process_path = state / f"{name}.process.json"
    process_claim = state / f"{name}.process-claim.json"
    terminal_path = state / f"{name}.terminal.json"
    terminal = None
    if process_claim.exists() and not process_path.exists():
        if not terminal_path.exists():
            raise ExternalCtfError("ambiguous_start_requires_reconciliation")
        terminal = _read_signed(terminal_path, "terminal")
        if (
            terminal.get("name") != name
            or any(terminal.get(key) != value for key, value in cell.items())
            or terminal.get("sandbox_id") != created["sandbox_id"]
            or terminal.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or terminal.get("outcome") != "infrastructure_invalid"
            or terminal.get("infrastructure_error_class")
            != "process_start_definitive_failure_absent"
        ):
            raise ExternalCtfError("ambiguous_start_requires_reconciliation")
    if process_path.exists():
        process, _path = _bound_process(
            protocol=protocol,
            authority=authority,
            state=state,
            cell=cell,
            name=name,
        )
        if not terminal_path.exists():
            raise ExternalCtfError("terminal_receipt_required_before_release")
        terminal = _read_signed(terminal_path, "terminal")
        if (
            terminal.get("sandbox_id") != process["sandbox_id"]
            or terminal.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
        ):
            raise ExternalCtfError("terminal_receipt_binding_mismatch")
    elif not process_claim.exists() and terminal_path.exists():
        terminal = _read_signed(terminal_path, "terminal")
        if (
            terminal.get("name") != name
            or any(terminal.get(key) != value for key, value in cell.items())
            or terminal.get("sandbox_id") != created["sandbox_id"]
            or terminal.get("execution_packet_receipt_sha256")
            != authority["execution_packet_receipt_sha256"]
            or terminal.get("outcome") != "infrastructure_invalid"
            or terminal.get("infrastructure_error_class")
            != "sandbox_unstarted_zero_process_reconciled"
        ):
            raise ExternalCtfError("terminal_receipt_binding_mismatch")
    if terminal is None:
        raise ExternalCtfError("terminal_receipt_required_before_release")
    if terminal.get("qualification_contract_sha256") != _qualification_contract_binding(
        protocol, cell
    ):
        raise ExternalCtfError("terminal_receipt_binding_mismatch")
    client = _client()
    client.request("DELETE", API + "/sandboxes/" + created["sandbox_id"], raw=True)
    for _ in range(60):
        detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
        if detail.get("status") == "terminated":
            receipt = _write_signed_once(
                state / f"{name}.released.json",
                {
                    "schema": "external_ctf_sandbox_release_v1",
                    **cell,
                    "name": name,
                    "sandbox_id": created["sandbox_id"],
                    "status": "terminated",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "execution_packet_receipt_sha256": authority["execution_packet_receipt_sha256"],
                    "terminal_receipt_sha256": (
                        None if terminal is None else terminal["receipt_sha256"]
                    ),
                },
            )
            with replica_set._shared_tensorlake_create_lock(authority["state"]):  # noqa: SLF001
                try:
                    replica_set.release_shared_capacity_slot(
                        state=authority["state"],
                        owned_names=authority["owned_names"],
                        sandbox_name=name,
                        reservation_receipt_sha256=created["capacity_reservation_receipt_sha256"],
                        provider_state="terminated",
                        terminal_receipt_sha256=receipt["receipt_sha256"],
                    )
                except replica_set.CollectionReplicaSetError as exc:
                    raise ExternalCtfError(str(exc)) from exc
            if (
                arm == "qualification"
                and benchmark == "cvebench_zero_day"
                and packet.get("packet_mode") == "scored"
                and terminal is not None
                and terminal.get("outcome") == "infrastructure_invalid"
            ):
                _seal_qualification_pair_skip(
                    state,
                    protocol=protocol,
                    task_index=task_index,
                    qualification_terminal=terminal,
                    execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
                )
            return receipt
        time.sleep(2)
    raise ExternalCtfError("sandbox_release_not_confirmed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--web-retry-execution", type=Path)
    parser.add_argument("--execution-packet", type=Path)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--route-preflight", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create")
    sub.add_parser("reconcile-create")
    sub.add_parser("abort-create-absent")
    sub.add_parser("start")
    sub.add_parser("reconcile-start")
    sub.add_parser("abort-start-absent")
    sub.add_parser("abort-unstarted")
    sub.add_parser("status")
    sub.add_parser("release")
    sub.add_parser("seal-cve-continuation")
    route_preflight = sub.add_parser("seal-route-preflight")
    route_preflight.add_argument("--kubernetes-context", required=True)
    route_preflight.add_argument("--output", type=Path, required=True)
    roster = sub.add_parser("seal-capacity-roster")
    roster.add_argument("--output", type=Path, required=True)
    roster_successor = sub.add_parser("seal-capacity-roster-successor")
    roster_successor.add_argument("--predecessor-roster", type=Path, required=True)
    roster_successor.add_argument("--expected-predecessor-roster-file-sha256", required=True)
    roster_successor.add_argument("--expected-predecessor-roster-receipt-sha256", required=True)
    roster_successor.add_argument("--retired-terminal", type=Path, required=True)
    roster_successor.add_argument("--retired-release", type=Path, required=True)
    roster_successor.add_argument("--retired-capacity-release", type=Path, required=True)
    for label in (
        "authority",
        "batch",
        "created",
        "release",
        "capacity-reserved",
        "capacity-release",
        "owner-ready",
        "owner-closed",
    ):
        roster_successor.add_argument(f"--web-{label}", type=Path, required=True)
    roster_successor.add_argument("--output", type=Path, required=True)
    bind_roster_successor = sub.add_parser("bind-capacity-roster-successor")
    bind_roster_successor.add_argument("--successor-roster", type=Path, required=True)
    bind_roster_successor.add_argument("--expected-successor-roster-file-sha256", required=True)
    bind_roster_successor.add_argument("--expected-successor-roster-receipt-sha256", required=True)
    args = parser.parse_args()
    if args.command == "seal-capacity-roster":
        print(
            json.dumps(
                seal_capacity_roster(protocol_path=args.protocol, output_path=args.output),
                sort_keys=True,
            )
        )
        return
    if args.command == "seal-capacity-roster-successor":
        print(
            json.dumps(
                seal_capacity_roster_successor(
                    protocol_path=args.protocol,
                    predecessor_roster_path=args.predecessor_roster,
                    expected_predecessor_roster_file_sha256=(
                        args.expected_predecessor_roster_file_sha256
                    ),
                    expected_predecessor_roster_receipt_sha256=(
                        args.expected_predecessor_roster_receipt_sha256
                    ),
                    retired_terminal_path=args.retired_terminal,
                    retired_release_path=args.retired_release,
                    retired_capacity_release_path=args.retired_capacity_release,
                    web_authority_path=args.web_authority,
                    web_batch_path=args.web_batch,
                    web_created_path=args.web_created,
                    web_release_path=args.web_release,
                    web_capacity_reserved_path=args.web_capacity_reserved,
                    web_capacity_release_path=args.web_capacity_release,
                    web_owner_ready_path=args.web_owner_ready,
                    web_owner_closed_path=args.web_owner_closed,
                    output_path=args.output,
                ),
                sort_keys=True,
            )
        )
        return
    if args.command == "bind-capacity-roster-successor":
        if args.web_retry_execution is None:
            parser.error("--web-retry-execution is required")
        print(
            json.dumps(
                bind_capacity_roster_successor(
                    protocol_path=args.protocol,
                    retry_execution_path=args.web_retry_execution,
                    successor_roster_path=args.successor_roster,
                    expected_successor_roster_file_sha256=(
                        args.expected_successor_roster_file_sha256
                    ),
                    expected_successor_roster_receipt_sha256=(
                        args.expected_successor_roster_receipt_sha256
                    ),
                ),
                sort_keys=True,
            )
        )
        return
    if args.command == "seal-cve-continuation":
        if args.web_retry_execution is None or args.execution_packet is None:
            parser.error("--web-retry-execution and --execution-packet are required")
        protocol = load_protocol(args.protocol)
        authority, _packet = _execution_context(
            protocol,
            args.web_retry_execution,
            args.execution_packet,
            require_live_owner=False,
            require_current_source=True,
        )
        print(
            json.dumps(
                seal_cve_continuation(
                    protocol=protocol,
                    state=authority["external_state"],
                    execution_packet_receipt_sha256=authority["execution_packet_receipt_sha256"],
                ),
                sort_keys=True,
            )
        )
        return
    if args.command == "seal-route-preflight":
        if (
            args.web_retry_execution is None
            or args.execution_packet is None
            or args.benchmark is None
            or args.task_index is None
            or args.arm is None
        ):
            parser.error(
                "--web-retry-execution, --execution-packet, --benchmark, --task-index "
                "and --arm are required"
            )
        print(
            json.dumps(
                seal_start_route_preflight(
                    protocol_path=args.protocol,
                    retry_execution_path=args.web_retry_execution,
                    execution_packet_path=args.execution_packet,
                    benchmark=args.benchmark,
                    task_index=args.task_index,
                    arm=args.arm,
                    kubernetes_context=args.kubernetes_context,
                    output_path=args.output,
                ),
                sort_keys=True,
            )
        )
        return
    if (
        args.web_retry_execution is None
        or args.execution_packet is None
        or args.benchmark is None
        or args.task_index is None
        or args.arm is None
    ):
        parser.error(
            "--web-retry-execution, --execution-packet, --benchmark, --task-index and --arm "
            "are required"
        )
    common = {
        "protocol_path": args.protocol,
        "retry_execution_path": args.web_retry_execution,
        "execution_packet_path": args.execution_packet,
        "benchmark": args.benchmark,
        "task_index": args.task_index,
        "arm": args.arm,
    }
    if args.command == "create":
        value = create(**common, route_preflight_path=args.route_preflight)
    elif args.command == "reconcile-create":
        value = reconcile_create(**common)
    elif args.command == "abort-create-absent":
        value = abort_create_absent(**common)
    elif args.command == "start":
        value = start(**common, route_preflight_path=args.route_preflight)
    elif args.command == "reconcile-start":
        value = reconcile_start(**common)
    elif args.command == "abort-start-absent":
        value = abort_start_absent(**common)
    elif args.command == "abort-unstarted":
        value = abort_unstarted(**common)
    elif args.command == "status":
        value = status(**common)
    elif args.command == "release":
        value = release(**common)
    else:  # pragma: no cover - argparse constrains this branch.
        raise ExternalCtfError("unsupported_command")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
