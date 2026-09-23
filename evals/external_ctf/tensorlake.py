"""Create one external-CTF TensorLake cell under the shared WEB capacity lock."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import json
import os
import re
import stat
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evals.webexploitbench.tensorlake.collection_replica_set import (
    PROJECT_ACTIVE_SANDBOX_LIMIT,
    _shared_tensorlake_create_lock,
)
from evals.webexploitbench.tensorlake.controller import API, TensorlakeClient

from .protocol import canonical, file_digest, load_protocol

WORKER = Path(__file__).with_name("worker.py")
ARMS = {"base": "b", "step_1000": "s1000"}
BENCHMARKS = {"cvebench_zero_day": "cve", "nyu_ctf_web_test": "nyu", "cybench_web": "cyb"}


class ExternalCtfError(RuntimeError):
    pass


def _read_signed(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    unsigned = dict(value)
    expected = unsigned.pop("receipt_sha256", None)
    if raw != canonical(value) + b"\n" or expected != file_digest(canonical(unsigned)):
        raise ExternalCtfError("invalid_signed_receipt")
    return value


def _read_bound(path: Path, expected: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if file_digest(raw) != expected:
        raise ExternalCtfError("bound_file_digest_mismatch")
    value = json.loads(raw)
    if raw != canonical(value) + b"\n":
        raise ExternalCtfError("bound_file_not_canonical")
    return value


def web_owned_names(set_path: Path, upgrade_path: Path) -> tuple[set[str], str]:
    """Return WEB's exact sealed names and carrier snapshot without trusting status rows."""

    receipt = _read_signed(set_path)
    names: set[str] = set()
    for replica in receipt["replicas"]:
        pair = _read_bound(Path(replica["pair_path"]), replica["pair_file_sha256"])
        for prefix in ("base", "candidate"):
            plan = _read_bound(
                Path(pair[f"{prefix}_plan_path"]), pair[f"{prefix}_plan_file_sha256"]
            )
            names.update(row["sandbox_name"] for row in plan["partitions"])
    for attempt in range(8):
        for task in range(15):
            for arm in ("base", "candidate"):
                names.add(f"wbe-p8-r{attempt:02d}-t{task:02d}-{arm}-gpt-exp-v1")
    upgrade = _read_signed(upgrade_path)
    names.update(upgrade["owned_roster"]["successor_export_names"])
    if len(names) != upgrade["project_owned_sandbox_name_count"]:
        raise ExternalCtfError("web_owned_name_count_mismatch")
    return names, receipt["snapshot_id"]


def external_names(protocol: dict[str, Any]) -> set[str]:
    return {
        cell_name(benchmark, index, arm)
        for benchmark, value in protocol["benchmarks"].items()
        for index, _task in enumerate(value["task_ids"])
        for arm in ARMS
    }


def cell_name(benchmark: str, task_index: int, arm: str) -> str:
    try:
        return f"extctf-{BENCHMARKS[benchmark]}-t{task_index:02d}-{ARMS[arm]}-v1"
    except KeyError as error:
        raise ExternalCtfError("invalid_cell_identity") from error


def active_project_count(rows: list[dict[str, Any]], names: set[str]) -> int:
    selected = [row for row in rows if row.get("name") in names]
    observed = [row.get("name") for row in selected]
    if len(observed) != len(set(observed)):
        raise ExternalCtfError("project_owned_sandbox_inventory_conflict")
    return sum(row.get("status") != "terminated" for row in selected)


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalCtfError(f"invalid_{label}_receipt") from exc
    if not isinstance(value, dict) or raw != canonical(value) + b"\n" or path.read_bytes() != raw:
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
    if task_id in value.get("source_unavailable_task_ids", []):
        raise ExternalCtfError("source_unavailable_cell_is_not_launchable")
    if task_id in value.get("execution_unavailable_task_ids", []):
        raise ExternalCtfError("execution_unavailable_cell_is_not_launchable")
    return {"benchmark": benchmark, "task_index": task_index, "task_id": task_id, "arm": arm}


def create(
    *,
    protocol_path: Path,
    set_path: Path,
    upgrade_path: Path,
    shared_state: Path,
    state: Path,
    benchmark: str,
    task_index: int,
    arm: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    web_names, snapshot_id = web_owned_names(set_path, upgrade_path)
    all_names = web_names | external_names(protocol)
    name = cell_name(benchmark, task_index, arm)
    client = _client()
    state = _private_state(state)
    claim = state / f"{name}.create-claim.json"
    created = state / f"{name}.created.json"
    if claim.exists() or created.exists():
        raise ExternalCtfError("cell_already_claimed")
    spec = {
        "name": name,
        "snapshot_id": snapshot_id,
        "resources": {"cpus": 8, "memory_mb": 32768, "disk_mb": 262144},
        "timeout_secs": 21600,
        "allow_unauthenticated_access": False,
        "network": {"allow_internet_access": True, "allow_out": []},
    }
    with _shared_tensorlake_create_lock(shared_state):
        _enforce_single_open_cell(state, target=name)
        rows = client.inventory()
        if any(row.get("name") == name for row in rows):
            raise ExternalCtfError("duplicate_sandbox_name")
        active = active_project_count(rows, all_names)
        if active >= PROJECT_ACTIVE_SANDBOX_LIMIT:
            raise ExternalCtfError("shared_project_active_sandbox_limit")
        _write_once(
            claim,
            {
                "schema": "external_ctf_sandbox_create_claim_v1",
                **cell,
                "name": name,
                "protocol_sha256": protocol["protocol_sha256"],
                "spec_sha256": file_digest(canonical(spec)),
                "active_before_create": active,
                "shared_limit": PROJECT_ACTIVE_SANDBOX_LIMIT,
            },
        )
        response = client.request("POST", API + "/sandboxes", spec)
    sandbox_id = response.get("sandbox_id")
    if not isinstance(sandbox_id, str) or re.fullmatch(r"[A-Za-z0-9._:-]+", sandbox_id) is None:
        raise ExternalCtfError("sandbox_create_response_invalid")
    receipt = {
        "schema": "external_ctf_sandbox_created_v1",
        **cell,
        "name": name,
        "sandbox_id": sandbox_id,
        "status": response.get("status"),
        "protocol_sha256": protocol["protocol_sha256"],
        "spec_sha256": file_digest(canonical(spec)),
    }
    _write_once(created, receipt)
    return receipt


def start(
    *, protocol_path: Path, state: Path, benchmark: str, task_index: int, arm: str
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    cell = _cell(protocol, benchmark, task_index, arm)
    name = cell_name(benchmark, task_index, arm)
    state = _private_state(state)
    process_claim = state / f"{name}.process-claim.json"
    process_receipt = state / f"{name}.process.json"
    with _state_lock(state):
        if process_claim.exists() or process_receipt.exists():
            raise ExternalCtfError("cell_start_already_claimed")
        _enforce_single_open_cell(state, target=name)
        created = _read_canonical(state / f"{name}.created.json", "created")
        if (
            created.get("schema") != "external_ctf_sandbox_created_v1"
            or any(created.get(key) != value for key, value in cell.items())
            or created.get("name") != name
            or created.get("protocol_sha256") != protocol["protocol_sha256"]
            or not isinstance(created.get("sandbox_id"), str)
        ):
            raise ExternalCtfError("created_receipt_binding_mismatch")
        client = _client()
        detail = client.request("GET", API + "/sandboxes/" + created["sandbox_id"])
        origin = detail.get("sandbox_url")
        if detail.get("status") != "running" or not isinstance(origin, str):
            raise ExternalCtfError("sandbox_not_runnable")
        fleet_key = os.environ.get("FLEET_API_KEY", "")
        if not fleet_key or fleet_key.strip() != fleet_key or any(
            character.isspace() for character in fleet_key
        ):
            raise ExternalCtfError("fleet_credential_missing_or_invalid")
        worker = WORKER.read_bytes()
        worker_sha256 = file_digest(worker)
        env = {
            "EXTERNAL_CTF_PROTOCOL_B64": base64.b64encode(protocol_path.read_bytes()).decode(),
            "EXTERNAL_CTF_BENCHMARK": benchmark,
            "EXTERNAL_CTF_TASK_ID": cell["task_id"],
            "EXTERNAL_CTF_ARM": arm,
            "FLEET_API_KEY": fleet_key,
            "WORKER_B64": base64.b64encode(worker).decode(),
            "WORKER_SHA256": worker_sha256,
        }
        script = (
            "import base64,hashlib,os,runpy,pathlib;"
            "b=base64.b64decode(os.environ.pop('WORKER_B64'),validate=True);"
            "assert 'sha256:'+hashlib.sha256(b).hexdigest()==os.environ.pop('WORKER_SHA256');"
            "p=pathlib.Path('/workspace/external_ctf_worker.py');p.write_bytes(b);"
            "runpy.run_path(str(p),run_name='__main__')"
        )
        _write_once(
            process_claim,
            {
                "schema": "external_ctf_process_claim_v1",
                **cell,
                "name": name,
                "sandbox_id": created["sandbox_id"],
                "protocol_sha256": protocol["protocol_sha256"],
                "worker_sha256": worker_sha256,
            },
        )
        response = client.request(
            "POST",
            origin + "/api/v1/processes",
            {
                "command": "/usr/bin/python3",
                "args": ["-c", script],
                "user": "root",
                "env": env,
                "stdin_mode": "closed",
                "stdout_mode": "discard",
                "stderr_mode": "discard",
            },
        )
        pid = response.get("pid")
        if type(pid) is not int or pid < 1:
            raise ExternalCtfError("process_create_response_invalid")
        receipt = {**created, "pid": pid, "worker_sha256": worker_sha256}
        _write_once(process_receipt, receipt)
        return receipt


def status(state: Path, benchmark: str, task_index: int, arm: str) -> dict[str, Any]:
    name = cell_name(benchmark, task_index, arm)
    state = _private_state(state)
    process = _read_canonical(state / f"{name}.process.json", "process")
    if process.get("name") != name or not isinstance(process.get("sandbox_id"), str):
        raise ExternalCtfError("process_receipt_binding_mismatch")
    client = _client()
    detail = client.request("GET", API + "/sandboxes/" + process["sandbox_id"])
    origin = detail.get("sandbox_url")
    if not isinstance(origin, str):
        return {"name": name, "sandbox_status": detail.get("status"), "process": "unknown"}
    rows = client.request("GET", origin + "/api/v1/processes").get("processes", [])
    row = next((item for item in rows if item.get("pid") == process["pid"]), None)
    result = None
    if row and row.get("status") == "exited":
        url = (
            origin
            + "/api/v1/files?"
            + urllib.parse.urlencode({"path": "/workspace/external-ctf-result.json"})
        )
        result = json.loads(client.request("GET", url, raw=True, max_response_bytes=65536))
    return {"name": name, "sandbox_status": detail.get("status"), "process": row, "result": result}


def release(state: Path, benchmark: str, task_index: int, arm: str) -> dict[str, Any]:
    name = cell_name(benchmark, task_index, arm)
    state = _private_state(state)
    process = _read_canonical(state / f"{name}.process.json", "process")
    if process.get("name") != name or not isinstance(process.get("sandbox_id"), str):
        raise ExternalCtfError("process_receipt_binding_mismatch")
    client = _client()
    client.request("DELETE", API + "/sandboxes/" + process["sandbox_id"], raw=True)
    for _ in range(60):
        detail = client.request("GET", API + "/sandboxes/" + process["sandbox_id"])
        if detail.get("status") == "terminated":
            receipt = {"name": name, "sandbox_id": process["sandbox_id"], "status": "terminated"}
            _write_once(state / f"{name}.released.json", receipt)
            return receipt
        time.sleep(2)
    raise ExternalCtfError("sandbox_release_not_confirmed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--web-set", type=Path, required=True)
    create_parser.add_argument("--web-source-upgrade", type=Path, required=True)
    create_parser.add_argument("--shared-state", type=Path, required=True)
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("release")
    args = parser.parse_args()
    common = {
        "state": args.state,
        "benchmark": args.benchmark,
        "task_index": args.task_index,
        "arm": args.arm,
    }
    if args.command == "create":
        value = create(
            protocol_path=args.protocol,
            set_path=args.web_set,
            upgrade_path=args.web_source_upgrade,
            shared_state=args.shared_state,
            **common,
        )
    elif args.command == "start":
        value = start(protocol_path=args.protocol, **common)
    elif args.command == "status":
        value = status(**common)
    else:
        value = release(**common)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
