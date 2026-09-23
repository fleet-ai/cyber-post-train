"""Small shared mechanics for provider-free external-CTF runtime gates."""

from __future__ import annotations

import copy
import io
import json
import os
import platform
import posixpath
import re
import secrets
import stat
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .protocol import canonical, git_no_replace_env, load_protocol

MODEL_CREDENTIALS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "FLEET_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HELM_API_KEY",
        "OPENAI_API_KEY",
        "TOGETHER_API_KEY",
    }
)


class RuntimeGateError(RuntimeError):
    pass


def failure_code(error: Exception) -> str:
    code = str(error)
    if isinstance(error, RuntimeGateError) and re.fullmatch(r"[a-z0-9_]+", code):
        return code
    if isinstance(error, subprocess.TimeoutExpired):
        return "runtime_command_timeout"
    if isinstance(error, subprocess.CalledProcessError):
        return "runtime_command_failed"
    return "unexpected_runtime_error"


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=check, capture_output=True, text=True, timeout=timeout, env=env
    )


def docker(*args: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return run(["docker", *args], check=check, timeout=timeout)


def require_docker_linux_amd64() -> None:
    observed = docker("version", "--format", "{{.Server.Os}}/{{.Server.Arch}}").stdout.strip()
    if observed != "linux/amd64":
        raise RuntimeGateError("docker_server_linux_amd64_required")


def require_task(
    checkout: Path, protocol_path: Path, benchmark_name: str, task_index: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeGateError("linux_amd64_required")
    if any(os.environ.get(name) for name in MODEL_CREDENTIALS):
        raise RuntimeGateError("model_credential_present")
    protocol = load_protocol(protocol_path)
    benchmark = protocol["benchmarks"][benchmark_name]
    task_ids = benchmark["task_ids"]
    if type(task_index) is not int or not 0 <= task_index < len(task_ids):
        raise RuntimeGateError("task_index_invalid")
    task_id = task_ids[task_index]
    unavailable = set(benchmark.get("source_unavailable_task_ids", ())) | set(
        benchmark.get("execution_unavailable_task_ids", ())
    )
    if task_id in unavailable:
        raise RuntimeGateError("task_runtime_unavailable")
    commit = benchmark["source"]["commit"]
    git_env = git_no_replace_env()
    head = run(
        ["git", "--no-replace-objects", "-C", str(checkout), "rev-parse", "HEAD"], env=git_env
    )
    status = run(
        ["git", "--no-replace-objects", "-C", str(checkout), "status", "--porcelain"],
        env=git_env,
    )
    if head.stdout.strip() != commit or status.stdout:
        raise RuntimeGateError("pinned_source_checkout_not_clean")
    return protocol, benchmark, task_id


def materialize(checkout: Path, commit: str, source_path: str, destination: Path) -> None:
    path = PurePosixPath(source_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeGateError("pinned_source_path_invalid")
    archive = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(checkout),
            "archive",
            "--format=tar",
            f"{commit}:{path.as_posix()}",
        ],
        check=True,
        capture_output=True,
        env=git_no_replace_env(),
    ).stdout
    destination.mkdir(mode=0o700, parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        members = bundle.getmembers()
        names = {PurePosixPath(member.name).as_posix().rstrip("/") for member in members}
        invalid = not members
        for member in members:
            member_path = PurePosixPath(member.name)
            if (
                member.islnk()
                or member.isdev()
                or member_path.is_absolute()
                or ".." in member_path.parts
                or not (member.isfile() or member.isdir() or member.issym())
            ):
                invalid = True
                break
            if not member.issym():
                continue
            target = PurePosixPath(
                posixpath.normpath((member_path.parent / member.linkname).as_posix())
            )
            target_name = target.as_posix().rstrip("/")
            if (
                PurePosixPath(member.linkname).is_absolute()
                or target_name == ".."
                or target_name.startswith("../")
                or not any(
                    name == target_name or name.startswith(target_name + "/") for name in names
                )
            ):
                invalid = True
                break
        if invalid:
            raise RuntimeGateError("pinned_source_archive_invalid")
        bundle.extractall(destination, members, filter="data")


def require_safe_compose(compose: object, root: Path) -> None:
    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict) or not services:
        raise RuntimeGateError("compose_contract_invalid")
    volumes = compose.get("volumes", {})
    if not isinstance(volumes, dict) or any(
        isinstance(value, dict) and (value.get("external") or value.get("name"))
        for value in volumes.values()
    ):
        raise RuntimeGateError("unsafe_compose_contract")
    dangerous = {
        "cap_add",
        "cgroup_parent",
        "device_cgroup_rules",
        "devices",
        "ipc",
        "network_mode",
        "pid",
        "privileged",
        "uts",
        "userns_mode",
        "volumes_from",
    }
    root = root.resolve()
    for service in services.values():
        if not isinstance(service, dict) or any(service.get(key) for key in dangerous):
            raise RuntimeGateError("unsafe_compose_contract")
        extra_hosts = service.get("extra_hosts") or []
        if isinstance(extra_hosts, dict):
            extra_hosts = [*extra_hosts.keys(), *extra_hosts.values()]
        if not isinstance(extra_hosts, list):
            raise RuntimeGateError("compose_contract_invalid")
        if any("host-gateway" in str(item).lower() for item in extra_hosts):
            raise RuntimeGateError("unsafe_compose_contract")
        if service.get("ports") is not None and not isinstance(service["ports"], list):
            raise RuntimeGateError("compose_contract_invalid")
        mounts = service.get("volumes") or []
        if not isinstance(mounts, list):
            raise RuntimeGateError("unsafe_compose_contract")
        for mount in mounts:
            if isinstance(mount, str):
                parts = mount.split(":")
                if len(parts) == 1:
                    continue
                source, mode = parts[0], parts[2:] or [""]
                if source.startswith("/") or "docker.sock" in source:
                    raise RuntimeGateError("unsafe_compose_contract")
                if source.startswith(".") and (
                    not any("ro" in item.split(",") for item in mode)
                    or not (root / source).resolve().is_relative_to(root)
                ):
                    raise RuntimeGateError("unsafe_compose_contract")
                continue
            if not isinstance(mount, dict):
                raise RuntimeGateError("unsafe_compose_contract")
            kind, source = mount.get("type", "volume"), mount.get("source")
            if kind == "bind" and (
                not isinstance(source, str)
                or source.startswith("/")
                or not mount.get("read_only")
                or not (root / source).resolve().is_relative_to(root)
            ):
                raise RuntimeGateError("unsafe_compose_contract")
            if kind not in {"bind", "volume", "tmpfs"}:
                raise RuntimeGateError("unsafe_compose_contract")


def contained_compose(compose: object, root: Path) -> tuple[Path, int, int]:
    require_safe_compose(compose, root)
    contained = copy.deepcopy(compose)
    services = contained["services"]
    removed = sum(len(service.pop("ports", []) or []) for service in services.values())
    removed_names = sum(
        service.pop("container_name", None) is not None for service in services.values()
    )
    path = root / ".qualification-source-compose.yml"
    try:
        with path.open("x") as stream:
            yaml.safe_dump(contained, stream, sort_keys=True)
    except FileExistsError as error:
        raise RuntimeGateError("qualification_control_path_exists") from error
    return path, removed, removed_names


def image_lock(reference: str, *, repository_digest: bool) -> dict[str, str | None]:
    rows = json.loads(docker("image", "inspect", reference).stdout)
    row = rows[0] if isinstance(rows, list) and len(rows) == 1 else {}
    image_id = row.get("Id")
    digests = row.get("RepoDigests") or []
    if (
        not isinstance(image_id, str)
        or len(image_id) != 71
        or not image_id.startswith("sha256:")
        or not isinstance(digests, list)
        or any(not isinstance(item, str) or "@sha256:" not in item for item in digests)
        or (repository_digest and not digests)
        or row.get("Os") != "linux"
        or row.get("Architecture") != "amd64"
    ):
        raise RuntimeGateError("resolved_linux_amd64_image_invalid")
    selected = None
    if digests:
        if "@sha256:" in reference and reference not in digests:
            raise RuntimeGateError("resolved_repository_digest_mismatch")
        selected = reference if reference in digests else sorted(digests)[0]
    return {
        "reference": reference,
        "image_id": image_id,
        "repository_digest": selected,
    }


def compose_rows(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeGateError("compose_state_invalid")
    return rows


class Sandbox:
    """One unique internal network, one Compose project, and verified teardown."""

    def __init__(self, root: Path, compose: Path, external_network: str, prefix: str):
        token = secrets.token_hex(6)
        self.project = f"{prefix}{token}"
        self.network = self.project + "net"
        override = root / ".qualification-compose.yml"
        try:
            stream = override.open("x")
        except FileExistsError as error:
            raise RuntimeGateError("qualification_control_path_exists") from error
        with stream:
            yaml.safe_dump(
                {"networks": {external_network: {"external": True, "name": self.network}}},
                stream,
                sort_keys=True,
            )
        self.command = [
            "docker",
            "compose",
            "-f",
            str(compose),
            "-f",
            str(override),
            "--project-name",
            self.project,
        ]
        self.containers: list[str] = []
        self.stack_attempted = False

    def __enter__(self) -> Sandbox:
        if docker("network", "inspect", self.network, check=False).returncode == 0:
            raise RuntimeGateError("qualification_network_already_exists")
        docker("network", "create", "--internal", self.network)
        return self

    def image_references(self) -> list[str]:
        output = run([*self.command, "config", "--images"]).stdout
        return sorted({line.strip() for line in output.splitlines() if line.strip()})

    def state(self) -> list[dict[str, Any]]:
        return compose_rows(run([*self.command, "ps", "--all", "--format", "json"]).stdout)

    def start(self, *, build: bool, timeout: int) -> None:
        self.stack_attempted = True
        flags = ["--build"] if build else ["--no-build"]
        run(
            [
                *self.command,
                "up",
                "-d",
                *flags,
                "--force-recreate",
                "--wait",
                "--wait-timeout",
                "300",
            ],
            timeout=timeout,
        )
        rows = self.state()
        if not rows or any(
            str(row.get("State", "")).lower() != "running"
            or str(row.get("Health", "")).lower() not in {"", "healthy"}
            for row in rows
        ):
            raise RuntimeGateError("challenge_startup_unhealthy")

    def running_images(self) -> list[dict[str, str | None]]:
        result = []
        for row in self.state():
            service, container = row.get("Service"), row.get("ID")
            if not isinstance(service, str) or not service or not isinstance(container, str):
                raise RuntimeGateError("challenge_container_identity_invalid")
            inspected = json.loads(docker("inspect", container).stdout)
            image_id = (
                inspected[0].get("Image")
                if isinstance(inspected, list)
                and len(inspected) == 1
                and isinstance(inspected[0], dict)
                else None
            )
            if not isinstance(image_id, str):
                raise RuntimeGateError("challenge_container_image_invalid")
            lock = image_lock(image_id, repository_digest=False)
            result.append({"service": service, **lock})
        if len({row["service"] for row in result}) != len(result):
            raise RuntimeGateError("challenge_service_identity_invalid")
        return sorted(result, key=lambda row: str(row["service"]))

    def down(self) -> None:
        run([*self.command, "down", "--volumes", "--remove-orphans"])

    def create(self, role: str, *args: str) -> str:
        name = self.project + role
        docker("create", "--name", name, *args)
        self.containers.append(name)
        docker("start", name)
        return name

    def __exit__(self, error_type, error, traceback) -> bool:
        codes = [docker("rm", "--force", name, check=False).returncode for name in self.containers]
        if self.stack_attempted:
            codes.append(
                run(
                    [*self.command, "down", "--volumes", "--remove-orphans"], check=False
                ).returncode
            )
        codes.append(docker("network", "rm", self.network, check=False).returncode)
        queries = [
            docker(
                "ps",
                "--all",
                "--filter",
                f"name=^/{name}$",
                "--format",
                "{{.ID}}",
                check=False,
            )
            for name in self.containers
        ]
        queries.append(
            docker(
                "network",
                "ls",
                "--filter",
                f"name=^{self.network}$",
                "--format",
                "{{.ID}}",
                check=False,
            )
        )
        residual = any(query.stdout.strip() for query in queries)
        query_failed = any(query.returncode for query in queries)
        if self.stack_attempted:
            compose = run([*self.command, "ps", "--all", "--format", "json"], check=False)
            query_failed |= compose.returncode != 0
            if compose.returncode == 0:
                try:
                    residual |= bool(compose_rows(compose.stdout))
                except (json.JSONDecodeError, RuntimeGateError):
                    query_failed = True
        if any(codes) or query_failed or residual:
            raise RuntimeGateError("qualification_cleanup_failed") from error
        return False


def probe(container: str, host: str, port: int) -> bool:
    script = (
        "const n=require('net'),h=process.argv[1],p=+process.argv[2];"
        "const s=n.createConnection({host:h,port:p},()=>{s.destroy();process.exit(0)});"
        "s.setTimeout(5000,()=>{s.destroy();process.exit(2)});s.on('error',()=>process.exit(3));"
    )
    for _ in range(60):
        if (
            docker("exec", container, "node", "-e", script, host, str(port), check=False).returncode
            == 0
        ):
            return True
        time.sleep(2)
    return False


def assert_isolated(
    container: str,
    network: str,
    mounts: set[tuple[str, str, bool]],
    *,
    hardened_user: bool,
) -> None:
    rows = json.loads(docker("inspect", container).stdout)
    row = rows[0] if isinstance(rows, list) and len(rows) == 1 else {}
    host = row.get("HostConfig", {})
    observed = {
        (mount.get("Source"), mount.get("Destination"), mount.get("RW"))
        for mount in row.get("Mounts") or []
        if isinstance(mount, dict)
    }
    hardened = (
        host.get("ReadonlyRootfs") is True
        and "ALL" in (host.get("CapDrop") or [])
        and "no-new-privileges" in (host.get("SecurityOpt") or [])
        and row.get("Config", {}).get("User") == "1000:1000"
    )
    if (
        host.get("Privileged") is not False
        or set(row.get("NetworkSettings", {}).get("Networks", {})) != {network}
        or observed != mounts
        or (hardened_user and not hardened)
    ):
        raise RuntimeGateError("container_isolation_invalid")


def write_once(path: Path, value: dict[str, Any]) -> None:
    ancestry = (path.parent, *path.parent.parents)
    for parent in ancestry:
        try:
            if stat.S_ISLNK(parent.lstat().st_mode):
                raise RuntimeGateError("receipt_parent_symlink_invalid")
        except FileNotFoundError:
            pass
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(stat.S_ISLNK(parent.lstat().st_mode) for parent in ancestry):
        raise RuntimeGateError("receipt_parent_symlink_invalid")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent = os.fstat(directory)
        if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
            raise RuntimeGateError("receipt_parent_not_private")
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)
