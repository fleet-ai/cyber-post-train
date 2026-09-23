"""Hermetic standard-library worker for the prod10 controller bootstrap."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

PACKET_SCHEMA = "cyber_skyrl_prod10_controller_packet_v1"
RECEIPT_SCHEMA = "cyber_skyrl_prod10_controller_bootstrap_v1"
LOG_PREFIX = "CYBER_SKYRL_PROD10_CONTROLLER_BOOTSTRAP="
ENV_DRIVER_SHA256 = "CYBER_SKYRL_CONTROLLER_DRIVER_SHA256"
ENV_DRIVER_SOURCE = "CYBER_SKYRL_CONTROLLER_DRIVER_SOURCE_B64"
ENV_PACKET = "CYBER_SKYRL_CONTROLLER_PACKET"
ENV_JOB_UID = "CYBER_SKYRL_CONTROLLER_JOB_UID"
ENV_POD_NAME = "CYBER_SKYRL_CONTROLLER_POD_NAME"
ENV_POD_UID = "CYBER_SKYRL_CONTROLLER_POD_UID"
IDENTITY_PATHS = {
    ENV_JOB_UID: "/controller/job_uid",
    ENV_POD_NAME: "/controller/pod_name",
    ENV_POD_UID: "/controller/pod_uid",
}
REQUIRED_ENV = (
    ENV_DRIVER_SHA256,
    ENV_DRIVER_SOURCE,
    ENV_PACKET,
    ENV_JOB_UID,
    ENV_POD_NAME,
    ENV_POD_UID,
)
RUNTIME_UID = 1000
RUNTIME_GID = 100
CONTROLS_MOUNT = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
CREATE_ONCE_ROOT = CONTROLS_MOUNT + "/prod9-create-once-v1"
BOOTSTRAP_ROOT = CONTROLS_MOUNT + "/prod10-controller-bootstrap-v1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,251}[a-z0-9])?")


class BootstrapError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_packet(value: object) -> dict:
    if not isinstance(value, dict) or value != _seal(value):
        raise BootstrapError("packet_digest_invalid")
    expected_keys = {
        "schema",
        "status",
        "source_commit",
        "identity_sha256",
        "plan_sha256",
        "request_sha256",
        "target_manifest_sha256",
        "target",
        "controller",
        "required_target_receipts",
        "platform_gates",
        "launch_authorized",
        "submitted",
        "sha256",
    }
    digests = (
        value.get("identity_sha256"),
        value.get("plan_sha256"),
        value.get("request_sha256"),
        value.get("target_manifest_sha256"),
    )
    target = value.get("target")
    controller = value.get("controller")
    receipts = value.get("required_target_receipts")
    gates = value.get("platform_gates")
    if (
        set(value) != expected_keys
        or value.get("schema") != PACKET_SCHEMA
        or value.get("status") != "prepared_not_authorized"
        or _COMMIT.fullmatch(value.get("source_commit", "")) is None
        or any(_SHA256.fullmatch(item or "") is None for item in digests)
        or target
        != {
            "name": "chris-q38-rlreward-prod10",
            "nodes": 1,
            "gpus": 8,
            "priority": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
        }
        or controller
        != {
            "name": "chris-q38-prod10-controller-v1",
            "context": "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
            "namespace": "fleet-train-jobs",
            "image": (
                "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/"
                "skyrl-train@sha256:"
                "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
            ),
            "runtime_user": {"uid": RUNTIME_UID, "gid": RUNTIME_GID},
            "controls_mount": CONTROLS_MOUNT,
            "create_once_root": CREATE_ONCE_ROOT,
            "bootstrap_root": BOOTSTRAP_ROOT,
            "bootstrap_receipt": (
                BOOTSTRAP_ROOT + "/" + value["plan_sha256"].removeprefix("sha256:") + ".json"
            ),
            "create_journal": (
                BOOTSTRAP_ROOT
                + "/"
                + value["plan_sha256"].removeprefix("sha256:")
                + ".create.jsonl"
            ),
        }
        or receipts
        != {
            "creator_exact_run_id_and_rayjob_uid": True,
            "observer_exact_uid_release": True,
            "terminal_reward_update_checkpoint": True,
        }
        or gates
        != {
            "immutable_run_id_and_rayjob_uid_release": False,
            "metadata_only_capacity_receipt": False,
            "dedicated_least_privilege_controller_service_account": False,
        }
        or value.get("launch_authorized") is not False
        or value.get("submitted") is not False
    ):
        raise BootstrapError("packet_contract_invalid")
    return value


def _driver_sha256(values: Mapping[str, str]) -> str:
    expected = values.get(ENV_DRIVER_SHA256, "")
    try:
        source = base64.b64decode(values.get(ENV_DRIVER_SOURCE, ""), validate=True)
    except ValueError:
        raise BootstrapError("driver_binding_invalid") from None
    actual = hashlib.sha256(source).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise BootstrapError("driver_binding_invalid")
    return actual


def _uuid(value: str, code: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        raise BootstrapError(code) from None


def _runtime_identity(
    values: Mapping[str, str], supplied: Mapping[str, str] | None
) -> dict[str, str]:
    if any(values.get(name) != path for name, path in IDENTITY_PATHS.items()):
        raise BootstrapError("runtime_identity_path_invalid")
    if supplied is None:
        try:
            result = {name: Path(path).read_text().strip() for name, path in IDENTITY_PATHS.items()}
        except OSError:
            raise BootstrapError("runtime_identity_read_failed") from None
    else:
        result = dict(supplied)
    if set(result) != set(IDENTITY_PATHS) or any(
        not isinstance(value, str) or not value for value in result.values()
    ):
        raise BootstrapError("runtime_identity_invalid")
    return result


def _real_directory(path: Path, *, owner: tuple[int, int] | None = None) -> os.stat_result:
    try:
        identity = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise BootstrapError("sfs_path_invalid") from None
    if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode) or resolved != path:
        raise BootstrapError("sfs_path_invalid")
    if owner is not None and (identity.st_uid, identity.st_gid) != owner:
        raise BootstrapError("sfs_owner_invalid")
    return identity


def _mount_options(path: str, mountinfo: str) -> set[str]:
    matches: list[set[str]] = []
    for row in mountinfo.splitlines():
        fields = row.split()
        if len(fields) < 6:
            continue
        mount_point = fields[4].replace("\\040", " ")
        if mount_point == path:
            matches.append(set(fields[5].split(",")))
    if len(matches) != 1:
        raise BootstrapError("sfs_mount_invalid")
    return matches[0]


def _write_once(path: Path, value: dict, *, owner: tuple[int, int]) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        _real_directory(path.parent, owner=owner)
    except OSError:
        raise BootstrapError("bootstrap_root_invalid") from None
    if path.exists() or path.is_symlink():
        raise BootstrapError("bootstrap_receipt_exists")
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent = os.open(path.parent, directory_flags)
        try:
            fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_json(value) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError:
        raise BootstrapError("bootstrap_receipt_write_failed") from None


def observe(
    environ: Mapping[str, str] | None = None,
    *,
    controls_root: Path | None = None,
    sfs_root: Path | None = None,
    mountinfo: str | None = None,
    runtime_user: tuple[int, int] | None = None,
    filesystem_owner: tuple[int, int] | None = None,
    runtime_identity: Mapping[str, str] | None = None,
    write: bool = True,
) -> dict:
    values = os.environ if environ is None else environ
    if not set(values).issuperset(REQUIRED_ENV):
        raise BootstrapError("environment_incomplete")
    driver_sha256 = _driver_sha256(values)
    try:
        packet = validate_packet(json.loads(values[ENV_PACKET]))
    except (json.JSONDecodeError, TypeError):
        raise BootstrapError("packet_json_invalid") from None
    user = (os.geteuid(), os.getegid()) if runtime_user is None else runtime_user
    if user != (RUNTIME_UID, RUNTIME_GID):
        raise BootstrapError("runtime_user_invalid")
    if values.get("CUDA_VISIBLE_DEVICES") != "" or values.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise BootstrapError("gpu_visibility_invalid")
    identity = _runtime_identity(values, runtime_identity)
    job_uid = _uuid(identity[ENV_JOB_UID], "job_uid_invalid")
    pod_uid = _uuid(identity[ENV_POD_UID], "pod_uid_invalid")
    pod_name = identity[ENV_POD_NAME]
    if _DNS_LABEL.fullmatch(pod_name) is None:
        raise BootstrapError("pod_name_invalid")

    controls = Path(CONTROLS_MOUNT) if controls_root is None else controls_root
    shared = Path("/mnt/sfs") if sfs_root is None else sfs_root
    expected_owner = (RUNTIME_UID, RUNTIME_GID) if filesystem_owner is None else filesystem_owner
    _real_directory(shared)
    _real_directory(controls)
    create_root = controls / "prod9-create-once-v1"
    create_identity = _real_directory(create_root, owner=expected_owner)
    mode = stat.S_IMODE(create_identity.st_mode)
    if not mode & stat.S_IRUSR or not mode & stat.S_IWUSR or not mode & stat.S_IXUSR:
        raise BootstrapError("create_once_root_mode_invalid")
    observed_mountinfo = (
        Path("/proc/self/mountinfo").read_text() if mountinfo is None else mountinfo
    )
    if "ro" not in _mount_options(str(shared), observed_mountinfo):
        raise BootstrapError("sfs_read_only_mount_invalid")
    if "rw" not in _mount_options(str(controls), observed_mountinfo):
        raise BootstrapError("controls_writable_mount_invalid")

    proof = _seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "passed_non_submitting_bootstrap",
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "packet_sha256": packet["sha256"],
            "source_commit": packet["source_commit"],
            "identity_sha256": packet["identity_sha256"],
            "plan_sha256": packet["plan_sha256"],
            "request_sha256": packet["request_sha256"],
            "target_manifest_sha256": packet["target_manifest_sha256"],
            "driver_sha256": driver_sha256,
            "job_uid": job_uid,
            "pod_name": pod_name,
            "pod_uid": pod_uid,
            "runtime_user": {"uid": RUNTIME_UID, "gid": RUNTIME_GID},
            "sfs_root_read_only": True,
            "controls_mount_writable": True,
            "create_once_root_owner": {"uid": RUNTIME_UID, "gid": RUNTIME_GID},
            "gpus": 0,
            "target_posts": 0,
            "launch_authorized": False,
            "private_rows_read": 0,
            "traces_read": 0,
        }
    )
    if write:
        receipt_path = (
            controls
            / "prod10-controller-bootstrap-v1"
            / Path(packet["controller"]["bootstrap_receipt"]).name
        )
        _write_once(receipt_path, proof, owner=expected_owner)
    return proof


def validate_receipt(
    value: object,
    *,
    packet: dict,
    job_uid: str,
    pod_name: str,
    pod_uid: str,
) -> dict:
    checked_packet = validate_packet(packet)
    if not isinstance(value, dict) or value != _seal(value):
        raise ValueError("prod10 controller runtime receipt digest changed")
    try:
        observed_at = datetime.fromisoformat(value.get("observed_at", "").replace("Z", "+00:00"))
        UUID(job_uid)
        UUID(pod_uid)
    except (TypeError, ValueError):
        raise ValueError("prod10 controller runtime receipt identity is invalid") from None
    if (
        observed_at.tzinfo is None
        or set(value)
        != {
            "schema",
            "status",
            "observed_at",
            "packet_sha256",
            "source_commit",
            "identity_sha256",
            "plan_sha256",
            "request_sha256",
            "target_manifest_sha256",
            "driver_sha256",
            "job_uid",
            "pod_name",
            "pod_uid",
            "runtime_user",
            "sfs_root_read_only",
            "controls_mount_writable",
            "create_once_root_owner",
            "gpus",
            "target_posts",
            "launch_authorized",
            "private_rows_read",
            "traces_read",
            "sha256",
        }
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") != "passed_non_submitting_bootstrap"
        or value.get("packet_sha256") != checked_packet["sha256"]
        or value.get("source_commit") != checked_packet["source_commit"]
        or value.get("identity_sha256") != checked_packet["identity_sha256"]
        or value.get("plan_sha256") != checked_packet["plan_sha256"]
        or value.get("request_sha256") != checked_packet["request_sha256"]
        or value.get("target_manifest_sha256") != checked_packet["target_manifest_sha256"]
        or not re.fullmatch(r"[0-9a-f]{64}", value.get("driver_sha256", ""))
        or value.get("job_uid") != job_uid
        or value.get("pod_name") != pod_name
        or value.get("pod_uid") != pod_uid
        or value.get("runtime_user") != {"uid": RUNTIME_UID, "gid": RUNTIME_GID}
        or value.get("sfs_root_read_only") is not True
        or value.get("controls_mount_writable") is not True
        or value.get("create_once_root_owner") != {"uid": RUNTIME_UID, "gid": RUNTIME_GID}
        or value.get("gpus") != 0
        or value.get("target_posts") != 0
        or value.get("launch_authorized") is not False
        or value.get("private_rows_read") != 0
        or value.get("traces_read") != 0
    ):
        raise ValueError("prod10 controller runtime receipt changed")
    return value


def _emit(value: dict) -> None:
    payload = canonical_json(value)
    if len(payload) > 4 * 1024:
        raise BootstrapError("termination_receipt_too_large")
    Path("/dev/termination-log").write_bytes(payload)
    print(LOG_PREFIX + payload.decode(), flush=True)


def main() -> None:
    os.umask(0o077)
    try:
        receipt = observe()
    except BootstrapError as exc:
        _emit(
            _seal(
                {
                    "schema": RECEIPT_SCHEMA,
                    "status": "rejected",
                    "error_code": exc.code,
                    "gpus": 0,
                    "target_posts": 0,
                    "launch_authorized": False,
                    "private_rows_read": 0,
                    "traces_read": 0,
                }
            )
        )
        raise SystemExit(1) from None
    _emit(receipt)


if __name__ == "__main__":
    main()
