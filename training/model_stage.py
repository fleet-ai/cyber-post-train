"""Create-once, zero-GPU streaming transfer of one accepted HF model export.

The producer's signed receipts are fetched before any weight.  Their exact file
and embedded digests select the complete top-level payload mapping; each payload
is then streamed directly into an inference-PVC transaction directory.  No ZIP,
temporary weight copy, model import, or tensor deserialization is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

PLAN_SCHEMA = "cyber_inference_model_stream_stage_plan_v1"
PORTABLE_PLAN_SCHEMA = "cyber_inference_model_stream_stage_plan_v2"
RECEIPT_SCHEMA = "cyber_inference_model_stream_stage_receipt_v1"
FILEBROWSER_ORIGIN = "http://filebrowser.fleet-train-data-plane.svc.cluster.local"
ACCEPTANCE = ".fleet-acceptance.json"
SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RECEIPT_LIMIT = 2 * 1024 * 1024
BLOCK = 8 * 1024 * 1024
OpenSource = Callable[[str], contextlib.AbstractContextManager[BinaryIO]]
FILEBROWSER_USER = "christopher@fleet.so"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _unsigned(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sha(value: Any, field: str, *, prefixed: bool = True) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be one SHA-256")
    if prefixed != value.startswith("sha256:"):
        form = "prefixed" if prefixed else "bare"
        raise ValueError(f"{field} must use the {form} SHA-256 form")
    return value


def read_plan(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("stage plan is not a regular file")
        raw = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            raw.extend(chunk)
            if len(raw) > RECEIPT_LIMIT:
                raise ValueError("stage plan exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stable_identity(before) != _stable_identity(after):
        raise ValueError("stage plan changed while it was read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("stage plan must be a JSON object")
    validate_plan(value)
    return value


def validate_plan(plan: Mapping[str, Any]) -> None:
    portable = plan.get("schema") == PORTABLE_PLAN_SCHEMA
    if plan.get("schema") not in {PLAN_SCHEMA, PORTABLE_PLAN_SCHEMA}:
        raise ValueError("unsupported model-stage plan schema")
    expected = _sha(plan.get("plan_sha256"), "plan_sha256")
    if digest_json(_unsigned(plan, "plan_sha256")) != expected:
        raise ValueError("model-stage plan digest does not validate")
    source = _mapping(plan.get("source"), "source")
    export = _mapping(source.get("export_receipt"), "export receipt binding")
    gpu = _mapping(source.get("gpu_check_receipt"), "GPU-check receipt binding")
    for label, binding, suffix in (
        ("export", export, "/EXPORT.json"),
        ("GPU check", gpu, "/GPU_CHECK.json"),
    ):
        path = binding.get("sfs_path")
        if (
            not isinstance(path, str)
            or not path.startswith("/jobs/")
            or not path.endswith(suffix)
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError(f"{label} receipt uses an unsafe SFS path")
        _sha(binding.get("file_sha256"), f"{label} receipt file_sha256")
        _sha(binding.get("receipt_sha256"), f"{label} receipt receipt_sha256")
        if not isinstance(binding.get("required_fields"), Mapping):
            raise ValueError(f"{label} receipt required_fields must be an object")
    export_fields = _mapping(export["required_fields"], "export required fields")
    gpu_fields = _mapping(gpu["required_fields"], "GPU-check required fields")
    if (
        export_fields.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or export_fields.get("optimizer_steps_executed") != 0
        or export_fields.get("gpu_reload_verified") is not False
        or export_fields.get("dtype") != "BF16"
        or export_fields.get("source_inventory_sizes_mtimes_unchanged") is not True
        or export_fields.get("all_output_tensors_reopened_equal") is not True
    ):
        raise ValueError("export binding does not preserve the accepted zero-update BF16 gate")
    if (
        gpu_fields.get("schema") != "cyber_hf_export_check_v1"
        or gpu_fields.get("status") != "passed"
        or gpu_fields.get("export_sha256") != export["file_sha256"].removeprefix("sha256:")
        or gpu_fields.get("export_receipt_sha256")
        != export["receipt_sha256"].removeprefix("sha256:")
        or gpu_fields.get("optimizer_steps_executed") != 0
        or gpu_fields.get("gpus") != 1
        or gpu_fields.get("gpu_reload_verified") is not True
        or gpu_fields.get("serving_qualified") is not False
        or gpu_fields.get("source_unchanged") is not True
        or gpu_fields.get("finite_logits") is not True
        or not isinstance(gpu_fields.get("generated_tokens"), int)
        or gpu_fields["generated_tokens"] <= 0
    ):
        raise ValueError("GPU-check binding does not preserve the accepted reload gate")
    payload = _mapping(source.get("payload"), "payload binding")
    count = payload.get("file_count")
    if (not portable and count != 29) or (
        portable and (type(count) is not int or not 1 <= count <= 4096)
    ):
        raise ValueError("payload file count is outside the reviewed bound")
    if not isinstance(payload.get("maximum_total_bytes"), int) or not (
        (1 if portable else 55_562_855_904) <= payload["maximum_total_bytes"] <= 64 * 1024**3
    ):
        raise ValueError("payload total-byte cap is missing or unreasonable")
    if not isinstance(payload.get("maximum_file_bytes"), int) or not (
        (1 if portable else 3 * 1024**3) <= payload["maximum_file_bytes"] <= 4 * 1024**3
    ):
        raise ValueError("payload per-file cap is missing or unreasonable")
    _sha(payload.get("manifest_sha256"), "payload manifest_sha256")
    execution = _mapping(plan.get("execution"), "execution")
    if (
        execution.get("namespace") != "inference"
        or execution.get("priority_class") != "c1"
        or execution.get("gpus") != 0
        or execution.get("restart_policy") != "Never"
        or not isinstance(execution.get("active_deadline_seconds"), int)
        or not 1 <= execution["active_deadline_seconds"] <= 5400
    ):
        raise ValueError("staging must be a bounded, zero-GPU c1 inference Pod")
    for field in ("job_name", "config_map_name") if portable else ("pod_name", "config_map_name"):
        value = execution.get(field)
        if (
            not isinstance(value, str)
            or len(value) > 63
            or re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value) is None
        ):
            raise ValueError(f"execution {field} is not a DNS label")
    image = execution.get("image")
    if not isinstance(image, str) or re.search(r"@sha256:[0-9a-f]{64}$", image) is None:
        raise ValueError("staging image must use an immutable digest")
    _sha(execution.get("runtime_source_sha256"), "runtime source SHA-256")
    resources = {
        "cpu_request": "4",
        "cpu_limit": "8",
        "memory_request": "8Gi",
        "memory_limit": "16Gi",
        "ephemeral_storage_request": "1Gi",
        "ephemeral_storage_limit": "2Gi",
    }
    if execution.get("resource_shape") != resources:
        raise ValueError("staging resource shape differs from the reviewed zero-GPU Pod")
    destination = _mapping(plan.get("destination"), "destination")
    path = destination.get("path")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or (not portable and PurePosixPath(path).name != destination.get("model_id"))
        or ".." in PurePosixPath(path).parts
        or PurePosixPath(path).name in {"", ".", ".."}
        or destination.get("must_be_absent") is not True
        or destination.get("acceptance_path") != str(Path(path) / ACCEPTANCE)
        or destination.get("atomic_transaction") != "directory_rename_noreplace_v1"
    ):
        raise ValueError("destination must bind one create-once absolute model directory")
    if portable:
        # Copying weights and registering an endpoint are independent operations.
        # Never rewrite a historical registration to fit a new experiment.
        if "registration_source" in plan or "desired_registration" in plan:
            raise ValueError("portable staging does not create or select a registration")
        if execution.get("kind") != "Job" or execution.get("backoff_limit") != 0:
            raise ValueError("portable staging requires a bounded zero-retry Job")
        if not re.fullmatch(r"chris-ar-[a-z0-9-]{1,44}", execution["job_name"]):
            raise ValueError("portable staging requires an owned Job name")
        if execution["config_map_name"] != execution["job_name"]:
            raise ValueError("portable staging config must share its owned Job name")
        if not re.fullmatch(r"/models/chris-autoresearch/[a-z0-9-]{1,63}", path):
            raise ValueError("portable staging destination must be an owned model directory")
        if (
            type(export_fields.get("optimizer_step")) is not int
            or export_fields["optimizer_step"] <= 0
            or not export_fields.get("model_repo")
            or not re.fullmatch(r"[0-9a-f]{40}", str(export_fields.get("model_revision", "")))
        ):
            raise ValueError("portable staging must bind the exact trained model and step")
    else:
        _validate_registration_clone(plan)
    scientific = _mapping(plan.get("scientific_boundary"), "scientific boundary")
    if scientific != {
        "optimizer_updates": 0,
        "external_benchmark_content_or_outcomes_used": False,
        "serving_qualified_by_staging": False,
        "registration_or_resume_performed_by_this_plan": False,
    }:
        raise ValueError("staging plan crosses its scientific or serving boundary")


def _replace_flag(args: list[Any], flag: str, value: str) -> None:
    if args.count(flag) != 1:
        raise ValueError(f"source registration must contain exactly one {flag}")
    index = args.index(flag)
    if index + 1 >= len(args):
        raise ValueError(f"source registration has no value for {flag}")
    args[index + 1] = value


def _validate_registration_clone(plan: Mapping[str, Any]) -> None:
    """Allow only the six reviewed identity substitutions from the paused source spec."""

    source = _mapping(plan.get("registration_source"), "registration source")
    desired = _mapping(plan.get("desired_registration"), "desired registration")
    source_spec = _mapping(source.get("spec"), "source registration spec")
    desired_spec = _mapping(desired.get("spec"), "desired registration spec")
    if source.get("id") != "chris-q38-fresh75-step20-v1":
        raise ValueError("registration source is not the accepted step-20 clone parent")
    if digest_json(source_spec) != source.get("spec_sha256"):
        raise ValueError("source registration spec digest does not validate")
    destination = _mapping(plan.get("destination"), "destination")
    model_id, staged = destination["model_id"], destination["path"]
    if desired.get("id") != model_id:
        raise ValueError("desired registration id differs from the staged model id")
    expected = json.loads(json.dumps(source_spec))
    expected["displayName"] = "Qwen3.8-27B Fresh75 teacher-max SFT V4 step 230"
    expected["model"]["sourcePath"] = staged
    expected["model"]["path"] = "/scratch/models/" + model_id
    expected["model"]["revision"] = plan["source"]["payload"]["manifest_sha256"]
    _replace_flag(expected["runtime"]["args"], "--model-path", expected["model"]["path"])
    _replace_flag(expected["runtime"]["args"], "--served-model-name", model_id)
    if dict(desired_spec) != expected:
        raise ValueError("desired registration changes fields outside the reviewed clone set")
    if (
        desired_spec.get("desiredState") != "paused"
        or desired_spec.get("scaling", {}).get("minReplicas") != 0
        or desired_spec.get("placement", {}).get("priorityClassName") != "c1"
    ):
        raise ValueError("desired registration must begin paused at zero replicas on c1")


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _receipt_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(_unsigned(value, "receipt_sha256"))).hexdigest()


def _required_fields(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} field {field} differs from the accepted binding")


@contextlib.contextmanager
def _filebrowser_stream(path: str, forwarded_user: str) -> Iterator[BinaryIO]:
    if not forwarded_user or any(character in forwarded_user for character in "\r\n"):
        raise ValueError("FILEBROWSER_USER is required and must fit one HTTP header")
    url = (
        FILEBROWSER_ORIGIN
        + "/api/resources/download?"
        + urllib.parse.urlencode({"file": path, "source": "sfs"})
    )
    request = urllib.request.Request(url, headers={"X-Forwarded-User": forwarded_user})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise ValueError("Filebrowser returned a non-success response")
        yield response


def _read_bound_receipt(
    binding: Mapping[str, Any], open_source: OpenSource, label: str
) -> dict[str, Any]:
    expected_file = _sha(binding.get("file_sha256"), f"{label} file SHA-256")
    with open_source(str(binding["sfs_path"])) as stream:
        raw = bytearray()
        while chunk := stream.read(1024 * 1024):
            raw.extend(chunk)
            if len(raw) > RECEIPT_LIMIT:
                raise ValueError(f"{label} exceeds its size bound")
    if _digest_bytes(raw) != expected_file:
        raise ValueError(f"{label} file digest differs from the accepted binding")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    actual = value.get("receipt_sha256")
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise ValueError(f"{label} has no valid embedded receipt digest")
    if _receipt_digest(value) != actual or "sha256:" + actual != binding.get("receipt_sha256"):
        raise ValueError(f"{label} embedded receipt digest differs from the accepted binding")
    _required_fields(value, _mapping(binding.get("required_fields"), label), label)
    return value


def _payload_files(export: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return _validated_payload_mapping(_mapping(export.get("files"), "EXPORT files"), plan)


def _validated_payload_mapping(
    raw: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    payload_plan = _mapping(_mapping(plan.get("source"), "source").get("payload"), "payload")
    if len(raw) != payload_plan.get("file_count") or digest_json(raw) != payload_plan.get(
        "manifest_sha256"
    ):
        raise ValueError("EXPORT payload mapping differs from the accepted manifest")
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for name, specification in raw.items():
        spec = _mapping(specification, f"payload specification {name}")
        size, digest = spec.get("bytes"), spec.get("sha256")
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or name in {"", ".", "..", ACCEPTANCE}
            or not isinstance(size, int)
            or not 0 <= size <= payload_plan["maximum_file_bytes"]
        ):
            raise ValueError("EXPORT contains an unsafe payload entry")
        _sha(digest, f"payload SHA-256 for {name}", prefixed=False)
        files[name] = {"bytes": size, "sha256": digest}
        total += size
    if total > payload_plan["maximum_total_bytes"]:
        raise ValueError("EXPORT payload exceeds its total-byte cap")
    return dict(sorted(files.items()))


def _exists_at(parent: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _open_directory_at(parent: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0),
        dir_fd=parent,
    )
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("staging transaction is not a directory")
    return descriptor


def _stable_hash_at(directory: int, name: str) -> tuple[int, str]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"staged payload {name} is not one regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, BLOCK):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stable_identity(before) != _stable_identity(after):
        raise ValueError(f"staged payload {name} changed while it was read")
    return before.st_size, digest.hexdigest()


def _read_regular_at(directory: int, name: str, limit: int) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{name} is not one regular file")
        value = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            value.extend(chunk)
            if len(value) > limit:
                raise ValueError(f"{name} exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stable_identity(before) != _stable_identity(after):
        raise ValueError(f"{name} changed while it was read")
    return bytes(value)


def _stream_payload(
    directory: int, name: str, specification: Mapping[str, Any], source: BinaryIO
) -> None:
    expected_size = int(specification["bytes"])
    expected_digest = str(specification["sha256"])
    headers = getattr(source, "headers", None)
    content_length = headers.get("Content-Length") if headers is not None else None
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise ValueError("Filebrowser returned an invalid Content-Length") from exc
        if declared != expected_size:
            raise ValueError("Filebrowser Content-Length differs from EXPORT")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    size = 0
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("new payload target is not one regular file")
        while chunk := source.read(BLOCK):
            size += len(chunk)
            if size > expected_size:
                raise ValueError("Filebrowser payload exceeds EXPORT size")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        size != expected_size
        or digest.hexdigest() != expected_digest
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_size != expected_size
    ):
        raise ValueError(f"streamed payload {name} differs from EXPORT")


def _write_new_at(directory: int, name: str, value: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(parent: int, source: str, destination: str, path: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("renameat2 is required for collision-free Linux promotion")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(parent, os.fsencode(source), parent, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("renamex_np is required for collision-free macOS promotion")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(path / source), os.fsencode(path / destination), 4)
    else:
        raise RuntimeError(f"atomic no-replace promotion is unsupported on {sys.platform}")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, "staging destination already exists", destination)
    raise OSError(error, os.strerror(error), destination)


def _execution_identity(plan: Mapping[str, Any], environment: Mapping[str, str]) -> dict[str, Any]:
    execution = _mapping(plan.get("execution"), "execution")
    portable = plan.get("schema") == PORTABLE_PLAN_SCHEMA
    if portable and (
        environment.get("JOB_NAME") != execution["job_name"]
        or not environment.get("POD_NAME", "").startswith(execution["job_name"] + "-")
    ):
        raise ValueError("downward-API Job identity differs from the stage plan")
    expected = {
        "namespace": execution["namespace"],
        "pod_name": environment.get("POD_NAME", "") if portable else execution["pod_name"],
    }
    observed = {
        "namespace": environment.get("POD_NAMESPACE", ""),
        "pod_name": environment.get("POD_NAME", ""),
    }
    if observed != expected:
        raise ValueError("downward-API Pod identity differs from the stage plan")
    pod_uid = environment.get("POD_UID", "")
    if UID.fullmatch(pod_uid) is None:
        raise ValueError("POD_UID is missing or malformed")
    source_path = Path(__file__)
    descriptor = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        source = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            source.extend(chunk)
            if len(source) > RECEIPT_LIMIT:
                raise ValueError("model-stage runtime source exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    source_digest = _digest_bytes(source)
    if (
        not stat.S_ISREG(before.st_mode)
        or _stable_identity(before) != _stable_identity(after)
        or source_digest != execution["runtime_source_sha256"]
    ):
        raise ValueError("executing source bytes differ from the stage plan")
    return {
        **observed,
        "pod_uid": pod_uid,
        "requested_image": execution["image"],
        "runtime_source_sha256": source_digest,
        "priority_class": "c1",
        "gpus": 0,
        "active_deadline_seconds": execution["active_deadline_seconds"],
    }


def _validate_staged(
    parent: int,
    final_name: str,
    plan: Mapping[str, Any],
    expected_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    directory = _open_directory_at(parent, final_name)
    try:
        raw = _read_regular_at(directory, ACCEPTANCE, RECEIPT_LIMIT)
        receipt = json.loads(raw)
        if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("staging acceptance receipt has an unsupported schema")
        actual = receipt.get("receipt_sha256")
        if digest_json(_unsigned(receipt, "receipt_sha256")) != actual:
            raise ValueError("staging acceptance receipt digest does not validate")
        files = _validated_payload_mapping(
            _mapping(receipt.get("payload", {}).get("files"), "acceptance payload files"),
            plan,
        )
        if expected_files is not None and files != expected_files:
            raise ValueError("staged model files differ from the fetched EXPORT receipt")
        if (
            receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("destination", {}).get("path") != plan["destination"]["path"]
            or receipt.get("payload", {}).get("manifest_sha256")
            != plan["source"]["payload"]["manifest_sha256"]
            or receipt.get("payload", {}).get("files") != files
        ):
            raise ValueError("staging acceptance receipt differs from the exact transaction")
        if sorted(os.listdir(directory)) != sorted([*files, ACCEPTANCE]):
            raise ValueError("staged model inventory differs from EXPORT")
        for name, spec in files.items():
            size, digest = _stable_hash_at(directory, name)
            if size != spec["bytes"] or digest != spec["sha256"]:
                raise ValueError(f"staged model payload differs for {name}")
        return receipt
    finally:
        os.close(directory)


def execute_stage(
    plan: Mapping[str, Any], *, open_source: OpenSource, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Fetch exact receipts/payload, publish once, or validate an identical final readback."""

    validate_plan(plan)
    destination = Path(str(plan["destination"]["path"]))
    parent_path = destination.parent
    # The dedicated shared parent may not exist for the first experiment. Open
    # each component without following symlinks; never mkdir arbitrary ancestors.
    if plan.get("schema") == PORTABLE_PLAN_SCHEMA:
        models = os.open("/models", os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
        try:
            try:
                os.mkdir("chris-autoresearch", 0o755, dir_fd=models)
                os.fsync(models)
            except FileExistsError:
                pass
            parent = _open_directory_at(models, "chris-autoresearch")
        finally:
            os.close(models)
    else:
        parent = os.open(parent_path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0))
    suffix = str(plan["plan_sha256"]).removeprefix("sha256:")[:12]
    partial_name = f".partial-{destination.name}-{suffix}"
    try:
        final_exists, partial_exists = (
            _exists_at(parent, destination.name),
            _exists_at(parent, partial_name),
        )
        if final_exists:
            if partial_exists:
                raise ValueError("both final and partial staging paths exist")
            return _validate_staged(parent, destination.name, plan)
        if partial_exists:
            receipt = _validate_staged(parent, partial_name, plan)
            _rename_noreplace(parent, partial_name, destination.name, parent_path)
            os.fsync(parent)
            _validate_staged(parent, destination.name, plan)
            return receipt
        identity = _execution_identity(plan, environment)
        source = _mapping(plan.get("source"), "source")
        export_binding = _mapping(source.get("export_receipt"), "export receipt binding")
        gpu_binding = _mapping(source.get("gpu_check_receipt"), "GPU-check receipt binding")
        export = _read_bound_receipt(export_binding, open_source, "EXPORT receipt")
        gpu = _read_bound_receipt(gpu_binding, open_source, "GPU_CHECK receipt")
        files = _payload_files(export, plan)
        os.mkdir(partial_name, mode=0o755, dir_fd=parent)
        directory = _open_directory_at(parent, partial_name)
        try:
            os.fchmod(directory, 0o755)
            source_root = str(export_binding["sfs_path"]).rsplit("/", 1)[0]
            for name, spec in files.items():
                with open_source(source_root + "/" + name) as stream:
                    _stream_payload(directory, name, spec, stream)
            payload = {
                "file_count": len(files),
                "total_bytes": sum(spec["bytes"] for spec in files.values()),
                "manifest_sha256": source["payload"]["manifest_sha256"],
                "files": files,
            }
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "plan_sha256": plan["plan_sha256"],
                "source": {
                    "export_file_sha256": export_binding["file_sha256"],
                    "export_receipt_sha256": export_binding["receipt_sha256"],
                    "gpu_check_file_sha256": gpu_binding["file_sha256"],
                    "gpu_check_receipt_sha256": gpu_binding["receipt_sha256"],
                    "gpu_reload_verified": gpu["gpu_reload_verified"],
                    "optimizer_steps_executed": gpu["optimizer_steps_executed"],
                },
                "execution": identity,
                "payload": payload,
                "destination": {
                    "path": str(destination),
                    "acceptance_path": str(destination / ACCEPTANCE),
                    "atomic_transaction": "directory_rename_noreplace_v1",
                    "create_once": True,
                },
            }
            receipt["receipt_sha256"] = digest_json(receipt)
            encoded = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
            _write_new_at(directory, ACCEPTANCE, encoded)
            os.fsync(directory)
        finally:
            os.close(directory)
        _validate_staged(parent, partial_name, plan, files)
        _rename_noreplace(parent, partial_name, destination.name, parent_path)
        os.fsync(parent)
        return _validate_staged(parent, destination.name, plan, files)
    finally:
        os.close(parent)


def render_objects(
    plan: Mapping[str, Any], source_path: Path, plan_path: Path
) -> list[dict[str, Any]]:
    """Render an immutable ConfigMap and bounded CPU worker; perform no submission.

    Legacy plans retain their original Pod contract; v2 uses a zero-retry Job.
    The inference namespace is required by its model PVC, not an alert exemption.
    Callers must journal creation and count terminal failures in their own policy.
    """

    validate_plan(plan)
    execution = plan["execution"]
    resources = execution["resource_shape"]
    if (
        source_path.is_symlink()
        or _digest_bytes(source_path.read_bytes()) != execution["runtime_source_sha256"]
    ):
        raise ValueError("renderer source differs from the stage plan")
    if read_plan(plan_path) != dict(plan):
        raise ValueError("renderer plan bytes differ from the supplied plan")
    portable = plan.get("schema") == PORTABLE_PLAN_SCHEMA
    name = execution["job_name" if portable else "pod_name"]
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": name,
        "cyber-post-train.fleet.ai/purpose": "model-stage",
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": execution["config_map_name"],
            "namespace": execution["namespace"],
            "labels": labels,
        },
        "immutable": True,
        "data": {
            "model_stage.py": source_path.read_text(encoding="utf-8"),
            "plan.json": plan_path.read_text(encoding="utf-8"),
        },
    }
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": execution["namespace"],
            "labels": labels,
        },
        "spec": {
            "activeDeadlineSeconds": execution["active_deadline_seconds"],
            "automountServiceAccountToken": False,
            "priorityClassName": "c1",
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 30,
            "nodeSelector": {"workload": "fleetai-training-ng-cpu"},
            "tolerations": [
                {
                    "key": "workload",
                    "operator": "Equal",
                    "value": "fleetai-training-ng-cpu",
                    "effect": "NoSchedule",
                }
            ],
            "containers": [
                {
                    "name": "stage",
                    "image": execution["image"],
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["python3", "/bundle/model_stage.py"],
                    "args": ["execute", "--plan", "/bundle/plan.json"],
                    "env": [
                        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                        {
                            "name": "FILEBROWSER_USER",
                            "value": FILEBROWSER_USER,
                        },
                        {
                            "name": "POD_NAME",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                        },
                        {
                            "name": "POD_NAMESPACE",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                        },
                        {
                            "name": "POD_UID",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                        },
                    ],
                    "resources": {
                        "requests": {
                            "cpu": resources["cpu_request"],
                            "memory": resources["memory_request"],
                            "ephemeral-storage": resources["ephemeral_storage_request"],
                        },
                        "limits": {
                            "cpu": resources["cpu_limit"],
                            "memory": resources["memory_limit"],
                            "ephemeral-storage": resources["ephemeral_storage_limit"],
                        },
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                        "readOnlyRootFilesystem": True,
                    },
                    "volumeMounts": [
                        {
                            "name": "bundle",
                            "mountPath": "/bundle/model_stage.py",
                            "subPath": "model_stage.py",
                            "readOnly": True,
                        },
                        {
                            "name": "bundle",
                            "mountPath": "/bundle/plan.json",
                            "subPath": "plan.json",
                            "readOnly": True,
                        },
                        {"name": "models", "mountPath": "/models"},
                    ],
                }
            ],
            "volumes": [
                {"name": "bundle", "configMap": {"name": execution["config_map_name"]}},
                {"name": "models", "persistentVolumeClaim": {"claimName": "hf-cache-shared"}},
            ],
        },
    }
    if portable:
        pod["spec"]["containers"][0]["env"].append(
            {
                "name": "JOB_NAME",
                "valueFrom": {
                    "fieldRef": {"fieldPath": "metadata.labels['batch.kubernetes.io/job-name']"}
                },
            }
        )
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": pod["metadata"],
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": execution["active_deadline_seconds"],
                "template": {"metadata": {"labels": labels}, "spec": pod["spec"]},
            },
        }
        return [config_map, job]
    return [config_map, pod]


def render_manifest(plan: Mapping[str, Any], source_path: Path, plan_path: Path) -> str:
    import yaml

    return yaml.safe_dump_all(render_objects(plan, source_path, plan_path), sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--plan", type=Path, required=True)
    render.add_argument("--source", type=Path, default=Path(__file__))
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.command == "render":
        print(render_manifest(plan, args.source, args.plan), end="")
        return
    user = os.environ.get("FILEBROWSER_USER", "")

    def opener(path: str) -> contextlib.AbstractContextManager[BinaryIO]:
        return _filebrowser_stream(path, user)

    result = execute_stage(plan, open_source=opener, environment=os.environ)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
