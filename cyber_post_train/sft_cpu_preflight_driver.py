"""Standard-library worker for the exact dense-SFT CPU preflight Job.

The launcher embeds this source verbatim and binds it by SHA-256.  The worker
unpacks one immutable, source-bound bundle into an ephemeral directory, runs
the repository's native CPU preflight with GPUs and W&B disabled, and emits
exactly one sanitized receipt line.  It never writes the training output.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

BUNDLE_SCHEMA = "cyber_sft_cpu_preflight_bundle_v2"
ENVELOPE_SCHEMA = "cyber_sft_cpu_preflight_observation_v1"
PREFLIGHT_SCHEMA = "cyber_sft_cpu_preflight_v1"
CHUNK_PREFIX = "CYBER_SFT_PREFLIGHT_BUNDLE_"
ENV_BUNDLE_SHA256 = "CYBER_SFT_PREFLIGHT_BUNDLE_SHA256"
ENV_DRIVER_SHA256 = "CYBER_SFT_PREFLIGHT_DRIVER_SHA256"
ENV_PLAN_SHA256 = "CYBER_SFT_PREFLIGHT_PLAN_SHA256"
ENV_REQUEST_SHA256 = "CYBER_SFT_PREFLIGHT_REQUEST_SHA256"
ENV_RUN_NAME = "CYBER_SFT_PREFLIGHT_RUN_NAME"
ENV_RUN_DIR = "CYBER_SFT_PREFLIGHT_RUN_DIR"
LOG_PREFIX = "CYBER_SFT_PREFLIGHT_RECEIPT "
MAX_CHUNKS = 32
MAX_CHUNK_BYTES = 48_000
MAX_FILES = 160
MAX_UNCOMPRESSED_BYTES = 8_000_000
MAX_PEAK_MEMORY_BYTES = 48 * 1024**3
PEAK_MEMORY_SOURCE = "cgroup_v2_memory.peak"
QWEN38_262K_4NODE_RUNTIME_VARIANT = "qwen38_sft_262k_4node_v1"
QWEN38_262K_4NODE_RUN_NAME = "chris-q38-t3k262-4n-can-v1"
QWEN38_262K_4NODE_PLAN_SHA256 = "3f96ba9d9233a969e47110f70ccf52c0e72d7c2f01d1ab5ce09678d982507a7e"
QWEN38_262K_4NODE_REQUEST_SHA256 = (
    "1679d4699b36bbd6e687ae9e84f6c0d6288620583bdb7a857ec873b87b908470"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STAGE = re.compile(r"[a-z0-9_.-]{1,96}")
_NATIVE_STAGES = frozenset(
    {
        "validate_262k_plan",
        "install_262k_runtime",
        "gpu_isolation",
        "validate_plan",
        "validate_runtime_sources",
        "build_runtime_configs",
        "validate_forward_backward_adapter",
        "load_model_config",
        "load_tokenizer",
        "read_dataset_train",
        "prepare_rows_train",
        "target_accounting_train",
        "read_dataset_dev",
        "prepare_rows_dev",
        "target_accounting_dev",
        "validate_training_split",
        "native_dataset_loader",
        "complete",
    }
)
FAILURE_STAGES = frozenset(
    {
        "unclassified",
        "runtime_isolation",
        "runtime_identity",
        "bundle_decode",
        "bundle_verify",
        "output_absence",
        "bundle_materialize",
        "source_import",
        "prepared_bindings",
        "native_preflight",
        "native_receipt",
        "cgroup_peak_memory",
        *("native." + stage for stage in _NATIVE_STAGES),
    }
)


class _PreflightFailure(Exception):
    """Sanitized stage binding for a private underlying preflight error."""

    def __init__(self, stage: str, error: BaseException):
        if _STAGE.fullmatch(stage) is None:
            stage = "unclassified"
        self.stage = stage
        self.error_class = type(error).__name__
        self.error_fingerprint = hashlib.sha256(
            (self.error_class + "\0" + str(error)).encode(errors="replace")
        ).hexdigest()
        super().__init__(stage)


@contextlib.contextmanager
def _failure_stage(stage: str):
    try:
        yield
    except _PreflightFailure:
        raise
    except BaseException as exc:
        raise _PreflightFailure(stage, exc) from None


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def is_qwen38_262k_4node_candidate(plan: dict, request: dict) -> bool:
    return (
        plan.get("runtime_variant") == QWEN38_262K_4NODE_RUNTIME_VARIANT
        and plan.get("run_name") == QWEN38_262K_4NODE_RUN_NAME
        and request.get("name") == QWEN38_262K_4NODE_RUN_NAME
        and digest(plan) == QWEN38_262K_4NODE_PLAN_SHA256
        and digest(request) == QWEN38_262K_4NODE_REQUEST_SHA256
    )


def _jobs_path(value: str, expected_name: str) -> Path:
    path = PurePosixPath(value)
    if (
        path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) != 5
        or path.name != expected_name
        or str(path) != value
    ):
        raise ValueError("training output is not the exact owned /mnt/sfs/jobs path")
    return Path(value)


def _require_output_absent(run_dir: Path) -> None:
    root = run_dir.parent
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("SFS jobs root is not a real directory")
    try:
        run_dir.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("training output could not be inspected") from exc
    raise FileExistsError("training output already exists")


def _bundle_from_environment() -> bytes:
    chunks: dict[int, str] = {}
    for name, value in os.environ.items():
        if name == ENV_BUNDLE_SHA256:
            continue
        if not name.startswith(CHUNK_PREFIX):
            continue
        suffix = name.removeprefix(CHUNK_PREFIX)
        if re.fullmatch(r"[0-9]{3}", suffix) is None or len(value.encode()) > MAX_CHUNK_BYTES:
            raise ValueError("invalid embedded preflight bundle chunk")
        index = int(suffix)
        if index in chunks:
            raise ValueError("duplicate embedded preflight bundle chunk")
        chunks[index] = value
    if not chunks or len(chunks) > MAX_CHUNKS or sorted(chunks) != list(range(len(chunks))):
        raise ValueError("embedded preflight bundle chunks are incomplete")
    try:
        return base64.b64decode(
            "".join(chunks[index] for index in range(len(chunks))), validate=True
        )
    except ValueError as exc:
        raise ValueError("embedded preflight bundle is not canonical base64") from exc


def _inspect_bundle(blob: bytes) -> tuple[dict, dict[str, str]]:
    expected = os.environ.get(ENV_BUNDLE_SHA256, "")
    if _SHA256.fullmatch(expected) is None or hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("immutable preflight bundle digest mismatch")
    try:
        payload = json.loads(gzip.decompress(blob))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("immutable preflight bundle is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != {"manifest", "files"}:
        raise ValueError("immutable preflight bundle envelope drifted")
    manifest = payload["manifest"]
    files = payload["files"]
    if not isinstance(manifest, dict) or not isinstance(files, dict):
        raise ValueError("immutable preflight bundle payload is malformed")
    receipt = manifest.pop("sha256", None)
    if manifest.get("schema") != BUNDLE_SCHEMA or receipt != digest(manifest):
        raise ValueError("immutable preflight bundle self-digest mismatch")
    inventory = manifest.get("files")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != set(files)
        or not 1 <= len(files) <= MAX_FILES
    ):
        raise ValueError("immutable preflight bundle file inventory drifted")
    total = 0
    for name, text in files.items():
        path = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not isinstance(text, str)
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or (
                name
                not in {
                    "prepared/plan.json",
                    "prepared/request.json",
                    "prepared/PREPARED.json",
                }
                and path.parts[:1] != ("src",)
            )
        ):
            raise ValueError("immutable preflight bundle contains an unsafe file")
        raw = text.encode()
        total += len(raw)
        if inventory.get(name) != hashlib.sha256(raw).hexdigest():
            raise ValueError("immutable preflight bundle file digest mismatch")
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("immutable preflight bundle exceeds its size bound")
    manifest["sha256"] = receipt
    return manifest, files


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for name, text in sorted(files.items()):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode())


def _cgroup_v2_memory_peak_path(
    cgroup_file: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Path:
    # v1 max_usage_in_bytes and process RSS/time(1) are different scopes.  An
    # exact 262K receipt therefore fails closed instead of mixing measurements.
    try:
        raw = cgroup_file.read_bytes()
    except OSError as exc:
        raise ValueError("cgroup v2 peak-memory evidence is unavailable") from exc
    if len(raw) > 4096:
        raise ValueError("cgroup v2 membership is malformed")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("cgroup v2 membership is malformed") from exc
    memberships = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[:2] == ["0", ""]:
            memberships.append(fields[2])
    if len(memberships) != 1:
        raise ValueError("cgroup v2 peak-memory evidence is unavailable")
    membership = PurePosixPath(memberships[0])
    if (
        not membership.is_absolute()
        or ".." in membership.parts
        or str(membership) != memberships[0]
    ):
        raise ValueError("cgroup v2 membership is malformed")
    return cgroup_root.joinpath(*membership.parts[1:], "memory.peak")


def _cgroup_v2_peak_memory_bytes(path: Path | None = None) -> int:
    path = _cgroup_v2_memory_peak_path() if path is None else path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("cgroup v2 peak-memory evidence is unavailable") from exc
    if len(raw) > 32 or re.fullmatch(rb"[0-9]+\n?", raw) is None:
        raise ValueError("cgroup v2 peak-memory evidence is malformed")
    value = int(raw)
    if not 0 < value <= MAX_PEAK_MEMORY_BYTES:
        raise ValueError("cgroup v2 peak-memory evidence is outside its bound")
    return value


def run_preflight() -> dict:
    with _failure_stage("runtime_isolation"):
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none"
            or os.environ.get("WANDB_MODE") != "disabled"
        ):
            raise ValueError("CPU preflight GPU/W&B isolation drifted")
    with _failure_stage("runtime_identity"):
        driver_sha = os.environ.get(ENV_DRIVER_SHA256, "")
        if _SHA256.fullmatch(driver_sha) is None:
            raise ValueError("CPU preflight driver digest is invalid")
        run_name = os.environ.get(ENV_RUN_NAME, "")
        run_dir = _jobs_path(os.environ.get(ENV_RUN_DIR, ""), run_name)
    with _failure_stage("bundle_decode"):
        blob = _bundle_from_environment()
    with _failure_stage("bundle_verify"):
        manifest, files = _inspect_bundle(blob)
        if (
            manifest.get("run_name") != run_name
            or manifest.get("training_output_root") != str(run_dir)
            or manifest.get("driver_sha256") != driver_sha
        ):
            raise ValueError("CPU preflight runtime identity drifted")
    with _failure_stage("output_absence"):
        _require_output_absent(run_dir)
    with _failure_stage("bundle_materialize"):
        temporary = tempfile.TemporaryDirectory(prefix="cyber-sft-preflight-", dir="/tmp")
        root = Path(temporary.name)
    try:
        with _failure_stage("bundle_materialize"):
            _write_tree(root, files)
        sys.path.insert(0, str(root / "src"))
        private_output = io.StringIO()
        with (
            contextlib.redirect_stdout(private_output),
            contextlib.redirect_stderr(private_output),
        ):
            with _failure_stage("source_import"):
                from cyber_post_train.cli import _prepared
                from training.sft_dispatch import compiler_for_plan

            with _failure_stage("prepared_bindings"):
                plan, request = _prepared(root / "prepared")
                compiler = compiler_for_plan(plan)
                if compiler.job_request(plan) != request:
                    raise ValueError("staged request differs from the source-bound renderer")
                if (
                    digest(plan) != manifest.get("plan_sha256")
                    or digest(request) != manifest.get("request_sha256")
                    or request.get("name") != run_name
                    or request.get("run_dir") != str(run_dir)
                    or request.get("failureAlerts") is not False
                    or request.get("priority_class") != "c1"
                ):
                    raise ValueError("staged plan/request identity drifted")
            native_stage = "native_preflight"

            def progress(stage: str) -> None:
                nonlocal native_stage
                if stage not in _NATIVE_STAGES:
                    raise ValueError("native preflight reported an invalid stage")
                native_stage = "native." + stage

            try:
                if is_qwen38_262k_4node_candidate(plan, request):
                    native = compiler.preflight(plan, progress=progress)
                else:
                    native = compiler.preflight(plan)
            except BaseException as exc:
                raise _PreflightFailure(native_stage, exc) from None
    finally:
        if sys.path and sys.path[0] == str(root / "src"):
            sys.path.pop(0)
        temporary.cleanup()
    with _failure_stage("native_receipt"):
        if (
            native.get("schema") != PREFLIGHT_SCHEMA
            or native.get("status") != "passed"
            or native.get("gpus") != 0
            or native.get("plan_sha256") != manifest["plan_sha256"]
            or native.get("request_sha256") != manifest["request_sha256"]
        ):
            raise ValueError("native dense-SFT CPU preflight receipt drifted")
    peak_memory_bytes = None
    receipt_body = dict(native)
    if is_qwen38_262k_4node_candidate(plan, request):
        with _failure_stage("cgroup_peak_memory"):
            peak_memory_bytes = _cgroup_v2_peak_memory_bytes()
        receipt_body["peak_memory_bytes"] = peak_memory_bytes
        receipt_body["peak_memory_source"] = PEAK_MEMORY_SOURCE
    receipt = {**receipt_body, "sha256": digest(receipt_body)}
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "job_name": os.environ["JOB_NAME"],
        "observed_at_unix": time.time(),
        "bundle_sha256": hashlib.sha256(blob).hexdigest(),
        "driver_sha256": driver_sha,
        "plan_sha256": manifest["plan_sha256"],
        "request_sha256": manifest["request_sha256"],
        "preflight": receipt,
    }
    if peak_memory_bytes is not None:
        envelope["peak_memory_bytes"] = peak_memory_bytes
        envelope["peak_memory_source"] = PEAK_MEMORY_SOURCE
    return envelope


def main() -> None:
    os.umask(0o077)
    try:
        value = run_preflight()
    except BaseException as exc:
        failure = (
            exc if isinstance(exc, _PreflightFailure) else _PreflightFailure("unclassified", exc)
        )
        print(
            LOG_PREFIX
            + json.dumps(
                {
                    "schema": ENVELOPE_SCHEMA,
                    "status": "failed",
                    "gpus": 0,
                    "job_name": os.environ.get("JOB_NAME", ""),
                    "observed_at_unix": time.time(),
                    "bundle_sha256": os.environ.get(ENV_BUNDLE_SHA256, ""),
                    "driver_sha256": os.environ.get(ENV_DRIVER_SHA256, ""),
                    "plan_sha256": os.environ.get(ENV_PLAN_SHA256, ""),
                    "request_sha256": os.environ.get(ENV_REQUEST_SHA256, ""),
                    "failure_stage": failure.stage,
                    "error_class": failure.error_class,
                    "error_fingerprint": failure.error_fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise SystemExit(1) from None
    print(LOG_PREFIX + canonical_json(value).decode(), flush=True)


if __name__ == "__main__":
    main()
