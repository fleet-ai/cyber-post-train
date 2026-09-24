"""Small in-image proof for the maintained Miles96 zero-update signal path."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cyber_qwen38_miles96_exact_image_preflight_v1"
LOG_PREFIX = "CYBER_MILES96_IMAGE_PREFLIGHT_RECEIPT "
ENV_DRIVER_SHA256 = "CYBER_MILES96_PREFLIGHT_DRIVER_SHA256"
ENV_PLAN_SHA256 = "CYBER_MILES96_PREFLIGHT_PLAN_SHA256"
ENV_REQUEST_SHA256 = "CYBER_MILES96_PREFLIGHT_REQUEST_SHA256"
ENV_SOURCE_CLOSURE_SHA256 = "CYBER_MILES96_PREFLIGHT_SOURCE_CLOSURE_SHA256"
ENV_RUNTIME_IMAGE = "CYBER_MILES96_PREFLIGHT_RUNTIME_IMAGE"
ENV_RECEIPT_PATH = "CYBER_MILES96_PREFLIGHT_RECEIPT_PATH"
RUNTIME_BUNDLE = "CYBER_RUNTIME_BUNDLE"
RUNTIME_BUNDLE_SHA256 = "CYBER_RUNTIME_BUNDLE_SHA256"
MAX_FILES = 32
MAX_UNCOMPRESSED_BYTES = 2_000_000


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _runtime_blob() -> bytes:
    if RUNTIME_BUNDLE in os.environ:
        encoded = os.environ[RUNTIME_BUNDLE]
    else:
        chunks = {
            int(name.removeprefix(RUNTIME_BUNDLE + "_")): value
            for name, value in os.environ.items()
            if name.startswith(RUNTIME_BUNDLE + "_")
            and name.removeprefix(RUNTIME_BUNDLE + "_").isdigit()
        }
        if not chunks or sorted(chunks) != list(range(len(chunks))):
            raise ValueError("Miles runtime bundle chunks are incomplete")
        encoded = "".join(chunks[index] for index in range(len(chunks)))
    try:
        blob = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Miles runtime bundle is not canonical base64") from exc
    expected = os.environ.get(RUNTIME_BUNDLE_SHA256, "")
    if expected != "sha256:" + hashlib.sha256(blob).hexdigest():
        raise ValueError("Miles runtime bundle digest changed")
    return blob


def _runtime_files(blob: bytes) -> dict[str, str]:
    try:
        payload = json.loads(gzip.decompress(blob))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Miles runtime bundle is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"files", "module", "argv"}
        or payload.get("module") != "training.miles96_signal_qualification"
        or not isinstance(payload.get("files"), dict)
        or not 1 <= len(payload["files"]) <= MAX_FILES
    ):
        raise ValueError("Miles runtime bundle envelope changed")
    total = 0
    files: dict[str, str] = {}
    for name, text in payload["files"].items():
        path = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not isinstance(text, str)
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
        ):
            raise ValueError("Miles runtime bundle contains an unsafe file")
        total += len(text.encode())
        files[name] = text
    if total > MAX_UNCOMPRESSED_BYTES or "plan.json" not in files:
        raise ValueError("Miles runtime bundle is incomplete or too large")
    return files


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for name, text in sorted(files.items()):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(text.encode())


def _session_open_check(mechanics: Any, runtime: Path, plan: dict[str, Any]) -> None:
    """Exercise the repaired one-read raw+visible tool gate without Fleet calls."""
    from fti.fleet.v1 import openai_tools

    raw = json.loads((runtime / mechanics.TOOL_CATALOG_PATH).read_text())
    session_type = mechanics._evidence_session_class()
    base = session_type.__mro__[1]
    original_open = base.open
    missing = object()
    original_close = session_type.__dict__.get("close", missing)

    class FakeInstance:
        def __init__(self, catalog: list[dict[str, Any]]) -> None:
            self.catalog = catalog
            self.list_calls = 0

        def list_tools(self):
            self.list_calls += 1
            return deepcopy(self.catalog)

    class FakeClient:
        def __init__(self, catalog: list[dict[str, Any]]) -> None:
            self.catalog = catalog
            self.instances: list[FakeInstance] = []

        def create_reward_instance(self, *_args: Any, **_kwargs: Any) -> FakeInstance:
            instance = FakeInstance(self.catalog)
            self.instances.append(instance)
            return instance

    def fake_base_open(self) -> None:
        self.instance = self.client.create_reward_instance()
        self.tools = openai_tools(self.instance.list_tools())

    def fake_close(self) -> None:
        self.close_calls += 1

    def new_session(catalog: list[dict[str, Any]]):
        session = session_type.__new__(session_type)
        session.client = FakeClient(catalog)
        session.task_key = plan["task_binding"]["task_key"]
        session.task_version_id = plan["task_binding"]["task_version_id"]
        session.verifier_version_id = plan["task_binding"]["verifier_version_id"]
        session.instance = None
        session.tools = []
        session.close_calls = 0
        return session

    base.open = fake_base_open
    session_type.close = fake_close
    try:
        exact = new_session(raw)
        exact.open()
        if (
            exact.close_calls != 0
            or len(exact.client.instances) != 1
            or exact.client.instances[0].list_calls != 1
        ):
            raise ValueError("exact session-open tool contract did not pass once")

        raw_drift = deepcopy(raw)
        raw_drift[0]["_meta"] = {"preflight": "raw-only-drift"}
        if openai_tools(raw_drift) != openai_tools(raw):
            raise ValueError("raw-only drift probe changed the projected catalog")
        rejected = new_session(raw_drift)
        try:
            rejected.open()
        except ValueError:
            pass
        else:
            raise ValueError("raw-only tool drift did not fail closed")
        if (
            rejected.close_calls != 1
            or len(rejected.client.instances) != 1
            or rejected.client.instances[0].list_calls != 1
        ):
            raise ValueError("raw-only drift cleanup was not exact")
    finally:
        base.open = original_open
        if original_close is missing:
            delattr(session_type, "close")
        else:
            session_type.close = original_close


def run_preflight() -> dict[str, Any]:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none"
        or os.environ.get("WANDB_MODE") != "disabled"
    ):
        raise ValueError("Miles CPU preflight isolation changed")
    blob = _runtime_blob()
    with tempfile.TemporaryDirectory(prefix="miles96-image-preflight-", dir="/tmp") as temp:
        runtime = Path(temp)
        _write_tree(runtime, _runtime_files(blob))
        sys.path.insert(0, str(runtime))
        os.environ["CYBER_RUNTIME_DIR"] = str(runtime)
        try:
            importlib.invalidate_caches()
            from training import miles96_mechanics_canary as mechanics
            from training import miles96_signal_qualification as signal

            plan = signal.validate_plan(json.loads((runtime / "plan.json").read_text()))
            plan_sha256 = "sha256:" + mechanics.digest(plan)
            if plan_sha256 != os.environ.get(ENV_PLAN_SHA256):
                raise ValueError("Miles signal plan digest changed")
            closure = "sha256:" + mechanics.digest(plan["runtime_sources"])
            if closure != os.environ.get(ENV_SOURCE_CLOSURE_SHA256):
                raise ValueError("Miles runtime source closure changed")
            binding = signal._runtime_signal_binding(plan)
            arguments = signal.native_arguments(plan)
            positions = {value: index for index, value in enumerate(arguments)}
            if (
                arguments[:2] != ["-m", "fti.trainers.miles.run_fleet"]
                or arguments[positions["--mode"] + 1] != "eval"
                or arguments[positions["--n-samples-per-prompt"] + 1] != "8"
                or any(value in arguments for value in ("--save", "--save-interval", "--post-save"))
                or plan["qualification"]["optimizer_steps"] != 0
                or binding["sample_count"] != 8
                or binding["max_concurrent_envs"] != 2
                or binding["outer_episode_replacements"] != 0
            ):
                raise ValueError("Miles zero-update entrypoint changed")
            _session_open_check(mechanics, runtime, plan)
            runtime_image = os.environ[ENV_RUNTIME_IMAGE]
            image_digest = runtime_image.rsplit("@", 1)[-1]
            body = {
                "schema": SCHEMA,
                "status": "passed",
                "gpus": 0,
                "job_name": os.environ["JOB_NAME"],
                "runtime_image": runtime_image,
                "image_digest": image_digest,
                "source_closure_sha256": closure,
                "driver_sha256": os.environ[ENV_DRIVER_SHA256],
                "runtime_bundle_sha256": os.environ[RUNTIME_BUNDLE_SHA256],
                "plan_sha256": plan_sha256,
                "request_sha256": os.environ[ENV_REQUEST_SHA256],
                "runtime_binding_sha256": binding["sha256"],
                "raw_tool_catalog_sha256": binding["raw_tool_catalog_sha256"],
                "openai_tool_catalog_sha256": binding["openai_tool_catalog_sha256"],
                "tool_transform_source_sha256": binding["tool_transform_source_sha256"],
                "checks": {
                    "pinned_runtime_binding": True,
                    "zero_update_entrypoint": True,
                    "session_open_exact_catalog": True,
                    "session_open_raw_drift_rejected_and_closed": True,
                },
                "observed_at_unix": time.time(),
            }
            return {**body, "sha256": "sha256:" + _digest(body)}
        finally:
            sys.path.remove(str(runtime))


def main() -> None:
    os.umask(0o077)
    try:
        receipt = run_preflight()
    except BaseException as exc:
        print(
            LOG_PREFIX
            + json.dumps(
                {"schema": SCHEMA, "status": "failed", "error_class": type(exc).__name__},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise SystemExit(1) from None
    payload = _canonical(receipt).decode()
    Path(os.environ[ENV_RECEIPT_PATH]).write_text(payload + "\n")
    print(LOG_PREFIX + payload, flush=True)


if __name__ == "__main__":
    main()
