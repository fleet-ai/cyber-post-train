"""One bounded process lifecycle for native RL backends, with private evidence."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import digest

from .miles_conversion import _write


def sealed(value, schema):
    if value.get("schema") != schema or value.get("sha256", "").removeprefix("sha256:") != digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("input receipt schema/digest mismatch")


def native_failure(plan, error):
    """Keep Ray's nested error types/code locations, never its private messages."""
    causes, seen = [], set()
    while isinstance(error, BaseException) and id(error) not in seen and len(causes) < 8:
        seen.add(id(error))
        # ActorDiedError flattens RayTaskError into its message, not .cause.
        # Parse only code locations/types; never persist that private message.
        detail = getattr(error, "traceback_str", "") or str(error)
        frames = re.findall(
            r'File "[^"\n]*/([\w.-]+\.py)", line (\d+), in ([\w<>]+)',
            detail,
        )
        causes.append(
            {
                "error_class": type(error).__name__,
                "actor_init_failed": getattr(error, "actor_init_failed", False) is True,
                "remote_error_classes": sorted(
                    set(re.findall(r"(?m)^\s*(\w+(?:Error|Exception)):", detail))
                ),
                "remote_frames": [
                    {"file": f, "line": int(n), "function": name} for f, n, name in frames
                ],
                "local_frames": [
                    {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                    for f in traceback.extract_tb(error.__traceback__)[-20:]
                ],
            }
        )
        error = getattr(error, "cause", None) or error.__cause__ or error.__context__
    return _write(
        Path(plan["output_root"]) / "NATIVE_FAILURE.json",
        {"plan_sha256": digest(plan), "causes": causes},
    )


def progress(root):
    files = []
    for parent in (root / "checkpoints", root / "episodes"):
        for path in parent.rglob("*"):
            try:
                stat = path.stat()
                if path.is_file():
                    files.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                pass  # native checkpoint rotation may race a read-only sample
    return tuple(sorted(files))


def run(plan, plan_path, *, backend):
    from .sft_runtime import WATCHDOG_POLL_SECONDS, ProgressWatchdog, _utilization_snapshot

    root = Path(plan["output_root"])
    if os.environ.get("RUN_DIR") != str(root):
        raise ValueError("Jobs API output binding mismatch")
    if any((root / name).exists() for name in ("checkpoints", "episodes")):
        raise FileExistsError("RL-from-base cannot consume an existing run directory")
    _write(root / "STARTED.json", {"plan_sha256": digest(plan), "started_at": time.time()})
    process = None
    try:
        backend.native_source()
        name = backend.MODULE.removeprefix("training.").removesuffix("_training")
        with (root / f"private-{name}.log").open("x") as log:
            os.chmod(log.name, 0o600)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    backend.MODULE,
                    "--plan",
                    str(plan_path.resolve()),
                    "--sha256",
                    digest(plan),
                    "--native",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            watchdog = ProgressWatchdog(time.monotonic())
            while process.poll() is None:
                try:
                    process.wait(timeout=WATCHDOG_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    gpu, io = _utilization_snapshot()
                    reason = watchdog.observe(time.monotonic(), progress(root), gpu, io)
                    if reason:
                        raise TimeoutError(reason) from None
            if process.returncode != 0:
                raise RuntimeError("native RL did not finish")
        return _write(root / "NATIVE_TRAINING_COMPLETE.json", backend.native_result(plan))
    except BaseException as exc:
        _write(
            root / "FAILED.json",
            {"status": "failed", "plan_sha256": digest(plan), "error_class": type(exc).__name__},
        )
        raise RuntimeError(
            "RL run failed; private evidence preserved; no automatic retry"
        ) from None
    finally:
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
