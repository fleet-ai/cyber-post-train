"""One bounded process lifecycle for native RL backends, with private evidence."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager, suppress
from pathlib import Path

from cyber_post_train.jobs import digest

from .miles_conversion import _write


class HardDeadlineExceeded(TimeoutError):
    """A process-local lifecycle deadline expired."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"{phase} exceeded its plan-bound hard deadline")


class HardDeadline:
    """Monotonic deadline shared by signal and API-level timeout guards."""

    def __init__(self, seconds: int, phase: str):
        if type(seconds) is not int or seconds < 1 or not phase:
            raise ValueError("hard deadline requires positive integer seconds and a phase")
        self.seconds = seconds
        self.phase = phase
        self.started_at = time.monotonic()
        self.expires_at = self.started_at + seconds
        self.expired = False

    def remaining(self) -> float:
        value = self.expires_at - time.monotonic()
        if value <= 0:
            self.raise_expired()
        return value

    def raise_expired(self) -> None:
        self.expired = True
        raise HardDeadlineExceeded(self.phase)


@contextmanager
def hard_deadline(seconds: int, phase: str):
    """Interrupt Python/Ray waits at one plan-bound wall-clock deadline.

    Diagnostic entrypoints run on the main thread of a dedicated Linux process.
    Refuse to replace an existing alarm or run from another thread: either case
    would make timeout ownership ambiguous.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hard lifecycle deadline requires the main thread")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("hard lifecycle deadline cannot replace an existing process alarm")
    previous_handler = signal.getsignal(signal.SIGALRM)
    deadline = HardDeadline(seconds, phase)

    def expired(_signum, _frame):
        deadline.raise_expired()

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def sealed(value, schema):
    if value.get("schema") != schema or value.get("sha256", "").removeprefix("sha256:") != digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("input receipt schema/digest mismatch")


def sanitized_causes(error):
    """Keep nested error types/code locations, never private messages."""
    causes, seen, pending = [], set(), [error]
    while pending and len(causes) < 8:
        error = pending.pop(0)
        if not isinstance(error, BaseException) or id(error) in seen:
            continue
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
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
        pending.append(getattr(error, "cause", None) or error.__cause__ or error.__context__)
    return causes


def native_failure(plan, error):
    """Persist Ray's sanitized failure chain for a real native training run."""
    return _write(
        Path(plan["output_root"]) / "NATIVE_FAILURE.json",
        {"plan_sha256": digest(plan), "causes": sanitized_causes(error)},
    )


def native_rejection(plan, error):
    from .rl_episode import budget_stop

    if not (reason := budget_stop(error)):
        return False
    _write(
        Path(plan["output_root"]) / "NATIVE_REJECTED.json",
        {
            "schema": "cyber_rl_native_rejection_v1",
            "status": "rejected",
            "reason": reason,
            "plan_sha256": digest(plan),
        },
    )
    return True


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
    process, reason = None, None
    try:
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
                previous_checkpoint = ()
                while process.poll() is None:
                    try:
                        process.wait(timeout=WATCHDOG_POLL_SECONDS)
                    except subprocess.TimeoutExpired:
                        gpu, io = _utilization_snapshot()
                        marker = progress(root)
                        checkpoint = tuple(
                            row for row in marker if row[0].startswith("checkpoints/")
                        )
                        reason = watchdog.observe(
                            time.monotonic(),
                            marker,
                            gpu,
                            io,
                            checkpoint_advancing=bool(checkpoint)
                            and checkpoint != previous_checkpoint,
                        )
                        previous_checkpoint = checkpoint
                        if reason:
                            raise TimeoutError(reason) from None
                if process.returncode != 0:
                    raise RuntimeError("native RL did not finish")
        finally:
            # Teardown must complete before any terminal success/rejection marker.
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        rejection = root / "NATIVE_REJECTED.json"
        if rejection.exists():
            from .rl_episode import BUDGET_STOPS

            result = json.loads(rejection.read_bytes())
            sealed(result, "cyber_rl_native_rejection_v1")
            if (
                result.get("plan_sha256") != digest(plan)
                or result.get("status") != "rejected"
                or result.get("reason") not in BUDGET_STOPS
                or any(
                    (root / name).exists()
                    for name in (
                        "FAILED.json",
                        "NATIVE_FAILURE.json",
                        "NATIVE_TRAINING_COMPLETE.json",
                        "ACCEPTED.json",
                    )
                )
            ):
                raise ValueError("native rejection is conflicting or not plan-bound")
            # Zero exit means the bounded operation shut down cleanly, not that
            # training completed. Do not fabricate a checkpoint or acceptance.
            return _write(
                root / "REJECTED.json", {k: v for k, v in result.items() if k != "sha256"}
            )
        return _write(root / "NATIVE_TRAINING_COMPLETE.json", backend.native_result(plan))
    except BaseException as exc:
        _write(
            root / "FAILED.json",
            {
                "status": "failed",
                "plan_sha256": digest(plan),
                "error_class": type(exc).__name__,
                "watchdog_reason": reason,
            },
        )
        raise RuntimeError(
            "RL run failed; private evidence preserved; no automatic retry"
        ) from None
