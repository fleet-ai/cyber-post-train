"""Versioned SFT runtime wrapper for native-save-return checkpoint telemetry.

This module deliberately wraps, rather than edits, ``sft_runtime.py``. Existing
plans keep executing the exact runtime bytes they already bind. A new plan must
opt into :data:`RUNTIME_VARIANT`, which binds both this entrypoint and the
unchanged base runtime before the request is rendered.

The filesystem snapshot is taken on the trainer driver immediately after the
native ``SFTTrainer.save_checkpoint`` call returns. It does not add a worker
barrier or finalize an asynchronous distributed writer, so every metric is
explicitly named ``native_save_return`` and makes no durability claim.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from training import sft_runtime as _base_runtime

RUNTIME_VARIANT = "native_save_return_v1"
RUNTIME_BINDING_SCHEMA = "cyber_sft_runtime_variant_v1"
METRIC_KEYS = (
    "checkpoint/native_save_return_duration_seconds",
    "checkpoint/native_save_return_bytes",
    "checkpoint/native_save_return_file_count",
    "checkpoint/native_save_return_stat_error_count",
)

_BASE_MAKE_TRAINER_CLASS = _base_runtime._make_trainer_class
_BASE_VALIDATE_PLAN = _base_runtime.validate_plan


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_binding(base_runtime_path: Path | None = None) -> dict[str, str]:
    """Return the immutable base-runtime binding required by the successor."""
    path = base_runtime_path or Path(_base_runtime.__file__)
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "name": RUNTIME_VARIANT,
        "base_runtime_sha256": _file_sha256(path),
    }


def validate_runtime_binding(
    plan: dict,
    *,
    runtime_path: Path | None = None,
    base_runtime_path: Path | None = None,
    check_files: bool = True,
) -> None:
    """Fail closed unless a plan selects this exact wrapper/base pair."""
    binding = plan.get("runtime_variant")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"schema", "name", "base_runtime_sha256"}
        or binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or binding.get("name") != RUNTIME_VARIANT
        or re.fullmatch(r"[a-f0-9]{64}", binding.get("base_runtime_sha256", "")) is None
        or re.fullmatch(r"[a-f0-9]{64}", plan.get("runtime_sha256", "")) is None
        or plan["runtime_sha256"] == binding["base_runtime_sha256"]
    ):
        raise ValueError("SFT runtime telemetry successor binding is invalid")
    if "lora" in plan:
        raise ValueError("native-save-return telemetry is qualified only for full-weight SFT")
    if check_files:
        wrapper = runtime_path or Path(__file__)
        base = base_runtime_path or Path(_base_runtime.__file__)
        if (
            _file_sha256(wrapper) != plan["runtime_sha256"]
            or _file_sha256(base) != binding["base_runtime_sha256"]
        ):
            raise ValueError("SFT runtime telemetry successor source digest mismatch")


def _observe_checkpoint_at_native_save_return(path: str | os.PathLike[str]) -> dict[str, int]:
    """Best-effort regular-file census that never follows links or raises."""
    files = 0
    total_bytes = 0
    stat_errors = 0
    try:
        pending = [os.fspath(path)]
    except Exception:
        pending = []
        stat_errors = 1
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        observed = entry.stat(follow_symlinks=False)
                    except Exception:
                        stat_errors += 1
                        continue
                    if stat.S_ISDIR(observed.st_mode):
                        pending.append(entry.path)
                    elif stat.S_ISREG(observed.st_mode):
                        files += 1
                        total_bytes += observed.st_size
        except Exception:
            stat_errors += 1
    return {
        "bytes": total_bytes,
        "file_count": files,
        "stat_error_count": stat_errors,
    }


def _pending_metrics(trainer) -> dict[int, dict[str, int | float]]:
    pending = getattr(trainer, "_native_save_return_metrics", None)
    if pending is None:
        pending = {}
        trainer._native_save_return_metrics = pending
    return pending


class _NativeSaveReturnTelemetryMixin:
    """Inserted between the audited trainer and its native SkyRL base class."""

    def save_checkpoint(self):
        started = None
        with contextlib.suppress(Exception):
            started = time.perf_counter()
        path = super().save_checkpoint()
        with contextlib.suppress(Exception):
            if started is not None:
                duration = time.perf_counter() - started
                observed = _observe_checkpoint_at_native_save_return(path)
                _pending_metrics(self)[int(self.global_step)] = {
                    METRIC_KEYS[0]: float(duration),
                    METRIC_KEYS[1]: int(observed["bytes"]),
                    METRIC_KEYS[2]: int(observed["file_count"]),
                    METRIC_KEYS[3]: int(observed["stat_error_count"]),
                }
        return path

    def _fire(self, event_name: str, **fields) -> None:
        # An in-loop periodic/final save is followed by the native same-step log.
        # Join that event so W&B receives one monotonic commit and the existing
        # local scalar callback records the checkpoint metrics exactly once.
        if event_name == "on_log" and isinstance(fields.get("logs"), dict):
            with contextlib.suppress(Exception):
                metrics = _pending_metrics(self).pop(int(self.global_step), None)
                if metrics is not None:
                    fields["logs"].update(metrics)
        return super()._fire(event_name, **fields)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            # The native postamble can perform a final-only save after the last
            # training log. Explicitly commit any remaining scalar event before
            # shutdown; telemetry failures never change training/checkpoint state.
            with contextlib.suppress(Exception):
                self._flush_native_save_return_metrics()

    def _flush_native_save_return_metrics(self) -> None:
        pending = _pending_metrics(self)
        events = sorted(pending.items())
        pending.clear()
        for optimizer_step, metrics in events:
            with (
                contextlib.suppress(Exception),
                (Path(self.output) / "metrics.jsonl").open("a") as stream,
            ):
                stream.write(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_step,
                            "time": time.time(),
                            **metrics,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            with contextlib.suppress(Exception):
                # The pinned tracker is W&B-backed. Omitting an explicit step
                # commits at its current monotonic step, even when the final
                # training event already advanced beyond optimizer_step.
                self.tracker.logger.log(data=metrics, commit=True)


def telemetry_trainer_class(base_trainer_class):
    """Insert telemetry at the exact native-save boundary or fail closed."""
    if (
        len(base_trainer_class.__bases__) != 1
        or "save_checkpoint" not in base_trainer_class.__dict__
        or "train" in base_trainer_class.__dict__
        or "_fire" in base_trainer_class.__dict__
    ):
        raise ValueError("SFT trainer inheritance changed at the telemetry boundary")
    native_trainer_class = base_trainer_class.__bases__[0]
    return type(
        "NativeSaveReturnTelemetryTrainer",
        (base_trainer_class, _NativeSaveReturnTelemetryMixin, native_trainer_class),
        {"__module__": __name__},
    )


def _make_trainer_class():
    return telemetry_trainer_class(_BASE_MAKE_TRAINER_CLASS())


def _validate_plan(plan: dict, *, check_files: bool = True) -> None:
    validate_runtime_binding(plan, check_files=check_files)
    _BASE_VALIDATE_PLAN(plan, check_files=check_files)


def main() -> None:
    _base_runtime._make_trainer_class = _make_trainer_class
    _base_runtime.validate_plan = _validate_plan
    _base_runtime.main()


if __name__ == "__main__":
    main()
