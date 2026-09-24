"""Offline regressions for shared Fleet source Jobs behind campaign cells."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.fleet import campaign_adapter as adapter
from evals.fleet import heldout_launch

KEY_A = "sha256:" + "a" * 64
KEY_B = "sha256:" + "b" * 64
JOB_UID = "11111111-2222-4333-8444-555555555555"
PACKET_SHA = "sha256:" + "c" * 64
ROUTE_SHA = "sha256:" + "d" * 64


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _campaign(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    state = tmp_path / "campaign"
    _write(state / "plan.json", {})
    packets = {
        key: _write(
            state / "targets" / key / "packet.json",
            {
                "experiment_key": key,
                "identity": {"attempt": 1, "seed": 46, "model": {"id": "base"}},
            },
        )
        for key in (KEY_A, KEY_B)
    }
    cells = {
        key: {
            "group": "base-seed46",
            "attempt": 1,
            "task_version_id": f"task-{index}",
            "model_id": "base",
            "model_revision": "revision-a",
            "source_attempt": 1,
            "campaign_model_id": "base",
        }
        for index, key in enumerate((KEY_A, KEY_B), 1)
    }
    bindings = {
        "schema": adapter.SCHEMA,
        "budget": {
            "date_utc": datetime.now(UTC).date().isoformat(),
            "cap": 500,
            "used": 0,
            "census_receipt_sha256": "sha256:" + "e" * 64,
        },
        "groups": {
            "base-seed46": {
                "packet": "source/LAUNCH_PACKET.json",
                "packet_sha256": PACKET_SHA,
                "leader": KEY_A,
                "cells": [KEY_A, KEY_B],
            }
        },
        "cells": cells,
    }
    bindings["sha256"] = adapter._digest(bindings)  # noqa: SLF001
    return packets, bindings


def _package() -> heldout_launch.Package:
    packet = SimpleNamespace(
        namespace="fleet-train-jobs",
        job_name="heldout-base-seed46",
        database="heldout_base_seed46",
        identity={"pass_k": 1, "retry_limit": 0, "sampling_seed": 46},
    )
    return SimpleNamespace(packet=packet, evaluation_config={})


class Cluster:
    def __init__(self, *, terminal: bool = True) -> None:
        self.terminal = terminal

    def get(self, _resource: str, _namespace: str, _name: str) -> dict[str, Any]:
        conditions = [{"type": "Complete", "status": "True"}] if self.terminal else []
        return {"metadata": {"uid": JOB_UID}, "status": {"conditions": conditions}}


class Database:
    def __init__(self, row: dict[str, Any] | None = None) -> None:
        self.row = row or {}

    def cell_status(self, _database: str, **_identity: Any) -> dict[str, Any]:
        return dict(self.row)


def _patch_source(monkeypatch: pytest.MonkeyPatch, bindings: dict[str, Any]) -> None:
    package = _package()

    def source(_bindings_path: Path, target: dict[str, Any]):
        return bindings, bindings["cells"][target["experiment_key"]], package, Path("source.json")

    monkeypatch.setattr(adapter, "_source", source)


def _phase_files(tmp_path: Path) -> tuple[Path, Path]:
    preview = _write(tmp_path / "preview.json", {"receipt_sha256": "sha256:" + "1" * 64})
    ready = _write(
        tmp_path / "ready.json",
        {"receipt_sha256": "sha256:" + "2" * 64, "route_profile_sha256": ROUTE_SHA},
    )
    return preview, ready


def test_shared_source_job_is_created_once_and_siblings_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, bindings = _campaign(tmp_path)
    _patch_source(monkeypatch, bindings)
    preview, ready = _phase_files(tmp_path)
    calls = 0

    def launch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "submitted": True,
            "gpus": 0,
            "job_name": "heldout-base-seed46",
            "job_uid": JOB_UID,
            "config_map_name": "heldout-base-seed46-code",
            "config_map_uid": "66666666-7777-4888-8999-aaaaaaaaaaaa",
            "evaluation_identity_sha256": "sha256:" + "3" * 64,
            "comparison_protocol_sha256": "sha256:" + "4" * 64,
        }

    monkeypatch.setattr(heldout_launch, "launch_once", launch)
    common = {
        "action": "launch",
        "phase": "rollout",
        "bindings_path": tmp_path / "bindings.json",
        "context": "fleet",
        "preview_receipt": preview,
        "readiness_receipt": ready,
        "cluster": Cluster(),
        "database": Database(),
        "budget_root": tmp_path / "budget",
    }
    first = adapter.run_action(packet_path=packets[KEY_A], **common)
    second = adapter.run_action(packet_path=packets[KEY_B], **common)
    assert calls == 1
    assert first["remote_id"] == second["remote_id"] == JOB_UID
    reservations = list(
        (
            tmp_path
            / "budget"
            / bindings["budget"]["date_utc"]
            / "reservations"
        ).glob("*.json")
    )
    assert len(reservations) == 1
    reservation = adapter._read(reservations[0])  # noqa: SLF001
    assert reservation["count"] == 2


def test_daily_used_plus_reserved_gate_is_atomic_and_idempotent(tmp_path: Path) -> None:
    _, bindings = _campaign(tmp_path)
    bindings["budget"]["used"] = 498
    bindings["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in bindings.items() if key != "sha256"}
    )
    root = tmp_path / "budget"
    first = adapter._budget(  # noqa: SLF001
        bindings, "base-seed46", 2, PACKET_SHA, root=root, reserve=True
    )
    second = adapter._budget(  # noqa: SLF001
        bindings, "base-seed46", 2, PACKET_SHA, root=root, reserve=True
    )
    assert first == second == {"used": 498, "reserved": 2, "cap": 500}
    with pytest.raises(adapter.CapacityUnavailable):
        adapter._budget(  # noqa: SLF001
            bindings, "another-group", 1, PACKET_SHA, root=root, reserve=True
        )


def test_unready_route_defers_without_duplicate_or_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, bindings = _campaign(tmp_path)
    _patch_source(monkeypatch, bindings)
    preview, _ = _phase_files(tmp_path)
    monkeypatch.setattr(
        heldout_launch,
        "duplicate_census",
        lambda *_args, **_kwargs: pytest.fail("duplicate census reached before route readiness"),
    )
    result = adapter.run_action(
        action="ready",
        phase="rollout",
        packet_path=packets[KEY_A],
        bindings_path=tmp_path / "bindings.json",
        context="fleet",
        preview_receipt=preview,
        cluster=Cluster(),
        database=Database(),
        route_check=lambda _package: (_ for _ in ()).throw(RuntimeError("unready")),
        budget_root=tmp_path / "budget",
    )
    assert result["status"] == "deferred_not_ready"
    assert result["defer_reason_code"] == "dependency_not_ready"


@pytest.mark.parametrize(
    ("row", "expected", "cleanup"),
    [
        (
            {
                "cell_id": "cell-a",
                "state": "accepted",
                "result_class": "valid",
                "local_results": 1,
                "retry_count": 0,
                "max_retries": 0,
                "receipt_digest": "sha256:" + "9" * 64,
            },
            "accepted",
            True,
        ),
        (
            {
                "cell_id": "cell-b",
                "state": "retry_review",
                "result_class": "infrastructure_invalid",
                "local_results": 0,
                "retry_count": 0,
                "max_retries": 0,
                "receipt_digest": None,
            },
            "infrastructure_invalid",
            False,
        ),
    ],
)
def test_terminal_cells_are_isolated_and_cleanup_is_proven_only_for_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, Any],
    expected: str,
    cleanup: bool,
) -> None:
    packets, bindings = _campaign(tmp_path)
    _patch_source(monkeypatch, bindings)
    root = packets[KEY_A].resolve().parents[2] / "fleet-source-jobs" / "base-seed46"
    launch_unsigned = {
        "schema": "cyber_fleet_source_launch_v1",
        "group_id": "base-seed46",
        "packet_sha256": PACKET_SHA,
        "route_profile_sha256": ROUTE_SHA,
        "job_uid": JOB_UID,
    }
    _write(root / "launch.json", {**launch_unsigned, "sha256": adapter._digest(launch_unsigned)})
    terminal_unsigned = {
        "schema": heldout_launch.TERMINAL_SCHEMA,
        "job": {"uid": JOB_UID},
    }
    _write(
        root / "terminal.json",
        {**terminal_unsigned, "sha256": adapter._digest(terminal_unsigned)},  # noqa: SLF001
    )
    launch = _write(
        tmp_path / "cell-launch.json",
        {"receipt_sha256": "sha256:" + "8" * 64, "remote_id": JOB_UID},
    )
    result = adapter.run_action(
        action="observe",
        phase="rollout",
        packet_path=packets[KEY_A],
        bindings_path=tmp_path / "bindings.json",
        context="fleet",
        launch_receipt=launch,
        cluster=Cluster(),
        database=Database(row),
    )
    assert result["status"] == expected
    assert result["evidence"]["cleanup_completed"] is cleanup
    assert "score" not in json.dumps(result).lower()
