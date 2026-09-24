"""Offline tests for the dedicated one-cell task-quality reservation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import campaign
from evals.fleet import daily_rollout_ledger as ledger
from evals.fleet import task_quality_reservation as reservation

NOW = datetime(2026, 9, 24, 12, tzinfo=UTC)
DATE = NOW.date().isoformat()
PLAN_SHA = "sha256:" + "1" * 64
PACKET_SHA = "sha256:" + "2" * 64
AUTH_SHA = "sha256:" + "3" * 64
CENSUS_SHA = "sha256:" + "4" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(campaign.canonical(value) + b"\n")
    return path


def _seal(value: dict) -> dict:
    return {**value, "sha256": ledger.digest(value)}


def _inputs() -> tuple[SimpleNamespace, dict, dict]:
    package = SimpleNamespace(
        packet=SimpleNamespace(value={"sha256": PACKET_SHA}, files={"plan": Path("plan")})
    )
    authorization = {"sha256": AUTH_SHA}
    plan = {
        "sha256": PLAN_SHA,
        "tasks": [{"task_key": "task-a", "task_version_id": "version-a"}],
        "selection": {
            "exact_task_identity": {
                "task_key": "task-a",
                "task_version_id": "version-a",
            }
        },
    }
    return package, authorization, plan


def test_cell_digest_has_an_independent_oracle_and_is_identity_sensitive() -> None:
    plan = _inputs()[2]
    exact = [{"task_key": "task-a", "task_version_id": "version-a"}]
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(exact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert reservation._cell_digest(plan) == expected  # noqa: SLF001
    changed = {
        **plan,
        "selection": {
            "exact_task_identity": {
                "task_key": "task-a",
                "task_version_id": "version-b",
            }
        },
    }
    assert reservation._cell_digest(changed) != expected  # noqa: SLF001


def _root(tmp_path: Path, *, used: int = 271) -> Path:
    root = tmp_path / "budget"
    day = root / DATE
    (day / "reservations").mkdir(parents=True)
    (day / ".lock").touch()
    baseline = _seal(
        {
            "schema": ledger.BUDGET_SCHEMA,
            "date_utc": DATE,
            "cap": ledger.CAP,
            "used": used,
            "census_receipt_sha256": CENSUS_SHA,
        }
    )
    _write(day / "baseline.json", baseline)
    return root


def _legacy(count: int) -> dict:
    return _seal(
        {
            "schema": ledger.LEGACY_RESERVATION_SCHEMA,
            "date_utc": DATE,
            "group_id": "legacy-group",
            "count": count,
            "packet_sha256": "sha256:" + "5" * 64,
            "bindings_sha256": "sha256:" + "6" * 64,
        }
    )


def _wave(count: int) -> dict:
    return _seal(
        {
            "schema": ledger.WAVE_RESERVATION_SCHEMA,
            "date_utc": DATE,
            "cap": ledger.CAP,
            "count": count,
            "reservation_id": "heldout-wave",
            "packet_set_sha256": "sha256:" + "7" * 64,
            "groups": [
                {
                    "group_id": "heldout-group",
                    "count": count,
                    "packet_sha256": "sha256:" + "7" * 64,
                }
            ],
            "bindings_sha256": "sha256:" + "8" * 64,
        }
    )


def _concurrent_reserve_worker(
    root: str,
    receipt: str,
    index: int,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    suffix = f"{index:x}"
    package = SimpleNamespace(packet=SimpleNamespace(value={"sha256": "sha256:" + suffix * 64}))
    authorization = {"sha256": "sha256:" + str(index + 2) * 64}
    plan = {
        "sha256": "sha256:" + str(index + 4) * 64,
        "tasks": [{"task_key": f"task-{index}", "task_version_id": f"version-{index}"}],
        "selection": {
            "exact_task_identity": {
                "task_key": f"task-{index}",
                "task_version_id": f"version-{index}",
            }
        },
    }
    reservation._inputs = lambda *_args: (package, authorization, plan)  # noqa: SLF001
    start.wait(timeout=10)
    try:
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=Path(receipt),
            root=Path(root),
            reserve=True,
            now=NOW,
        )
    except reservation.ReservationError as exc:
        results.put(("blocked", str(exc)))
    else:
        results.put(("reserved", ""))


@pytest.fixture()
def fake_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reservation, "_inputs", lambda *_args: _inputs())


def test_reserve_counts_current_mixed_ledger_and_exact_replay(
    tmp_path: Path, fake_inputs: None
) -> None:
    root = _root(tmp_path)
    reservations = root / DATE / "reservations"
    _write(reservations / "legacy.json", _legacy(27))
    _write(reservations / "wave.json", _wave(160))
    receipt_path = tmp_path / "reservation-receipt.json"
    preview = reservation.reserve_qualification_cell(
        Path("packet"), Path("authorization"), root=root, reserve=False, now=NOW
    )
    assert preview["committed"] == 459 and preview["remaining"] == 41
    assert not receipt_path.exists()
    first = reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt_path,
        root=root,
        reserve=True,
        now=NOW,
    )
    second = reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt_path,
        root=root,
        reserve=True,
        now=NOW,
    )
    assert first == second
    assert first["committed"] == 459 and first["remaining"] == 41
    assert len(list(reservations.glob("qa-*.json"))) == 1
    assert "task-a" not in receipt_path.read_text()
    assert "version-a" not in receipt_path.read_text()
    assert "task-a" not in next(reservations.glob("qa-*.json")).read_text()
    loaded = reservation.load_reservation_receipt(
        receipt_path, Path("packet"), Path("authorization"), root=root, now=NOW
    )
    assert loaded == first


@pytest.mark.parametrize("used,accepted", [(499, True), (500, False)])
def test_capacity_boundary_and_replay_at_cap(
    tmp_path: Path, fake_inputs: None, used: int, accepted: bool
) -> None:
    root = _root(tmp_path, used=used)
    receipt = tmp_path / "receipt.json"
    if not accepted:
        with pytest.raises(reservation.ReservationError, match="cannot admit"):
            reservation.reserve_qualification_cell(
                Path("packet"),
                Path("authorization"),
                receipt_path=receipt,
                root=root,
                reserve=True,
                now=NOW,
            )
        assert not list((root / DATE / "reservations").glob("qa-*.json"))
        return
    first = reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    assert first["committed"] == 500
    assert (
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=receipt,
            root=root,
            reserve=True,
            now=NOW,
        )
        == first
    )


def test_missing_receipt_path_fails_before_ledger_mutation(
    tmp_path: Path, fake_inputs: None
) -> None:
    root = _root(tmp_path)
    with pytest.raises(reservation.ReservationError, match="receipt path"):
        reservation.reserve_qualification_cell(
            Path("packet"), Path("authorization"), root=root, reserve=True, now=NOW
        )
    assert not list((root / DATE / "reservations").glob("qa-*.json"))


def test_conflicting_receipt_fails_before_ledger_mutation(
    tmp_path: Path, fake_inputs: None
) -> None:
    root = _root(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(reservation.ReservationError, match="must be absent"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=receipt,
            root=root,
            reserve=True,
            now=NOW,
        )
    assert ledger.reservation_index(root / DATE / "reservations", date_utc=DATE).total == 0


@pytest.mark.parametrize("parent_kind", ["missing", "symlink"])
def test_invalid_receipt_parent_fails_before_ledger_mutation(
    tmp_path: Path, fake_inputs: None, parent_kind: str
) -> None:
    root = _root(tmp_path)
    parent = tmp_path / "receipt-parent"
    if parent_kind == "symlink":
        target = tmp_path / "receipt-target"
        target.mkdir()
        parent.symlink_to(target, target_is_directory=True)
    receipt = parent / "receipt.json"
    with pytest.raises(reservation.ReservationError, match="receipt"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=receipt,
            root=root,
            reserve=True,
            now=NOW,
        )
    assert ledger.reservation_index(root / DATE / "reservations", date_utc=DATE).total == 0


@pytest.mark.parametrize("location", ["reservations", "exact-row", "day"])
def test_receipt_inside_canonical_ledger_fails_before_mutation(
    tmp_path: Path, fake_inputs: None, location: str
) -> None:
    root = _root(tmp_path)
    day = root / DATE
    cell = reservation._cell_digest(_inputs()[2])  # noqa: SLF001
    if location == "reservations":
        receipt = day / "reservations" / "receipt.json"
    elif location == "exact-row":
        receipt = day / "reservations" / f"qa-{cell[7:]}.json"
    else:
        receipt = day / "receipt.json"
    with pytest.raises(reservation.ReservationError, match="unsafe"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=receipt,
            root=root,
            reserve=True,
            now=NOW,
        )
    assert ledger.reservation_index(day / "reservations", date_utc=DATE).total == 0


def test_two_processes_competing_for_last_slot_cannot_overbook(tmp_path: Path) -> None:
    root = _root(tmp_path, used=499)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_concurrent_reserve_worker,
            args=(str(root), str(tmp_path / f"receipt-{index}.json"), index, start, results),
        )
        for index in (1, 2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    outcomes = sorted(results.get(timeout=2)[0] for _ in workers)
    assert outcomes == ["blocked", "reserved"]
    index = ledger.reservation_index(root / DATE / "reservations", date_utc=DATE)
    assert index.total == 1


def test_utc_rollover_before_publish_writes_nothing(
    tmp_path: Path, fake_inputs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)

    class RolloverClock:
        calls = 0

        @classmethod
        def now(cls, _timezone):
            cls.calls += 1
            return NOW if cls.calls < 3 else NOW + timedelta(days=1)

    monkeypatch.setattr(reservation, "datetime", RolloverClock)
    with pytest.raises(reservation.ReservationError, match="before reservation publication"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=tmp_path / "receipt.json",
            root=root,
            reserve=True,
        )
    assert not list((root / DATE / "reservations").glob("qa-*.json"))


def test_crash_after_install_recovers_exact_receipt(
    tmp_path: Path, fake_inputs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    receipt = tmp_path / "receipt.json"
    real_write = reservation._write_receipt_once  # noqa: SLF001
    monkeypatch.setattr(
        reservation,
        "_write_receipt_once",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("crash after install")),
    )
    with pytest.raises(RuntimeError, match="after install"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=receipt,
            root=root,
            reserve=True,
            now=NOW,
        )
    assert len(list((root / DATE / "reservations").glob("qa-*.json"))) == 1
    monkeypatch.setattr(reservation, "_write_receipt_once", real_write)
    recovered = reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    assert recovered["committed"] == 272 and receipt.is_file()


def test_exact_replay_returns_original_receipt_after_unrelated_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    first_inputs = _inputs()
    second_inputs = (
        SimpleNamespace(
            packet=SimpleNamespace(
                value={"sha256": "sha256:" + "9" * 64}, files={"plan": Path("plan-b")}
            )
        ),
        {"sha256": "sha256:" + "a" * 64},
        {
            "sha256": "sha256:" + "b" * 64,
            "tasks": [{"task_key": "task-b", "task_version_id": "version-b"}],
            "selection": {
                "exact_task_identity": {
                    "task_key": "task-b",
                    "task_version_id": "version-b",
                }
            },
        },
    )
    inputs = {"packet-a": first_inputs, "packet-b": second_inputs}
    monkeypatch.setattr(reservation, "_inputs", lambda packet, _authorization: inputs[packet.name])
    first_receipt = tmp_path / "first.json"
    first = reservation.reserve_qualification_cell(
        Path("packet-a"),
        Path("authorization-a"),
        receipt_path=first_receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    first_bytes = first_receipt.read_bytes()
    reservation.reserve_qualification_cell(
        Path("packet-b"),
        Path("authorization-b"),
        receipt_path=tmp_path / "second.json",
        root=root,
        reserve=True,
        now=NOW,
    )
    replay = reservation.reserve_qualification_cell(
        Path("packet-a"),
        Path("authorization-a"),
        receipt_path=first_receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    assert replay == first
    assert first_receipt.read_bytes() == first_bytes
    assert ledger.reservation_index(root / DATE / "reservations", date_utc=DATE).total == 2


@pytest.mark.parametrize("defect", ["missing", "tampered", "stale"])
def test_missing_tampered_or_stale_receipt_blocks(
    tmp_path: Path, fake_inputs: None, defect: str
) -> None:
    root = _root(tmp_path)
    receipt = tmp_path / "receipt.json"
    reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    if defect == "missing":
        receipt.unlink()
    elif defect == "tampered":
        value = json.loads(receipt.read_text())
        value["packet_sha256"] = "sha256:" + "f" * 64
        value["sha256"] = ledger.digest({k: v for k, v in value.items() if k != "sha256"})
        receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((reservation.ReservationError, OSError)):
        reservation.load_reservation_receipt(
            receipt,
            Path("packet"),
            Path("authorization"),
            root=root,
            now=NOW - timedelta(days=1) if defect == "stale" else NOW,
        )


@pytest.mark.parametrize(("field", "value"), [("count", True), ("count", 1.0), ("cap", 500.0)])
def test_receipt_count_and_cap_require_exact_integers(
    tmp_path: Path, fake_inputs: None, field: str, value: object
) -> None:
    root = _root(tmp_path)
    receipt = tmp_path / "receipt.json"
    reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt,
        root=root,
        reserve=True,
        now=NOW,
    )
    row = json.loads(receipt.read_text())
    row[field] = value
    row["sha256"] = ledger.digest({key: item for key, item in row.items() if key != "sha256"})
    _write(receipt, row)
    with pytest.raises(reservation.ReservationError, match="receipt is invalid"):
        reservation.load_reservation_receipt(
            receipt, Path("packet"), Path("authorization"), root=root, now=NOW
        )


def test_receipt_validation_rejects_utc_rollover_after_lock(
    tmp_path: Path, fake_inputs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    receipt = tmp_path / "receipt.json"
    reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=receipt,
        root=root,
        reserve=True,
        now=NOW,
    )

    class RolloverClock:
        calls = 0

        @classmethod
        def now(cls, _timezone):
            cls.calls += 1
            return NOW if cls.calls == 1 else NOW + timedelta(days=1)

    monkeypatch.setattr(reservation, "datetime", RolloverClock)
    with pytest.raises(reservation.ReservationError, match="UTC date changed"):
        reservation.load_reservation_receipt(
            receipt, Path("packet"), Path("authorization"), root=root
        )


@pytest.mark.parametrize("field", ["plan", "packet", "authorization"])
def test_conflicting_cell_plan_packet_or_authorization_fails_closed(
    tmp_path: Path, fake_inputs: None, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    root = _root(tmp_path)
    reservation.reserve_qualification_cell(
        Path("packet"),
        Path("authorization"),
        receipt_path=tmp_path / "first.json",
        root=root,
        reserve=True,
        now=NOW,
    )
    package, authorization, plan = _inputs()
    if field == "plan":
        plan = {**plan, "sha256": "sha256:" + "9" * 64}
    elif field == "packet":
        package.packet.value["sha256"] = "sha256:" + "9" * 64
    else:
        authorization = {"sha256": "sha256:" + "9" * 64}
    monkeypatch.setattr(reservation, "_inputs", lambda *_args: (package, authorization, plan))
    with pytest.raises(reservation.ReservationError, match="different reservation"):
        reservation.reserve_qualification_cell(
            Path("packet"),
            Path("authorization"),
            receipt_path=tmp_path / "second.json",
            root=root,
            reserve=True,
            now=NOW,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"tasks": []},
        {"execution": {"concurrency": 2}},
        {"execution": {"model_calls": 1}},
        {"execution": {"automatic_retry": True}},
    ],
)
def test_input_contract_rejects_nonexact_cell(
    monkeypatch: pytest.MonkeyPatch, change: dict
) -> None:
    package, authorization, plan = _inputs()
    plan = {
        **plan,
        "selection": {
            **plan["selection"],
            "requested_limit": 1,
            "selected_task_versions": 1,
        },
        "execution": {
            "concurrency": 1,
            "model_calls": 0,
            "external_mutations_authorized": True,
            "mutating_request_attempts": 1,
            "automatic_retry": False,
            "ambiguous_mutation_replay": False,
        },
        **change,
    }
    package.packet.value["execution"] = {
        "create_attempts_maximum": 1,
        "automatic_create_retry": False,
        "root_annotation": {"fleet.ai/failure-alerts": "off"},
    }
    fake_job = SimpleNamespace(
        FAILURE_ALERT_ANNOTATION="fleet.ai/failure-alerts",
        build_package=lambda _path: package,
        load_authorization=lambda _path, _package: authorization,
        read_json=lambda _path, _label: plan,
    )
    monkeypatch.setattr(reservation, "_job_module", lambda: fake_job)
    with pytest.raises(reservation.ReservationError, match="one exact"):
        reservation._inputs(Path("packet"), Path("authorization"))  # noqa: SLF001


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("selection", "requested_limit", True),
        ("selection", "requested_limit", 1.0),
        ("selection", "selected_task_versions", True),
        ("selection", "selected_task_versions", 1.0),
        ("execution", "concurrency", True),
        ("execution", "concurrency", 1.0),
        ("execution", "model_calls", False),
        ("execution", "model_calls", 0.0),
        ("execution", "mutating_request_attempts", True),
        ("execution", "mutating_request_attempts", 1.0),
        ("packet", "create_attempts_maximum", True),
        ("packet", "create_attempts_maximum", 1.0),
    ],
)
def test_input_contract_numeric_fields_require_exact_integers(
    monkeypatch: pytest.MonkeyPatch, scope: str, field: str, value: object
) -> None:
    package, authorization, plan = _inputs()
    plan = {
        **plan,
        "selection": {
            **plan["selection"],
            "requested_limit": 1,
            "selected_task_versions": 1,
        },
        "execution": {
            "concurrency": 1,
            "model_calls": 0,
            "external_mutations_authorized": True,
            "mutating_request_attempts": 1,
            "automatic_retry": False,
            "ambiguous_mutation_replay": False,
        },
    }
    packet_execution = {
        "create_attempts_maximum": 1,
        "automatic_create_retry": False,
        "root_annotation": {"fleet.ai/failure-alerts": "off"},
    }
    if scope == "selection":
        plan["selection"][field] = value
    elif scope == "execution":
        plan["execution"][field] = value
    else:
        packet_execution[field] = value
    package.packet.value["execution"] = packet_execution
    fake_job = SimpleNamespace(
        FAILURE_ALERT_ANNOTATION="fleet.ai/failure-alerts",
        build_package=lambda _path: package,
        load_authorization=lambda _path, _package: authorization,
        read_json=lambda _path, _label: plan,
    )
    monkeypatch.setattr(reservation, "_job_module", lambda: fake_job)
    with pytest.raises(reservation.ReservationError, match="one exact"):
        reservation._inputs(Path("packet"), Path("authorization"))  # noqa: SLF001
