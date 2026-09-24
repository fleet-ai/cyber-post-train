"""Regressions for shared accounting across Fleet daily reservation schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import campaign
from evals.fleet import daily_rollout_ledger as ledger

DATE = "2026-09-24"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(campaign.canonical(value) + b"\n")
    return path


def _seal(value: dict) -> dict:
    return {**value, "sha256": ledger.digest(value)}


def _legacy(count: int = 27) -> dict:
    return _seal(
        {
            "schema": ledger.LEGACY_RESERVATION_SCHEMA,
            "date_utc": DATE,
            "group_id": "legacy-group",
            "count": count,
            "packet_sha256": SHA_A,
            "bindings_sha256": SHA_B,
        }
    )


def _wave(count: int = 160) -> dict:
    return _seal(
        {
            "schema": ledger.WAVE_RESERVATION_SCHEMA,
            "date_utc": DATE,
            "cap": 500,
            "count": count,
            "reservation_id": "heldout20-base-step1000-v6",
            "packet_set_sha256": SHA_C,
            "groups": [
                {
                    "group_id": "heldout-wave",
                    "count": count,
                    "packet_sha256": SHA_C,
                }
            ],
            "bindings_sha256": SHA_D,
        }
    )


def _qa(**changes) -> dict:
    body = {
        "schema": ledger.QA_RESERVATION_SCHEMA,
        "date_utc": DATE,
        "cap": 500,
        "count": 1,
        "plan_sha256": SHA_A,
        "packet_sha256": SHA_B,
        "authorization_sha256": SHA_C,
        "cell_universe_sha256": SHA_E,
        **changes,
    }
    return _seal(body)


def test_shared_index_counts_legacy_wave_and_qa_without_qa_becoming_wave(
    tmp_path: Path,
) -> None:
    reservations = tmp_path / "reservations"
    _write(reservations / "legacy.json", _legacy())
    _write(reservations / "wave.json", _wave())
    _write(reservations / f"qa-{SHA_E[7:]}.json", _qa())
    index = ledger.reservation_index(reservations, date_utc=DATE)
    assert index.total == 188
    assert set(index.covered) == {(SHA_B, "legacy-group"), (SHA_D, "heldout-wave")}
    assert index.waves == {SHA_D: _wave()}
    assert index.qualifications == {SHA_E: _qa()}
    assert index.qa_plans == {SHA_A}
    assert index.qa_packets == {SHA_B}


@pytest.mark.parametrize(
    "change,match",
    [
        ({"schema": "unknown"}, "schema"),
        ({"count": 2}, "task-quality"),
        ({"count": True}, "task-quality"),
        ({"count": 1.0}, "task-quality"),
        ({"cap": 499}, "task-quality"),
        ({"cap": 500.0}, "task-quality"),
        ({"date_utc": "2026-09-23"}, "digest/date"),
        ({"plan_sha256": "bad"}, "plan"),
    ],
)
def test_malformed_qa_reservation_fails_closed(tmp_path: Path, change: dict, match: str) -> None:
    value = _qa(**change)
    path = tmp_path / "reservations" / f"qa-{value.get('cell_universe_sha256', SHA_E)[7:]}.json"
    _write(path, value)
    with pytest.raises(ledger.LedgerError, match=match):
        ledger.reservation_index(path.parent, date_utc=DATE)


def test_qa_path_and_self_digest_are_exact(tmp_path: Path) -> None:
    reservations = tmp_path / "reservations"
    value = _qa()
    _write(reservations / "wrong.json", value)
    with pytest.raises(ledger.LedgerError, match="path"):
        ledger.reservation_index(reservations, date_utc=DATE)
    (reservations / "wrong.json").unlink()
    value["count"] = 2
    _write(reservations / f"qa-{SHA_E[7:]}.json", value)
    with pytest.raises(ledger.LedgerError, match="digest"):
        ledger.reservation_index(reservations, date_utc=DATE)


@pytest.mark.parametrize("field", ["plan_sha256", "packet_sha256"])
def test_qa_exact_identity_cannot_repeat(tmp_path: Path, field: str) -> None:
    reservations = tmp_path / "reservations"
    first = _qa()
    changes = {
        "plan_sha256": SHA_D,
        "packet_sha256": SHA_D,
        "cell_universe_sha256": SHA_D,
    }
    changes[field] = first[field]
    second = _qa(**changes)
    _write(reservations / f"qa-{first['cell_universe_sha256'][7:]}.json", first)
    _write(reservations / f"qa-{second['cell_universe_sha256'][7:]}.json", second)
    with pytest.raises(ledger.LedgerError, match="duplicated"):
        ledger.reservation_index(reservations, date_utc=DATE)


def test_unknown_final_entry_fails_but_known_atomic_temp_is_ignored(tmp_path: Path) -> None:
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    (reservations / ".qa-dead.json.abc123.tmp").write_text("partial")
    index = ledger.reservation_index(reservations, date_utc=DATE)
    assert index.total == 0
    (reservations / "notes.txt").write_text("unexpected")
    with pytest.raises(ledger.LedgerError, match="unknown entry"):
        ledger.reservation_index(reservations, date_utc=DATE)


def test_baseline_and_wave_cap_require_exact_integers(tmp_path: Path) -> None:
    baseline = _seal(
        {
            "schema": ledger.BUDGET_SCHEMA,
            "date_utc": DATE,
            "cap": 500.0,
            "used": 0,
            "census_receipt_sha256": SHA_A,
        }
    )
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    with pytest.raises(ledger.LedgerError, match="baseline"):
        ledger.validate_baseline(baseline_path, DATE)

    reservations = tmp_path / "reservations"
    wave = _wave()
    wave["cap"] = 500.0
    wave["sha256"] = ledger.digest({key: value for key, value in wave.items() if key != "sha256"})
    _write(reservations / "wave.json", wave)
    with pytest.raises(ledger.LedgerError, match="wave"):
        ledger.reservation_index(reservations, date_utc=DATE)
