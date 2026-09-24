"""Strict parser for the one shared Fleet daily rollout ledger."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import campaign

BUDGET_SCHEMA = "cyber_fleet_daily_rollout_budget_v1"
LEGACY_RESERVATION_SCHEMA = "cyber_fleet_daily_rollout_reservation_v1"
WAVE_RESERVATION_SCHEMA = "cyber_fleet_daily_rollout_wave_reservation_v1"
QA_RESERVATION_SCHEMA = "cyber_fleet_daily_task_quality_qualification_reservation_v1"
CAP = 500
ROOT = campaign.CANONICAL_REGISTRY / "fleet-daily-rollouts-v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_ATOMIC_TEMP = re.compile(r"\..+\.json\.[A-Za-z0-9_-]+\.tmp")


class LedgerError(ValueError):
    """The shared daily ledger is malformed or internally contradictory."""


@dataclass(frozen=True)
class ReservationIndex:
    total: int
    covered: dict[tuple[str, str], dict[str, Any]]
    waves: dict[str, dict[str, Any]]
    qualifications: dict[str, dict[str, Any]]
    qa_plans: frozenset[str]
    qa_packets: frozenset[str]


def digest(value: object) -> str:
    return campaign.digest(value)


def validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise LedgerError(f"{label} digest is invalid")
    return value


def read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LedgerError(f"{label} is absent")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"{label} is invalid") from exc
    if not isinstance(value, dict) or raw != campaign.canonical(value) + b"\n":
        raise LedgerError(f"{label} is not canonical JSON")
    return value


def validate_baseline(path: Path, date_utc: str) -> dict[str, Any]:
    value = read_canonical(path, "Fleet daily rollout baseline")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if (
        set(value) != {"schema", "date_utc", "cap", "used", "census_receipt_sha256", "sha256"}
        or value.get("schema") != BUDGET_SCHEMA
        or value.get("sha256") != digest(unsigned)
        or value.get("date_utc") != date_utc
        or type(value.get("cap")) is not int
        or value.get("cap") != CAP
        or type(value.get("used")) is not int
        or not 0 <= value["used"] <= CAP
    ):
        raise LedgerError("Fleet daily rollout baseline is invalid")
    validate_digest(value.get("census_receipt_sha256"), "Fleet census receipt")
    return value


def _legacy_groups(value: dict[str, Any]) -> list[dict[str, Any]]:
    if set(value) != {
        "schema",
        "date_utc",
        "group_id",
        "count",
        "packet_sha256",
        "bindings_sha256",
        "sha256",
    }:
        raise LedgerError("Fleet legacy reservation fields are invalid")
    return [
        {
            "group_id": value.get("group_id"),
            "count": value.get("count"),
            "packet_sha256": value.get("packet_sha256"),
        }
    ]


def _wave_groups(value: dict[str, Any]) -> list[dict[str, Any]]:
    if set(value) != {
        "schema",
        "date_utc",
        "cap",
        "count",
        "reservation_id",
        "packet_set_sha256",
        "groups",
        "bindings_sha256",
        "sha256",
    }:
        raise LedgerError("Fleet wave reservation fields are invalid")
    groups = value.get("groups")
    if (
        type(value.get("cap")) is not int
        or value.get("cap") != CAP
        or not isinstance(value.get("reservation_id"), str)
        or _ID.fullmatch(value["reservation_id"]) is None
        or not isinstance(groups, list)
        or not groups
    ):
        raise LedgerError("Fleet wave reservation is invalid")
    validate_digest(value.get("packet_set_sha256"), "Fleet wave packet set")
    return groups


def _validate_qualification(path: Path, value: dict[str, Any]) -> tuple[str, str, str]:
    if (
        set(value)
        != {
            "schema",
            "date_utc",
            "cap",
            "count",
            "plan_sha256",
            "packet_sha256",
            "authorization_sha256",
            "cell_universe_sha256",
            "sha256",
        }
        or type(value.get("cap")) is not int
        or value.get("cap") != CAP
        or type(value.get("count")) is not int
        or value.get("count") != 1
    ):
        raise LedgerError("Fleet task-quality reservation is invalid")
    plan = validate_digest(value.get("plan_sha256"), "qualification plan")
    packet = validate_digest(value.get("packet_sha256"), "qualification packet")
    validate_digest(value.get("authorization_sha256"), "qualification authorization")
    cell = validate_digest(value.get("cell_universe_sha256"), "qualification cell universe")
    if path.name != f"qa-{cell.removeprefix('sha256:')}.json":
        raise LedgerError("Fleet task-quality reservation path differs from its cell")
    return cell, plan, packet


def reservation_index(reservations: Path, *, date_utc: str) -> ReservationIndex:
    if reservations.is_symlink() or not reservations.is_dir():
        raise LedgerError("Fleet daily rollout reservation directory is invalid")
    total = 0
    covered: dict[tuple[str, str], dict[str, Any]] = {}
    waves: dict[str, dict[str, Any]] = {}
    qualifications: dict[str, dict[str, Any]] = {}
    qa_plans: set[str] = set()
    qa_packets: set[str] = set()
    entries = sorted(reservations.iterdir())
    for path in entries:
        if _ATOMIC_TEMP.fullmatch(path.name):
            if path.is_symlink() or not path.is_file():
                raise LedgerError("Fleet reservation temporary entry is invalid")
            continue
        if path.suffix != ".json":
            raise LedgerError("Fleet reservation directory contains an unknown entry")
        value = read_canonical(path, "Fleet daily rollout reservation")
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        if value.get("sha256") != digest(unsigned) or value.get("date_utc") != date_utc:
            raise LedgerError("Fleet daily rollout reservation digest/date is invalid")
        schema = value.get("schema")
        if schema == QA_RESERVATION_SCHEMA:
            cell, plan, packet = _validate_qualification(path, value)
            if cell in qualifications or plan in qa_plans or packet in qa_packets:
                raise LedgerError("Fleet task-quality reservation identity is duplicated")
            qualifications[cell] = value
            qa_plans.add(plan)
            qa_packets.add(packet)
            total += 1
            continue
        if schema == LEGACY_RESERVATION_SCHEMA:
            groups = _legacy_groups(value)
        elif schema == WAVE_RESERVATION_SCHEMA:
            groups = _wave_groups(value)
        else:
            raise LedgerError("Fleet daily rollout reservation schema is invalid")
        bindings = validate_digest(value.get("bindings_sha256"), "Fleet reservation binding")
        if schema == WAVE_RESERVATION_SCHEMA:
            if bindings in waves:
                raise LedgerError("Fleet binding has more than one wave reservation")
            waves[bindings] = value
        if type(value.get("count")) is not int or not 1 <= value["count"] <= CAP:
            raise LedgerError("Fleet daily rollout reservation count is invalid")
        group_total = 0
        for group in groups:
            if (
                not isinstance(group, dict)
                or set(group) != {"group_id", "count", "packet_sha256"}
                or not isinstance(group.get("group_id"), str)
                or _ID.fullmatch(group["group_id"]) is None
                or type(group.get("count")) is not int
                or not 1 <= group["count"] <= CAP
            ):
                raise LedgerError("Fleet daily rollout reservation group is invalid")
            validate_digest(group.get("packet_sha256"), "Fleet reservation packet")
            key = (bindings, group["group_id"])
            if key in covered:
                raise LedgerError("Fleet daily rollout reservation group is duplicated")
            covered[key] = group
            group_total += group["count"]
        if group_total != value["count"]:
            raise LedgerError("Fleet daily rollout reservation count differs from its groups")
        total += value["count"]
    return ReservationIndex(
        total=total,
        covered=covered,
        waves=waves,
        qualifications=qualifications,
        qa_plans=frozenset(qa_plans),
        qa_packets=frozenset(qa_packets),
    )
