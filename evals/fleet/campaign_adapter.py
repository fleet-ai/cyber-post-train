"""Fleet source-Job adapter for the generic resumable pass@4 controller.

One sealed CPU Job may evaluate several held-out tasks for one model and seed.
This adapter lets each statistical cell remain independently resumable while
electing exactly one cell to create that shared Job.  It never reads scores,
prompts, traces, answers, or flags.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals import campaign
from evals.fleet import evaluate, heldout_launch, outcome_validity
from evals.fleet import opencode_self_hosted as harness

SCHEMA = "cyber_fleet_campaign_bindings_v1"
WAVE_BINDING_SCHEMA = "cyber_fleet_campaign_bindings_v2"
STRICT_PROFILE_SCHEMA = "cyber_fleet_strict_wave_profile_v1"
BUDGET_SCHEMA = "cyber_fleet_daily_rollout_budget_v1"
RESERVATION_SCHEMA = "cyber_fleet_daily_rollout_reservation_v1"
WAVE_RESERVATION_SCHEMA = "cyber_fleet_daily_rollout_wave_reservation_v1"
BUDGET_ROOT = campaign.CANONICAL_REGISTRY / "fleet-daily-rollouts-v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
WAVE_GROUP_COUNT = 16
WAVE_CELL_COUNT = 160
STRICT_PROFILE_DIGEST_ENV = "CYBER_FLEET_STRICT_PROFILE_FILE_SHA256"


class AdapterError(ValueError):
    """The frozen Fleet cell mapping or its external evidence is invalid."""


class CapacityUnavailable(AdapterError):
    """The bound daily rollout allowance cannot admit this source group."""


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _control_source_sha256() -> str:
    """Bind every module that can advance or create the strict Fleet wave."""
    return _digest(
        {
            "evals/campaign.py": "sha256:"
            + hashlib.sha256(Path(campaign.__file__).read_bytes()).hexdigest(),
            "evals/fleet/campaign_adapter.py": "sha256:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evals/fleet/heldout_launch.py": "sha256:"
            + hashlib.sha256(Path(heldout_launch.__file__).read_bytes()).hexdigest(),
        }
    )


def _packet_set_sha256(groups: dict[str, Any]) -> str:
    return _digest(
        [
            {"group_id": group_id, "packet_sha256": group["packet_sha256"]}
            for group_id, group in sorted(groups.items())
        ]
    )


def _strict_profile(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    expected_file_sha256 = _validate_digest(
        expected_file_sha256, "strict-wave profile file"
    )
    if path.is_symlink() or not path.is_file():
        raise AdapterError("Fleet strict-wave profile is absent")
    try:
        payload = path.read_bytes()
        file_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError("Fleet strict-wave profile is unreadable") from exc
    if file_sha256 != expected_file_sha256:
        raise AdapterError("Fleet strict-wave profile file differs from its authorization")
    if not isinstance(value, dict):
        raise AdapterError("Fleet strict-wave profile is invalid")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if (
        set(value)
        != {
            "schema",
            "campaign_id",
            "binding_schema",
            "campaign_plan_sha256",
            "bindings_sha256",
            "reservation_id",
            "packet_set_sha256",
            "group_count",
            "cell_count",
            "control_source_sha256",
            "sha256",
        }
        or value.get("schema") != STRICT_PROFILE_SCHEMA
        or value.get("sha256") != _digest(unsigned)
    ):
        raise AdapterError("Fleet strict-wave profile is invalid")
    for field in (
        "campaign_plan_sha256",
        "bindings_sha256",
        "packet_set_sha256",
        "control_source_sha256",
    ):
        _validate_digest(value.get(field), f"strict-wave profile {field}")
    return value


def _validate_wave_contract(
    binding: dict[str, Any],
    plan: dict[str, Any],
    strict_profile_path: Path,
    strict_profile_file_sha256: str,
) -> dict[str, Any]:
    binding = _validate_bindings(binding)
    if binding["schema"] != WAVE_BINDING_SCHEMA:
        raise AdapterError("Fleet wave coordinator requires a wave-bound campaign")
    targets = plan.get("targets")
    if (
        not isinstance(targets, list)
        or any(not isinstance(target, dict) for target in targets)
        or len(targets) != WAVE_CELL_COUNT
    ):
        raise AdapterError("Fleet wave campaign plan is not the exact 160-cell plan")
    keys = [target.get("experiment_key") for target in targets]
    control_source_sha256 = _control_source_sha256()
    profile = _strict_profile(strict_profile_path, strict_profile_file_sha256)
    sources = [
        target.get("drivers", {}).get(phase, {}).get("source_sha256")
        for target in targets
        for phase in campaign.PHASES
    ]
    if (
        any(not isinstance(key, str) or _SHA.fullmatch(key) is None for key in keys)
        or len(set(keys)) != WAVE_CELL_COUNT
        or set(keys) != set(binding["cells"])
        or plan.get("plan_sha256") != binding["wave"]["campaign_plan_sha256"]
        or binding["wave"]["control_source_sha256"] != control_source_sha256
        or any(source != control_source_sha256 for source in sources)
        or profile
        != {
            "schema": STRICT_PROFILE_SCHEMA,
            "campaign_id": plan.get("campaign_id"),
            "binding_schema": WAVE_BINDING_SCHEMA,
            "campaign_plan_sha256": plan.get("plan_sha256"),
            "bindings_sha256": binding["sha256"],
            "reservation_id": binding["wave"]["reservation_id"],
            "packet_set_sha256": binding["wave"]["packet_set_sha256"],
            "group_count": WAVE_GROUP_COUNT,
            "cell_count": WAVE_CELL_COUNT,
            "control_source_sha256": control_source_sha256,
            "sha256": profile["sha256"],
        }
    ):
        raise AdapterError("Fleet wave campaign control contract differs")
    return binding


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError("required JSON evidence is absent")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError("required JSON evidence is invalid") from exc
    if not isinstance(value, dict):
        raise AdapterError("required JSON evidence is invalid")
    return value


def _mkdir_durable(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise AdapterError("state directory must not be a symlink")
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise AdapterError("state directory parent is invalid")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise AdapterError("state directory is invalid") from None
        parent = os.open(directory.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    _mkdir_durable(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _reservation_index(
    reservations: Path, *, date_utc: str
) -> tuple[int, dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    total = 0
    covered: dict[tuple[str, str], dict[str, Any]] = {}
    waves: dict[str, dict[str, Any]] = {}
    for path in reservations.glob("*.json"):
        row = _read(path)
        unsigned = {key: value for key, value in row.items() if key != "sha256"}
        if row.get("sha256") != _digest(unsigned):
            raise AdapterError("Fleet daily rollout reservation is invalid")
        schema = row.get("schema")
        if schema == RESERVATION_SCHEMA:
            if set(row) != {
                "schema",
                "date_utc",
                "group_id",
                "count",
                "packet_sha256",
                "bindings_sha256",
                "sha256",
            }:
                raise AdapterError("Fleet daily rollout reservation is invalid")
            groups = [
                {
                    "group_id": row.get("group_id"),
                    "count": row.get("count"),
                    "packet_sha256": row.get("packet_sha256"),
                }
            ]
        elif schema == WAVE_RESERVATION_SCHEMA:
            if set(row) != {
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
                raise AdapterError("Fleet daily rollout wave reservation is invalid")
            _validate_digest(row.get("packet_set_sha256"), "wave packet set")
            groups = row.get("groups")
            if (
                row.get("cap") != 500
                or not isinstance(row.get("reservation_id"), str)
                or _ID.fullmatch(row["reservation_id"]) is None
                or not isinstance(groups, list)
                or not groups
            ):
                raise AdapterError("Fleet daily rollout wave reservation is invalid")
        else:
            raise AdapterError("Fleet daily rollout reservation schema is invalid")
        bindings_sha256 = _validate_digest(row.get("bindings_sha256"), "reservation bindings")
        if schema == WAVE_RESERVATION_SCHEMA:
            if bindings_sha256 in waves:
                raise AdapterError("Fleet binding has more than one wave reservation")
            waves[bindings_sha256] = row
        if (
            type(row.get("count")) is not int
            or not 1 <= row["count"] <= 500
            or row.get("date_utc") != date_utc
        ):
            raise AdapterError("Fleet daily rollout reservation is invalid")
        group_total = 0
        for group in groups:
            if (
                not isinstance(group, dict)
                or set(group) != {"group_id", "count", "packet_sha256"}
                or not isinstance(group["group_id"], str)
                or _ID.fullmatch(group["group_id"]) is None
                or type(group["count"]) is not int
                or not 1 <= group["count"] <= 500
            ):
                raise AdapterError("Fleet daily rollout reservation group is invalid")
            _validate_digest(group["packet_sha256"], "reservation packet")
            key = (bindings_sha256, group["group_id"])
            if key in covered:
                raise AdapterError("Fleet daily rollout reservation group is duplicated")
            covered[key] = group
            group_total += group["count"]
        if group_total != row["count"]:
            raise AdapterError("Fleet daily rollout reservation count differs from its groups")
        total += row["count"]
    return total, covered, waves


def reserve_wave(
    binding: dict[str, Any],
    plan: dict[str, Any],
    strict_profile_path: Path,
    strict_profile_file_sha256: str,
    reservation_id: str,
    *,
    expected_sessions: int,
    packet_set_sha256: str,
    root: Path = BUDGET_ROOT,
) -> dict[str, Any]:
    """Atomically reserve every source group in one campaign before any create."""
    if not isinstance(reservation_id, str) or _ID.fullmatch(reservation_id) is None:
        raise AdapterError("Fleet wave reservation id is invalid")
    binding = _validate_wave_contract(
        binding, plan, strict_profile_path, strict_profile_file_sha256
    )
    packet_set_sha256 = _validate_digest(packet_set_sha256, "wave packet set")
    wave = binding["wave"]
    if (
        reservation_id != wave["reservation_id"]
        or expected_sessions != wave["expected_sessions"]
        or packet_set_sha256 != wave["packet_set_sha256"]
    ):
        raise AdapterError("Fleet wave reservation request differs from its binding")
    budget = binding["budget"]
    if budget["date_utc"] != datetime.now(UTC).date().isoformat():
        raise AdapterError("Fleet daily budget census is not from today UTC")
    groups = [
        {
            "group_id": group_id,
            "count": len(group["cells"]),
            "packet_sha256": group["packet_sha256"],
        }
        for group_id, group in sorted(binding["groups"].items())
    ]
    count = sum(group["count"] for group in groups)
    if (
        type(expected_sessions) is not int
        or count != expected_sessions
        or count != len(binding["cells"])
        or not 1 <= count <= 500
    ):
        raise AdapterError("Fleet wave reservation does not cover every campaign cell")
    day = root / budget["date_utc"]
    if root.is_symlink() or day.is_symlink():
        raise AdapterError("Fleet daily budget path must not be a symlink")
    _mkdir_durable(day)
    lock_path = day / ".lock"
    if lock_path.is_symlink():
        raise AdapterError("Fleet daily budget lock must not be a symlink")
    lock_fd = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(lock_fd, "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if budget["date_utc"] != datetime.now(UTC).date().isoformat():
            raise AdapterError("Fleet daily budget UTC date changed while waiting for its lock")
        baseline_path = day / "baseline.json"
        baseline = {"schema": BUDGET_SCHEMA, **budget}
        baseline = {**baseline, "sha256": _digest(baseline)}
        if baseline_path.exists():
            if _read(baseline_path) != baseline:
                raise AdapterError("Fleet daily budget baseline differs")
        else:
            _write_once(baseline_path, baseline)
        reservations = day / "reservations"
        _mkdir_durable(reservations)
        key = hashlib.sha256(
            f"{reservation_id}:wave".encode()
        ).hexdigest()
        path = reservations / f"wave-{key}.json"
        reservation = {
            "schema": WAVE_RESERVATION_SCHEMA,
            "date_utc": budget["date_utc"],
            "cap": budget["cap"],
            "count": count,
            "reservation_id": reservation_id,
            "packet_set_sha256": packet_set_sha256,
            "groups": groups,
            "bindings_sha256": binding["sha256"],
        }
        reservation = {**reservation, "sha256": _digest(reservation)}
        existing = _read(path) if path.exists() else None
        if existing is not None and existing != reservation:
            raise AdapterError("Fleet daily rollout wave reservation differs")
        total_reserved, covered, waves = _reservation_index(
            reservations, date_utc=budget["date_utc"]
        )
        if budget["used"] + total_reserved > budget["cap"]:
            raise AdapterError("Fleet daily rollout ledger exceeds its bound cap")
        expected_keys = {(binding["sha256"], group["group_id"]) for group in groups}
        already_covered = expected_keys & covered.keys()
        if existing is None and already_covered:
            raise AdapterError("Fleet wave overlaps an earlier partial reservation")
        if existing is None:
            if budget["used"] + total_reserved + count > budget["cap"]:
                raise CapacityUnavailable("Fleet daily 500-rollout budget is exhausted")
            _write_once(path, reservation)
            total_reserved += count
        elif already_covered != expected_keys:
            raise AdapterError("Fleet wave reservation coverage is incomplete")
        persisted_wave = waves.get(binding["sha256"])
        if existing is not None and persisted_wave != reservation:
            raise AdapterError("Fleet wave reservation differs after durable readback")
        result = {
            "schema": "cyber_fleet_daily_rollout_wave_reservation_receipt_v1",
            "date_utc": budget["date_utc"],
            "baseline_sha256": baseline["sha256"],
            "bindings_sha256": binding["sha256"],
            "reservation_id": reservation_id,
            "packet_set_sha256": packet_set_sha256,
            "used": budget["used"],
            "reserved": total_reserved,
            "committed_after": budget["used"] + total_reserved,
            "cap": budget["cap"],
            "wave_count": count,
            "wave_sha256": reservation["sha256"],
            "wave_path": str(path),
            "replayed": existing is not None,
        }
        return {**result, "sha256": _digest(result)}


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise AdapterError(f"{label} is not a SHA-256 digest")
    return value


def _validate_bindings(value: dict[str, Any]) -> dict[str, Any]:
    schema = value.get("schema")
    expected_keys = {"schema", "budget", "groups", "cells", "sha256"}
    if schema == WAVE_BINDING_SCHEMA:
        expected_keys.add("wave")
    if set(value) != expected_keys:
        raise AdapterError("Fleet campaign bindings have unknown or missing fields")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if schema not in {SCHEMA, WAVE_BINDING_SCHEMA} or value["sha256"] != _digest(unsigned):
        raise AdapterError("Fleet campaign bindings identity changed")
    budget = value["budget"]
    if not isinstance(budget, dict) or set(budget) != {
        "date_utc",
        "cap",
        "used",
        "census_receipt_sha256",
    }:
        raise AdapterError("Fleet daily budget binding is invalid")
    try:
        datetime.strptime(budget["date_utc"], "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise AdapterError("Fleet daily budget date is invalid") from exc
    if budget["cap"] != 500 or type(budget["used"]) is not int or not 0 <= budget["used"] <= 500:
        raise AdapterError("Fleet daily budget must bind the 500-rollout limit")
    _validate_digest(budget["census_receipt_sha256"], "budget census receipt")
    if schema == WAVE_BINDING_SCHEMA:
        wave = value["wave"]
        if (
            not isinstance(wave, dict)
            or set(wave)
            != {
                "reservation_id",
                "expected_sessions",
                "packet_set_sha256",
                "campaign_plan_sha256",
                "control_source_sha256",
            }
            or not isinstance(wave["reservation_id"], str)
            or _ID.fullmatch(wave["reservation_id"]) is None
            or type(wave["expected_sessions"]) is not int
            or not 1 <= wave["expected_sessions"] <= 500
        ):
            raise AdapterError("Fleet campaign wave binding is invalid")
        _validate_digest(wave["packet_set_sha256"], "wave packet set")
        _validate_digest(wave["campaign_plan_sha256"], "wave campaign plan")
        _validate_digest(wave["control_source_sha256"], "wave control source")
    groups, cells = value["groups"], value["cells"]
    if not isinstance(groups, dict) or not groups or not isinstance(cells, dict) or not cells:
        raise AdapterError("Fleet campaign source groups and cells are required")
    assigned: set[str] = set()
    for group_id, group in groups.items():
        if (
            not isinstance(group_id, str)
            or _ID.fullmatch(group_id) is None
            or not isinstance(group, dict)
        ):
            raise AdapterError("Fleet source group is invalid")
        if set(group) != {"packet", "packet_sha256", "leader", "cells"}:
            raise AdapterError("Fleet source group shape is invalid")
        _validate_digest(group["packet_sha256"], "source packet")
        members = group["cells"]
        if (
            not isinstance(group["packet"], str)
            or not isinstance(members, list)
            or not members
            or any(
                not isinstance(member, str) or _SHA.fullmatch(member) is None
                for member in members
            )
            or len(members) != len(set(members))
            or group["leader"] not in members
            or any(member in assigned for member in members)
        ):
            raise AdapterError("Fleet source group membership is invalid")
        assigned.update(members)
    if assigned != set(cells):
        raise AdapterError("Fleet source groups do not partition campaign cells")
    if schema == WAVE_BINDING_SCHEMA and (
        len(groups) != WAVE_GROUP_COUNT
        or len(cells) != WAVE_CELL_COUNT
        or value["wave"]["expected_sessions"] != WAVE_CELL_COUNT
        or value["wave"]["packet_set_sha256"] != _packet_set_sha256(groups)
    ):
        raise AdapterError("Fleet campaign wave is not the exact 16-group/160-cell packet set")
    required = {
        "group",
        "attempt",
        "task_version_id",
        "model_id",
        "model_revision",
        "source_attempt",
        "campaign_model_id",
    }
    for key, cell in cells.items():
        if (
            not isinstance(key, str)
            or _SHA.fullmatch(key) is None
            or not isinstance(cell, dict)
            or set(cell) != required
            or not isinstance(cell["group"], str)
            or cell["group"] not in groups
            or key not in groups[cell["group"]]["cells"]
            or type(cell["attempt"]) is not int
            or not 1 <= cell["attempt"] <= 4
            or type(cell["source_attempt"]) is not int
            or cell["source_attempt"] < 1
            or any(
                not isinstance(cell[field], str) or not cell[field]
                for field in required - {"group", "attempt", "source_attempt"}
            )
        ):
            raise AdapterError("Fleet campaign cell binding is invalid")
    return value


def load_bindings(path: Path) -> dict[str, Any]:
    return _validate_bindings(_read(path))


def _source(
    bindings_path: Path, target: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], heldout_launch.Package, Path]:
    bindings = load_bindings(bindings_path)
    key = target.get("experiment_key")
    if key not in bindings["cells"]:
        raise AdapterError("campaign cell has no Fleet source binding")
    cell = bindings["cells"][key]
    group = bindings["groups"][cell["group"]]
    packet = (bindings_path.resolve().parent / group["packet"]).resolve()
    if bindings_path.resolve().parent not in packet.parents:
        raise AdapterError("sealed Fleet source packet escapes its binding root")
    package = heldout_launch.build_package(packet)
    if package.packet.packet_sha256 != group["packet_sha256"]:
        raise AdapterError("sealed Fleet source packet differs from its binding")
    sealed = heldout_launch.sealed_evaluation(package)
    rows = {
        (row["task_version_id"], row["model_id"], row["model_revision"], row["attempt"])
        for row in sealed.rows
    }
    expected = {
        (
            bindings["cells"][member]["task_version_id"],
            bindings["cells"][member]["model_id"],
            bindings["cells"][member]["model_revision"],
            bindings["cells"][member]["source_attempt"],
        )
        for member in group["cells"]
    }
    identity = target.get("identity", {})
    if (
        rows != expected
        or len(rows) != len(group["cells"])
        or package.packet.identity["pass_k"] != 1
        or package.packet.identity["retry_limit"] != 0
        or identity.get("attempt") != cell["attempt"]
        or identity.get("seed") != package.packet.identity["sampling_seed"]
        or identity.get("model", {}).get("id") != cell["campaign_model_id"]
    ):
        raise AdapterError("campaign cell and sealed Fleet source Job differ")
    return bindings, cell, package, packet


def _group_root(packet: Path, group_id: str) -> Path:
    # packet.json is <campaign>/targets/<experiment-key>/packet.json.
    state = packet.resolve().parents[2]
    if not (state / "plan.json").is_file():
        raise AdapterError("campaign state root is invalid")
    return state / "fleet-source-jobs" / group_id


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": _digest(value)}


def _receipt(
    target: dict[str, Any], phase: str, action: str, status: str, **extra: Any
) -> dict[str, Any]:
    return _signed(
        {
            "schema": campaign.RECEIPT_SCHEMA,
            "experiment_key": target["experiment_key"],
            "phase": phase,
            "action": action,
            "provider": "fleet",
            "status": status,
            **extra,
        }
    )


def _route_ready(package: heldout_launch.Package) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise AdapterError("Fleet route check requires FLEET_API_KEY")
    route = next(iter(package.evaluation_config["routes"].values()))
    model = package.evaluation_config["models"][route["model"]]
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        account = harness._request(client, "GET", "/v1/account")  # noqa: SLF001
        if account.get("team_name") != "fleet" or account.get("team_id") != harness.FLEET_TEAM_ID:
            raise AdapterError("Fleet team identity is required")
        return evaluate.check_route(route, model, client)


class _AbsentDatabase:
    def __init__(self, database: str) -> None:
        self.database = database

    def exists(self, database: str) -> bool:
        if database != self.database:
            raise AdapterError("cluster-side database-absence evidence name differs")
        return False


def _cluster_duplicate_absence(
    path: Path,
    *,
    bindings: dict[str, Any],
    group_id: str,
    package: heldout_launch.Package,
) -> tuple[Callable[[str], bool], _AbsentDatabase]:
    evidence = _read(path)
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    group = evidence.get("groups", {}).get(group_id, {})
    packet = package.packet
    if (
        evidence.get("schema") != "cyber_fleet_cluster_duplicate_gate_binding_v1"
        or evidence.get("sha256") != _digest(unsigned)
        or evidence.get("date_utc") != datetime.now(UTC).date().isoformat()
        or evidence.get("bindings_sha256") != bindings["sha256"]
        or evidence.get("source_gate_receipt_sha256")
        != bindings["budget"]["census_receipt_sha256"]
        or not isinstance(group, dict)
        or group.get("packet_sha256") != bindings["groups"][group_id]["packet_sha256"]
        or group.get("job_name") != packet.job_name
        or group.get("config_map_name") != packet.config_map_name
        or group.get("output_root") != packet.output_root
        or group.get("database") != packet.database
        or group.get("evaluation_identity_sha256") != packet.identity_sha256
        or group.get("sfs_output_absent") is not True
        or group.get("database_absent") is not True
    ):
        raise AdapterError("cluster-side duplicate-gate evidence differs from this source Job")

    def output_exists(output_root: str) -> bool:
        if output_root != packet.output_root:
            raise AdapterError("cluster-side output-absence evidence path differs")
        return False

    return output_exists, _AbsentDatabase(packet.database)


def _budget(
    binding: dict[str, Any],
    group_id: str,
    count: int,
    packet_sha256: str,
    *,
    root: Path = BUDGET_ROOT,
    reserve: bool,
) -> dict[str, Any]:
    budget = binding["budget"]
    if type(count) is not int or not 1 <= count <= 500:
        raise AdapterError("Fleet rollout reservation count is invalid")
    if budget["date_utc"] != datetime.now(UTC).date().isoformat():
        raise AdapterError("Fleet daily budget census is not from today UTC")
    day = root / budget["date_utc"]
    if root.is_symlink() or day.is_symlink():
        raise AdapterError("Fleet daily budget path must not be a symlink")
    _mkdir_durable(day)
    lock_path = day / ".lock"
    if lock_path.is_symlink():
        raise AdapterError("Fleet daily budget lock must not be a symlink")
    lock_fd = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(lock_fd, "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if budget["date_utc"] != datetime.now(UTC).date().isoformat():
            raise AdapterError("Fleet daily budget UTC date changed while waiting for its lock")
        baseline_path = day / "baseline.json"
        baseline = {"schema": BUDGET_SCHEMA, **budget}
        baseline = {**baseline, "sha256": _digest(baseline)}
        if baseline_path.exists():
            if _read(baseline_path) != baseline:
                raise AdapterError("Fleet daily budget baseline differs")
        elif reserve:
            _write_once(baseline_path, baseline)
        reservations = day / "reservations"
        _mkdir_durable(reservations)
        reservation_key = hashlib.sha256(
            f"{binding['sha256']}:{group_id}".encode()
        ).hexdigest()
        path = reservations / f"{reservation_key}.json"
        reservation = {
            "schema": RESERVATION_SCHEMA,
            "date_utc": budget["date_utc"],
            "group_id": group_id,
            "count": count,
            "packet_sha256": packet_sha256,
            "bindings_sha256": binding["sha256"],
        }
        reservation = {**reservation, "sha256": _digest(reservation)}
        existing = _read(path) if path.exists() else None
        if existing is not None and existing != reservation:
            raise AdapterError("Fleet daily rollout reservation differs")
        total_reserved, covered, waves = _reservation_index(
            reservations, date_utc=budget["date_utc"]
        )
        if budget["used"] + total_reserved > budget["cap"]:
            raise AdapterError("Fleet daily rollout ledger exceeds its bound cap")
        coverage = covered.get((binding["sha256"], group_id))
        if coverage is not None and coverage != {
            "group_id": group_id,
            "count": count,
            "packet_sha256": packet_sha256,
        }:
            raise AdapterError("Fleet daily rollout reservation differs")
        if binding["schema"] == WAVE_BINDING_SCHEMA:
            wave = waves.get(binding["sha256"])
            expected_wave = binding["wave"]
            expected_groups = [
                {
                    "group_id": expected_group_id,
                    "count": len(expected_group["cells"]),
                    "packet_sha256": expected_group["packet_sha256"],
                }
                for expected_group_id, expected_group in sorted(binding["groups"].items())
            ]
            if (
                wave is None
                or wave.get("reservation_id") != expected_wave["reservation_id"]
                or wave.get("packet_set_sha256") != expected_wave["packet_set_sha256"]
                or wave.get("count") != expected_wave["expected_sessions"]
                or wave.get("groups") != expected_groups
                or coverage is None
            ):
                raise CapacityUnavailable(
                    "Fleet whole-wave reservation is absent; no source Job may be created"
                )
            return {"used": budget["used"], "reserved": total_reserved, "cap": budget["cap"]}
        if existing is None and coverage is None:
            if budget["used"] + total_reserved + count > budget["cap"]:
                raise CapacityUnavailable("Fleet daily 500-rollout budget is exhausted")
            if reserve:
                _write_once(path, reservation)
                total_reserved += count
        return {"used": budget["used"], "reserved": total_reserved, "cap": budget["cap"]}


def _launch_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read(path)
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != "cyber_fleet_source_launch_v1" or value.get(
        "sha256"
    ) != _digest(unsigned):
        raise AdapterError("shared Fleet source launch record is invalid")
    return value


def run_action(
    *,
    action: str,
    phase: str,
    packet_path: Path,
    bindings_path: Path,
    strict_profile_path: Path,
    strict_profile_file_sha256: str,
    context: str,
    preview_receipt: Path | None = None,
    readiness_receipt: Path | None = None,
    launch_receipt: Path | None = None,
    terminal_receipt: Path | None = None,
    duplicate_gate_evidence: Path | None = None,
    cluster: heldout_launch.Cluster | None = None,
    database: heldout_launch.Database | None = None,
    route_check: Callable[[heldout_launch.Package], dict[str, Any]] = _route_ready,
    budget_root: Path = BUDGET_ROOT,
) -> dict[str, Any]:
    target = _read(packet_path)
    state = packet_path.resolve().parents[2]
    plan = campaign.load_plan(state)
    prebound = load_bindings(bindings_path)
    _validate_wave_contract(
        prebound, plan, strict_profile_path, strict_profile_file_sha256
    )
    expected_packet = state / "targets" / target["experiment_key"] / "packet.json"
    planned_target = next(
        (
            item
            for item in plan["targets"]
            if item["experiment_key"] == target["experiment_key"]
        ),
        None,
    )
    driver = target.get("drivers", {}).get(phase, {})
    source_sha256 = _control_source_sha256()
    if (
        packet_path.resolve() != expected_packet
        or planned_target != target
        or bindings_path.resolve() != state.parent / "fleet-bindings.json"
        or strict_profile_path.resolve() != state.parent / "strict-wave-profile.json"
        or duplicate_gate_evidence is None
        or duplicate_gate_evidence.resolve() != state.parent / "duplicate-gate-evidence.json"
        or driver.get("source_sha256") != source_sha256
    ):
        raise AdapterError("Fleet campaign control plane differs from its strict profile")
    bindings, cell, package, source_packet = _source(bindings_path, target)
    if bindings != prebound:
        raise AdapterError("Fleet strict-wave binding changed during package capture")
    group_id = cell["group"]
    group = bindings["groups"][group_id]
    root = _group_root(packet_path, group_id)
    cluster = cluster or heldout_launch.KubectlCluster(context)
    database = database or heldout_launch.PostgresDatabase()
    duplicate_absence = (
        _cluster_duplicate_absence(
            duplicate_gate_evidence,
            bindings=bindings,
            group_id=group_id,
            package=package,
        )
        if duplicate_gate_evidence is not None
        else None
    )
    output_exists = duplicate_absence[0] if duplicate_absence else None
    duplicate_database = duplicate_absence[1] if duplicate_absence else database

    if phase == "score":
        collection_path = terminal_receipt or (
            packet_path.resolve().parents[2]
            / "targets"
            / target["experiment_key"]
            / "rollout"
            / "observe.json"
        )
        collection = _read(collection_path)
        if not collection or collection.get("status") != "accepted":
            raise AdapterError("Fleet score phase requires an accepted rollout receipt")
        common = {"collection_terminal_receipt_sha256": collection["receipt_sha256"]}
        if action == "preview":
            return _receipt(target, phase, action, "accepted", **common)
        if action == "ready":
            return _receipt(
                target,
                phase,
                action,
                "ready",
                preview_receipt_sha256=_read(preview_receipt)["receipt_sha256"],
                defer_reason_code=None,
                **common,
            )
        if action == "launch":
            ready = _read(readiness_receipt)
            return _receipt(
                target,
                phase,
                action,
                "created",
                preview_receipt_sha256=_read(preview_receipt)["receipt_sha256"],
                readiness_receipt_path=str(readiness_receipt),
                readiness_receipt_sha256=ready["receipt_sha256"],
                remote_id="score:" + target["experiment_key"],
                **common,
            )
        launch = _read(launch_receipt)
        return _receipt(
            target,
            phase,
            action,
            "accepted",
            launch_receipt_sha256=launch["receipt_sha256"],
            remote_id=launch["remote_id"],
            terminal_evidence_sha256=collection["terminal_evidence_sha256"],
            **common,
        )

    shared = _launch_record(root / "launch.json")
    if action == "preview":
        server_preview = heldout_launch.preview_package(package, cluster=cluster)
        return _receipt(
            target,
            phase,
            action,
            "accepted",
            rendered_objects=[
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {"annotations": {heldout_launch.FAILURE_ALERT_ANNOTATION: "off"}},
                }
            ],
            server_preview_sha256=server_preview,
            source_packet_sha256=group["packet_sha256"],
        )
    if action == "ready":
        preview = _read(preview_receipt)
        try:
            if shared is not None:
                route = {"profile_sha256": shared["route_profile_sha256"]}
            else:
                try:
                    route = route_check(package)
                except (AdapterError, RuntimeError):
                    return _receipt(
                        target,
                        phase,
                        action,
                        "deferred_not_ready",
                        preview_receipt_sha256=preview["receipt_sha256"],
                        defer_reason_code="dependency_not_ready",
                    )
                if target["experiment_key"] != group["leader"]:
                    raise LookupError
                _budget(
                    bindings,
                    group_id,
                    len(group["cells"]),
                    group["packet_sha256"],
                    root=budget_root,
                    reserve=False,
                )
                census_kwargs = {"output_exists": output_exists} if output_exists else {}
                heldout_launch.duplicate_census(
                    package, cluster=cluster, database=duplicate_database, **census_kwargs
                )
        except LookupError:
            return _receipt(
                target,
                phase,
                action,
                "deferred_not_ready",
                preview_receipt_sha256=preview["receipt_sha256"],
                defer_reason_code="dependency_not_ready",
            )
        except CapacityUnavailable:
            return _receipt(
                target,
                phase,
                action,
                "deferred_not_ready",
                preview_receipt_sha256=preview["receipt_sha256"],
                defer_reason_code="capacity_unavailable",
            )
        return _receipt(
            target,
            phase,
            action,
            "ready",
            preview_receipt_sha256=preview["receipt_sha256"],
            defer_reason_code=None,
            route_profile_sha256=route["profile_sha256"],
        )
    if action == "launch":
        preview, ready = _read(preview_receipt), _read(readiness_receipt)
        if shared is None:
            if target["experiment_key"] != group["leader"]:
                raise AdapterError("non-leader Fleet cell cannot create its source Job")
            _budget(
                bindings,
                group_id,
                len(group["cells"]),
                group["packet_sha256"],
                root=budget_root,
                reserve=True,
            )
            if root.is_symlink():
                raise AdapterError("Fleet source group path must not be a symlink")
            _mkdir_durable(root)
            launch_kwargs = {"output_exists": output_exists} if output_exists else {}
            result = heldout_launch.launch_package_once(
                package,
                cluster=cluster,
                database=duplicate_database,
                journal=root / "create-intent.jsonl",
                **launch_kwargs,
            )
            unsigned = {
                "schema": "cyber_fleet_source_launch_v1",
                "group_id": group_id,
                "packet_sha256": group["packet_sha256"],
                "route_profile_sha256": ready["route_profile_sha256"],
                **result,
            }
            shared = {**unsigned, "sha256": _digest(unsigned)}
            _write_once(root / "launch.json", shared)
        return _receipt(
            target,
            phase,
            action,
            "created",
            preview_receipt_sha256=preview["receipt_sha256"],
            readiness_receipt_path=str(readiness_receipt),
            readiness_receipt_sha256=ready["receipt_sha256"],
            remote_id=shared["job_uid"],
            source_launch_sha256=shared["sha256"],
        )
    if action != "observe" or shared is None:
        raise AdapterError("Fleet source Job has not been launched")
    launch = _read(launch_receipt)
    job = cluster.get("jobs.batch", package.packet.namespace, package.packet.job_name)
    config_map = cluster.get(
        "configmaps", package.packet.namespace, package.packet.config_map_name
    )
    if (
        job.get("metadata", {}).get("uid") != shared["job_uid"]
        or config_map.get("metadata", {}).get("uid") != shared["config_map_uid"]
    ):
        raise AdapterError("Fleet source Job or ConfigMap UID changed")
    conditions = {
        row.get("type")
        for row in job.get("status", {}).get("conditions", [])
        if isinstance(row, dict) and row.get("status") == "True"
    }
    terminal = conditions & {"Complete", "Failed"}
    if not terminal:
        return _receipt(
            target,
            phase,
            action,
            "running",
            launch_receipt_sha256=launch["receipt_sha256"],
            remote_id=shared["job_uid"],
        )
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        source_terminal = _read(terminal_path)
    else:
        source_terminal = heldout_launch.collect_terminal(
            source_packet,
            cluster=cluster,
            database=database,
            receipt_path=terminal_path,
        )
    terminal_unsigned = {key: value for key, value in source_terminal.items() if key != "sha256"}
    if (
        source_terminal.get("schema") != heldout_launch.TERMINAL_SCHEMA
        or source_terminal.get("sha256") != _digest(terminal_unsigned)
        or source_terminal.get("job", {}).get("uid") != shared["job_uid"]
        or source_terminal.get("config_map", {}).get("uid") != shared["config_map_uid"]
        or source_terminal.get("evaluation_identity_sha256")
        != shared["evaluation_identity_sha256"]
    ):
        raise AdapterError("shared Fleet terminal evidence is invalid")
    final_job = cluster.get("jobs.batch", package.packet.namespace, package.packet.job_name)
    final_config_map = cluster.get(
        "configmaps", package.packet.namespace, package.packet.config_map_name
    )
    try:
        final_pods = heldout_launch._owned_pods_for_job(  # noqa: SLF001
            cluster, package.packet, shared["job_uid"]
        )
    except heldout_launch.HeldoutLaunchError as exc:
        raise AdapterError("Fleet terminal resource identity changed during collection") from exc
    final_pod_rows = heldout_launch._terminal_resource_rows(  # noqa: SLF001
        final_pods, kind="Pod"
    )
    if (
        final_job.get("metadata", {}).get("uid") != shared["job_uid"]
        or final_config_map.get("metadata", {}).get("uid") != shared["config_map_uid"]
        or source_terminal.get("pods") != final_pod_rows
    ):
        raise AdapterError("Fleet terminal resource identity changed during collection")
    try:
        row = database.cell_status(
            package.packet.database,
            task_version_id=cell["task_version_id"],
            model_id=cell["model_id"],
            model_revision=cell["model_revision"],
            attempt=cell["source_attempt"],
        )
    except heldout_launch.HeldoutLaunchError:
        # The source Job is terminal and its collector already completed all
        # score-blind reads. A missing exact row is terminal infrastructure
        # evidence for this cell, not a reason to hold its siblings forever.
        row = {"cell_id": None, "state": "absent", "receipt_digest": None}
    normally_completed = True
    try:
        outcome_validity.require_normal_completion(row)
    except outcome_validity.OutcomeValidityError:
        normally_completed = False
    accepted = (
        row.get("state") == "accepted"
        and row.get("result_class") == "valid"
        and row.get("local_results") == 1
        and row.get("retry_count") == 0
        and row.get("max_retries") == 0
        and normally_completed
        and isinstance(row.get("receipt_digest"), str)
        and _SHA.fullmatch(row["receipt_digest"]) is not None
    )
    evidence = {
        "source_terminal_sha256": source_terminal["sha256"],
        "cell_id": row.get("cell_id"),
        "cell_state": row.get("state"),
        "cell_receipt_sha256": row.get("receipt_digest"),
        "normal_completion": normally_completed,
        "cleanup_completed": accepted,
    }
    return _receipt(
        target,
        phase,
        action,
        "accepted" if accepted else "infrastructure_invalid",
        launch_receipt_sha256=launch["receipt_sha256"],
        remote_id=shared["job_uid"],
        terminal_evidence_sha256=_digest(evidence),
        evidence=evidence,
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["reserve-wave"]:
        reserve_parser = argparse.ArgumentParser()
        reserve_parser.add_argument("command", choices=["reserve-wave"])
        reserve_parser.add_argument("bindings", type=Path)
        reserve_parser.add_argument("campaign_state", type=Path)
        reserve_parser.add_argument("strict_profile", type=Path)
        reserve_parser.add_argument("receipt", type=Path)
        reserve_parser.add_argument("--reservation-id", required=True)
        reserve_parser.add_argument("--expected-sessions", required=True, type=int)
        reserve_parser.add_argument("--packet-set-sha256", required=True)
        reserve_parser.add_argument("--strict-profile-file-sha256", required=True)
        args = reserve_parser.parse_args(argv)
        result = reserve_wave(
            load_bindings(args.bindings),
            campaign.load_plan(args.campaign_state),
            args.strict_profile,
            args.strict_profile_file_sha256,
            args.reservation_id,
            expected_sessions=args.expected_sessions,
            packet_set_sha256=args.packet_set_sha256,
        )
        _write_once(args.receipt, result)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=campaign.ACTIONS)
    parser.add_argument("phase", choices=campaign.PHASES)
    parser.add_argument("packet", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--strict-profile", required=True, type=Path)
    parser.add_argument("--strict-profile-file-sha256")
    parser.add_argument("--context", required=True)
    parser.add_argument("--preview-receipt", type=Path)
    parser.add_argument("--readiness-receipt", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--duplicate-gate-evidence", type=Path)
    args = parser.parse_args(argv)
    strict_profile_file_sha256 = args.strict_profile_file_sha256 or os.environ.get(
        STRICT_PROFILE_DIGEST_ENV
    )
    if strict_profile_file_sha256 is None:
        parser.error(
            f"--strict-profile-file-sha256 or {STRICT_PROFILE_DIGEST_ENV} is required"
        )
    result = run_action(
        action=args.action,
        phase=args.phase,
        packet_path=args.packet,
        bindings_path=args.bindings,
        strict_profile_path=args.strict_profile,
        strict_profile_file_sha256=strict_profile_file_sha256,
        context=args.context,
        preview_receipt=args.preview_receipt,
        readiness_receipt=args.readiness_receipt,
        launch_receipt=args.launch_receipt,
        terminal_receipt=args.terminal_receipt,
        duplicate_gate_evidence=args.duplicate_gate_evidence,
    )
    _write_once(args.receipt, result)


if __name__ == "__main__":
    main()
