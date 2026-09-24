"""Reserve one exact zero-model task-quality cell under the shared Fleet cap."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import campaign
from evals.fleet import daily_rollout_ledger as ledger

RECEIPT_SCHEMA = "cyber_fleet_daily_task_quality_qualification_reservation_receipt_v1"


class ReservationError(RuntimeError):
    """The exact qualification cell cannot safely consume daily capacity."""


def _job_module():
    from evals.fleet import task_quality_qualification_job

    return task_quality_qualification_job


def _inputs(packet_path: Path, authorization_path: Path) -> tuple[Any, dict[str, Any], dict]:
    job = _job_module()
    package = job.build_package(packet_path)
    authorization = job.load_authorization(authorization_path, package)
    plan = job.read_json(package.packet.files["plan"], "qualification plan")
    tasks = plan.get("tasks")
    selection = plan.get("selection")
    execution = plan.get("execution")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], dict)
        or not isinstance(selection, dict)
        or type(selection.get("requested_limit")) is not int
        or selection.get("requested_limit") != 1
        or type(selection.get("selected_task_versions")) is not int
        or selection.get("selected_task_versions") != 1
        or selection.get("exact_task_identity")
        != {
            "task_key": tasks[0].get("task_key"),
            "task_version_id": tasks[0].get("task_version_id"),
        }
        or not isinstance(execution, dict)
        or type(execution.get("concurrency")) is not int
        or execution.get("concurrency") != 1
        or type(execution.get("model_calls")) is not int
        or execution.get("model_calls") != 0
        or execution.get("external_mutations_authorized") is not True
        or type(execution.get("mutating_request_attempts")) is not int
        or execution.get("mutating_request_attempts") != 1
        or execution.get("automatic_retry") is not False
        or execution.get("ambiguous_mutation_replay") is not False
        or type(package.packet.value.get("execution", {}).get("create_attempts_maximum")) is not int
        or package.packet.value.get("execution", {}).get("create_attempts_maximum") != 1
        or package.packet.value.get("execution", {}).get("automatic_create_retry") is not False
        or package.packet.value.get("execution", {}).get("root_annotation")
        != {job.FAILURE_ALERT_ANNOTATION: "off"}
    ):
        raise ReservationError("qualification inputs are not one exact zero-model cell")
    return package, authorization, plan


def _cell_digest(plan: dict[str, Any]) -> str:
    exact = plan["selection"]["exact_task_identity"]
    return campaign.digest(
        [
            {
                "task_key": exact["task_key"],
                "task_version_id": exact["task_version_id"],
            }
        ]
    )


def _candidate(
    package: Any, authorization: dict[str, Any], plan: dict[str, Any], date_utc: str
) -> dict[str, Any]:
    body = {
        "schema": ledger.QA_RESERVATION_SCHEMA,
        "date_utc": date_utc,
        "cap": ledger.CAP,
        "count": 1,
        "plan_sha256": plan["sha256"],
        "packet_sha256": package.packet.value["sha256"],
        "authorization_sha256": authorization["sha256"],
        "cell_universe_sha256": _cell_digest(plan),
    }
    return {**body, "sha256": ledger.digest(body)}


def _reservation_path(root: Path, date_utc: str, cell_sha256: str) -> Path:
    return root / date_utc / "reservations" / f"qa-{cell_sha256.removeprefix('sha256:')}.json"


def _publish_once(path: Path, value: dict[str, Any], *, label: str) -> bool:
    """Publish without replacement; return True for an exact prior install."""
    payload = campaign.canonical(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if ledger.read_canonical(path, f"existing {label}") != value:
                raise ReservationError(f"existing {label} differs") from None
            return True
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return False
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _locked_day(root: Path, date_utc: str, *, exclusive: bool):
    day = root / date_utc
    lock_path = day / ".lock"
    if root.is_symlink() or day.is_symlink() or lock_path.is_symlink():
        raise ReservationError("canonical daily ledger path is unsafe")
    try:
        descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except (OSError, RuntimeError) as exc:
        raise ReservationError("canonical daily ledger lock is unavailable") from exc
    lock = os.fdopen(descriptor, "r+")
    fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return lock, day


def _receipt(
    *,
    baseline: dict[str, Any],
    reservation: dict[str, Any],
    path: Path,
    total: int,
) -> dict[str, Any]:
    body = {
        "schema": RECEIPT_SCHEMA,
        "date_utc": reservation["date_utc"],
        "baseline_sha256": baseline["sha256"],
        "reservation_sha256": reservation["sha256"],
        "reservation_path": str(path.resolve()),
        "plan_sha256": reservation["plan_sha256"],
        "packet_sha256": reservation["packet_sha256"],
        "authorization_sha256": reservation["authorization_sha256"],
        "cell_universe_sha256": reservation["cell_universe_sha256"],
        "count": 1,
        "used": baseline["used"],
        "reserved": total,
        "committed": baseline["used"] + total,
        "remaining": ledger.CAP - baseline["used"] - total,
        "cap": ledger.CAP,
        "rollback_policy": "durable_reservation_is_never_deleted_replaced_or_reused",
    }
    return {**body, "sha256": ledger.digest(body)}


def _write_receipt_once(path: Path, value: dict[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReservationError("qualification reservation receipt directory is invalid")
    _publish_once(path, value, label="qualification reservation receipt")


def _receipt_destination(path: Path, *, root: Path, must_be_absent: bool) -> tuple[Path, bool]:
    try:
        lexical_parent = Path(os.path.abspath(path.parent))
        resolved_parent = path.parent.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ReservationError("qualification reservation receipt directory is invalid") from exc
    destination = resolved_parent / path.name
    if (
        path.parent.is_symlink()
        or resolved_parent != lexical_parent
        or not resolved_parent.is_dir()
        or destination == resolved_root
        or resolved_root in destination.parents
        or destination.is_symlink()
    ):
        raise ReservationError("qualification reservation receipt destination is unsafe")
    exists = destination.exists()
    if must_be_absent and exists:
        raise ReservationError("qualification reservation receipt must be absent before reserve")
    if exists and not destination.is_file():
        raise ReservationError("qualification reservation receipt must be a regular file")
    return destination, exists


def reserve_qualification_cell(
    packet_path: Path,
    authorization_path: Path,
    *,
    receipt_path: Path | None = None,
    root: Path = ledger.ROOT,
    reserve: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Preview or durably charge exactly one authorized qualification cell."""
    if reserve and receipt_path is None:
        raise ReservationError("durable reservation requires a receipt path")
    package, authorization, plan = _inputs(packet_path, authorization_path)
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    date_utc = instant.date().isoformat()
    candidate = _candidate(package, authorization, plan, date_utc)
    path = _reservation_path(root, date_utc, candidate["cell_universe_sha256"])
    reuse_receipt = False
    lock, day = _locked_day(root, date_utc, exclusive=reserve)
    with lock:
        if datetime.now(UTC).date().isoformat() != date_utc and now is None:
            raise ReservationError("UTC date changed while waiting for daily ledger lock")
        try:
            baseline = ledger.validate_baseline(day / "baseline.json", date_utc)
            index = ledger.reservation_index(day / "reservations", date_utc=date_utc)
        except ledger.LedgerError as exc:
            raise ReservationError(str(exc)) from exc
        existing = index.qualifications.get(candidate["cell_universe_sha256"])
        if existing is not None and existing != candidate:
            raise ReservationError("task-quality cell already has a different reservation")
        if existing is None and (
            candidate["plan_sha256"] in index.qa_plans
            or candidate["packet_sha256"] in index.qa_packets
        ):
            raise ReservationError("qualification plan or packet is already reserved elsewhere")
        if reserve:
            assert receipt_path is not None
            receipt_path, receipt_exists = _receipt_destination(
                receipt_path,
                root=root,
                must_be_absent=existing is None,
            )
            reuse_receipt = existing is not None and receipt_exists
        committed = baseline["used"] + index.total
        if committed > ledger.CAP or (existing is None and committed + 1 > ledger.CAP):
            raise ReservationError("daily Fleet session cap cannot admit the qualification cell")
        if reserve and existing is None:
            if now is None and datetime.now(UTC).date().isoformat() != date_utc:
                raise ReservationError("UTC date changed before reservation publication")
            _publish_once(path, candidate, label="task-quality reservation")
            try:
                index = ledger.reservation_index(day / "reservations", date_utc=date_utc)
            except ledger.LedgerError as exc:
                raise ReservationError(str(exc)) from exc
            existing = index.qualifications.get(candidate["cell_universe_sha256"])
            if existing != candidate:
                raise ReservationError("task-quality reservation readback differs")
        if now is None and datetime.now(UTC).date().isoformat() != date_utc:
            raise ReservationError("UTC date changed before reservation receipt")
        total = index.total if reserve or existing is not None else index.total + 1
        receipt = _receipt(
            baseline=baseline,
            reservation=candidate,
            path=path,
            total=total,
        )
    if reserve:
        assert receipt_path is not None
        if reuse_receipt:
            return load_reservation_receipt(
                receipt_path,
                packet_path,
                authorization_path,
                root=root,
                now=now,
            )
        try:
            _write_receipt_once(receipt_path, receipt)
        except ReservationError:
            return load_reservation_receipt(
                receipt_path,
                packet_path,
                authorization_path,
                root=root,
                now=now,
            )
    return receipt


def load_reservation_receipt(
    receipt_path: Path,
    packet_path: Path,
    authorization_path: Path,
    *,
    root: Path = ledger.ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reopen the exact canonical row before any Kubernetes create."""
    job = _job_module()
    try:
        value = job.read_json(receipt_path, "qualification reservation receipt")
    except (OSError, ValueError, job.QualificationJobError) as exc:
        raise ReservationError("qualification reservation receipt is unavailable") from exc
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    expected_fields = {
        "schema",
        "date_utc",
        "baseline_sha256",
        "reservation_sha256",
        "reservation_path",
        "plan_sha256",
        "packet_sha256",
        "authorization_sha256",
        "cell_universe_sha256",
        "count",
        "used",
        "reserved",
        "committed",
        "remaining",
        "cap",
        "rollback_policy",
        "sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("sha256") != ledger.digest(unsigned)
        or type(value.get("count")) is not int
        or value.get("count") != 1
        or type(value.get("cap")) is not int
        or value.get("cap") != ledger.CAP
        or type(value.get("used")) is not int
        or type(value.get("reserved")) is not int
        or type(value.get("committed")) is not int
        or type(value.get("remaining")) is not int
        or not 0 <= value["used"] <= ledger.CAP
        or not 1 <= value["reserved"] <= ledger.CAP
        or value["committed"] != value["used"] + value["reserved"]
        or value["remaining"] != ledger.CAP - value["committed"]
        or not 1 <= value["committed"] <= ledger.CAP
        or value.get("rollback_policy") != "durable_reservation_is_never_deleted_replaced_or_reused"
    ):
        raise ReservationError("qualification reservation receipt is invalid")
    package, authorization, plan = _inputs(packet_path, authorization_path)
    date_utc = (now or datetime.now(UTC)).astimezone(UTC).date().isoformat()
    candidate = _candidate(package, authorization, plan, date_utc)
    path = _reservation_path(root, date_utc, candidate["cell_universe_sha256"])
    if (
        value.get("date_utc") != date_utc
        or value.get("reservation_path") != str(path.resolve())
        or value.get("reservation_sha256") != candidate["sha256"]
        or value.get("plan_sha256") != candidate["plan_sha256"]
        or value.get("packet_sha256") != candidate["packet_sha256"]
        or value.get("authorization_sha256") != candidate["authorization_sha256"]
        or value.get("cell_universe_sha256") != candidate["cell_universe_sha256"]
    ):
        raise ReservationError("qualification reservation receipt differs from launch inputs")
    lock, day = _locked_day(root, date_utc, exclusive=False)
    with lock:
        if now is None and datetime.now(UTC).date().isoformat() != date_utc:
            raise ReservationError("UTC date changed while validating daily reservation")
        try:
            baseline = ledger.validate_baseline(day / "baseline.json", date_utc)
            index = ledger.reservation_index(day / "reservations", date_utc=date_utc)
        except ledger.LedgerError as exc:
            raise ReservationError(str(exc)) from exc
        if (
            baseline["sha256"] != value.get("baseline_sha256")
            or baseline["used"] != value.get("used")
            or index.total < value["reserved"]
            or index.qualifications.get(candidate["cell_universe_sha256"]) != candidate
            or baseline["used"] + index.total > ledger.CAP
        ):
            raise ReservationError("canonical qualification reservation is absent or changed")
        if now is None and datetime.now(UTC).date().isoformat() != date_utc:
            raise ReservationError("UTC date changed before daily reservation validation finished")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preview", "reserve"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = reserve_qualification_cell(
            args.packet,
            args.authorization,
            receipt_path=args.receipt,
            reserve=args.operation == "reserve",
        )
    except ReservationError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
