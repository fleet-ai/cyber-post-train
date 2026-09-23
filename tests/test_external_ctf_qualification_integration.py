from __future__ import annotations

import base64
import copy
import json
import lzma
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.external_ctf import (
    cybench_runtime_qualification,
    execution_packet,
    nyu_runtime_qualification,
    runtime_qualification,
    tensorlake,
)
from evals.external_ctf import protocol as external_protocol

PROTOCOL = Path("configs/evaluation/qwen38-external-ctf-paired-v1.json")


def _protocol() -> dict:
    return external_protocol.load_protocol(PROTOCOL)


def _signed(value: dict) -> dict:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return {**unsigned, "receipt_sha256": external_protocol.digest(unsigned)}


def test_closed_adapters_expose_only_model_free_qualification_cells() -> None:
    protocol = _protocol()
    nyu = tensorlake._cell(protocol, "nyu_ctf_web_test", 6, "qualification")  # noqa: SLF001
    cybench = tensorlake._cell(protocol, "cybench_web", 2, "qualification")  # noqa: SLF001
    assert nyu["arm"] == cybench["arm"] == "qualification"

    with pytest.raises(tensorlake.ExternalCtfError, match="not_launchable"):
        tensorlake._cell(protocol, "nyu_ctf_web_test", 4, "qualification")  # noqa: SLF001
    with pytest.raises(tensorlake.ExternalCtfError, match="not_launchable"):
        tensorlake._cell(protocol, "cybench_web", 1, "qualification")  # noqa: SLF001
    with pytest.raises(tensorlake.ExternalCtfError, match="invalid_cell_identity"):
        tensorlake._cell(protocol, "nyu_ctf_web_test", 6, "base")  # noqa: SLF001


def test_qualification_packet_forbids_every_scored_cell() -> None:
    protocol = _protocol()
    schedule = execution_packet.qualification_schedule(protocol)
    assert [(row["benchmark"], row["task_index"]) for row in schedule[:3]] == [
        ("cvebench_zero_day", 5),
        ("nyu_ctf_web_test", 6),
        ("cybench_web", 2),
    ]
    assert (
        len(schedule)
        == len({(row["benchmark"], row["task_index"], row["arm"]) for row in schedule})
        == 61
    )
    packet = {
        "packet_mode": "qualification_only",
        "execution": {"qualification_cells": schedule},
    }
    qualification = tensorlake._cell(  # noqa: SLF001
        protocol, "nyu_ctf_web_test", 6, "qualification"
    )
    tensorlake._require_packet_allows_cell(packet, protocol, qualification)  # noqa: SLF001
    scored = {
        "benchmark": "cvebench_zero_day",
        "task_index": 0,
        "task_id": protocol["benchmarks"]["cvebench_zero_day"]["task_ids"][0],
        "arm": "base",
    }
    with pytest.raises(tensorlake.ExternalCtfError, match="forbids_cell"):
        tensorlake._require_packet_allows_cell(packet, protocol, scored)  # noqa: SLF001


def test_qualification_schedule_gates_canaries_then_bounds_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    rows = execution_packet.qualification_schedule(protocol)
    stages = execution_packet.qualification_stages(protocol)
    packet = {
        "receipt_sha256": "sha256:" + "9" * 64,
        "protocol": {"receipt_sha256": protocol["protocol_sha256"]},
        "execution": {
            "qualification_cells": rows,
            "staged_concurrency": stages,
        },
    }
    capacity_state = tmp_path / "capacity"
    capacity_state.mkdir()
    owned_names = {
        tensorlake.cell_name(row["benchmark"], row["task_index"], row["arm"]) for row in rows
    }

    def enforce(target: str) -> None:
        tensorlake._enforce_qualification_stage(  # noqa: SLF001
            tmp_path,
            protocol=protocol,
            packet=packet,
            target=target,
            capacity_state=capacity_state,
            owned_names=owned_names,
        )

    monkeypatch.setattr(
        tensorlake,
        "_result_value",
        lambda raw, **_kwargs: json.loads(raw),
    )

    def accept(row: dict) -> None:
        cell = {key: row[key] for key in ("benchmark", "task_index", "task_id", "arm")}
        name = tensorlake.cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
        terminal = _signed(
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": row["qualification_contract_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "result": {"status": "runtime_qualified"},
                "outcome": "runtime_preflight_passed",
                "infrastructure_error_class": None,
            }
        )
        release = _signed(
            {
                "schema": "external_ctf_sandbox_release_v1",
                **cell,
                "name": name,
                "status": "terminated",
                "protocol_sha256": protocol["protocol_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "terminal_receipt_sha256": terminal["receipt_sha256"],
            }
        )
        (tmp_path / f"{name}.terminal.json").write_bytes(tensorlake.canonical(terminal) + b"\n")
        (tmp_path / f"{name}.released.json").write_bytes(tensorlake.canonical(release) + b"\n")

    canary = tensorlake.cell_name("cvebench_zero_day", 5, "qualification")
    enforce(canary)
    with pytest.raises(tensorlake.ExternalCtfError, match="canary_order_violation"):
        enforce(tensorlake.cell_name("cvebench_zero_day", 0, "qualification"))
    first_canary = stages["canary"]["ordered_cells"][0]
    first_cell = {key: first_canary[key] for key in ("benchmark", "task_index", "task_id", "arm")}
    first_name = tensorlake.cell_name(
        first_cell["benchmark"], first_cell["task_index"], first_cell["arm"]
    )
    failed = _signed(
        {
            "schema": "external_ctf_cell_terminal_v1",
            **first_cell,
            "name": first_name,
            "protocol_sha256": protocol["protocol_sha256"],
            "qualification_contract_sha256": first_canary["qualification_contract_sha256"],
            "execution_packet_receipt_sha256": packet["receipt_sha256"],
            "result": None,
            "outcome": "infrastructure_invalid",
            "infrastructure_error_class": "synthetic_canary_failure",
        }
    )
    failed_release = _signed(
        {
            "schema": "external_ctf_sandbox_release_v1",
            **first_cell,
            "name": first_name,
            "status": "terminated",
            "protocol_sha256": protocol["protocol_sha256"],
            "execution_packet_receipt_sha256": packet["receipt_sha256"],
            "terminal_receipt_sha256": failed["receipt_sha256"],
        }
    )
    terminal_path = tmp_path / f"{first_name}.terminal.json"
    release_path = tmp_path / f"{first_name}.released.json"
    terminal_path.write_bytes(tensorlake.canonical(failed) + b"\n")
    release_path.write_bytes(tensorlake.canonical(failed_release) + b"\n")
    with pytest.raises(tensorlake.ExternalCtfError, match="canary_not_accepted"):
        second = stages["canary"]["ordered_cells"][1]
        enforce(tensorlake.cell_name(second["benchmark"], second["task_index"], second["arm"]))
    terminal_path.unlink()
    release_path.unlink()
    for row in stages["canary"]["ordered_cells"]:
        accept(row)
    first_batch = stages["batch"]["cells"][0]
    enforce(
        tensorlake.cell_name(
            first_batch["benchmark"], first_batch["task_index"], first_batch["arm"]
        )
    )
    for row in stages["batch"]["cells"][:4]:
        name = tensorlake.cell_name(row["benchmark"], row["task_index"], row["arm"])
        (tmp_path / f"{name}.create-claim.json").write_text("claimed", encoding="utf-8")
        enforce(name)
    fifth = stages["batch"]["cells"][4]
    with pytest.raises(tensorlake.ExternalCtfError, match="stage_parallel_limit"):
        enforce(tensorlake.cell_name(fifth["benchmark"], fifth["task_index"], fifth["arm"]))


def test_qualification_batch_counts_orphan_pending_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    rows = execution_packet.qualification_schedule(protocol)
    stages = execution_packet.qualification_stages(protocol)
    packet = {
        "receipt_sha256": "sha256:" + "9" * 64,
        "protocol": {"receipt_sha256": protocol["protocol_sha256"]},
        "execution": {
            "qualification_cells": rows,
            "staged_concurrency": stages,
        },
    }
    state = tmp_path / "state"
    capacity_state = tmp_path / "capacity"
    state.mkdir()
    capacity_state.mkdir()
    owned_names = {
        tensorlake.cell_name(row["benchmark"], row["task_index"], row["arm"]) for row in rows
    }
    monkeypatch.setattr(tensorlake, "_result_value", lambda raw, **_kwargs: json.loads(raw))

    for row in stages["canary"]["ordered_cells"]:
        cell = {key: row[key] for key in ("benchmark", "task_index", "task_id", "arm")}
        name = tensorlake.cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
        terminal = _signed(
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": row["qualification_contract_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "result": {"status": "runtime_qualified"},
                "outcome": "runtime_preflight_passed",
                "infrastructure_error_class": None,
            }
        )
        release = _signed(
            {
                "schema": "external_ctf_sandbox_release_v1",
                **cell,
                "name": name,
                "status": "terminated",
                "protocol_sha256": protocol["protocol_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "terminal_receipt_sha256": terminal["receipt_sha256"],
            }
        )
        (state / f"{name}.terminal.json").write_bytes(tensorlake.canonical(terminal) + b"\n")
        (state / f"{name}.released.json").write_bytes(tensorlake.canonical(release) + b"\n")

    pending_rows = stages["batch"]["cells"][:4]
    for row in pending_rows:
        name = tensorlake.cell_name(row["benchmark"], row["task_index"], row["arm"])
        tensorlake.replica_set.reserve_shared_capacity_slot(
            state=capacity_state,
            rows=[],
            owned_names=owned_names,
            sandbox_name=name,
            creator="external_ctf",
            authority_receipt_sha256="sha256:" + "a" * 64,
            spec_sha256="sha256:" + "b" * 64,
        )

    adopted = tensorlake.cell_name(
        pending_rows[0]["benchmark"], pending_rows[0]["task_index"], pending_rows[0]["arm"]
    )
    tensorlake._enforce_qualification_stage(  # noqa: SLF001
        state,
        protocol=protocol,
        packet=packet,
        target=adopted,
        capacity_state=capacity_state,
        owned_names=owned_names,
    )
    fifth = stages["batch"]["cells"][4]
    with pytest.raises(tensorlake.ExternalCtfError, match="stage_parallel_limit"):
        tensorlake._enforce_qualification_stage(  # noqa: SLF001
            state,
            protocol=protocol,
            packet=packet,
            target=tensorlake.cell_name(fifth["benchmark"], fifth["task_index"], fifth["arm"]),
            capacity_state=capacity_state,
            owned_names=owned_names,
        )


def test_qualification_start_allows_four_precreated_batch_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    rows = execution_packet.qualification_schedule(protocol)
    stages = execution_packet.qualification_stages(protocol)
    state = tmp_path / "state"
    capacity_state = tmp_path / "capacity"
    state.mkdir(mode=0o700)
    capacity_state.mkdir(mode=0o700)
    packet = {
        "receipt_sha256": "sha256:" + "9" * 64,
        "packet_mode": "qualification_only",
        "protocol": {"receipt_sha256": protocol["protocol_sha256"]},
        "execution": {
            "qualification_cells": rows,
            "staged_concurrency": stages,
        },
    }
    owned_names = {
        tensorlake.cell_name(row["benchmark"], row["task_index"], row["arm"]) for row in rows
    }
    authority = {
        "state": capacity_state,
        "external_state": state,
        "owned_names": owned_names,
        "capacity_successor_receipt_sha256": "sha256:" + "1" * 64,
        "capacity_successor_state_receipt_sha256": "sha256:" + "2" * 64,
        "shared_capacity_roster_receipt_sha256": "sha256:" + "3" * 64,
        "execution_packet_receipt_sha256": packet["receipt_sha256"],
    }

    class Client:
        def request(self, method: str, url: str, body: dict | None = None, **_kwargs):
            if method == "GET":
                return {"status": "running", "sandbox_url": "https://sandbox.invalid"}
            assert method == "POST"
            assert url == "https://sandbox.invalid/api/v1/processes"
            assert body is not None
            return {"pid": 100 + len(list(state.glob("*.process.json")))}

    monkeypatch.setattr(tensorlake, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        tensorlake,
        "_execution_context",
        lambda *_args, **_kwargs: (authority, packet),
    )
    monkeypatch.setattr(
        tensorlake,
        "_accepted_qualification_release",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(tensorlake, "_qualification_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tensorlake, "_client", Client)

    for row in stages["batch"]["cells"][:4]:
        cell = {key: row[key] for key in ("benchmark", "task_index", "task_id", "arm")}
        name = tensorlake.cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
        (state / f"{name}.create-claim.json").write_text("claimed", encoding="utf-8")
        created = {
            "schema": "external_ctf_sandbox_created_v1",
            **cell,
            "name": name,
            "sandbox_id": f"sandbox-{name}",
            "protocol_sha256": protocol["protocol_sha256"],
            "execution_packet_receipt_sha256": packet["receipt_sha256"],
            "capacity_successor_receipt_sha256": authority["capacity_successor_receipt_sha256"],
            "capacity_successor_state_receipt_sha256": authority[
                "capacity_successor_state_receipt_sha256"
            ],
            "shared_capacity_roster_receipt_sha256": authority[
                "shared_capacity_roster_receipt_sha256"
            ],
            "live_owner_receipt_sha256": "sha256:" + "4" * 64,
            "capacity_reservation_receipt_sha256": "sha256:" + "5" * 64,
            "create_route_preflight_receipt_sha256": None,
        }
        (state / f"{name}.created.json").write_bytes(tensorlake.canonical(created) + b"\n")

    for row in stages["batch"]["cells"][:4]:
        result = tensorlake.start(
            protocol_path=PROTOCOL,
            retry_execution_path=tmp_path / "retry.json",
            execution_packet_path=tmp_path / "packet.json",
            benchmark=row["benchmark"],
            task_index=row["task_index"],
            arm="qualification",
        )
        assert result["pid"] >= 100

    assert len(list(state.glob("*.process-claim.json"))) == 4
    assert len(list(state.glob("*.process.json"))) == 4


@pytest.mark.parametrize(
    ("verb", "needs_route_preflight"),
    [
        (tensorlake.create, True),
        (tensorlake.reconcile_create, False),
        (tensorlake.abort_create_absent, False),
        (tensorlake.start, True),
        (tensorlake.reconcile_start, False),
        (tensorlake.abort_start_absent, False),
        (tensorlake.abort_unstarted, False),
        (tensorlake.status, False),
        (tensorlake.release, False),
    ],
)
def test_qualification_packet_rejects_scored_provider_verb_before_client(
    verb: object,
    needs_route_preflight: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    packet = {
        "receipt_sha256": "sha256:" + "9" * 64,
        "packet_mode": "qualification_only",
        "execution": {"qualification_cells": execution_packet.qualification_schedule(protocol)},
    }
    authority = {"external_state": state}
    monkeypatch.setattr(tensorlake, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(tensorlake, "capacity_authority", lambda *_args, **_kwargs: authority)
    monkeypatch.setattr(
        tensorlake,
        "_load_execution_packet",
        lambda *_args, **_kwargs: (packet, state),
    )
    monkeypatch.setattr(
        tensorlake,
        "_client",
        lambda: (_ for _ in ()).throw(AssertionError("provider client must not be created")),
    )
    kwargs = {
        "protocol_path": PROTOCOL,
        "retry_execution_path": tmp_path / "retry.json",
        "execution_packet_path": tmp_path / "packet.json",
        "benchmark": "cvebench_zero_day",
        "task_index": 0,
        "arm": "base",
    }
    if needs_route_preflight:
        kwargs["route_preflight_path"] = tmp_path / "route.json"
    with pytest.raises(tensorlake.ExternalCtfError, match="forbids_cell"):
        verb(**kwargs)  # type: ignore[operator]


def test_qualification_packet_rejects_scored_route_preflight_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    packet = {
        "receipt_sha256": "sha256:" + "9" * 64,
        "packet_mode": "qualification_only",
        "execution": {"qualification_cells": execution_packet.qualification_schedule(protocol)},
    }
    monkeypatch.setattr(tensorlake, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        tensorlake,
        "_execution_context",
        lambda *_args, **_kwargs: ({"external_state": tmp_path}, packet),
    )
    monkeypatch.setattr(
        tensorlake.live_parity,
        "capture_start_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not probe serving")),
    )
    with pytest.raises(tensorlake.ExternalCtfError, match="forbids_cell"):
        tensorlake.seal_start_route_preflight(
            protocol_path=PROTOCOL,
            retry_execution_path=tmp_path / "retry.json",
            execution_packet_path=tmp_path / "packet.json",
            benchmark="cvebench_zero_day",
            task_index=0,
            arm="base",
            kubernetes_context="never-read",
            output_path=tmp_path / "route.json",
        )


def test_qualification_packet_seals_and_loads_without_serving_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    state = tmp_path / "external-state"
    state.mkdir(mode=0o700)
    authority = {
        "state": tmp_path / "shared-state",
        "external_state": state,
        "snapshot_id": "snapshot-1",
        "owned_names": set(),
        "retry_execution_receipt_sha256": "sha256:" + "1" * 64,
        "capacity_successor_receipt_sha256": "sha256:" + "2" * 64,
        "capacity_successor_state_receipt_sha256": "sha256:" + "3" * 64,
        "shared_capacity_roster_receipt_sha256": "sha256:" + "4" * 64,
        "live_owner_receipt_sha256": "sha256:" + "5" * 64,
    }
    authority["state"].mkdir(mode=0o700)
    prohibition = {
        "receipt_sha256": "sha256:" + "6" * 64,
        "source_sha256": {
            name: external_protocol.file_digest(path.read_bytes())
            for name, path in sorted(execution_packet.UNMIGRATED_CREATORS.items())
        },
    }
    monkeypatch.setattr(
        execution_packet.shared_capacity_exclusion,
        "load_prohibition",
        lambda **_kwargs: prohibition,
    )
    monkeypatch.setattr(execution_packet, "_process_matches", lambda: [])
    monkeypatch.setattr(
        execution_packet.replica_set, "shared_project_capacity_count", lambda *_args: 0
    )
    fake_tensorlake = SimpleNamespace(
        capacity_authority=lambda *_args, **_kwargs: authority,
        external_names=lambda _protocol: set(),
        _client=lambda: SimpleNamespace(inventory=lambda: []),
    )
    output = tmp_path / "qualification-packet.json"
    schedule = execution_packet.qualification_schedule(protocol)
    sealed = execution_packet._seal_under_shared_lock(  # noqa: SLF001
        packet_mode="qualification_only",
        protocol=protocol,
        authority=authority,
        retry_execution_path=tmp_path / "retry.json",
        tensorlake=fake_tensorlake,
        source={"commit": "a" * 40},
        official_sources={"verified": True},
        base_clone_serving=None,
        parity=None,
        parity_bytes=None,
        quiescence={"receipt_sha256": "sha256:" + "7" * 64},
        quiescence_bytes=b"quiescence\n",
        retry_bytes=b"retry\n",
        protocol_bytes=PROTOCOL.read_bytes(),
        schedule=schedule,
        current=execution_packet.datetime(2026, 9, 23, tzinfo=execution_packet.UTC),
        output=output,
    )
    assert sealed["packet_mode"] == "qualification_only"
    assert sealed["models"] is None
    assert sealed["base_clone_serving"] is None
    assert sealed["live_parity"] is None
    assert len(sealed["execution"]["qualification_cells"]) == 61
    assert sealed["execution"]["staged_concurrency"]["canary"]["max_parallel_cells"] == 1
    assert (
        sealed["execution"]["staged_concurrency"]["batch"]["max_parallel_cells"]
        == execution_packet.QUALIFICATION_BATCH_MAX_PARALLEL
    )
    monkeypatch.setattr(
        execution_packet.live_parity,
        "validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    loaded, packet_state = execution_packet.load(
        output, protocol=protocol, authority=authority, require_current_source=False
    )
    assert loaded["receipt_sha256"] == sealed["receipt_sha256"]
    assert packet_state.name == sealed["receipt_sha256"].removeprefix("sha256:")


def test_qualification_and_scored_packets_use_distinct_create_once_markers(tmp_path: Path) -> None:
    qualification = execution_packet._bound_marker(tmp_path, "qualification_only")  # noqa: SLF001
    scored = execution_packet._bound_marker(tmp_path, "scored")  # noqa: SLF001
    assert qualification != scored
    qualification.write_text("bound", encoding="utf-8")
    execution_packet._initial_state_absent(tmp_path, {"extctf-cve-t00-b-v1"}, scored)  # noqa: SLF001


def test_qualification_summary_binds_terminal_and_release_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    state = tmp_path / "packet-state"
    state.mkdir()
    packet_path = tmp_path / "qualification-packet.json"
    packet_path.write_bytes(b"sealed-qualification-packet\n")
    packet = {
        "packet_mode": "qualification_only",
        "receipt_sha256": "sha256:" + "9" * 64,
    }
    monkeypatch.setattr(
        tensorlake,
        "capacity_authority",
        lambda *_args, **_kwargs: {"external_state": tmp_path},
    )
    monkeypatch.setattr(
        execution_packet,
        "load",
        lambda *_args, **_kwargs: (packet, state),
    )
    for row in execution_packet.qualification_schedule(protocol):
        if row["benchmark"] != "nyu_ctf_web_test":
            continue
        cell = {key: row[key] for key in ("benchmark", "task_index", "task_id", "arm")}
        name = tensorlake.cell_name(cell["benchmark"], cell["task_index"], cell["arm"])
        terminal = _signed(
            {
                "schema": "external_ctf_cell_terminal_v1",
                **cell,
                "name": name,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": row["qualification_contract_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "result": None,
                "outcome": "infrastructure_invalid",
                "infrastructure_error_class": "synthetic_infrastructure_hold",
            }
        )
        release = _signed(
            {
                "schema": "external_ctf_sandbox_release_v1",
                **cell,
                "name": name,
                "status": "terminated",
                "protocol_sha256": protocol["protocol_sha256"],
                "execution_packet_receipt_sha256": packet["receipt_sha256"],
                "terminal_receipt_sha256": terminal["receipt_sha256"],
            }
        )
        (state / f"{name}.terminal.json").write_bytes(tensorlake.canonical(terminal) + b"\n")
        (state / f"{name}.released.json").write_bytes(tensorlake.canonical(release) + b"\n")
    summary = execution_packet.seal_qualification_summary(
        protocol_path=PROTOCOL,
        retry_execution_path=tmp_path / "retry.json",
        qualification_packet_path=packet_path,
        benchmark="nyu_ctf_web_test",
        output=tmp_path / "summary.json",
    )
    assert summary["scheduled_task_count"] == 16
    assert summary["runtime_preflight_passed_count"] == 0
    assert summary["infrastructure_invalid_count"] == 16
    assert summary["score_reads"] == summary["model_requests"] == 0
    assert all(set(row).isdisjoint({"result", "score"}) for row in summary["rows"])

    first = next(
        row
        for row in execution_packet.qualification_schedule(protocol)
        if row["benchmark"] == "nyu_ctf_web_test"
    )
    first_name = tensorlake.cell_name(first["benchmark"], first["task_index"], first["arm"])
    terminal_path = state / f"{first_name}.terminal.json"
    terminal = json.loads(terminal_path.read_bytes())
    terminal["result"] = {"status": "forged_infrastructure_result"}
    terminal_path.write_bytes(tensorlake.canonical(_signed(terminal)) + b"\n")
    with pytest.raises(execution_packet.ExecutionPacketError, match="cell_binding_invalid"):
        execution_packet.seal_qualification_summary(
            protocol_path=PROTOCOL,
            retry_execution_path=tmp_path / "retry.json",
            qualification_packet_path=packet_path,
            benchmark="nyu_ctf_web_test",
            output=tmp_path / "tampered-summary.json",
        )


@pytest.mark.parametrize(
    ("benchmark", "task_index"),
    [("nyu_ctf_web_test", 6), ("cybench_web", 2)],
)
def test_provider_precondition_receipt_is_coordinator_compatible(
    benchmark: str, task_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_qualification.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_qualification.platform, "machine", lambda: "arm64")
    protocol = _protocol()
    cell = tensorlake._cell(protocol, benchmark, task_index, "qualification")  # noqa: SLF001
    if benchmark == "nyu_ctf_web_test":
        receipt = nyu_runtime_qualification.qualify(Path("/not-read"), task_index, PROTOCOL)
    else:
        controller = protocol["benchmarks"][benchmark]["runtime_qualification"]["controller_image"]
        receipt = cybench_runtime_qualification.qualify(
            Path("/not-read"), task_index, controller, PROTOCOL
        )
    assert receipt["status"] == "precondition_failed"
    assert (
        tensorlake._result_value(  # noqa: SLF001
            tensorlake.canonical(receipt) + b"\n", protocol=protocol, cell=cell
        )
        == receipt
    )


def test_qualification_contract_excludes_models_and_scored_budget() -> None:
    protocol = _protocol()
    original = {
        name: external_protocol.runtime_qualification_contract_sha256(protocol, name)
        for name in protocol["benchmarks"]
    }
    changed = copy.deepcopy(protocol)
    changed["arms"]["base"]["served_model"] = "unrelated-future-route"
    changed["benchmarks"]["nyu_ctf_web_test"]["budget"]["wall_seconds"] += 1
    changed["execution"]["worker_sha256"] = "sha256:" + "0" * 64
    assert {
        name: external_protocol.runtime_qualification_contract_sha256(changed, name)
        for name in changed["benchmarks"]
    } == original

    changed["benchmarks"]["cybench_web"]["runtime_qualification"]["controller_image"] = (
        "docker.io/cybench/cybench@sha256:" + "0" * 64
    )
    assert (
        external_protocol.runtime_qualification_contract_sha256(changed, "cybench_web")
        != original["cybench_web"]
    )


def test_execution_packet_source_identity_covers_every_qualification_executor() -> None:
    packet_sources = set(execution_packet.SOURCE_FILES.values())
    assert {
        path
        for paths in external_protocol.RUNTIME_QUALIFICATION_SOURCE_PATHS.values()
        for path in paths.values()
    } <= packet_sources


def _nyu_result(protocol: dict) -> tuple[dict, dict]:
    cell = tensorlake._cell(protocol, "nyu_ctf_web_test", 6, "qualification")  # noqa: SLF001
    benchmark = protocol["benchmarks"][cell["benchmark"]]
    task_source = {
        "challenge_sha256": "sha256:" + "1" * 64,
        "compose_sha256": "sha256:" + "2" * 64,
        "attachment_manifest_sha256": "sha256:" + "3" * 64,
    }
    images = [
        {
            "reference": "registry.invalid/challenge:v1",
            "image_id": "sha256:" + "4" * 64,
            "repository_digest": "registry.invalid/challenge@sha256:" + "5" * 64,
        }
    ]
    qualification = _signed(
        {
            "schema": "external_ctf_nyu_runtime_qualification_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "qualification_contract_sha256": benchmark["runtime_qualification"]["contract_sha256"],
            "benchmark": cell["benchmark"],
            "task_id": cell["task_id"],
            "source_commit": benchmark["source"]["commit"],
            **task_source,
            "target_host": "challenge",
            "target_port": 8080,
            "status": "runtime_preflight_passed",
            "model_requests": 0,
            "startup_healthy": True,
            "target_reachable": True,
            "grader_negative_control": False,
            "grader_positive_control": True,
            "agent_image_id": benchmark["harness"]["image_id"],
            "opencode_version": benchmark["harness"]["version"],
            "platform": "linux/amd64",
            "isolated_task_network": True,
            "images": images,
        }
    )
    harness = {
        "reference": benchmark["harness"]["image_id"],
        "image_id": benchmark["harness"]["image_id"],
        "repository_digest": None,
    }
    result = _signed(
        {
            "schema": "external_ctf_nyu_runtime_qualification_execution_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "qualification_contract_sha256": benchmark["runtime_qualification"]["contract_sha256"],
            "benchmark": cell["benchmark"],
            "qualification_name": tensorlake.cell_name(
                cell["benchmark"], cell["task_index"], "qualification"
            ),
            "task_index": cell["task_index"],
            "task_id_sha256": external_protocol.file_digest(cell["task_id"].encode()),
            "source_commit": benchmark["source"]["commit"],
            "task_source": task_source,
            "task_source_sha256": external_protocol.digest(task_source),
            "published_port_entries_removed": 1,
            "container_names_removed": 0,
            "harness_image_id": benchmark["harness"]["image_id"],
            "harness_image": harness,
            "harness_image_sha256": external_protocol.digest(harness),
            "challenge_runtime_images": images,
            "challenge_runtime_images_sha256": external_protocol.digest(images),
            "platform": "linux/amd64",
            "status": "runtime_qualified",
            "target_address_binding": "compose_alias",
            "qualification": qualification,
            "error_code": None,
            "official_solution_solvability_claimed": False,
            "provider_calls": 0,
            "model_requests": 0,
            "scored_attempts": 0,
            "contains_prompts_flags_solutions_traces_or_scores": False,
        }
    )
    return cell, result


def test_nyu_qualification_result_binds_contract_and_rejects_tamper() -> None:
    protocol = _protocol()
    cell, result = _nyu_result(protocol)
    raw = tensorlake.canonical(result) + b"\n"
    assert tensorlake._result_value(raw, protocol=protocol, cell=cell) == result  # noqa: SLF001

    changed = copy.deepcopy(result)
    changed["qualification_contract_sha256"] = "sha256:" + "0" * 64
    unsigned = {key: value for key, value in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = external_protocol.digest(unsigned)
    assert (  # noqa: SLF001
        tensorlake._result_value(
            tensorlake.canonical(changed) + b"\n", protocol=protocol, cell=cell
        )
        is None
    )


def test_cybench_predeclared_host_gateway_is_infrastructure_not_score() -> None:
    protocol = _protocol()
    cell = tensorlake._cell(protocol, "cybench_web", 0, "qualification")  # noqa: SLF001
    benchmark = protocol["benchmarks"][cell["benchmark"]]
    task_source = {"path_sha256": "sha256:" + "1" * 64}
    result = _signed(
        {
            "schema": "cybench_remote_runtime_qualification_task_execution_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "qualification_contract_sha256": benchmark["runtime_qualification"]["contract_sha256"],
            "benchmark": cell["benchmark"],
            "qualification_name": tensorlake.cell_name(
                cell["benchmark"], cell["task_index"], "qualification"
            ),
            "task_index": cell["task_index"],
            "task_id_sha256": external_protocol.file_digest(cell["task_id"].encode()),
            "source_commit": benchmark["source"]["commit"],
            "task_source": task_source,
            "task_source_sha256": external_protocol.digest(task_source),
            "published_port_entries_removed": None,
            "container_names_removed": None,
            "controller_image_reference": benchmark["runtime_qualification"]["controller_image"],
            "controller_image": None,
            "controller_image_sha256": None,
            "harness_image_id": benchmark["harness"]["image_id"],
            "harness_image": None,
            "harness_image_sha256": None,
            "challenge_images": None,
            "challenge_image_set_sha256": None,
            "official_solution_manifest_sha256": None,
            "official_solution_helper_sha256": None,
            "platform": "linux/amd64",
            "status": "infrastructure_invalid",
            "task_evidence": None,
            "error_code": "task_host_gateway_unsupported",
            "provider_calls": 0,
            "model_requests": 0,
            "scored_attempts": 0,
            "contains_prompts_flags_solutions_traces_or_scores": False,
        }
    )
    assert (
        tensorlake._result_value(  # noqa: SLF001
            tensorlake.canonical(result) + b"\n", protocol=protocol, cell=cell
        )
        == result
    )
    assert result["status"] == "infrastructure_invalid"
    assert "score" not in result


def test_qualification_bundle_rejects_executor_source_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    bundle = tensorlake._qualification_bundle(protocol, "nyu_ctf_web_test")  # noqa: SLF001
    files = json.loads(lzma.decompress(bundle))
    assert "evals/external_ctf/nyu_runtime_qualification.py" in files
    assert "evals/external_ctf/cybench_runtime_qualification.py" in files

    changed = tmp_path / "changed.py"
    changed.write_text("# changed\n")
    paths = dict(tensorlake.RUNTIME_QUALIFICATION_SOURCE_PATHS["nyu_ctf_web_test"])
    paths["executor"] = changed
    monkeypatch.setitem(tensorlake.RUNTIME_QUALIFICATION_SOURCE_PATHS, "nyu_ctf_web_test", paths)
    with pytest.raises(tensorlake.ExternalCtfError, match="executor_source_drifted"):
        tensorlake._qualification_bundle(protocol, "nyu_ctf_web_test")  # noqa: SLF001


def test_cve_qualification_contract_uses_stable_dedicated_executor() -> None:
    protocol = _protocol()
    paths = tensorlake.RUNTIME_QUALIFICATION_SOURCE_PATHS["cvebench_zero_day"]
    assert set(paths) == {"executor"}
    assert paths["executor"].name == "cvebench_runtime_qualification.py"
    bundle = tensorlake._qualification_bundle(protocol, "cvebench_zero_day")  # noqa: SLF001
    files = json.loads(lzma.decompress(bundle))
    assert "evals/external_ctf/cvebench_runtime_qualification.py" in files
    expected = {
        "evals/external_ctf/analyze.py",
        "evals/external_ctf/protocol.py",
        "evals/external_ctf/tensorlake.py",
        "evals/external_ctf/worker.py",
        *(
            path.relative_to(external_protocol.ROOT).as_posix()
            for paths in external_protocol.RUNTIME_QUALIFICATION_SOURCE_PATHS.values()
            for path in paths.values()
        ),
    }
    assert files.keys() == expected


@pytest.mark.parametrize("benchmark", ["cvebench_zero_day", "nyu_ctf_web_test", "cybench_web"])
def test_qualification_bundle_is_a_complete_protocol_runtime(
    benchmark: str, tmp_path: Path
) -> None:
    bundle = tensorlake._qualification_bundle(_protocol(), benchmark)  # noqa: SLF001
    root = tmp_path / "runtime"
    assert len(base64.b64encode(bundle)) < 128 * 1024
    for relative, encoded in json.loads(lzma.decompress(bundle)).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded, validate=True))
    for package in (root / "evals", root / "evals/external_ctf"):
        (package / "__init__.py").write_bytes(b"")
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys;"
                "from pathlib import Path;"
                "sys.path.insert(0, sys.argv[1]);"
                "from evals.external_ctf import protocol;"
                "assert Path(protocol.__file__).resolve().is_relative_to("
                "Path(sys.argv[1]).resolve());"
                "protocol.load_protocol(Path(sys.argv[2]))"
            ),
            str(root),
            str(PROTOCOL.resolve()),
        ],
        check=True,
        cwd=tmp_path,
    )
