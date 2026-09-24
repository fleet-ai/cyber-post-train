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
CONFIG_MAP_UID = "66666666-7777-4888-8999-aaaaaaaaaaaa"
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
        config_map_name="heldout-base-seed46-code",
        database="heldout_base_seed46",
        identity_sha256="sha256:" + "3" * 64,
        identity={"pass_k": 1, "retry_limit": 0, "sampling_seed": 46},
    )
    return SimpleNamespace(packet=packet, evaluation_config={})


class Cluster:
    def __init__(self, *, terminal: bool = True) -> None:
        self.terminal = terminal

    def get(self, resource: str, _namespace: str, _name: str) -> dict[str, Any]:
        if resource == "configmaps":
            return {"metadata": {"uid": CONFIG_MAP_UID}}
        conditions = [{"type": "Complete", "status": "True"}] if self.terminal else []
        return {"metadata": {"uid": JOB_UID}, "status": {"conditions": conditions}}

    def list(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"items": []}


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
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    _patch_source(monkeypatch, bindings)
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
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
            "config_map_uid": CONFIG_MAP_UID,
            "evaluation_identity_sha256": "sha256:" + "3" * 64,
            "comparison_protocol_sha256": "sha256:" + "4" * 64,
        }

    monkeypatch.setattr(heldout_launch, "launch_package_once", launch)
    adapter.reserve_wave(
        bindings,
        plan,
        profile,
        _profile_file_sha256(profile),
        bindings["wave"]["reservation_id"],
        expected_sessions=160,
        packet_set_sha256=bindings["wave"]["packet_set_sha256"],
        root=tmp_path / "budget",
    )
    common = {
        "action": "launch",
        "phase": "rollout",
        "bindings_path": tmp_path / "fleet-bindings.json",
        "strict_profile_path": profile,
        "strict_profile_file_sha256": _profile_file_sha256(profile),
        "context": "fleet",
        "preview_receipt": preview,
        "readiness_receipt": ready,
        "cluster": Cluster(),
        "database": Database(),
        "budget_root": tmp_path / "budget",
        "duplicate_gate_evidence": tmp_path / "duplicate-gate-evidence.json",
    }
    first = adapter.run_action(packet_path=packets[KEY_A], **common)
    second = adapter.run_action(packet_path=packets[KEY_B], **common)
    assert calls == 1
    assert first["remote_id"] == second["remote_id"] == JOB_UID
    reservations = list(
        (tmp_path / "budget" / bindings["budget"]["date_utc"] / "reservations").glob("*.json")
    )
    assert len(reservations) == 1
    reservation = adapter._read(reservations[0])  # noqa: SLF001
    assert reservation["count"] == 160


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


def _wave_binding(tmp_path: Path, *, used: int = 271) -> dict[str, Any]:
    _, binding = _campaign(tmp_path)
    groups: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    for index in range(16):
        count = 13 if index < 8 else 7
        group_id = f"group-{index:02d}"
        members = []
        for offset in range(count):
            key = "sha256:" + f"{index * 13 + offset + 1:064x}"
            members.append(key)
            cells[key] = {
                "group": group_id,
                "attempt": offset % 4 + 1,
                "task_version_id": f"task-{index}-{offset}",
                "model_id": "base",
                "model_revision": "revision-a",
                "source_attempt": offset + 1,
                "campaign_model_id": "base",
            }
        groups[group_id] = {
            "packet": f"source/{group_id}/LAUNCH_PACKET.json",
            "packet_sha256": "sha256:" + f"{index + 1:064x}",
            "leader": members[0],
            "cells": members,
        }
    binding["schema"] = adapter.WAVE_BINDING_SCHEMA
    binding["budget"]["used"] = used
    binding["wave"] = {
        "reservation_id": "heldout20-base-step1000-v3",
        "expected_sessions": 160,
        "packet_set_sha256": adapter._packet_set_sha256(groups),  # noqa: SLF001
        "campaign_plan_sha256": "sha256:" + "7" * 64,
        "control_source_sha256": adapter._control_source_sha256(),  # noqa: SLF001
    }
    binding["groups"] = groups
    binding["cells"] = cells
    binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    return binding


def _wave_plan(binding: dict[str, Any], *targets: dict[str, Any]) -> dict[str, Any]:
    by_key = {target["experiment_key"]: target for target in targets}
    source = adapter._control_source_sha256()  # noqa: SLF001
    rows = []
    for key in binding["cells"]:
        rows.append(
            by_key.get(
                key,
                {
                    "experiment_key": key,
                    "drivers": {
                        phase: {"source_sha256": source} for phase in adapter.campaign.PHASES
                    },
                },
            )
        )
    return {
        "campaign_id": "heldout20-base-step1000-test",
        "plan_sha256": binding["wave"]["campaign_plan_sha256"],
        "targets": rows,
    }


def _strict_profile(tmp_path: Path, binding: dict[str, Any], plan: dict[str, Any]) -> Path:
    value = {
        "schema": adapter.STRICT_PROFILE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "binding_schema": adapter.WAVE_BINDING_SCHEMA,
        "campaign_plan_sha256": plan["plan_sha256"],
        "bindings_sha256": binding["sha256"],
        "reservation_id": binding["wave"]["reservation_id"],
        "packet_set_sha256": binding["wave"]["packet_set_sha256"],
        "group_count": adapter.WAVE_GROUP_COUNT,
        "cell_count": adapter.WAVE_CELL_COUNT,
        "control_source_sha256": adapter._control_source_sha256(),  # noqa: SLF001
    }
    value["sha256"] = adapter._digest(value)  # noqa: SLF001
    return _write(tmp_path / "strict-wave-profile.json", value)


def _profile_file_sha256(path: Path) -> str:
    return "sha256:" + adapter.hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_strict_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    binding: dict[str, Any],
    plan: dict[str, Any],
) -> Path:
    _write(tmp_path / "fleet-bindings.json", binding)
    profile = _strict_profile(tmp_path, binding, plan)
    monkeypatch.setattr(adapter.campaign, "load_plan", lambda _state: plan)
    return profile


def _strict_wave_campaign(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    packets, _ = _campaign(tmp_path)
    binding = _wave_binding(tmp_path)
    group = binding["groups"]["group-00"]
    replaced = group["cells"][:2]
    group["cells"][0] = KEY_A
    group["cells"][1] = KEY_B
    group["leader"] = KEY_A
    group["packet_sha256"] = PACKET_SHA
    for key, old in zip((KEY_A, KEY_B), replaced, strict=True):
        binding["cells"][key] = {
            **binding["cells"].pop(old),
            "group": "group-00",
        }
    binding["wave"]["packet_set_sha256"] = adapter._packet_set_sha256(  # noqa: SLF001
        binding["groups"]
    )
    source = adapter._control_source_sha256()  # noqa: SLF001
    targets = []
    for key in (KEY_A, KEY_B):
        target = json.loads(packets[key].read_text())
        target["drivers"] = {phase: {"source_sha256": source} for phase in adapter.campaign.PHASES}
        _write(packets[key], target)
        targets.append(target)
    binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    return packets, binding, _wave_plan(binding, *targets)


def test_wave_control_digest_binds_all_three_creator_modules() -> None:
    expected = adapter._digest(  # noqa: SLF001
        {
            "evals/campaign.py": "sha256:"
            + adapter.hashlib.sha256(Path(adapter.campaign.__file__).read_bytes()).hexdigest(),
            "evals/fleet/campaign_adapter.py": "sha256:"
            + adapter.hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
            "evals/fleet/heldout_launch.py": "sha256:"
            + adapter.hashlib.sha256(Path(heldout_launch.__file__).read_bytes()).hexdigest(),
        }
    )
    assert adapter._control_source_sha256() == expected  # noqa: SLF001


def test_strict_wave_rejects_partial_binding_and_arbitrary_packet_set(tmp_path: Path) -> None:
    binding = _wave_binding(tmp_path)
    one_group = next(iter(binding["groups"].values()))
    one_key = one_group["cells"][0]
    binding["groups"] = {
        "one-group": {
            **one_group,
            "leader": one_key,
            "cells": [one_key],
        }
    }
    binding["cells"] = {one_key: {**binding["cells"][one_key], "group": "one-group"}}
    binding["wave"]["expected_sessions"] = 1
    binding["wave"]["packet_set_sha256"] = adapter._packet_set_sha256(  # noqa: SLF001
        binding["groups"]
    )
    binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    with pytest.raises(adapter.AdapterError, match="16-group/160-cell"):
        adapter.reserve_wave(
            binding,
            _wave_plan(binding),
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            binding["wave"]["reservation_id"],
            expected_sessions=1,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=tmp_path / "budget-partial",
        )
    assert not (tmp_path / "budget-partial").exists()

    complete = _wave_binding(tmp_path)
    complete["wave"]["packet_set_sha256"] = "sha256:" + "9" * 64
    complete["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in complete.items() if key != "sha256"}
    )
    with pytest.raises(adapter.AdapterError, match="16-group/160-cell"):
        adapter.reserve_wave(
            complete,
            _wave_plan(complete),
            (profile := _strict_profile(tmp_path, complete, _wave_plan(complete))),
            _profile_file_sha256(profile),
            complete["wave"]["reservation_id"],
            expected_sessions=160,
            packet_set_sha256=complete["wave"]["packet_set_sha256"],
            root=tmp_path / "budget-arbitrary",
        )
    assert not (tmp_path / "budget-arbitrary").exists()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "source"])
def test_wave_reservation_requires_the_exact_plan_before_state_write(
    tmp_path: Path, defect: str
) -> None:
    binding = _wave_binding(tmp_path)
    plan = _wave_plan(binding)
    if defect == "missing":
        plan["targets"].pop()
    elif defect == "duplicate":
        plan["targets"][-1] = json.loads(json.dumps(plan["targets"][0]))
    else:
        plan["targets"][0]["drivers"]["rollout"]["source_sha256"] = "sha256:" + "6" * 64
    root = tmp_path / "budget-plan-drift"
    with pytest.raises(adapter.AdapterError, match="wave campaign"):
        adapter.reserve_wave(
            binding,
            plan,
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            binding["wave"]["reservation_id"],
            expected_sessions=160,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=root,
        )
    assert not root.exists()


def test_wave_reservation_counts_existing_reservations_and_every_group_adopts(
    tmp_path: Path,
) -> None:
    binding = _wave_binding(tmp_path)
    binding["budget"]["census_receipt_sha256"] = (
        "sha256:9b2d3bccc2c86c383ef9c7fbbe46ba66226ca2e91b89346c7a7871448e0b00e4"
    )
    binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    root = tmp_path / "budget"
    old_binding = _wave_binding(tmp_path)
    old_binding["schema"] = adapter.SCHEMA
    old_binding.pop("wave")
    old_binding["budget"] = json.loads(json.dumps(binding["budget"]))
    old_group_ids = ["group-00", "group-08", "group-09"]
    old_binding["groups"] = {key: old_binding["groups"][key] for key in old_group_ids}
    old_cells = {
        key: value
        for key, value in old_binding["cells"].items()
        if value["group"] in old_binding["groups"]
    }
    old_binding["cells"] = old_cells
    old_binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in old_binding.items() if key != "sha256"}
    )
    for group_id, group in old_binding["groups"].items():
        adapter._budget(  # noqa: SLF001
            old_binding,
            group_id,
            len(group["cells"]),
            group["packet_sha256"],
            root=root,
            reserve=True,
        )
    old_reserved = sum(len(group["cells"]) for group in old_binding["groups"].values())
    assert old_reserved == 27
    first = adapter.reserve_wave(
        binding,
        _wave_plan(binding),
        (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
        _profile_file_sha256(profile),
        "heldout20-base-step1000-v3",
        expected_sessions=160,
        packet_set_sha256=binding["wave"]["packet_set_sha256"],
        root=root,
    )
    second = adapter.reserve_wave(
        binding,
        _wave_plan(binding),
        (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
        _profile_file_sha256(profile),
        "heldout20-base-step1000-v3",
        expected_sessions=160,
        packet_set_sha256=binding["wave"]["packet_set_sha256"],
        root=root,
    )
    assert first["used"] == 271
    assert first["reserved"] == old_reserved + 160
    assert first["used"] + first["reserved"] == 271 + old_reserved + 160
    assert first["committed_after"] == 271 + old_reserved + 160
    assert first["committed_after"] == 458
    assert first["cap"] - first["committed_after"] == 42
    assert first["wave_count"] == 160
    assert first["replayed"] is False
    assert first["sha256"] == adapter._digest(  # noqa: SLF001
        {key: value for key, value in first.items() if key != "sha256"}
    )
    assert second["replayed"] is True
    assert second["wave_sha256"] == first["wave_sha256"]
    assert second["committed_after"] == first["committed_after"]
    assert second["sha256"] == adapter._digest(  # noqa: SLF001
        {key: value for key, value in second.items() if key != "sha256"}
    )
    reservations = list((root / binding["budget"]["date_utc"] / "reservations").glob("*.json"))
    assert len(reservations) == 4
    assert (
        sum(
            adapter._read(path)["schema"] == adapter.WAVE_RESERVATION_SCHEMA  # noqa: SLF001
            for path in reservations
        )
        == 1
    )
    for group_id, group in binding["groups"].items():
        adopted = adapter._budget(  # noqa: SLF001
            binding,
            group_id,
            len(group["cells"]),
            group["packet_sha256"],
            root=root,
            reserve=True,
        )
        assert adopted == {
            "used": 271,
            "reserved": old_reserved + 160,
            "cap": 500,
        }
    assert len(reservations) == 4


def test_wave_reservation_over_cap_writes_no_reservation(tmp_path: Path) -> None:
    binding = _wave_binding(tmp_path, used=341)
    root = tmp_path / "budget"
    with pytest.raises(adapter.CapacityUnavailable):
        adapter.reserve_wave(
            binding,
            _wave_plan(binding),
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            "heldout20-base-step1000-v3",
            expected_sessions=160,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=root,
        )
    reservations = root / binding["budget"]["date_utc"] / "reservations"
    assert not list(reservations.glob("*.json"))


def test_wave_reservation_conflict_and_partial_overlap_fail_closed(tmp_path: Path) -> None:
    binding = _wave_binding(tmp_path)
    root = tmp_path / "budget"
    day = root / binding["budget"]["date_utc"]
    baseline = {"schema": adapter.BUDGET_SCHEMA, **binding["budget"]}
    baseline = {**baseline, "sha256": adapter._digest(baseline)}  # noqa: SLF001
    adapter._write_once(day / "baseline.json", baseline)  # noqa: SLF001
    reservation = {
        "schema": adapter.RESERVATION_SCHEMA,
        "date_utc": binding["budget"]["date_utc"],
        "group_id": "group-00",
        "count": 13,
        "packet_sha256": binding["groups"]["group-00"]["packet_sha256"],
        "bindings_sha256": binding["sha256"],
    }
    reservation = {**reservation, "sha256": adapter._digest(reservation)}  # noqa: SLF001
    adapter._write_once(day / "reservations" / "partial.json", reservation)  # noqa: SLF001
    with pytest.raises(adapter.AdapterError, match="partial reservation"):
        adapter.reserve_wave(
            binding,
            _wave_plan(binding),
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            "heldout20-base-step1000-v3",
            expected_sessions=160,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=root,
        )


def test_wave_reservation_same_id_with_changed_binding_fails_closed(tmp_path: Path) -> None:
    binding = _wave_binding(tmp_path)
    root = tmp_path / "budget"
    adapter.reserve_wave(
        binding,
        _wave_plan(binding),
        (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
        _profile_file_sha256(profile),
        "heldout20-base-step1000-v3",
        expected_sessions=160,
        packet_set_sha256=binding["wave"]["packet_set_sha256"],
        root=root,
    )
    changed = json.loads(json.dumps(binding))
    changed["groups"]["group-00"]["packet_sha256"] = "sha256:" + "8" * 64
    changed["wave"]["packet_set_sha256"] = adapter._packet_set_sha256(  # noqa: SLF001
        changed["groups"]
    )
    changed["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    with pytest.raises(adapter.AdapterError, match="wave reservation differs"):
        adapter.reserve_wave(
            changed,
            _wave_plan(changed),
            (profile := _strict_profile(tmp_path, changed, _wave_plan(changed))),
            _profile_file_sha256(profile),
            "heldout20-base-step1000-v3",
            expected_sessions=160,
            packet_set_sha256=changed["wave"]["packet_set_sha256"],
            root=root,
        )


def test_wave_reservation_write_failure_leaves_no_final_wave(tmp_path: Path, monkeypatch) -> None:
    binding = _wave_binding(tmp_path)
    root = tmp_path / "budget"
    real_write_once = adapter._write_once  # noqa: SLF001

    def fail_wave(path: Path, value: dict) -> None:
        if value.get("schema") == adapter.WAVE_RESERVATION_SCHEMA:
            raise OSError("injected before durable publish")
        real_write_once(path, value)

    monkeypatch.setattr(adapter, "_write_once", fail_wave)
    with pytest.raises(OSError, match="injected"):
        adapter.reserve_wave(
            binding,
            _wave_plan(binding),
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            "heldout20-base-step1000-v3",
            expected_sessions=160,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=root,
        )
    reservations = root / binding["budget"]["date_utc"] / "reservations"
    assert not list(reservations.glob("wave-*.json"))


def test_atomic_writer_never_exposes_final_name_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "state" / "receipt.json"
    monkeypatch.setattr(
        adapter.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected before install")),
    )
    with pytest.raises(OSError, match="before install"):
        adapter._write_once(destination, {"accepted": True})  # noqa: SLF001
    assert not destination.exists()


def test_wave_bound_budget_rejects_forged_partial_coverage(tmp_path: Path) -> None:
    binding = _wave_binding(tmp_path)
    root = tmp_path / "budget"
    day = root / binding["budget"]["date_utc"]
    baseline = {"schema": adapter.BUDGET_SCHEMA, **binding["budget"]}
    baseline = {**baseline, "sha256": adapter._digest(baseline)}  # noqa: SLF001
    adapter._write_once(day / "baseline.json", baseline)  # noqa: SLF001
    forged = {
        "schema": adapter.WAVE_RESERVATION_SCHEMA,
        "date_utc": binding["budget"]["date_utc"],
        "cap": 500,
        "count": 160,
        "reservation_id": binding["wave"]["reservation_id"],
        "packet_set_sha256": binding["wave"]["packet_set_sha256"],
        "groups": [
            {
                "group_id": "group-00",
                "count": 13,
                "packet_sha256": binding["groups"]["group-00"]["packet_sha256"],
            },
            {
                "group_id": "fake-group",
                "count": 147,
                "packet_sha256": "sha256:" + "f" * 64,
            },
        ],
        "bindings_sha256": binding["sha256"],
    }
    forged = {**forged, "sha256": adapter._digest(forged)}  # noqa: SLF001
    adapter._write_once(day / "reservations" / "forged-wave.json", forged)  # noqa: SLF001
    with pytest.raises(adapter.CapacityUnavailable, match="whole-wave reservation"):
        adapter._budget(  # noqa: SLF001
            binding,
            "group-00",
            13,
            binding["groups"]["group-00"]["packet_sha256"],
            root=root,
            reserve=True,
        )


@pytest.mark.parametrize("defect", ["cap", "partition"])
def test_wave_reservation_validates_full_binding_before_state_write(
    tmp_path: Path, defect: str
) -> None:
    binding = _wave_binding(tmp_path)
    if defect == "cap":
        binding["budget"]["cap"] = 600
    else:
        binding["groups"]["group-00"]["cells"] = binding["groups"]["group-00"]["cells"][1:]
    binding["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    root = tmp_path / "budget"
    with pytest.raises(adapter.AdapterError):
        adapter.reserve_wave(
            binding,
            _wave_plan(binding),
            (profile := _strict_profile(tmp_path, binding, _wave_plan(binding))),
            _profile_file_sha256(profile),
            "heldout20-base-step1000-v3",
            expected_sessions=160,
            packet_set_sha256=binding["wave"]["packet_set_sha256"],
            root=root,
        )
    assert not root.exists()


def test_wave_bound_launch_cannot_reach_create_before_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    _patch_source(monkeypatch, bindings)
    monkeypatch.setattr(
        heldout_launch,
        "launch_package_once",
        lambda *_args, **_kwargs: pytest.fail("create reached without whole-wave reservation"),
    )
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
    preview, ready = _phase_files(tmp_path)
    with pytest.raises(adapter.CapacityUnavailable, match="whole-wave reservation"):
        adapter.run_action(
            action="launch",
            phase="rollout",
            packet_path=packets[KEY_A],
            bindings_path=tmp_path / "fleet-bindings.json",
            strict_profile_path=profile,
            strict_profile_file_sha256=_profile_file_sha256(profile),
            context="fleet",
            preview_receipt=preview,
            readiness_receipt=ready,
            cluster=Cluster(),
            database=Database(),
            budget_root=tmp_path / "budget",
            duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
        )


@pytest.mark.parametrize("drift", ["plan", "target", "source"])
def test_wave_bound_control_plane_drift_fails_before_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    target = json.loads(packets[KEY_A].read_text())
    _patch_source(monkeypatch, bindings)
    planned_target = json.loads(json.dumps(target))
    plan_sha256 = bindings["wave"]["campaign_plan_sha256"]
    if drift == "plan":
        plan_sha256 = "sha256:" + "6" * 64
    elif drift == "target":
        planned_target["canary"] = False
    else:
        target["drivers"]["rollout"]["source_sha256"] = "sha256:" + "5" * 64
        _write(packets[KEY_A], target)
    plan["plan_sha256"] = plan_sha256
    plan["targets"] = [
        planned_target if item["experiment_key"] == KEY_A else item for item in plan["targets"]
    ]
    monkeypatch.setattr(adapter.campaign, "load_plan", lambda _state: plan)
    monkeypatch.setattr(
        heldout_launch,
        "preview_package",
        lambda *_args, **_kwargs: pytest.fail("preview reached after control-plane drift"),
    )
    with pytest.raises(adapter.AdapterError, match="control .* differs"):
        adapter.run_action(
            action="preview",
            phase="rollout",
            packet_path=packets[KEY_A],
            bindings_path=tmp_path / "fleet-bindings.json",
            strict_profile_path=profile,
            strict_profile_file_sha256=_profile_file_sha256(profile),
            context="fleet",
            duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
            cluster=Cluster(),
            database=Database(),
        )


def test_strict_wave_launch_rejects_legacy_binding_downgrade_before_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, binding, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, binding, plan)
    downgraded = json.loads(json.dumps(binding))
    downgraded["schema"] = adapter.SCHEMA
    downgraded.pop("wave")
    leader = downgraded["groups"]["group-00"]["leader"]
    downgraded["groups"] = {
        "group-00": {
            **downgraded["groups"]["group-00"],
            "cells": [leader],
        }
    }
    downgraded["cells"] = {leader: downgraded["cells"][leader]}
    downgraded["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in downgraded.items() if key != "sha256"}
    )
    _write(tmp_path / "fleet-bindings.json", downgraded)
    _patch_source(monkeypatch, downgraded)
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
    create_calls = 0

    def launch(*_args: Any, **_kwargs: Any) -> None:
        nonlocal create_calls
        create_calls += 1
        pytest.fail("legacy binding reached source Job create")

    monkeypatch.setattr(
        heldout_launch,
        "launch_package_once",
        launch,
    )
    preview, ready = _phase_files(tmp_path)
    budget_root = tmp_path / "budget"
    with pytest.raises(adapter.AdapterError, match="wave-bound campaign"):
        adapter.run_action(
            action="launch",
            phase="rollout",
            packet_path=packets[KEY_A],
            bindings_path=tmp_path / "fleet-bindings.json",
            strict_profile_path=profile,
            strict_profile_file_sha256=_profile_file_sha256(profile),
            context="fleet",
            preview_receipt=preview,
            readiness_receipt=ready,
            duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
            cluster=Cluster(),
            database=Database(),
            budget_root=budget_root,
        )
    assert create_calls == 0
    assert not budget_root.exists()


def test_strict_wave_action_rejects_coherent_alternate_plan_profile_and_packet_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, binding, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, binding, plan)
    authorized_profile_sha256 = _profile_file_sha256(profile)
    alternate = json.loads(json.dumps(binding))
    alternate["groups"]["group-15"]["packet_sha256"] = "sha256:" + "f" * 64
    alternate["wave"]["packet_set_sha256"] = adapter._packet_set_sha256(  # noqa: SLF001
        alternate["groups"]
    )
    alternate_plan = json.loads(json.dumps(plan))
    alternate_plan["plan_sha256"] = "sha256:" + "6" * 64
    alternate["wave"]["campaign_plan_sha256"] = alternate_plan["plan_sha256"]
    alternate["sha256"] = adapter._digest(  # noqa: SLF001
        {key: value for key, value in alternate.items() if key != "sha256"}
    )
    _write(tmp_path / "fleet-bindings.json", alternate)
    _strict_profile(tmp_path, alternate, alternate_plan)
    monkeypatch.setattr(adapter.campaign, "load_plan", lambda _state: alternate_plan)
    monkeypatch.setattr(
        adapter,
        "_source",
        lambda *_args, **_kwargs: pytest.fail("alternate packet set reached source capture"),
    )
    with pytest.raises(adapter.AdapterError, match="differs from its authorization"):
        adapter.run_action(
            action="preview",
            phase="rollout",
            packet_path=packets[KEY_A],
            bindings_path=tmp_path / "fleet-bindings.json",
            strict_profile_path=profile,
            strict_profile_file_sha256=authorized_profile_sha256,
            context="fleet",
            duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
            cluster=Cluster(),
            database=Database(),
        )


def test_cluster_output_absence_is_bound_to_exact_source_group(tmp_path: Path) -> None:
    _, bindings = _campaign(tmp_path)
    packet = SimpleNamespace(
        job_name="heldout-base-seed46",
        config_map_name="heldout-base-seed46-code",
        output_root="/mnt/sfs/jobs/heldout-base-seed46",
        database="heldout_base_seed46",
        identity_sha256="sha256:" + "4" * 64,
    )
    evidence = {
        "schema": "cyber_fleet_cluster_duplicate_gate_binding_v1",
        "date_utc": datetime.now(UTC).date().isoformat(),
        "bindings_sha256": bindings["sha256"],
        "source_gate_plan_sha256": "sha256:" + "5" * 64,
        "source_gate_terminal_file_sha256": "sha256:" + "6" * 64,
        "source_gate_terminal_sha256": "sha256:" + "7" * 64,
        "source_gate_receipt_sha256": bindings["budget"]["census_receipt_sha256"],
        "packet_set_sha256": "sha256:" + "8" * 64,
        "groups": {
            "base-seed46": {
                "packet_sha256": PACKET_SHA,
                "job_name": packet.job_name,
                "config_map_name": packet.config_map_name,
                "output_root": packet.output_root,
                "database": packet.database,
                "evaluation_identity_sha256": packet.identity_sha256,
                "sfs_output_absent": True,
                "database_absent": True,
            }
        },
    }
    evidence["sha256"] = adapter._digest(evidence)  # noqa: SLF001
    path = _write(tmp_path / "duplicate-gate.json", evidence)
    check, database = adapter._cluster_duplicate_absence(  # noqa: SLF001
        path,
        bindings=bindings,
        group_id="base-seed46",
        package=SimpleNamespace(packet=packet),
    )
    assert check(packet.output_root) is False
    assert database.exists(packet.database) is False
    with pytest.raises(adapter.AdapterError):
        check("/mnt/sfs/jobs/another-output")
    with pytest.raises(adapter.AdapterError):
        database.exists("another_database")


def test_unready_route_defers_without_duplicate_or_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    _patch_source(monkeypatch, bindings)
    preview, _ = _phase_files(tmp_path)
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
    monkeypatch.setattr(
        heldout_launch,
        "duplicate_census",
        lambda *_args, **_kwargs: pytest.fail("duplicate census reached before route readiness"),
    )
    result = adapter.run_action(
        action="ready",
        phase="rollout",
        packet_path=packets[KEY_A],
        bindings_path=tmp_path / "fleet-bindings.json",
        strict_profile_path=profile,
        strict_profile_file_sha256=_profile_file_sha256(profile),
        context="fleet",
        preview_receipt=preview,
        duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
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
                "agent_exit_code": 0,
                "agent_termination": "completed",
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
                "agent_exit_code": 1,
                "agent_termination": "process_error",
            },
            "infrastructure_invalid",
            False,
        ),
        (
            {
                "cell_id": "cell-c",
                "state": "accepted",
                "result_class": "valid",
                "local_results": 1,
                "retry_count": 0,
                "max_retries": 0,
                "receipt_digest": "sha256:" + "7" * 64,
                "agent_exit_code": 0,
                "agent_termination": "output_limit",
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
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    _patch_source(monkeypatch, bindings)
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
    root = packets[KEY_A].resolve().parents[2] / "fleet-source-jobs" / "group-00"
    launch_unsigned = {
        "schema": "cyber_fleet_source_launch_v1",
        "group_id": "group-00",
        "packet_sha256": PACKET_SHA,
        "route_profile_sha256": ROUTE_SHA,
        "job_uid": JOB_UID,
        "config_map_uid": CONFIG_MAP_UID,
        "evaluation_identity_sha256": "sha256:" + "3" * 64,
    }
    _write(root / "launch.json", {**launch_unsigned, "sha256": adapter._digest(launch_unsigned)})
    terminal_unsigned = {
        "schema": heldout_launch.TERMINAL_SCHEMA,
        "job": {"uid": JOB_UID},
        "config_map": {"uid": CONFIG_MAP_UID},
        "evaluation_identity_sha256": "sha256:" + "3" * 64,
        "pods": [],
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
        bindings_path=tmp_path / "fleet-bindings.json",
        strict_profile_path=profile,
        strict_profile_file_sha256=_profile_file_sha256(profile),
        context="fleet",
        launch_receipt=launch,
        duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
        cluster=Cluster(),
        database=Database(row),
    )
    assert result["status"] == expected
    assert result["evidence"]["cleanup_completed"] is cleanup
    assert "score" not in json.dumps(result).lower()


@pytest.mark.parametrize("drift", ["config_map", "pod"])
def test_terminal_observation_rejects_resource_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    packets, bindings, plan = _strict_wave_campaign(tmp_path)
    profile = _patch_strict_contract(monkeypatch, tmp_path, bindings, plan)
    _patch_source(monkeypatch, bindings)
    monkeypatch.setattr(
        adapter,
        "_cluster_duplicate_absence",
        lambda *_args, **_kwargs: (
            lambda _path: False,
            adapter._AbsentDatabase("heldout_base_seed46"),
        ),
    )
    root = packets[KEY_A].resolve().parents[2] / "fleet-source-jobs" / "group-00"
    launch_unsigned = {
        "schema": "cyber_fleet_source_launch_v1",
        "group_id": "group-00",
        "packet_sha256": PACKET_SHA,
        "route_profile_sha256": ROUTE_SHA,
        "job_uid": JOB_UID,
        "config_map_uid": CONFIG_MAP_UID,
        "evaluation_identity_sha256": "sha256:" + "3" * 64,
    }
    _write(root / "launch.json", {**launch_unsigned, "sha256": adapter._digest(launch_unsigned)})
    terminal_unsigned = {
        "schema": heldout_launch.TERMINAL_SCHEMA,
        "job": {"uid": JOB_UID},
        "config_map": {"uid": CONFIG_MAP_UID},
        "evaluation_identity_sha256": "sha256:" + "3" * 64,
        "pods": [],
    }
    _write(
        root / "terminal.json",
        {**terminal_unsigned, "sha256": adapter._digest(terminal_unsigned)},
    )
    launch = _write(
        tmp_path / "cell-launch.json",
        {"receipt_sha256": "sha256:" + "8" * 64, "remote_id": JOB_UID},
    )

    class DriftingCluster(Cluster):
        def __init__(self) -> None:
            super().__init__()
            self.config_map_reads = 0

        def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]:
            if resource == "configmaps":
                self.config_map_reads += 1
                uid = (
                    "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
                    if drift == "config_map" and self.config_map_reads > 1
                    else CONFIG_MAP_UID
                )
                return {"metadata": {"uid": uid}}
            return super().get(resource, namespace, name)

        def list(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            if drift == "pod":
                return {
                    "items": [
                        {
                            "apiVersion": "v1",
                            "kind": "Pod",
                            "metadata": {
                                "name": "replacement-pod",
                                "namespace": "fleet-train-jobs",
                                "uid": "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
                                "labels": {"job-name": "heldout-base-seed46"},
                            },
                            "status": {"phase": "Succeeded"},
                        }
                    ]
                }
            return {"items": []}

    with pytest.raises(adapter.AdapterError, match="identity changed"):
        adapter.run_action(
            action="observe",
            phase="rollout",
            packet_path=packets[KEY_A],
            bindings_path=tmp_path / "fleet-bindings.json",
            strict_profile_path=profile,
            strict_profile_file_sha256=_profile_file_sha256(profile),
            context="fleet",
            launch_receipt=launch,
            duplicate_gate_evidence=tmp_path / "duplicate-gate-evidence.json",
            cluster=DriftingCluster(),
            database=Database(),
        )
