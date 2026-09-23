"""Regressions for the seed-62/63/64 wave with seed 65 deferred."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_protocol_v2_seed62_seed63_seed64_successor as successor
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source


def _predecessor() -> tuple[dict, dict, dict[str, dict]]:
    mapping = [
        (46, 54),
        (47, 55),
        (49, 61),
        (50, 57),
        (51, 58),
        (52, 59),
        (53, 60),
    ]
    definition = {
        "schema": "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v3",
        "protocol_study_id": protocol_v2.PROTOCOL_V2_STUDY_ID,
        "predecessor_comparison_definition_sha256": "sha256:" + "1" * 64,
        "predecessor_migration_receipt_sha256": "sha256:" + "2" * 64,
        "superseded_replacement": {
            "invalid_original_seed": 49,
            "superseded_replacement_seed": 56,
            "successor_seed": 61,
            "whole_pair_excluded": True,
            "evidence_receipt_sha256s": ["sha256:" + "3" * 64],
        },
        "replacement_mapping": [
            {"invalid_original_seed": original, "replacement_seed": replacement}
            for original, replacement in mapping
        ],
        "included_seeds": [48, 54, 55, 57, 58, 59, 60, 61],
        "excluded_protocol_v2_seeds": [56],
        "replica_protocols": [
            {
                "seed": seed,
                "origin": "retained_original" if seed == 48 else "whole_pair_replacement",
                "protocol_id": f"protocol-{seed}",
                "comparison_protocol_sha256": "sha256:" + f"{seed:064x}",
            }
            for seed in (48, 54, 55, 57, 58, 59, 60, 61)
        ],
        "sessions_per_arm": 136,
        "total_sessions": 272,
        "harness": {"harness": "opencode"},
    }
    definition["sha256"] = protocol_v2._canonical(definition)  # noqa: SLF001
    receipt = {
        "sha256": successor.PREDECESSOR_RECEIPT_SHA256,
        "included_seeds": definition["included_seeds"],
    }
    base_template, *_rest = source._inputs()  # noqa: SLF001
    base, candidate = protocol_v2._configs(base_template, 62)  # noqa: SLF001
    contracts = {
        "base": successor._frozen_config_contract(base),  # noqa: SLF001
        "candidate": successor._frozen_config_contract(candidate),  # noqa: SLF001
    }
    return receipt, definition, contracts


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("seed626364")
    live = root / "live.json"
    budget = root / "budget.json"
    absence = root / "absence.json"
    for path in (live, budget, absence):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(successor, "_predecessor", lambda _path: _predecessor())
    monkeypatch.setattr(
        successor,
        "_budget",
        lambda _path: (
            {
                "window_utc": {
                    "start_inclusive": "2026-09-23T00:00:00Z",
                    "end_exclusive": "2026-09-24T00:00:00Z",
                },
                "budget": {
                    "actual_started_rollouts": 196,
                    "daily_cap": 500,
                    "deferred_seed65_pair_additional_cost": 34,
                    "immediate_seed62_seed63_seed64_reserved_not_started": 289,
                    "projected_after_immediate_seed62_seed63_seed64": 485,
                    "projected_with_deferred_seed65_pair": 519,
                    "remaining_after_immediate_seed62_seed63_seed64": 15,
                    "seed65_over_cap_if_launched_same_day": 19,
                    "within_cap_for_immediate_seed62_seed63_seed64": True,
                    "within_cap_with_deferred_seed65_pair": False,
                },
            },
            {
                "path": "budget.json",
                "file_sha256": protocol_v2._file_sha(budget),  # noqa: SLF001
                "sha256": "sha256:" + "5" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        successor,
        "_absence",
        lambda _path: {
            "path": "absence.json",
            "file_sha256": protocol_v2._file_sha(absence),  # noqa: SLF001
            "sha256": "sha256:" + "7" * 64,
        },
    )
    monkeypatch.setattr(
        source.shared,
        "_live_parity",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "sha256:" + "8" * 64,
            "observed_at": "2026-09-23T12:00:00Z",
            "fixed_probe_logit_projection_differs_between_weights": True,
        },
    )
    first = root / "first"
    second = root / "second"
    try:
        for output in (first, second):
            successor.prepare(
                predecessor_packets=root / "unused",
                live_parity=live,
                budget_refresh=budget,
                absence_receipt=absence,
                output=output,
                now=datetime(2026, 9, 23, 12, tzinfo=UTC),
            )
    finally:
        monkeypatch.undo()
    return first, second


def _reseal_job(packet_path: Path, mutate) -> None:
    raw = json.loads(packet_path.read_text(encoding="utf-8"))
    job_path = packet_path.parent / raw["files"]["job"]["path"]
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    mutate(job)
    job_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    raw["files"]["job"]["sha256"] = protocol_v2._file_sha(job_path)  # noqa: SLF001
    source._write_json(packet_path, raw)  # noqa: SLF001


def test_checked_in_seed55_seed57_seed58_seed59_evidence_is_exact() -> None:
    evidence = successor._evidence()  # noqa: SLF001
    assert set(evidence) == {*successor.EVIDENCE_FILES, "seed59_invalid_post_v1"}
    assert all(row["file_sha256"].startswith("sha256:") for row in evidence.values())
    invalid = evidence["seed59_invalid_post_v1"]
    assert invalid["authoritative"] is False
    assert invalid["embedded_self_sha256"] != invalid["canonical_self_sha256"]
    assert evidence["seed58_classification"] == {
        "path": ("docs/evidence/qwen38-study/2026-09-23-q38-dev17-seed58-pair-classification.json"),
        "file_sha256": successor.EVIDENCE_FILES["seed58_classification"][1],
        "sha256": successor.EVIDENCE_FILES["seed58_classification"][2],
    }


def test_live_parity_requires_a_distinct_weight_probe() -> None:
    successor._require_weight_probe_difference(  # noqa: SLF001
        {"fixed_probe_logit_projection_differs_between_weights": True}
    )
    for value in (False, None):
        with pytest.raises(ValueError, match="distinct base and candidate weights"):
            successor._require_weight_probe_difference(  # noqa: SLF001
                {"fixed_probe_logit_projection_differs_between_weights": value}
            )


def test_packet_partition_fails_closed_on_seed65_packet_or_false_readiness() -> None:
    rows = [
        {"successor_seed": seed, "arm_id": arm_id}
        for seed in successor.RENDERED_SUCCESSOR_SEEDS
        for arm_id in protocol_v2.ARMS
    ]
    protocols = {seed: {} for seed in successor.FINAL_SUCCESSOR_SEEDS}
    state = successor._full_comparison_packet_state()  # noqa: SLF001
    successor._validate_packet_partition(rows, protocols, state)  # noqa: SLF001

    ready = {**state, "full_roster_launch_ready": True}
    with pytest.raises(ValueError, match="rendered and deferred"):
        successor._validate_packet_partition(rows, protocols, ready)  # noqa: SLF001

    with pytest.raises(ValueError, match="rendered and deferred"):
        successor._validate_packet_partition(  # noqa: SLF001
            [*rows, {"successor_seed": 65, "arm_id": "base"}], protocols, state
        )

    with pytest.raises(ValueError, match="rendered and deferred"):
        successor._validate_packet_partition(  # noqa: SLF001
            rows, {seed: {} for seed in successor.RENDERED_SUCCESSOR_SEEDS}, state
        )


def test_prepare_is_deterministic_and_preserves_exact_pass8(
    rendered: tuple[Path, Path],
) -> None:
    first, second = rendered
    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
    receipt = json.loads((first / "SUCCESSOR_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["sha256"] == protocol_v2._canonical(  # noqa: SLF001
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert receipt["included_seeds"] == list(successor.INCLUDED_SEEDS)
    assert receipt["excluded_protocol_v2_seeds"] == [55, 56, 57, 58, 59]
    assert receipt["capacity"]["immediate_successor_rollouts"] == 102
    assert receipt["capacity"]["projected_after_immediate_seed62_seed63_seed64"] == 485
    assert receipt["capacity"]["remaining_after_immediate_seed62_seed63_seed64"] == 15
    assert receipt["capacity"]["projected_with_deferred_seed65_pair"] == 519
    assert receipt["capacity"]["seed65_over_cap_if_launched_same_day"] == 19
    assert receipt["capacity"]["seed58_accounting"] == {
        "released_unstarted_reservations": 1,
        "credit_claimed_for_started_excluded_sessions": 0,
    }
    assert receipt["capacity"]["deferred_seed65"] == {
        "rollouts": 34,
        "reserved_in_this_window": False,
        "included_in_projected_after_reservation": False,
        "projected_if_reserved_in_same_window": 519,
        "within_current_utc_cap": False,
        "over_cap_by": 19,
    }
    budget_copy = first / receipt["capacity"]["budget_refresh"]["path"]
    absence_copy = first / receipt["destination_absence"]["path"]
    assert budget_copy.read_bytes() == b"{}\n"
    assert absence_copy.read_bytes() == b"{}\n"
    assert budget_copy.stat().st_mode & 0o777 == 0o600
    assert absence_copy.stat().st_mode & 0o777 == 0o600
    assert receipt["external_mutations"] == 0
    assert receipt["server_preview_performed"] is False
    assert receipt["launch_performed"] is False
    assert len(receipt["replacement_arms"]) == 6
    assert len(receipt["replacement_protocols"]) == 4
    assert [row["seed"] for row in receipt["replacement_protocols"]] == [62, 63, 64, 65]
    for field in (
        "packet_path",
        "job_name",
        "config_map_name",
        "output_root",
        "database",
        "ledger_identity",
    ):
        assert len({row[field] for row in receipt["replacement_arms"]}) == 6
    assert {row["packet_path"] for row in receipt["replacement_arms"]} == {
        f"seed{seed}/{arm}/LAUNCH_PACKET.json"
        for seed in successor.RENDERED_SUCCESSOR_SEEDS
        for arm in protocol_v2.ARMS
    }
    assert len(list(first.rglob("LAUNCH_PACKET.json"))) == 6
    assert not any((first / "seed65" / arm).exists() for arm in protocol_v2.ARMS)
    seed65_protocol = (
        first / "seed65" / ("qwen38-fleet-dev17-seed65-base-step1000-replacement-protocol-v2.json")
    )
    seed65_value = json.loads(seed65_protocol.read_text(encoding="utf-8"))
    assert seed65_value["protocol_id"] == "q38-dev17-s65-base-t3k32s1000-replacement-p1-v2"
    assert seed65_value["sha256"] == (
        "sha256:7ff2c0764b9f6abb8c87e90020ef773849c3674be10be55e0bde113dbf3c918e"
    )
    assert protocol_v2._file_sha(seed65_protocol) == (  # noqa: SLF001
        "sha256:f9ef38fffc005e8bafa874b4cf6e93223777e1bda70249f794a70acdd545aaff"
    )

    definition = receipt["comparison_definition"]
    assert definition["schema"] == successor.COMPARISON_SCHEMA
    assert definition["included_seeds"] == list(successor.INCLUDED_SEEDS)
    assert definition["excluded_protocol_v2_seeds"] == [55, 56, 57, 58, 59]
    assert [
        (row["invalid_original_seed"], row["replacement_seed"])
        for row in definition["replacement_mapping"]
    ] == [(46, 54), (47, 62), (49, 61), (50, 63), (51, 65), (52, 64), (53, 60)]
    assert [row["seed"] for row in definition["replica_protocols"]] == list(
        successor.INCLUDED_SEEDS
    )
    assert definition["sessions_per_arm"] == 136
    assert definition["total_sessions"] == 272
    assert receipt["full_comparison_packet_state"] == successor._full_comparison_packet_state()  # noqa: SLF001
    assert definition["full_comparison_packet_state"] == receipt["full_comparison_packet_state"]
    assert receipt["full_comparison_packet_state"]["full_roster_launch_ready"] is False
    assert receipt["full_comparison_packet_state"]["missing_packet_cells"] == [
        {"seed": 65, "arm_id": "base"},
        {"seed": 65, "arm_id": "candidate"},
    ]
    seed58_lineage = next(
        row
        for row in definition["superseded_replacements"]
        if row["superseded_replacement_seed"] == 58
    )
    assert seed58_lineage == {
        "invalid_original_seed": 51,
        "superseded_replacement_seed": 58,
        "successor_seed": 65,
        "whole_pair_excluded": True,
        "evidence_receipt_sha256s": [successor.EVIDENCE_FILES["seed58_classification"][2]],
    }
    seed59_lineage = next(
        row
        for row in definition["superseded_replacements"]
        if row["superseded_replacement_seed"] == 59
    )
    assert seed59_lineage["evidence_receipt_sha256s"] == [
        successor.EVIDENCE_FILES[key][2] for key in successor.EVIDENCE_DIGEST_KEYS_BY_SEED[59]
    ]
    assert successor.SEED59_INVALID_V1[2] not in seed59_lineage["evidence_receipt_sha256s"]

    rows = {(row["successor_seed"], row["arm_id"]): row for row in receipt["replacement_arms"]}
    task_rosters = []
    for seed in successor.RENDERED_SUCCESSOR_SEEDS:
        for arm in protocol_v2.ARMS:
            packet_path = first / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            package = successor._validate_successor_package(packet_path)  # noqa: SLF001
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            job_path = packet_path.parent / packet["files"]["job"]["path"]
            pod = package.job["spec"]["template"]["spec"]
            labels = package.job["spec"]["template"]["metadata"]["labels"]
            assert pod["automountServiceAccountToken"] is False
            assert pod["priorityClassName"] == "c1"
            assert labels[heldout_launch.POSTGRES_CLIENT_LABEL] == "true"
            assert package.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
            assert "nvidia.com/gpu" not in json.dumps(package.job)
            assert package.packet.identity["sampling_seed"] == seed
            assert package.packet.identity["pass_k"] == 1
            assert package.packet.identity["retry_limit"] == 0
            assert package.evaluation_config["max_reviewed_infrastructure_retries"] == 0
            assert packet["files"]["job"]["sha256"] == protocol_v2._file_sha(job_path)  # noqa: SLF001
            assert rows[(seed, arm)]["packet_file_sha256"] == protocol_v2._file_sha(  # noqa: SLF001
                packet_path
            )
            route = next(iter(package.evaluation_config["routes"].values()))
            task_rosters.append(route["task_versions"])
    assert all(roster == task_rosters[0] for roster in task_rosters)
    assert len(task_rosters[0]) == 17


@pytest.mark.parametrize("value", [None, True])
def test_successor_validation_rejects_missing_or_true_automount(
    rendered: tuple[Path, Path], tmp_path: Path, value: bool | None
) -> None:
    source_dir = rendered[0] / "seed62" / "base"
    directory = tmp_path / "arm"
    shutil.copytree(source_dir, directory)
    packet = directory / "LAUNCH_PACKET.json"

    def mutate(job: dict) -> None:
        pod = job["spec"]["template"]["spec"]
        if value is None:
            pod.pop("automountServiceAccountToken")
        else:
            pod["automountServiceAccountToken"] = value

    _reseal_job(packet, mutate)
    with pytest.raises(ValueError, match="disable service-account token automount"):
        successor._validate_successor_package(packet)  # noqa: SLF001


def test_successor_hardening_is_idempotent_and_rejects_conflicting_true(
    rendered: tuple[Path, Path], tmp_path: Path
) -> None:
    directory = tmp_path / "arm"
    shutil.copytree(rendered[0] / "seed62" / "base", directory)
    packet = directory / "LAUNCH_PACKET.json"
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    successor._harden_and_reseal(packet)  # noqa: SLF001
    after = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    assert before == after

    _reseal_job(
        packet,
        lambda job: job["spec"]["template"]["spec"].__setitem__(
            "automountServiceAccountToken", True
        ),
    )
    with pytest.raises(ValueError, match="conflicting service-account token policy"):
        successor._harden_and_reseal(packet)  # noqa: SLF001


@pytest.mark.parametrize("value", [None, "wrong"])
def test_successor_rejects_missing_or_wrong_postgres_label(
    rendered: tuple[Path, Path], tmp_path: Path, value: str | None
) -> None:
    directory = tmp_path / "arm"
    shutil.copytree(rendered[0] / "seed62" / "base", directory)
    packet = directory / "LAUNCH_PACKET.json"

    def mutate(job: dict) -> None:
        labels = job["spec"]["template"]["metadata"]["labels"]
        if value is None:
            labels.pop(heldout_launch.POSTGRES_CLIENT_LABEL)
        else:
            labels[heldout_launch.POSTGRES_CLIENT_LABEL] = value

    _reseal_job(packet, mutate)
    with pytest.raises(heldout_launch.HeldoutLaunchError, match="PostgreSQL"):
        successor._validate_successor_package(packet)  # noqa: SLF001
