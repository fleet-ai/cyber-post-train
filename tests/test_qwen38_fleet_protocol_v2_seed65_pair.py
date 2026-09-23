"""Regressions for the deferred, unsealed seed-65/66 matched pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_protocol_v2_seed62_seed63_seed64_successor as successor
from scripts import prepare_qwen38_fleet_protocol_v2_seed65_pair as seed65
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source

RETIREMENT = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/qwen38-study/2026-09-23-q38-dev17-seed60-whole-pair-retirement.json"
)


def _runtime_files() -> dict[str, str]:
    runtime_root = Path(seed65.__file__).resolve().parents[1] / "evals" / "fleet"
    return {
        name: hashlib.sha256((runtime_root / name).read_bytes()).hexdigest()
        for name in protocol_v2.SEALED_RUNTIME_IDENTITY_FILES
    }


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("seed65")
    proof = root / "proof.json"
    proof.write_text("{}\n", encoding="utf-8")
    base_template, *_rest = source._inputs()  # noqa: SLF001
    base, candidate = protocol_v2._configs(  # noqa: SLF001
        base_template, seed65.FROZEN_DEFERRED_SEED
    )
    protocol = protocol_v2._protocol(base, seed65.FROZEN_DEFERRED_SEED)  # noqa: SLF001
    frozen_definition = {
        "sha256": seed65.FROZEN_DEFINITION_SHA256,
        "excluded_protocol_v2_seeds": [55, 56, 57, 58, 59],
        "replacement_mapping": [{"invalid_original_seed": 53, "replacement_seed": 60}],
        "replica_protocols": [
            {
                "comparison_protocol_sha256": (
                    protocol["sha256"] if seed == 65 else f"sha256:protocol-{seed}"
                ),
                "origin": "whole_pair_successor",
                "protocol_id": (protocol["protocol_id"] if seed == 65 else f"protocol-seed-{seed}"),
                "seed": seed,
            }
            for seed in seed65.FROZEN_INCLUDED_SEEDS
        ],
        "superseded_replacements": [],
    }
    contracts = {
        "base": {
            "config": successor._frozen_config_contract(base),  # noqa: SLF001
            "runtime_files_sha256": _runtime_files(),
        },
        "candidate": {
            "config": successor._frozen_config_contract(candidate),  # noqa: SLF001
            "runtime_files_sha256": _runtime_files(),
        },
    }
    frozen = root / "frozen"
    frozen.mkdir()
    source._write_json(  # noqa: SLF001
        frozen / "SUCCESSOR_RECEIPT.json", {"sha256": seed65.FROZEN_RECEIPT_SHA256}
    )
    source._write_json(  # noqa: SLF001
        frozen / "COMPARISON_DEFINITION.json", {"sha256": seed65.FROZEN_DEFINITION_SHA256}
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        seed65,
        "_frozen_source",
        lambda _path: (
            {"sha256": seed65.FROZEN_RECEIPT_SHA256, "replacement_arms": []},
            frozen_definition,
            protocol,
            contracts,
            proof,
        ),
    )
    monkeypatch.setattr(
        seed65,
        "_validate_frozen_parity",
        lambda *_args: {"receipt_sha256": "sha256:" + "8" * 64},
    )
    first = root / "first"
    second = root / "second"
    try:
        seed65.prepare(frozen_packets=frozen, seed60_retirement=RETIREMENT, output=first)
        seed65.prepare(frozen_packets=frozen, seed60_retirement=RETIREMENT, output=second)
    finally:
        monkeypatch.undo()
    return first, second


def test_prepare_is_deterministic_and_stays_unsealed(rendered: tuple[Path, Path]) -> None:
    first, second = rendered
    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
    receipt = json.loads(
        (first / "SEED65_SEED66_PREPARATION_RECEIPT_V2.json").read_text(encoding="utf-8")
    )
    assert receipt["sha256"] == protocol_v2._canonical(  # noqa: SLF001
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert receipt["included_seeds"] == list(seed65.INCLUDED_SEEDS)
    assert receipt["scientific_identity"] == {
        "task_count_per_arm": 17,
        "comparison_arms": ["base", "candidate"],
        "pass_k": 1,
        "retry_limit": 0,
        "harness": "opencode",
        "sampling_seeds": [65, 66],
        "rollouts_requiring_fresh_reservation": 68,
        "frozen_parity_file_sha256": seed65._file_sha(  # noqa: SLF001
            first / "seed65" / "base" / "serving-route-proof.json"
        ),
        "frozen_parity_receipt_sha256": "sha256:" + "8" * 64,
    }
    assert receipt["launch_readiness"] == {
        "state": "prepared_unsealed_next_utc_gates_required",
        "launch_ready": False,
        "current_utc_capacity_claimed": False,
        "destination_absence_checked": False,
        "fresh_live_parity_checked": False,
        "server_preview_performed": False,
        "required_before_create": [
            "fresh_next_utc_global_budget_census",
            "fresh_all_dimension_destination_absence",
            "fresh_content_free_live_parity",
            "two_identical_server_previews",
            "exact_head_ci_and_independent_review",
        ],
        "packet_bytes_may_change_at_launch": False,
    }
    assert receipt["provider_requests"] == 0
    assert receipt["external_mutations"] == 0
    assert receipt["server_preview_performed"] is False
    assert receipt["launch_performed"] is False
    assert "actual_started_rollouts" not in json.dumps(receipt)
    assert len(receipt["arms"]) == 4


def test_seed65_seed66_inner_and_outer_jobs_preserve_the_exact_policy(
    rendered: tuple[Path, Path],
) -> None:
    first, _second = rendered
    receipt = json.loads(
        (first / "SEED65_SEED66_PREPARATION_RECEIPT_V2.json").read_text(encoding="utf-8")
    )
    task_rosters = []
    for seed in seed65.DEFERRED_SEEDS:
        for arm in protocol_v2.ARMS:
            packet_path = first / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            package = successor._validate_successor_package(packet_path)  # noqa: SLF001
            binding = protocol_v2._evaluation_binding(packet_path)  # noqa: SLF001
            identity = package.packet.identity
            job = package.job
            pod = job["spec"]["template"]["spec"]
            labels = job["spec"]["template"]["metadata"]["labels"]
            assert identity["sampling_seed"] == seed
            assert identity["arm_id"] == arm
            assert identity["harness"] == "opencode"
            assert identity["pass_k"] == 1
            assert identity["retry_limit"] == 0
            assert binding["runtime_files_sha256"] == _runtime_files()
            assert pod["automountServiceAccountToken"] is False
            assert pod["priorityClassName"] == "c1"
            assert labels[heldout_launch.POSTGRES_CLIENT_LABEL] == "true"
            assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
            assert "nvidia.com/gpu" not in json.dumps(job)
            route = next(iter(package.evaluation_config["routes"].values()))
            task_rosters.append(route["task_versions"])
    assert all(roster == task_rosters[0] for roster in task_rosters)
    assert len(task_rosters[0]) == 17

    bundle = yaml.safe_load((first / "launchers.yaml").read_text(encoding="utf-8"))
    jobs = [item for item in bundle["items"] if item["kind"] == "Job"]
    config_maps = [item for item in bundle["items"] if item["kind"] == "ConfigMap"]
    assert len(jobs) == len(config_maps) == 4
    assert len({item["metadata"]["name"] for item in bundle["items"]}) == 4
    for job in jobs:
        annotations = job["metadata"]["annotations"]
        labels = job["spec"]["template"]["metadata"]["labels"]
        assert annotations[heldout_launch.FAILURE_ALERT_ANNOTATION] == "off"
        assert annotations[heldout_launch.CREATE_ONCE_ANNOTATION] == "true"
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        assert labels[heldout_launch.POSTGRES_CLIENT_LABEL] == "true"
        assert "nvidia.com/gpu" not in json.dumps(job)
    assert receipt["launcher_bundle"]["object_count"] == 8


def test_roster_v2_excludes_seed60_and_binds_the_exact_retirement(
    rendered: tuple[Path, Path], tmp_path: Path
) -> None:
    first, _second = rendered
    receipt = json.loads(
        (first / "SEED65_SEED66_PREPARATION_RECEIPT_V2.json").read_text(encoding="utf-8")
    )
    definition_path = first / "COMPARISON_DEFINITION_V6.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    assert definition["sha256"] == protocol_v2._canonical(  # noqa: SLF001
        {key: value for key, value in definition.items() if key != "sha256"}
    )
    assert definition["included_seeds"] == list(seed65.INCLUDED_SEEDS)
    assert 60 not in definition["included_seeds"]
    assert 60 in definition["excluded_protocol_v2_seeds"]
    replica_seeds = {row["seed"] for row in definition["replica_protocols"]}
    assert replica_seeds == set(seed65.INCLUDED_SEEDS)
    assert (
        next(
            row for row in definition["replacement_mapping"] if row["invalid_original_seed"] == 53
        )["replacement_seed"]
        == 66
    )
    retirement = definition["seed60_pair_retirement_evidence"]
    assert retirement == {
        "file_sha256": seed65.SEED60_RETIREMENT_FILE_SHA256,
        "sha256": seed65.SEED60_RETIREMENT_SHA256,
        "whole_pair_excluded": True,
        "replacement_seed": 66,
    }
    superseded = definition["superseded_replacements"][-1]
    assert superseded["superseded_replacement_seed"] == 60
    assert superseded["successor_seed"] == 66
    assert superseded["whole_pair_excluded"] is True
    assert seed65.SEED60_RETIREMENT_SHA256 in superseded["evidence_receipt_sha256s"]
    assert receipt["comparison_definition"] == {
        "path": definition_path.name,
        "file_sha256": seed65._file_sha(definition_path),  # noqa: SLF001
        "sha256": definition["sha256"],
    }
    assert receipt["seed60_pair_retirement"] == {
        "path": "evidence/SEED60_RETIREMENT_RECEIPT.json",
        "file_sha256": seed65.SEED60_RETIREMENT_FILE_SHA256,
        "sha256": seed65.SEED60_RETIREMENT_SHA256,
        "whole_pair_excluded": True,
        "replacement_seed": 66,
    }
    assert {row["seed"] for row in receipt["replacement_protocols"]} == {65, 66}
    assert {(row["seed"], row["arm_id"]) for row in receipt["arms"]} == {
        (seed, arm) for seed in seed65.DEFERRED_SEEDS for arm in protocol_v2.ARMS
    }

    drifted = tmp_path / "retirement.json"
    value = json.loads(RETIREMENT.read_text(encoding="utf-8"))
    value["lineage"]["replacement_seed"] = 67
    source._write_json(drifted, value)  # noqa: SLF001
    with pytest.raises(ValueError, match="retirement"):
        seed65._seed60_retirement(drifted)  # noqa: SLF001


def test_frozen_source_requires_exact_bytes_and_complete_prior_pair_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    protocol_dir = root / "seed65"
    protocol_dir.mkdir()
    protocol = {
        "protocol_id": "q38-dev17-s65-base-t3k32s1000-replacement-p1-v2",
    }
    protocol["sha256"] = protocol_v2._canonical(protocol)  # noqa: SLF001
    source._write_json(  # noqa: SLF001
        protocol_dir / "qwen38-fleet-dev17-seed65-base-step1000-replacement-protocol-v2.json",
        protocol,
    )
    definition = {"included_seeds": list(seed65.FROZEN_INCLUDED_SEEDS)}
    definition["sha256"] = protocol_v2._canonical(definition)  # noqa: SLF001
    source._write_json(root / "COMPARISON_DEFINITION.json", definition)  # noqa: SLF001
    rows = [
        {
            "successor_seed": prior_seed,
            "arm_id": arm,
            "packet_file_sha256": f"sha256:packet-{prior_seed}-{arm}",
            "evaluation_identity_sha256": f"sha256:identity-{prior_seed}-{arm}",
        }
        for prior_seed in seed65.PRIOR_RENDERED_SEEDS
        for arm in protocol_v2.ARMS
    ]
    receipt = {
        "comparison_definition": definition,
        "included_seeds": list(seed65.FROZEN_INCLUDED_SEEDS),
        "external_mutations": 0,
        "server_preview_performed": False,
        "launch_performed": False,
        "full_comparison_packet_state": {
            "rendered_successor_seeds": list(seed65.PRIOR_RENDERED_SEEDS),
            "deferred_successor_seeds": [65],
            "missing_packet_cells": [{"seed": 65, "arm_id": arm} for arm in protocol_v2.ARMS],
            "full_roster_launch_ready": False,
        },
        "replacement_protocols": [
            {
                "seed": 65,
                "protocol_id": protocol["protocol_id"],
                "sha256": protocol["sha256"],
                "file_sha256": "pending",
            }
        ],
        "replacement_arms": rows,
    }
    receipt["sha256"] = protocol_v2._canonical(receipt)  # noqa: SLF001
    source._write_json(root / "SUCCESSOR_RECEIPT.json", receipt)  # noqa: SLF001
    receipt["replacement_protocols"][0]["file_sha256"] = seed65._file_sha(  # noqa: SLF001
        protocol_dir / "qwen38-fleet-dev17-seed65-base-step1000-replacement-protocol-v2.json"
    )
    receipt["sha256"] = protocol_v2._canonical(  # noqa: SLF001
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    source._write_json(root / "SUCCESSOR_RECEIPT.json", receipt)  # noqa: SLF001
    proof = tmp_path / "proof.json"
    proof.write_text("{}\n", encoding="utf-8")
    base_template, *_rest = source._inputs()  # noqa: SLF001
    base, candidate = protocol_v2._configs(base_template, 62)  # noqa: SLF001
    configs = {"base": base, "candidate": candidate}

    monkeypatch.setattr(
        seed65, "FROZEN_RECEIPT_FILE_SHA256", seed65._file_sha(root / "SUCCESSOR_RECEIPT.json")
    )  # noqa: SLF001
    monkeypatch.setattr(seed65, "FROZEN_RECEIPT_SHA256", receipt["sha256"])
    monkeypatch.setattr(
        seed65,
        "FROZEN_DEFINITION_FILE_SHA256",
        seed65._file_sha(root / "COMPARISON_DEFINITION.json"),
    )  # noqa: SLF001
    monkeypatch.setattr(seed65, "FROZEN_DEFINITION_SHA256", definition["sha256"])
    monkeypatch.setattr(
        seed65, "FROZEN_PROTOCOL_FILE_SHA256", receipt["replacement_protocols"][0]["file_sha256"]
    )
    monkeypatch.setattr(seed65, "FROZEN_PROTOCOL_SHA256", protocol["sha256"])

    def fake_package(packet_path: Path) -> SimpleNamespace:
        prior_seed = int(packet_path.parts[-3].removeprefix("seed"))
        arm = packet_path.parts[-2]
        return SimpleNamespace(
            evaluation_config=configs[arm],
            packet=SimpleNamespace(
                identity={
                    "sampling_seed": prior_seed,
                    "arm_id": arm,
                    "harness": "opencode",
                    "pass_k": 1,
                    "retry_limit": 0,
                },
                files={"serving_route_proof": proof},
            ),
        )

    monkeypatch.setattr(successor, "_validate_successor_package", fake_package)

    def fake_binding(packet_path: Path) -> dict[str, object]:
        prior_seed = packet_path.parts[-3].removeprefix("seed")
        arm = packet_path.parts[-2]
        return {
            "packet_file_sha256": f"sha256:packet-{prior_seed}-{arm}",
            "evaluation_identity_sha256": f"sha256:identity-{prior_seed}-{arm}",
            "runtime_files_sha256": _runtime_files(),
        }

    monkeypatch.setattr(protocol_v2, "_evaluation_binding", fake_binding)
    _receipt, _definition, _protocol, contracts, actual_proof = seed65._frozen_source(root)  # noqa: SLF001
    assert set(contracts) == set(protocol_v2.ARMS)
    assert actual_proof == proof

    receipt["full_comparison_packet_state"]["full_roster_launch_ready"] = True
    receipt["sha256"] = protocol_v2._canonical(  # noqa: SLF001
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    source._write_json(root / "SUCCESSOR_RECEIPT.json", receipt)  # noqa: SLF001
    monkeypatch.setattr(
        seed65, "FROZEN_RECEIPT_FILE_SHA256", seed65._file_sha(root / "SUCCESSOR_RECEIPT.json")
    )  # noqa: SLF001
    monkeypatch.setattr(seed65, "FROZEN_RECEIPT_SHA256", receipt["sha256"])
    with pytest.raises(ValueError, match="frozen seed-62/63/64 successor bytes differ"):
        seed65._frozen_source(root)  # noqa: SLF001
