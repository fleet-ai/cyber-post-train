"""Focused no-network tests for the Fleet final8 campaign adapter."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from evals import campaign
from evals.fleet import campaign_adapter as adapter
from evals.fleet import heldout_launch, rollout_postgres

DIGEST = "sha256:" + "a" * 64
JOB_UID = "11111111-2222-4333-8444-555555555555"
CONFIG_MAP_UID = "66666666-7777-4888-8999-aaaaaaaaaaaa"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _reseal_bindings(path: Path, value: dict[str, Any]) -> None:
    value["sha256"] = campaign.digest({key: item for key, item in value.items() if key != "sha256"})
    _write(path, value)


def _set_real_packet_seed(packet: Path, seed: int) -> None:
    raw = json.loads(packet.read_text())
    config_path = packet.parent / raw["files"]["evaluation_config"]["path"]
    protocol_path = packet.parent / raw["files"]["comparison_protocol"]["path"]
    config_map_path = packet.parent / raw["files"]["config_map"]["path"]
    config = json.loads(config_path.read_text())
    config["sampling"]["seed"] = seed
    _write(config_path, config)
    protocol = json.loads(protocol_path.read_text())
    protocol["protocol_id"] = f"heldout-protocol-{seed}"
    protocol["sampling"] = config["sampling"]
    protocol["sha256"] = heldout_launch._canonical_digest(  # noqa: SLF001
        {key: item for key, item in protocol.items() if key != "sha256"}
    )
    _write(protocol_path, protocol)
    config_map = yaml.safe_load(config_map_path.read_text())
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_map["data"]["config.json"] = config_text
    config_map["data"]["heldout.json"] = config_text
    config_map_path.write_text(yaml.safe_dump(config_map), encoding="utf-8")
    raw["files"]["evaluation_config"]["sha256"] = adapter._file_sha256(  # noqa: SLF001
        config_path
    )
    raw["files"]["comparison_protocol"]["sha256"] = adapter._file_sha256(  # noqa: SLF001
        protocol_path
    )
    raw["files"]["config_map"]["sha256"] = adapter._file_sha256(  # noqa: SLF001
        config_map_path
    )
    identity = raw["evaluation_identity"]
    identity["protocol_id"] = protocol["protocol_id"]
    identity["sampling_seed"] = seed
    identity["evaluation_config_sha256"] = raw["files"]["evaluation_config"]["sha256"]
    identity["comparison_protocol_file_sha256"] = raw["files"]["comparison_protocol"]["sha256"]
    identity["comparison_protocol_sha256"] = protocol["sha256"]
    raw["evaluation_identity_sha256"] = heldout_launch._canonical_digest(identity)  # noqa: SLF001
    _write(packet, raw)


class FakeCluster:
    def __init__(self) -> None:
        self.preview_calls = 0

    def server_dry_run(self, namespace, bundle):
        assert namespace == "fleet-train-jobs"
        self.preview_calls += 1
        return copy.deepcopy(bundle)

    def get(self, resource, namespace, name):
        assert (resource, namespace) == ("jobs.batch", "fleet-train-jobs")
        return {
            "metadata": {"name": name, "uid": JOB_UID},
            "status": {"conditions": [{"type": "Complete", "status": "True"}]},
        }


class FakeDatabase:
    def __init__(self, rejected_task: str, retried_task: str | None = None) -> None:
        self.rejected_task = rejected_task
        self.retried_task = retried_task

    def campaign_cell_status(self, _database, *, task_version_id, model_revision, attempt):
        assert model_revision and attempt == 1
        self.last_task_version = task_version_id
        accepted = task_version_id != self.rejected_task
        return {
            "cell_id": DIGEST,
            "state": "accepted" if accepted else "retry_review",
            "result_class": "valid" if accepted else "infrastructure_invalid",
            "receipt_digest": DIGEST if accepted else None,
            "failure_code": None if accepted else "worker_failed",
            "reconciliation_digest": None,
            "local_result_present": accepted,
            "retry_count": int(task_version_id == self.retried_task),
            "max_retries": 0,
        }


def _fixture(tmp_path: Path, monkeypatch):
    model_ids = list(adapter.FINAL8_OPERATIONAL_ARMS)
    targets = sorted(adapter.FINAL8_TASK_VERSIONS)
    target_receipts = {target: DIGEST for target in targets}
    models = {
        model_id: {
            "artifact_id": adapter.FINAL8_OPERATIONAL_ARMS[model_id],
            "checkpoint_id": adapter.FINAL8_OPERATIONAL_ARMS[model_id],
            "weights_sha256": DIGEST,
            "model_revision": f"revision-{model_id}",
            "checkpoint_provenance_sha256": DIGEST,
            "serving_route_proof_sha256": DIGEST,
            "serving_route_receipt_sha256": DIGEST,
            "live_parity_receipt_sha256": DIGEST,
            "selection_artifact_receipt_sha256": adapter.FINAL8_SELECTION_RECEIPTS[model_id],
            "serving_model_id": adapter.FINAL8_SERVING_MODEL_IDS[model_id],
        }
        for model_id in model_ids
    }
    bindings_root = tmp_path / "bindings"
    selection = {
        "schema": "cyber_eval_task_selection_v2",
        "selection_role": "final_test",
        "sha256": adapter.FINAL8_SELECTION_SHA256,
        "tasks": [{"task_version_id": target} for target in targets],
    }
    selection_path = bindings_root / "final8.json"
    _write(selection_path, selection)
    packages = {}
    groups: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    packets: dict[tuple[str, int, str], Path] = {}
    for model_id in model_ids:
        for attempt, seed in enumerate(adapter.ATTEMPT_SEEDS, start=1):
            group_id = f"{model_id.replace('_', '-')}-attempt-{attempt}"
            launch_packet = bindings_root / "packets" / group_id / "packet.json"
            comparison_protocol = launch_packet.parent / "comparison-protocol.json"
            protocol_digest = "sha256:" + f"{seed:064x}"
            _write(launch_packet, {"group": group_id})
            _write(
                comparison_protocol,
                {
                    "model_revisions": {
                        arm: binding["model_revision"] for arm, binding in models.items()
                    },
                    "sha256": protocol_digest,
                },
            )
            launch_identity = {
                "arm_id": model_id,
                "model_revision": models[model_id]["model_revision"],
                "checkpoint_provenance_sha256": DIGEST,
                "serving_route_proof_sha256": DIGEST,
                "comparison_arms": model_ids,
                "comparison_protocol_sha256": protocol_digest,
                "harness": "opencode",
                "harness_version": "1.18.27",
                "context_management": adapter.FINAL8_HARNESS["context_management"],
                "pass_k": 1,
                "retry_limit": 0,
                "sampling_seed": seed,
                "split_manifest_sha256": adapter.FINAL8_SPLIT_SHA256,
            }
            packet = heldout_launch.LaunchPacket(
                path=launch_packet.resolve(),
                namespace="fleet-train-jobs",
                job_name=f"fleet-{model_id}-{attempt}",
                config_map_name=f"fleet-{model_id}-{attempt}-config",
                output_root=f"/mnt/sfs/jobs/fleet-{model_id}-{attempt}",
                database=f"fleet_{model_id.replace('-', '_')}_{attempt}",
                files={
                    "comparison_protocol": comparison_protocol,
                    "task_set": selection_path,
                },
                file_sha256={"split_manifest": adapter.FINAL8_SPLIT_FILE_SHA256},
                identity=launch_identity,
                identity_sha256=campaign.digest(launch_identity),
            )
            job = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": packet.job_name,
                    "namespace": packet.namespace,
                    "labels": {
                        adapter.QUEUE_NAME_LABEL: "training-lq",
                        adapter.QUEUE_PRIORITY_LABEL: "q1",
                    },
                    "annotations": {"fleet.ai/failure-alerts": "off"},
                },
                "spec": {"template": {"spec": {"priorityClassName": "c1"}}},
            }
            packages[launch_packet.resolve()] = heldout_launch.Package(
                packet=packet,
                evaluation_config={
                    "models": {model_id: {"revision": models[model_id]["model_revision"]}},
                    "routes": {
                        model_id: {
                            "catalog": {
                                "engine": "sglang",
                                "precision": "bf16",
                                "tensor_parallel_size": 1,
                            },
                            "served_id": models[model_id]["serving_model_id"],
                            "server_info": {
                                "context_length": 262144,
                                "dp_size": 8,
                                "tp_size": 1,
                                "kv_cache_dtype": "fp8_e4m3",
                                "quantization": None,
                                "reasoning_parser": "qwen3",
                                "tool_call_parser": "qwen3_coder",
                                "load_balance_method": "total_tokens",
                            },
                            "task_versions": targets,
                        }
                    },
                    "harness": adapter.FINAL8_HARNESS,
                    "images": adapter.FINAL8_IMAGES,
                    "sampling": {"seed": seed, "temperature": 0.6, "top_p": 0.95},
                    "concurrency": 8,
                    "pass_k": 1,
                    "max_reviewed_infrastructure_retries": 0,
                },
                config_map={"apiVersion": "v1", "kind": "ConfigMap", "metadata": {}},
                job=job,
            )
            group_cells = []
            for target in targets:
                identity = {
                    "model": {
                        "id": model_id,
                        "checkpoint_id": models[model_id]["checkpoint_id"],
                        "weights_sha256": DIGEST,
                        "matched_treatment_receipt_sha256": DIGEST,
                    },
                    "benchmark": {
                        "id": "fleet-final8",
                        "task_set_sha256": adapter.FINAL8_SELECTION_SHA256,
                        "harness": {"name": "opencode", "identity_receipt_sha256": DIGEST},
                        "scoring_protocol_sha256": DIGEST,
                    },
                    "target": {"id": target, "identity_receipt_sha256": DIGEST},
                    "attempt": attempt,
                    "seed": seed,
                    "sampling": {"temperature": 0.6, "top_p": 0.95},
                    "budgets_sha256": DIGEST,
                }
                key = campaign.digest(identity)
                campaign_packet = {
                    "experiment_key": key,
                    "identity": identity,
                    "matrix_sha256": adapter.FINAL8_MATRIX_SHA256,
                    "canary": False,
                    "serving_evidence": {
                        "serving_route_receipt_sha256": DIGEST,
                        "live_parity_receipt_sha256": DIGEST,
                    },
                }
                path = tmp_path / "campaign" / "targets" / key / "packet.json"
                _write(path, campaign_packet)
                packets[(model_id, attempt, target)] = path
                cells[key] = {
                    "campaign_identity_sha256": key,
                    "group_id": group_id,
                    "model_id": model_id,
                    "task_version_id": target,
                    "attempt": attempt,
                    "canary": False,
                }
                group_cells.append(key)
            groups[group_id] = {
                "comparison_protocol_sha256": protocol_digest,
                "leader_experiment_key": group_cells[0],
                "launch_packet": str(launch_packet.relative_to(bindings_root)),
                "launch_packet_sha256": adapter._file_sha256(launch_packet),  # noqa: SLF001
            }
    unsigned = {
        "schema": adapter.BINDINGS_SCHEMA,
        "matrix_sha256": adapter.FINAL8_MATRIX_SHA256,
        "benchmark_id": "fleet-final8",
        "task_set_sha256": adapter.FINAL8_SELECTION_SHA256,
        "scoring_protocol_sha256": DIGEST,
        "budgets_sha256": DIGEST,
        "matched_treatment_receipt_sha256": DIGEST,
        "split_manifest_file_sha256": adapter.FINAL8_SPLIT_FILE_SHA256,
        "split_manifest_sha256": adapter.FINAL8_SPLIT_SHA256,
        "evidence_boundary_label": adapter.FINAL8_EVIDENCE_BOUNDARY,
        "exposure_audit": adapter.FINAL8_EXPOSURE_AUDIT,
        "harness": {"name": "opencode", "identity_receipt_sha256": DIGEST},
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "attempt_seeds": list(adapter.ATTEMPT_SEEDS),
        },
        "model_bindings": models,
        "target_identity_receipts": target_receipts,
        "groups": groups,
        "cells": cells,
    }
    bindings = {**unsigned, "sha256": campaign.digest(unsigned)}
    bindings_path = bindings_root / "bindings.json"
    _write(bindings_path, bindings)
    monkeypatch.setattr(
        heldout_launch, "build_package", lambda path: packages[Path(path).resolve()]
    )
    monkeypatch.setattr(heldout_launch, "_validate_server_preview", lambda *_: DIGEST)
    return bindings_path, packets, groups


def _terminal(
    receipt_path: Path,
    package,
    job_uid: str,
    config_map_uid: str = CONFIG_MAP_UID,
) -> dict[str, Any]:
    value = {
        "schema": heldout_launch.TERMINAL_SCHEMA,
        "evaluation_identity_sha256": package.packet.identity_sha256,
        "job": {"uid": job_uid},
        "config_map": {"uid": config_map_uid},
        "privacy": {
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "score_values_included": False,
            "credentials_included": False,
        },
    }
    value["sha256"] = heldout_launch._canonical_digest(value)  # noqa: SLF001
    _write(receipt_path, value)
    return value


def test_terminal_receipt_binds_created_job_and_config_map_uids(tmp_path, monkeypatch):
    bindings, _packets, groups = _fixture(tmp_path, monkeypatch)
    group = next(iter(groups.values()))
    package = heldout_launch.build_package(bindings.parent / group["launch_packet"])
    receipt = tmp_path / "terminal.json"

    _terminal(receipt, package, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    with pytest.raises(adapter.FleetCampaignError, match="invalid_fleet_terminal_receipt"):
        adapter._terminal_file(  # noqa: SLF001
            receipt,
            package,
            JOB_UID,
            CONFIG_MAP_UID,
        )

    receipt.unlink()
    _terminal(receipt, package, JOB_UID, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    with pytest.raises(adapter.FleetCampaignError, match="invalid_fleet_terminal_receipt"):
        adapter._terminal_file(  # noqa: SLF001
            receipt,
            package,
            JOB_UID,
            CONFIG_MAP_UID,
        )


def test_final8_groups_create_once_and_isolate_failed_cells(tmp_path, monkeypatch):
    bindings, packets, groups = _fixture(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    target_a, target_b, target_c = sorted(adapter.FINAL8_TASK_VERSIONS)[:3]
    leader = packets[("base", 1, target_a)]
    sibling = packets[("base", 1, target_b)]
    retried_sibling = packets[("base", 1, target_c)]
    assert (
        json.loads(bindings.read_text())["cells"][
            campaign.digest(json.loads(leader.read_text())["identity"])
        ]["group_id"]
        == "base-attempt-1"
    )
    cluster = FakeCluster()
    database = FakeDatabase(rejected_task=target_b, retried_task=target_c)
    create_calls = []

    def launch_once(path, *, validated_package, expected_packet_sha256, **_kwargs):
        assert path == validated_package.packet.path
        assert expected_packet_sha256 == adapter._file_sha256(path)  # noqa: SLF001
        package = validated_package
        create_calls.append(package.packet.path)
        return {
            "job_name": package.packet.job_name,
            "job_uid": JOB_UID,
            "config_map_name": package.packet.config_map_name,
            "config_map_uid": CONFIG_MAP_UID,
        }

    monkeypatch.setattr(heldout_launch, "launch_once", launch_once)
    monkeypatch.setattr(heldout_launch, "duplicate_census", lambda *_args, **_kwargs: {})

    def collect_terminal(path, *, receipt_path, **_kwargs):
        return _terminal(receipt_path, heldout_launch.build_package(path), JOB_UID)

    monkeypatch.setattr(heldout_launch, "collect_terminal", collect_terminal)

    def rollout(path: Path, stem: str):
        root = path.parent / "rollout"
        preview_path, ready_path, launch_path, observe_path = [
            root / f"{name}-{stem}.json" for name in ("preview", "ready", "launch", "observe")
        ]
        preview = adapter.run_action(
            "preview",
            "rollout",
            path,
            preview_path,
            bindings_path=bindings,
            context="prod",
            cluster=cluster,
            database=database,
        )
        assert preview["rendered_objects"][0]["metadata"]["annotations"] == {
            "fleet.ai/failure-alerts": "off"
        }
        assert preview["rendered_objects"][0]["metadata"]["labels"] == {
            adapter.QUEUE_NAME_LABEL: "training-lq",
            adapter.QUEUE_PRIORITY_LABEL: "q1",
        }
        assert preview["rendered_objects"][0]["spec"]["template"]["spec"] == {
            "priorityClassName": "c1"
        }
        adapter.run_action(
            "ready",
            "rollout",
            path,
            ready_path,
            bindings_path=bindings,
            context="prod",
            preview_receipt=preview_path,
            cluster=cluster,
            database=database,
        )
        relative_ready_path = ready_path.relative_to(tmp_path)
        launch = adapter.run_action(
            "launch",
            "rollout",
            path,
            launch_path,
            bindings_path=bindings,
            context="prod",
            preview_receipt=preview_path,
            readiness_receipt=relative_ready_path,
            cluster=cluster,
            database=database,
        )
        assert launch["readiness_receipt_path"] == str(relative_ready_path)
        observed = adapter.run_action(
            "observe",
            "rollout",
            path,
            observe_path,
            bindings_path=bindings,
            context="prod",
            launch_receipt=launch_path,
            cluster=cluster,
            database=database,
        )
        return preview_path, ready_path, launch_path, observe_path, observed

    leader_records = rollout(leader, "leader")
    sibling_records = rollout(sibling, "sibling")
    retried_records = rollout(retried_sibling, "retried")
    assert len(create_calls) == 1
    assert cluster.preview_calls == 2
    assert leader_records[-1]["status"] == "accepted"
    assert sibling_records[-1]["status"] == "infrastructure_invalid"
    assert retried_records[-1]["status"] == "infrastructure_invalid"

    preview_path = leader.parent / "score-preview.json"
    ready_path = leader.parent / "score-ready.json"
    launch_path = leader.parent / "score-launch.json"
    observe_path = leader.parent / "score-observe.json"
    score_preview = adapter.run_action(
        "preview",
        "score",
        leader,
        preview_path,
        bindings_path=bindings,
        context="prod",
        terminal_receipt=leader_records[3],
        cluster=cluster,
        database=database,
    )
    assert score_preview["provider"] == "local"
    adapter.run_action(
        "ready",
        "score",
        leader,
        ready_path,
        bindings_path=bindings,
        context="prod",
        preview_receipt=preview_path,
        cluster=cluster,
        database=database,
    )
    adapter.run_action(
        "launch",
        "score",
        leader,
        launch_path,
        bindings_path=bindings,
        context="prod",
        preview_receipt=preview_path,
        readiness_receipt=ready_path,
        cluster=cluster,
        database=database,
    )
    score_observe = adapter.run_action(
        "observe",
        "score",
        leader,
        observe_path,
        bindings_path=bindings,
        context="prod",
        launch_receipt=launch_path,
        terminal_receipt=leader_records[3],
        cluster=cluster,
        database=database,
    )
    assert score_observe["status"] == "accepted"


def test_bindings_reject_any_seed_other_than_reviewed_pass4(tmp_path, monkeypatch):
    bindings, _packets, _groups = _fixture(tmp_path, monkeypatch)
    value = json.loads(bindings.read_text())
    value["sampling"]["attempt_seeds"][-1] = 50
    value["sha256"] = campaign.digest({key: item for key, item in value.items() if key != "sha256"})
    _write(bindings, value)
    with pytest.raises(adapter.FleetCampaignError, match="invalid_fleet_campaign_bindings"):
        adapter._load_bindings(bindings)  # noqa: SLF001


def test_bindings_require_one_distinct_protocol_per_attempt(tmp_path, monkeypatch):
    bindings, _packets, _groups = _fixture(tmp_path, monkeypatch)
    value = json.loads(bindings.read_text())
    value["groups"]["step1000-attempt-1"]["comparison_protocol_sha256"] = "sha256:" + "f" * 64
    _reseal_bindings(bindings, value)
    with pytest.raises(
        adapter.FleetCampaignError,
        match="attempt_groups_have_different_comparison_protocols",
    ):
        adapter._load_bindings(bindings)  # noqa: SLF001

    bindings, _packets, _groups = _fixture(tmp_path / "same", monkeypatch)
    value = json.loads(bindings.read_text())
    attempt_one = value["groups"]["base-attempt-1"]["comparison_protocol_sha256"]
    for name, group in value["groups"].items():
        if name.endswith("attempt-2"):
            group["comparison_protocol_sha256"] = attempt_one
    _reseal_bindings(bindings, value)
    with pytest.raises(
        adapter.FleetCampaignError,
        match="comparison_protocols_are_not_distinct_per_attempt",
    ):
        adapter._load_bindings(bindings)  # noqa: SLF001


def test_real_launch_packets_bind_distinct_seed_protocols(tmp_path):
    from tests.test_fleet_heldout_launch import (  # noqa: PLC0415
        _enable_rollout_database,
        _packet,
    )

    packets = []
    for seed in (46, 47):
        root = tmp_path / f"seed-{seed}"
        root.mkdir()
        packet = _packet(root)
        _enable_rollout_database(packet)
        _set_real_packet_seed(packet, seed)
        packets.append(packet)
    packages = [heldout_launch.build_package(packet) for packet in packets]
    digests = [package.packet.identity["comparison_protocol_sha256"] for package in packages]
    assert len(set(digests)) == 2

    first_protocol = packets[0].parent / "comparison-protocol.json"
    second_raw = json.loads(packets[1].read_text())
    second_protocol = packets[1].parent / second_raw["files"]["comparison_protocol"]["path"]
    second_protocol.write_bytes(first_protocol.read_bytes())
    second_raw["files"]["comparison_protocol"]["sha256"] = adapter._file_sha256(  # noqa: SLF001
        second_protocol
    )
    _write(packets[1], second_raw)
    with pytest.raises(heldout_launch.HeldoutLaunchError):
        heldout_launch.build_package(packets[1])


def test_create_uses_already_validated_package_without_second_build(tmp_path, monkeypatch):
    from tests.test_fleet_heldout_launch import (  # noqa: PLC0415
        FakeCluster as LaunchCluster,
    )
    from tests.test_fleet_heldout_launch import (
        FakeDatabase as LaunchDatabase,
    )
    from tests.test_fleet_heldout_launch import (
        _enable_rollout_database,
        _packet,
    )

    packet = _packet(tmp_path)
    _enable_rollout_database(packet)
    package = heldout_launch.build_package(packet)
    monkeypatch.setattr(
        heldout_launch,
        "build_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("second build")),
    )
    result = heldout_launch.launch_once(
        packet,
        validated_package=package,
        expected_packet_sha256=adapter._file_sha256(packet),  # noqa: SLF001
        cluster=LaunchCluster(),
        database=LaunchDatabase(),
        journal=tmp_path / "create.jsonl",
        output_exists=lambda _path: False,
    )
    assert result["job_uid"] == JOB_UID
    assert result["config_map_uid"] == CONFIG_MAP_UID


def test_create_rejects_wrong_prevalidated_path_or_digest_before_live_checks(
    tmp_path, monkeypatch
):
    from tests.test_fleet_heldout_launch import (  # noqa: PLC0415
        FakeCluster as LaunchCluster,
    )
    from tests.test_fleet_heldout_launch import (
        FakeDatabase as LaunchDatabase,
    )
    from tests.test_fleet_heldout_launch import (
        _enable_rollout_database,
        _packet,
    )

    packet = _packet(tmp_path)
    _enable_rollout_database(packet)
    package = heldout_launch.build_package(packet)
    other = tmp_path / "other.json"
    other.write_bytes(packet.read_bytes())
    cluster = LaunchCluster()
    for path, packet_sha256 in (
        (other, adapter._file_sha256(other)),  # noqa: SLF001
        (packet, "sha256:" + "0" * 64),
    ):
        with pytest.raises(
            heldout_launch.HeldoutLaunchError,
            match="validated package differs from bound packet",
        ):
            heldout_launch.launch_once(
                path,
                validated_package=package,
                expected_packet_sha256=packet_sha256,
                cluster=cluster,
                database=LaunchDatabase(),
                journal=tmp_path / "create.jsonl",
                output_exists=lambda _path: False,
            )
    assert cluster.list_calls == []
    assert cluster.preview_calls == 0
    assert cluster.create_calls == 0


def test_binding_rejects_peer_revision_drift(tmp_path, monkeypatch):
    bindings, packets, groups = _fixture(tmp_path, monkeypatch)
    protocol_path = (
        bindings.parent / groups["base-attempt-1"]["launch_packet"]
    ).parent / "comparison-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["model_revisions"]["step1000"] = "different-revision"
    _write(protocol_path, protocol)

    packet = packets[("base", 1, sorted(adapter.FINAL8_TASK_VERSIONS)[0])]
    with pytest.raises(
        adapter.FleetCampaignError,
        match="heldout_source_job_differs_from_eight_cell_group",
    ):
        adapter._binding(packet, bindings)  # noqa: SLF001


def test_terminal_rejects_any_private_content_claim(tmp_path, monkeypatch):
    bindings, _packets, groups = _fixture(tmp_path, monkeypatch)
    group = groups["base-attempt-1"]
    package = heldout_launch.build_package(bindings.parent / group["launch_packet"])
    receipt = tmp_path / "terminal-private.json"
    value = _terminal(receipt, package, JOB_UID)
    value["privacy"]["credentials_included"] = True
    value["sha256"] = heldout_launch._canonical_digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "sha256"}
    )
    _write(receipt, value)
    with pytest.raises(adapter.FleetCampaignError, match="invalid_fleet_terminal_receipt"):
        adapter._terminal_file(  # noqa: SLF001
            receipt,
            package,
            JOB_UID,
            CONFIG_MAP_UID,
        )


@pytest.mark.parametrize(
    ("label", "value"),
    (
        (adapter.QUEUE_NAME_LABEL, "other-queue"),
        (adapter.QUEUE_PRIORITY_LABEL, "q0"),
    ),
)
def test_binding_rejects_wrong_queue_or_priority(tmp_path, monkeypatch, label, value):
    bindings, packets, groups = _fixture(tmp_path, monkeypatch)
    group = next(iter(groups.values()))
    package = heldout_launch.build_package(bindings.parent / group["launch_packet"])
    package.job["metadata"]["labels"][label] = value
    packet = packets[("base", 1, sorted(adapter.FINAL8_TASK_VERSIONS)[0])]

    with pytest.raises(
        adapter.FleetCampaignError,
        match="heldout_source_job_differs_from_eight_cell_group",
    ):
        adapter._binding(packet, bindings)  # noqa: SLF001


def test_binding_rejects_other_matrix_or_exposure_claim(tmp_path, monkeypatch):
    bindings, packets, _groups = _fixture(tmp_path, monkeypatch)
    packet = packets[("base", 1, sorted(adapter.FINAL8_TASK_VERSIONS)[0])]
    value = json.loads(packet.read_text())
    value["matrix_sha256"] = DIGEST
    _write(packet, value)
    with pytest.raises(adapter.FleetCampaignError, match="campaign_cell_is_not_bound"):
        adapter._binding(packet, bindings)  # noqa: SLF001

    bindings, _packets, _groups = _fixture(tmp_path / "second", monkeypatch)
    value = json.loads(bindings.read_text())
    value["exposure_audit"] = {**value["exposure_audit"], "sha256": DIGEST}
    value["sha256"] = campaign.digest({key: item for key, item in value.items() if key != "sha256"})
    _write(bindings, value)
    with pytest.raises(adapter.FleetCampaignError, match="bindings_do_not_select_locked_final8"):
        adapter._load_bindings(bindings)  # noqa: SLF001


def test_deferred_sibling_cannot_bypass_group_leader(tmp_path, monkeypatch):
    bindings, packets, _groups = _fixture(tmp_path, monkeypatch)
    sibling_target = sorted(adapter.FINAL8_TASK_VERSIONS)[1]
    packet = packets[("base", 1, sibling_target)]
    cluster = FakeCluster()
    database = FakeDatabase(rejected_task="none")
    preview_path = packet.parent / "preview.json"
    ready_path = packet.parent / "ready.json"
    adapter.run_action(
        "preview",
        "rollout",
        packet,
        preview_path,
        bindings_path=bindings,
        context="prod",
        cluster=cluster,
        database=database,
    )
    ready = adapter.run_action(
        "ready",
        "rollout",
        packet,
        ready_path,
        bindings_path=bindings,
        context="prod",
        preview_receipt=preview_path,
        cluster=cluster,
        database=database,
    )
    assert ready["status"] == "deferred_not_ready"
    with pytest.raises(adapter.FleetCampaignError, match="previous_campaign_receipt_mismatch"):
        adapter.run_action(
            "launch",
            "rollout",
            packet,
            packet.parent / "launch.json",
            bindings_path=bindings,
            context="prod",
            preview_receipt=preview_path,
            readiness_receipt=ready_path,
            cluster=cluster,
            database=database,
        )


def test_postgres_cell_status_reads_presence_but_not_score(monkeypatch):
    class Result:
        def __init__(self, *, rows=None, row=None):
            self.rows = rows
            self.row = row

        def fetchall(self):
            return self.rows

        def fetchone(self):
            return self.row

    class Connection:
        def execute(self, query, parameters):
            if "FROM rollout_cells" in query:
                assert parameters == ("task-version", "model-revision")
                assert "score" not in query.casefold()
                return Result(
                    rows=[
                        {
                            "cell_id": DIGEST,
                            "state": "accepted",
                            "result_class": "valid",
                            "receipt_digest": DIGEST,
                            "failure_code": None,
                            "reconciliation_digest": None,
                            "retry_count": 0,
                            "max_retries": 0,
                        }
                    ]
                )
            assert "SELECT EXISTS" in query and parameters == (DIGEST,)
            return Result(row={"present": True})

    @contextmanager
    def read_transaction(_dsn):
        yield Connection()

    monkeypatch.setattr(rollout_postgres, "_read_transaction", read_transaction)
    monkeypatch.setenv("TEST_ROLLOUT_DATABASE_URL", "postgresql://example/original")
    status = adapter._campaign_cell_status(  # noqa: SLF001
        object(),
        "database",
        dsn_env="TEST_ROLLOUT_DATABASE_URL",
        task_version_id="task-version",
        model_revision="model-revision",
    )
    assert status["local_result_present"] is True
    assert status["retry_count"] == status["max_retries"] == 0
    assert "score" not in json.dumps(status)
