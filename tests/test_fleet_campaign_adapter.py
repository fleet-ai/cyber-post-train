"""Focused no-network tests for the Fleet final8 campaign adapter."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from evals import campaign
from evals.fleet import campaign_adapter as adapter
from evals.fleet import heldout_launch, rollout_postgres

DIGEST = "sha256:" + "a" * 64
JOB_UID = "11111111-2222-4333-8444-555555555555"
CONFIG_MAP_UID = "66666666-7777-4888-8999-aaaaaaaaaaaa"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


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
    def __init__(self, rejected_task: str) -> None:
        self.rejected_task = rejected_task

    def cell_status(self, _database, *, task_version_id, model_revision, attempt):
        assert model_revision and attempt == 1
        accepted = task_version_id != self.rejected_task
        return {
            "cell_id": DIGEST,
            "state": "accepted" if accepted else "retry_review",
            "result_class": "valid" if accepted else "infrastructure_invalid",
            "receipt_digest": DIGEST if accepted else None,
            "failure_code": None if accepted else "worker_failed",
            "reconciliation_digest": None,
            "local_result_present": accepted,
        }


def _fixture(tmp_path: Path, monkeypatch):
    model_ids = ["base", "candidate-a", "candidate-b", "candidate-c", "candidate-d", "candidate-e"]
    targets = sorted(adapter.FINAL8_TASK_VERSIONS)
    target_receipts = {target: DIGEST for target in targets}
    models = {
        model_id: {
            "checkpoint_id": f"checkpoint-{model_id}",
            "weights_sha256": DIGEST,
            "model_revision": f"revision-{model_id}",
            "checkpoint_provenance_sha256": DIGEST,
            "serving_route_proof_sha256": DIGEST,
            "serving_route_receipt_sha256": DIGEST,
            "live_parity_receipt_sha256": DIGEST,
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
            group_id = f"{model_id}-attempt-{attempt}"
            launch_packet = bindings_root / "packets" / group_id / "packet.json"
            _write(launch_packet, {"group": group_id})
            launch_identity = {
                "arm_id": model_id,
                "model_revision": models[model_id]["model_revision"],
                "checkpoint_provenance_sha256": DIGEST,
                "serving_route_proof_sha256": DIGEST,
                "comparison_arms": model_ids,
                "comparison_protocol_sha256": DIGEST,
                "harness": "opencode",
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
                files={"task_set": selection_path},
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
                    "annotations": {"fleet.ai/failure-alerts": "off"},
                },
                "spec": {"template": {"spec": {"priorityClassName": "c1"}}},
            }
            packages[launch_packet.resolve()] = heldout_launch.Package(
                packet=packet,
                evaluation_config={
                    "routes": {model_id: {"task_versions": targets}},
                    "sampling": {"seed": seed, "temperature": 0.6, "top_p": 0.95},
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
                    "canary": target == targets[0],
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
                    "canary": target == targets[0],
                }
                group_cells.append(key)
            groups[group_id] = {
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
        "fleet_comparison_protocol_sha256": DIGEST,
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


def _terminal(receipt_path: Path, package, job_uid: str) -> dict[str, Any]:
    value = {
        "schema": heldout_launch.TERMINAL_SCHEMA,
        "evaluation_identity_sha256": package.packet.identity_sha256,
        "job": {"uid": job_uid},
        "privacy": {"score_values_included": False},
    }
    value["sha256"] = heldout_launch._canonical_digest(value)  # noqa: SLF001
    _write(receipt_path, value)
    return value


def test_final8_groups_create_once_and_isolate_failed_cells(tmp_path, monkeypatch):
    bindings, packets, groups = _fixture(tmp_path, monkeypatch)
    target_a, target_b = sorted(adapter.FINAL8_TASK_VERSIONS)[:2]
    leader = packets[("base", 1, target_a)]
    sibling = packets[("base", 1, target_b)]
    assert (
        json.loads(bindings.read_text())["cells"][
            campaign.digest(json.loads(leader.read_text())["identity"])
        ]["group_id"]
        == "base-attempt-1"
    )
    cluster = FakeCluster()
    database = FakeDatabase(rejected_task=target_b)
    create_calls = []

    def launch_once(path, **_kwargs):
        create_calls.append(path)
        package = heldout_launch.build_package(path)
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
        adapter.run_action(
            "launch",
            "rollout",
            path,
            launch_path,
            bindings_path=bindings,
            context="prod",
            preview_receipt=preview_path,
            readiness_receipt=ready_path,
            cluster=cluster,
            database=database,
        )
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
    assert len(create_calls) == 1
    assert cluster.preview_calls == 2
    assert leader_records[-1]["status"] == "accepted"
    assert sibling_records[-1]["status"] == "infrastructure_invalid"

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
                assert parameters == ("task-version", "model-revision", 1)
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
                        }
                    ]
                )
            assert "SELECT EXISTS" in query and parameters == (DIGEST,)
            return Result(row={"present": True})

    @contextmanager
    def read_transaction(_dsn):
        yield Connection()

    monkeypatch.setattr(rollout_postgres, "_read_transaction", read_transaction)
    status = rollout_postgres.cell_status(
        "postgresql://example/database",
        task_version_id="task-version",
        model_revision="model-revision",
        attempt=1,
    )
    assert status["local_result_present"] is True
    assert "score" not in json.dumps(status)
