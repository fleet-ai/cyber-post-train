from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from cyber_post_train import public_eval_import
from evals.fleet import final_pass8_aggregate as final
from scripts import render_qwen38_fleet_pass8_final_aggregate as renderer

ROOT = Path(__file__).resolve().parents[1]


class FakeSnapshot:
    def __init__(self, gate: final.GateSnapshot, scored: list[dict[str, Any]]) -> None:
        self.value = gate
        self.scored = scored
        self.score_reads = 0

    def gate(self) -> final.GateSnapshot:
        return self.value

    def scored_results(self) -> list[dict[str, Any]]:
        self.score_reads += 1
        return copy.deepcopy(self.scored)


def _signed(value: dict[str, Any], field: str = "sha256") -> dict[str, Any]:
    return {**value, field: final._digest(value)}  # noqa: SLF001


def _plan(tmp_path: Path) -> dict[str, Any]:
    plan = final.build_current_study_plan(
        task_set_path=renderer.TASK_SET,
        roster_path=renderer.ROSTER,
        base_config_path=renderer.BASE_CONFIG,
    )
    plan = copy.deepcopy(plan)
    for replica in plan["replicas"]:
        root = tmp_path / f"s{replica['seed']}-{replica['arm']}"
        root.mkdir()
        replica["output_root"] = str(root)
        replica["terminal_receipt_path"] = str(root / "TERMINAL_OBSERVATION.json")
    plan["private_output_root"] = str(tmp_path / "final")
    plan.pop("sha256")
    return _signed(plan)


def _evaluation(plan: dict[str, Any], replica: dict[str, Any]) -> dict[str, Any]:
    arm = plan["arms"][replica["arm"]]
    value = {
        "schema": "cyber_fleet_eval_v1",
        "campaign_id": replica["experiment_id"],
        "tasks": sorted(plan["tasks"], key=lambda row: row["task_version_id"]),
        "models": {arm["model_id"]: arm["model"]},
        "routes": {arm["serving_block"]: arm["route"]},
        "treatment": plan["harness"],
        "images": plan["images"],
        "sampling": {**plan["sampling"], "seed": replica["seed"]},
        "pass_k": 1,
        "automatic_retry": False,
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": False,
        "runtime_files": {
            name: f"{index + 1:064x}" for index, name in enumerate(sorted(final.RUNTIME_FILES))
        },
    }
    value["sha256"] = final._plain_digest(value)  # noqa: SLF001
    return value


def _local_record(cell_id: str, session_id: str, score: float, index: int) -> dict[str, Any]:
    digest = f"{index + 1:064x}"[-64:]
    value = {
        "execution_id": "sha256:" + f"{index + 300:064x}"[-64:],
        "cell_id": cell_id,
        "execution_generation": 1,
        "run_id": f"run-{index}",
        "session_id": session_id,
        "verifier_execution_id": f"verifier-{index}",
        "score": score,
        "config_sha256": digest,
        "artifact_directory": f"attempts/{index}",
        "trace_path": "trace.json",
        "trace_sha256": digest,
        "result_path": "result.json",
        "result_sha256": digest,
        "reward_path": "reward-result.json",
        "reward_sha256": digest,
        "session_ingest_path": "session-ingest.json",
        "session_ingest_sha256": digest,
        "cleanup_path": "cleanup.json",
        "cleanup_sha256": digest,
        "session_ingest_status": "completed",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "elapsed_seconds": 10.0,
    }
    return {**value, "record_sha256": final._plain_digest(value)}  # noqa: SLF001


def _study(tmp_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], FakeSnapshot]]:
    plan = _plan(tmp_path)
    snapshots: dict[tuple[int, str], FakeSnapshot] = {}
    for replica in plan["replicas"]:
        evaluation = _evaluation(plan, replica)
        root = Path(replica["output_root"])
        (root / "EVAL.json").write_text(json.dumps(evaluation, sort_keys=True) + "\n")
        arm = plan["arms"][replica["arm"]]
        stored_plan = []
        for task in sorted(plan["tasks"], key=lambda row: row["task_version_id"]):
            row = {
                "experiment_id": replica["experiment_id"],
                "task_key": task["task_key"],
                "task_version_id": task["task_version_id"],
                "model_id": arm["model_id"],
                "model_revision": arm["model"]["revision"],
                "serving_block": arm["serving_block"],
                "endpoint_model_id": arm["route"]["served_id"],
                "harness_id": "protocol-" + evaluation["sha256"],
                "attempt": 1,
                "max_retries": 0,
            }
            stored_plan.append({"cell_id": final._cell_id(row), **row})  # noqa: SLF001
        plan_digest = final._plain_digest(stored_plan)  # noqa: SLF001
        protocol = "sha256:" + f"{replica['seed']:064x}"[-64:]
        terminal = _signed(
            {
                "schema": final.TERMINAL_SCHEMA,
                "evaluation_identity_sha256": "sha256:" + "e" * 64,
                "comparison_protocol_sha256": protocol,
                "protocol_id": f"q38-dev17-s{replica['seed']}-base-t3k32s1000-p1-v1",
                "arm_id": replica["arm"],
                "job": {
                    "name": replica["job_name"],
                    "uid": "11111111-1111-4111-8111-111111111111",
                    "terminal_condition": "Complete",
                    "succeeded": 1,
                    "failed": 0,
                },
                "config_map": {
                    "name": replica["job_name"],
                    "uid": "22222222-2222-4222-8222-222222222222",
                },
                "workloads": [],
                "pods": [],
                "database": {
                    "name": replica["database"],
                    "summary": {"plan_sha256": plan_digest},
                },
                "output_root": {"path": replica["output_root"], "exists": True},
                "decision": {
                    "capability_result_status": "not_interpreted",
                    "score_blind_reconciliation_required": False,
                    "unresolved_cells": 0,
                    "rollout_retry_performed": False,
                    "score_read_or_generated": False,
                },
                "privacy": {
                    "prompts_responses_flags_rewards_or_trace_content_included": False,
                    "score_values_included": False,
                    "credentials_included": False,
                },
            }
        )
        Path(replica["terminal_receipt_path"]).write_text(
            json.dumps(terminal, sort_keys=True) + "\n"
        )
        cells = []
        local = []
        events = []
        planned_by_task = {row["task_version_id"]: row for row in stored_plan}
        for index, task in enumerate(plan["tasks"]):
            planned = planned_by_task[task["task_version_id"]]
            cell_id = planned["cell_id"]
            session_id = f"private-session-{replica['seed']}-{replica['arm']}-{index}"
            receipt = f"{index + 500:064x}"[-64:]
            cells.append(
                {
                    "cell_id": cell_id,
                    **planned,
                    "state": "accepted",
                    "session_id": session_id,
                    "lease_expires_at": None,
                    "retry_count": 0,
                    "max_retries": 0,
                    "result_class": "valid",
                    "receipt_digest": receipt,
                    "failure_code": None,
                    "reconciliation_digest": None,
                }
            )
            record = _local_record(
                cell_id,
                session_id,
                float(index == 0 and replica["arm"] == "candidate"),
                index,
            )
            local.append(record)
            events.append(
                {
                    "cell_id": cell_id,
                    "event": "accepted",
                    "to_state": "accepted",
                    "detail_json": json.dumps({"receipt_digest": receipt}),
                }
            )
        gate = final.GateSnapshot(
            plan_sha256=plan_digest,
            cells=cells,
            local_metadata=[
                {key: value for key, value in row.items() if key != "score"} for row in local
            ],
            events=events,
            reconciliations=[],
        )
        snapshots[(replica["seed"], replica["arm"])] = FakeSnapshot(gate, local)
    return plan, snapshots


def test_final_gate_opens_scores_only_after_all_replicas_and_emits_safe_public_input(
    tmp_path: Path,
) -> None:
    plan, snapshots = _study(tmp_path)

    result = final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert result["status"] == "final"
    assert result["valid_outcomes_per_task_arm"] == 8
    assert all(snapshot.score_reads == 1 for snapshot in snapshots.values())
    public_path = Path(plan["private_output_root"]) / "SANITIZED_AGGREGATE.json"
    public = json.loads(public_path.read_text())
    validated = public_eval_import._validate(public, final._file_digest(public_path))  # noqa: SLF001
    assert validated["summary"]["paired_valid_tasks"] == 17
    assert all(
        row[arm]["valid_attempts"] == 8 and row[arm]["infrastructure_invalid_attempts"] == 0
        for row in public["task_rows"]
        for arm in final.ARMS
    )
    public_text = public_path.read_text().lower()
    for forbidden in (
        "task_key",
        "task_version_id",
        "session_id",
        "cell_id",
        "prompt",
        "trace",
        "verifier",
        "private-session",
    ):
        assert forbidden not in public_text
    assert oct(public_path.stat().st_mode & 0o777) == "0o600"


def test_one_unready_cell_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    first = snapshots[(46, "base")]
    first.value.cells[0]["state"] = "retry_review"
    first.value.cells[0]["result_class"] = "infrastructure_invalid"

    with pytest.raises(final.FinalAggregateError, match="non-authoritative"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())
    assert not Path(plan["private_output_root"]).exists()


def _rewrite_terminal(replica: dict[str, Any], mutate: Any) -> None:
    path = Path(replica["terminal_receipt_path"])
    receipt = json.loads(path.read_text())
    receipt.pop("sha256")
    mutate(receipt)
    path.write_text(json.dumps(_signed(receipt), sort_keys=True) + "\n")


def test_wrong_protocol_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 46 and row["arm"] == "base")
    _rewrite_terminal(replica, lambda receipt: receipt.update(protocol_id="wrong"))

    with pytest.raises(final.FinalAggregateError, match="terminal receipt identity"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


def test_live_ledger_plan_must_recompute_from_exact_cells(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    identity = (46, "base")
    snapshot = snapshots[identity]
    wrong = "f" * 64
    snapshot.value = final.GateSnapshot(
        plan_sha256=wrong,
        cells=snapshot.value.cells,
        local_metadata=snapshot.value.local_metadata,
        events=snapshot.value.events,
        reconciliations=snapshot.value.reconciliations,
    )
    replica = next(row for row in plan["replicas"] if (row["seed"], row["arm"]) == identity)
    _rewrite_terminal(
        replica,
        lambda receipt: receipt["database"]["summary"].update(plan_sha256=wrong),
    )

    with pytest.raises(final.FinalAggregateError, match="immutable ledger plan"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_scored_query_cannot_change_score_blind_metadata(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    snapshot = snapshots[(46, "base")]
    snapshot.scored[0]["trace_sha256"] = "a" * 64

    with pytest.raises(final.FinalAggregateError, match="score-blind gate"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert snapshot.score_reads == 1
    assert not Path(plan["private_output_root"]).exists()


def test_reviewed_stored_session_needs_matching_private_reconciliation(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    snapshot = snapshots[(46, "base")]
    cell = snapshot.value.cells[0]
    intent = "9" * 64
    cell["reconciliation_digest"] = intent
    snapshot.value.events[0] = {
        "cell_id": cell["cell_id"],
        "event": "stored_scored_session_reconciled",
        "to_state": "accepted",
        "detail_json": json.dumps(
            {
                "reviewed_intent_sha256": intent,
                "cell_receipt_sha256": cell["receipt_digest"],
            }
        ),
    }
    snapshot.value.reconciliations.append(
        {
            "reviewed_intent_sha256": intent,
            "accepted_existing_completed_session_count": 1,
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
        }
    )

    final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert Path(plan["private_output_root"], "FINAL.json").is_file()


def test_renderer_is_cpu_only_c1_alert_suppressed_and_module_bound(tmp_path: Path) -> None:
    root = tmp_path / "render"

    receipt = renderer.render(output=root)

    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    config_map, job = bundle["items"]
    assert receipt["external_mutations"] == 0
    assert receipt["launch_performed"] is False
    assert receipt["two_server_previews_required_before_create"] is True
    assert job["metadata"]["annotations"] == {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(job)
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    assert "aggregate.py" in files and "study.json" in files and "run.py" in files
    assert receipt["source_files"]["aggregate.py"] == (
        "sha256:" + hashlib.sha256(files["aggregate.py"].encode()).hexdigest()
    )


def base64_decode(value: str) -> bytes:
    import base64

    return base64.b64decode(value, validate=True)


def _server_preview(bundle: dict[str, Any], uid: str) -> dict[str, Any]:
    value = copy.deepcopy(bundle)
    for item in value["items"]:
        item["metadata"].update(
            uid=uid,
            creationTimestamp="2026-09-23T00:00:00Z",
            resourceVersion="1",
        )
        if item["kind"] == "Job":
            item["status"] = {"active": 0}
            item["spec"]["selector"] = {"matchLabels": {"controller-uid": uid}}
            item["spec"]["template"]["metadata"]["labels"].update(
                {
                    "controller-uid": uid,
                    "batch.kubernetes.io/controller-uid": uid,
                }
            )
    return value


def test_two_preview_validator_normalizes_only_server_job_identity(tmp_path: Path) -> None:
    root = tmp_path / "render"
    renderer.render(output=root)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "1" * 36)))
    second.write_text(json.dumps(_server_preview(bundle, "2" * 36)))

    receipt = renderer.validate_previews(
        render_root=root,
        first=first,
        second=second,
        output=tmp_path / "previews.json",
    )

    assert receipt["root_failure_alerts"] == "off"
    assert receipt["priority_class"] == "c1"
    assert receipt["gpu_requests"] == 0
    assert receipt["create_performed"] is False


def test_two_preview_validator_rejects_render_receipt_drift(tmp_path: Path) -> None:
    root = tmp_path / "render"
    renderer.render(output=root)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "1" * 36)))
    second.write_text(json.dumps(_server_preview(bundle, "2" * 36)))
    receipt = json.loads((root / "RENDER.json").read_text())
    receipt["gpu_requests"] = 1
    (root / "RENDER.json").write_text(json.dumps(receipt))

    with pytest.raises(renderer.RenderError, match="self digest"):
        renderer.validate_previews(
            render_root=root,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )
