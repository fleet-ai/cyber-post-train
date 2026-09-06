import json
import subprocess
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import hosted_glm_rank29_a3a4_c2_package_v1 as package
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v1 as release
from evals.fleet import hosted_glm_rank29_a3a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_rank29_successor_selects_only_unstarted_cells_with_fresh_executions() -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    source_plan = source.validate_all(ROOT)[successor.SOURCE_CONTROLLER]
    source_rows = {
        row["attempt"]: row
        for row in source_plan["attempts"]
        if row["selection_rank"] == 29
    }
    assert [row["attempt"] for row in plan["attempts"]] == [3, 4]
    assert all(row["execution_generation"] == 2 for row in plan["attempts"])
    assert all(row["run_id"].find("-g2-") >= 0 for row in plan["attempts"])
    assert all(
        row["cell_id"] == source_rows[row["attempt"]]["cell_id"]
        and row["execution_id"] != source_rows[row["attempt"]]["execution_id"]
        for row in plan["attempts"]
    )
    assert plan["partition"]["attempt_1_accepted"] is True
    assert plan["partition"]["attempt_2_nonrepeatable_blocked"] is True
    assert plan["partition"]["attempt_2_claim_sha256"] == successor.BLOCKED_A2_CLAIM_SHA
    assert plan["partition"]["source_generation_1_retired_unclaimed"] == [
        {
            "attempt": attempt,
            "cell_id": source_rows[attempt]["cell_id"],
            "execution_id": source_rows[attempt]["execution_id"],
            "run_id": source_rows[attempt]["run_id"],
        }
        for attempt in (3, 4)
    ]


def test_scientific_treatment_matches_known_good_s2() -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    source_plan = source.validate_all(ROOT)[successor.SOURCE_CONTROLLER]
    assert plan["model"] == source_plan["model"]
    assert plan["harness"] == source_plan["harness"]
    assert plan["treatment"] == source_plan["treatment"]
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["execution"]["endpoint_lease"] == {
        "lease_root": str(successor.LEASE_ROOT),
        "endpoint_key": successor.LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
    }
    assert plan["execution"]["priority_class"] == "fleet-serve-low"
    assert plan["execution"]["preemption_policy"] == "Never"


def test_release_package_is_held_create_once_and_score_blind() -> None:
    rendered = package.render_release(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == package.RELEASE_JOB_NAME
    assert (
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
        == "false"
    )
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert "release.py" in configmap["data"]
    assert "successor_runtime.py" in configmap["data"]
    env = {
        row["name"]: row.get("value")
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["SCORED_SOURCE_SHA256"] == rendered["scored_source_sha256"]
    assert (
        env["SCORED_PACKAGE_TEMPLATE_SHA256"]
        == rendered["scored_package_template_sha256"]
    )


def test_release_package_import_closure_materializes_in_isolated_tree(tmp_path: Path) -> None:
    configmap = package.render_release(ROOT)["objects"]["items"][0]
    install = {
        **package.release_base.INSTALL_PATHS,
        "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
        "successor.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "release.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v1.py",
    }
    for key, relative in install.items():
        if key not in configmap["data"]:
            continue
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(configmap["data"][key])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; "
            f"sys.path.insert(0, {str(tmp_path)!r}); "
            "from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_deadline_guard_stops_before_claim_and_emits_sanitized_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2033, 5, 18, tzinfo=UTC).timestamp()
    monkeypatch.setattr(runtime, "_job_timing", lambda: (start, 86_400))

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(start + 86_400 - 32_399, tz=tz)

    monkeypatch.setattr(runtime, "datetime", FixedDateTime)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    plan = {"plan_sha256": "sha256:" + "1" * 64}
    (tmp_path / "PLAN.json").write_text(json.dumps(plan))
    observer = runtime._deadline_observer(tmp_path)  # noqa: SLF001
    item = {
        "cell_id": "sha256:" + "2" * 64,
        "execution_id": "sha256:" + "3" * 64,
    }
    with pytest.raises(RuntimeError, match="stopped before"):
        observer("06-route-valid", item)
    receipt = json.loads((tmp_path / "STOP_BEFORE_DEADLINE.json").read_text())
    assert receipt["claim_written"] is False
    assert receipt["model_request_started"] is False
    assert receipt["next_cell_id"] == item["cell_id"]
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )


def test_deadline_guard_allows_claim_with_full_task_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2033, 5, 18, tzinfo=UTC).timestamp()
    monkeypatch.setattr(runtime, "_job_timing", lambda: (start, 86_400))

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(start + 1, tz=tz)

    monkeypatch.setattr(runtime, "datetime", FixedDateTime)
    observer = runtime._deadline_observer(Path("/does-not-need-to-exist"))  # noqa: SLF001
    observer(
        "06-route-valid",
        {"cell_id": "sha256:" + "2" * 64, "execution_id": "sha256:" + "3" * 64},
    )


def test_runtime_uses_exact_engine_with_deadline_observer() -> None:
    source_text = (ROOT / "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py").read_text()
    assert "engine.run_controller(" in source_text
    assert "stage_observer=_deadline_observer(out)" in source_text
    assert runtime.CLAIM_GUARD_SECONDS > 28_800
    assert runtime.ACTIVE_DEADLINE_SECONDS > 2 * runtime.CLAIM_GUARD_SECONDS
    engine.validate_bulk_adapter(successor)


def _write_receipt(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def test_global_duplicate_scan_is_clear_for_unused_generations(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    result = release._global_evidence(plan, jobs_root=jobs, claim_root=claims)  # noqa: SLF001
    assert all(value == 0 for value in result.values())


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("retired_claim", "retired_generation_claim_collisions"),
        ("fresh_claim", "fresh_generation_claim_collisions"),
        ("cell_claim", "global_cell_claim_collisions"),
        ("accepted", "global_accepted_evidence_collisions"),
        ("accepted_path", "global_accepted_evidence_collisions"),
        ("output", "global_output_evidence_collisions"),
    ],
)
def test_global_duplicate_scan_rejects_each_evidence_class(
    tmp_path: Path, kind: str, expected: str
) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    fresh = plan["attempts"][0]
    retired = plan["partition"]["source_generation_1_retired_unclaimed"][0]
    if kind == "retired_claim":
        _write_receipt(
            claims / f"{retired['execution_id'].removeprefix('sha256:')}.json",
            execution_id="sha256:" + "f" * 64,
        )
    elif kind == "fresh_claim":
        _write_receipt(
            claims / f"{fresh['execution_id'].removeprefix('sha256:')}.json",
            execution_id="sha256:" + "f" * 64,
        )
    elif kind == "cell_claim":
        _write_receipt(claims / "cell.json", cell_id=fresh["cell_id"])
    elif kind == "accepted":
        _write_receipt(jobs / "other" / "accepted" / "other.json", cell_id=fresh["cell_id"])
    elif kind == "accepted_path":
        _write_receipt(
            jobs / "other" / "accepted" / f"{retired['run_id']}.json",
            cell_id="sha256:" + "f" * 64,
        )
    else:
        (jobs / "other" / "attempts" / retired["run_id"]).mkdir(parents=True)
    result = release._global_evidence(plan, jobs_root=jobs, claim_root=claims)  # noqa: SLF001
    assert result[expected] == 1


def test_global_duplicate_scan_refuses_protected_receipt_keys(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    _write_receipt(claims / "unsafe.json", score=0)
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    with pytest.raises(RuntimeError, match="protected key"):
        release._global_evidence(plan, jobs_root=jobs, claim_root=claims)  # noqa: SLF001


def test_release_builder_seals_exact_global_and_package_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    clear = {
        "global_claim_files_examined": 85,
        "global_accepted_files_examined": 53,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
    }
    monkeypatch.setenv("FLEET_API_KEY", "not-serialized")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv("SCORED_SOURCE_SHA256", "sha256:" + "3" * 64)
    monkeypatch.setenv("SCORED_PACKAGE_TEMPLATE_SHA256", "sha256:" + "4" * 64)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    monkeypatch.setattr(release.source_runtime, "INVENTORY_PATH", inventory)
    monkeypatch.setattr(successor, "build_runtime_plan", lambda *_args: plan)
    monkeypatch.setattr(successor, "SFS_ROOT", tmp_path / "unused-output")
    monkeypatch.setattr(release, "_validate_predecessors", lambda: None)
    monkeypatch.setattr(release, "_validate_rank29_history", lambda: None)
    monkeypatch.setattr(release, "_global_evidence", lambda _plan: clear)
    monkeypatch.setattr(release, "_kube_get", lambda *_args: (404, {}))
    monkeypatch.setattr(engine, "_fresh_route_check", lambda *_args: None)
    monkeypatch.setattr(
        engine, "_task_for_item", lambda *_args: {"task": {"key": "safe-task-key"}}
    )
    monkeypatch.setattr(
        engine,
        "_attempt_config",
        lambda _plan, _task, item: {"run_id": item["run_id"]},
    )
    monkeypatch.setattr(engine, "_client", lambda _key: nullcontext(object()))
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda *_args: [])
    receipt = release.build(ROOT)
    runtime.validate_release_receipt(plan, receipt)
    assert receipt["global_claim_files_examined"] == 85
    assert receipt["global_accepted_files_examined"] == 53
    assert receipt["scored_source_sha256"] == "sha256:" + "3" * 64
    assert receipt["scored_package_template_sha256"] == "sha256:" + "4" * 64
    assert "not-serialized" not in json.dumps(receipt)


def test_scored_renderer_rejects_source_or_template_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    bindings = package.scored_bindings(ROOT)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    monkeypatch.setattr(package.base.runtime, "INVENTORY_PATH", inventory)
    monkeypatch.setattr(
        successor,
        "build_runtime_plan",
        lambda *_args, **_kwargs: {"plan_sha256": plan["plan_sha256"]},
    )
    body = {
        "schema_version": runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": 2,
        "source_failed_job_uid": successor.SOURCE_FAILED_JOB_UID,
        "source_failed_pod_uid": successor.SOURCE_FAILED_POD_UID,
        "blocked_a2_claim_sha256": successor.BLOCKED_A2_CLAIM_SHA,
        "blocked_a2_session_collisions": 0,
        "peer_job_uid": runtime.PEER_JOB_UID,
        "peer_pod_uid": "0da05643-b6a5-4cf0-8498-d715e2cb3422",
        "peer_active": True,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "serving_load_block": successor.SERVING_LOAD_BLOCK,
        "job_active_deadline_seconds": runtime.ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": runtime.CLAIM_GUARD_SECONDS,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "global_claim_files_examined": 85,
        "global_accepted_files_examined": 53,
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "observer_job_uid": "11111111-1111-4111-8111-111111111111",
        "observer_pod_uid": "22222222-2222-4222-8222-222222222222",
        **bindings,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    receipt = tmp_path / "release.json"
    receipt.write_text(json.dumps(body))
    rendered = package.render_scored(ROOT, receipt)
    assert rendered["held_plan_sha256"] == plan["plan_sha256"]
    assert rendered["scored_source_sha256"] == bindings["scored_source_sha256"]
    configmap = rendered["objects"]["items"][0]
    assert configmap["immutable"] is True
    assert configmap["data"]["release.json"] == receipt.read_text()
    for field in (*bindings, "retired_generation_claim_collisions"):
        drifted = dict(body)
        drifted[field] = (
            "sha256:" + "0" * 64 if field in bindings else 1
        )
        drifted["receipt_sha256"] = self_hosted.digest_without(
            drifted, "receipt_sha256"
        )
        receipt.write_text(json.dumps(drifted))
        with pytest.raises(RuntimeError, match="release drifted"):
            package.render_scored(ROOT, receipt)


def test_tracked_held_receipt_binds_plan_package_and_no_launch() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-hosted-rank29-a3a4-successor-held-v1.json"
    )
    value = json.loads(path.read_text())
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    rendered = package.render_release(ROOT)
    assert value["launch_authorized"] is False
    assert value["plan_sha256"] == plan["plan_sha256"]
    assert value["package_sha256"] == rendered["package_sha256"]
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
