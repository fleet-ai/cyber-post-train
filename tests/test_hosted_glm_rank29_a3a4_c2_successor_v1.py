import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import hosted_glm_rank29_a3a4_c2_package_v1 as package
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
    assert job["metadata"]["name"] == package.release.JOB_NAME
    assert (
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
        == "false"
    )
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert "release.py" in configmap["data"]
    assert "successor_runtime.py" in configmap["data"]


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
