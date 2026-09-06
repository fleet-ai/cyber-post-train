from pathlib import Path

from evals.fleet import hosted_glm_rank3_a1_c2_canary_v1 as canary
from evals.fleet import hosted_glm_rank3_a1_c2_canary_runtime_v1 as runtime
from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import self_hosted


ROOT = Path(__file__).resolve().parents[1]


def test_rank3_canary_is_held_one_cell_on_a_wholly_unstarted_task_boundary() -> None:
    plan = canary.validate_all(ROOT)[canary.CONTROLLER]
    assert plan["launch_authorized"] is False
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (3, 1)
    ]
    boundary = plan["partition"]["whole_task_cells"]
    assert [(row["attempt"], row["cell_id"]) for row in boundary] == [
        (1, "sha256:7568e59b6949088e67f3a98a566a22927640771abfd331ce5094364eb7cbac04"),
        (2, "sha256:57f0ec78f93760d980f31df4dac3fc6daf38f2849ee2949d943e3a41f4a61e2d"),
        (3, "sha256:85d8da4373e2dbd30100aca88b08b884f3ef65febf71f77060ae67246df37c8b"),
        (4, "sha256:f5b26b9037dd0f02ad2bb22401c9d0be8b9ab4d98641fdb091cc5f1bf6cddb44"),
    ]
    assert plan["partition"]["remaining_attempts_held"] == [2, 3, 4]
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")


def test_rank3_canary_binds_exact_live_s2_runner_and_failure_evidence() -> None:
    plan = canary.validate_all(ROOT)[canary.CONTROLLER]
    runtime = plan["known_good_s2_runtime"]
    assert runtime["uid"] == "1b9b171e-5ea8-4f9d-b014-e4e0bd106789"
    assert runtime["immutable"] is True
    assert runtime["critical_data_sha256"]["engine.py"] == (
        "sha256:06f9857c99ee1773d505ea60c06c335b373d91b7dbf267d2074bbdc39c4aa20f"
    )
    assert plan["failure_authority"]["commit"] == "640f430"
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["preemption_policy"] == "Never"


def test_rank3_runtime_release_is_fail_closed(tmp_path, monkeypatch) -> None:
    plan = canary.validate_all(ROOT)[canary.CONTROLLER]
    receipt = {
        "schema_version": runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": canary.JOB_NAME,
        "successor_configmap": canary.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "selection_rank": 3,
        "attempt": 1,
        "whole_task_boundary_clear": True,
        "base_configmap_uid": canary.KNOWN_GOOD_BASE_CONFIGMAP["uid"],
        "active_scored_lease_slots_before_create": 1,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "cell_id": plan["attempts"][0]["cell_id"],
        "execution_id": plan["attempts"][0]["execution_id"],
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    path = tmp_path / "release.json"
    self_hosted.write_json_once(path, receipt)
    monkeypatch.setattr(runtime, "RELEASE_PATH", path)
    runtime.validate_release(plan)
    receipt["fleet_session_collisions"] = 1
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    path.write_bytes(self_hosted.canonical_json(receipt) + b"\n")
    import pytest

    with pytest.raises(RuntimeError):
        runtime.validate_release(plan)


def test_rank3_adapter_is_complete_for_the_exact_s2_engine() -> None:
    engine.validate_bulk_adapter(canary)


def test_v2_copies_projected_release_to_a_private_regular_file() -> None:
    script = (ROOT / "evals/fleet/scripts/run_hosted_glm_rank3_a1_c2_canary_v2.sh").read_text()
    assert 'install -m 0600 /bootstrap/release.json "$ROOT/.runtime/release.json"' in script
