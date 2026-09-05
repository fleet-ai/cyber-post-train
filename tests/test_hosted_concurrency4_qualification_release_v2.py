from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import hosted_concurrency4_qualification_package_v1 as package
from evals.fleet import hosted_concurrency4_qualification_release_v2 as release

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_g6_gated_authority_is_held_and_self_digesting() -> None:
    receipt = release.load(ROOT / release.HELD_PATH)
    release.validate_held(receipt)
    assert receipt == release.expected_held()
    assert receipt["launch_authorized"] is False
    assert receipt["objects_created"] is False
    assert receipt["preserved_non_scoring_contract"] == {
        "chat_completion_requests": 24,
        "task_instance_session_scoring_or_verifier_calls": 0,
        "maximum_concurrent_requests_per_model": 4,
        "models_run_sequentially": True,
    }


def test_append_only_release_binds_exact_unchanged_package() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt = release.build_release(ROOT, commit)
    release.validate_release(receipt, ROOT)
    assert receipt["package"] == {
        "commit": release.PACKAGE_COMMIT,
        "schema_version": package.SCHEMA,
        "sha256": package.build_configmap(ROOT, release.PACKAGE_COMMIT)["data"][
            "package_sha256"
        ],
        "held_launch_authorized": False,
    }
    assert receipt["runtime_prerequisites"][
        "generation6_qwen_and_glm_exclusively_complete_and_accepted"
    ] is True


def test_released_objects_are_cpu_only_and_do_not_change_probe_payload() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt = release.build_release(ROOT, commit)
    bootstrap, intent, job = release.released_objects(receipt, ROOT)
    package_value = json.loads(bootstrap["data"]["package.json"])
    assert package_value["launch_authorized"] is False
    assert intent["metadata"]["name"] == release.RELEASE_INTENT
    assert intent["data"]["generation6_evidence_required"] == "true"
    assert job["metadata"]["name"] == package.JOB_NAME
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert job["spec"]["backoffLimit"] == 0


def _job(*, complete: bool = True, failed: bool = False, active: int = 0) -> dict:
    conditions = []
    if complete:
        conditions.append({"type": "Complete", "status": "True"})
    if failed:
        conditions.append({"type": "Failed", "status": "True"})
    return {
        "metadata": {"name": "g6", "uid": JOB_UID},
        "status": {"conditions": conditions, "active": active},
    }


def _pods(
    *, phase: str = "Succeeded", restarts: int = 0, exit_code: int = 0
) -> dict:
    return {
        "items": [
            {
                "metadata": {"uid": POD_UID},
                "status": {
                    "phase": phase,
                    "containerStatuses": [
                        {
                            "name": "evaluator",
                            "restartCount": restarts,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }
                    ],
                    "initContainerStatuses": [
                        {
                            "name": name,
                            "restartCount": restarts,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }
                        for name in ("docker-cli", "dind")
                    ],
                },
            }
        ]
    }


def test_generation6_job_must_be_exclusively_and_cleanly_complete() -> None:
    assert release.validate_terminal_job(_job(), _pods(), job_name="g6") == (
        JOB_UID,
        POD_UID,
    )
    for job, pods in (
        (_job(complete=False), _pods()),
        (_job(failed=True), _pods()),
        (_job(active=1), _pods()),
        (_job(), _pods(phase="Running")),
        (_job(), _pods(restarts=1)),
        (_job(), _pods(exit_code=1)),
    ):
        with pytest.raises(release.ReleaseError):
            release.validate_terminal_job(job, pods, job_name="g6")


def test_release_writer_is_create_once(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    release.write_json_once(path, {"safe": True})
    with pytest.raises(FileExistsError):
        release.write_json_once(path, {"safe": False})


def test_submitter_is_held_until_release_and_has_fixed_g6_gates() -> None:
    script = (ROOT / release.SUBMIT_PATH).read_text()
    assert "HELD: append-only G6-gated v2 launch release has not been rendered" in script
    assert 'kubectl -n "$namespace" create -f "$work/bundle.yaml"' in script
    assert "validate-generation6-evidence" in script
    assert "CANARY-TERMINAL.json" in script
    assert "cell-execution-claims/opencode11827-autocontinue-v1" in script
    assert release.INVALID_V1_RELEASE_INTENT in script
    assert "validate_identity" in script
    assert "task_instance_session_scoring_or_verifier_calls:0" in script
    for name in (
        "chris-q38-ac-exact100-bulk-a199-v1",
        "chris-q38-ac-exact100-bulk-b200-v1",
        "chris-glm53-ac-exact100-bulk-a199-v1",
        "chris-glm53-ac-exact100-bulk-b200-v1",
    ):
        assert name in script
