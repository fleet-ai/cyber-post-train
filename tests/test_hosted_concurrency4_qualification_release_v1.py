from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import hosted_concurrency4_qualification_package_v1 as package
from evals.fleet import hosted_concurrency4_qualification_release_v1 as release

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def _release(commit: str) -> dict:
    return release.build_release(ROOT, commit)


def test_release_is_append_only_and_binds_exact_held_package() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = _release(commit)
    release.validate_release(receipt, ROOT)
    assert receipt["package"] == {
        "commit": release.PACKAGE_COMMIT,
        "schema_version": package.SCHEMA,
        "sha256": package.build_configmap(ROOT, release.PACKAGE_COMMIT)["data"]["package_sha256"],
        "held_launch_authorized": False,
    }
    assert receipt["authorization"] == {
        "launch_authorized": True,
        "create_once": True,
        "non_scoring": True,
        "chat_completion_requests": 24,
        "task_instance_session_scoring_or_verifier_calls": 0,
        "scored_bulk_launch_authorized": False,
        "maximum_concurrent_requests_per_model": 4,
        "models_run_sequentially": True,
        "priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
        "cpu_only": True,
    }


def test_released_objects_only_release_exact_package() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = _release(commit)
    bootstrap, intent, job = release.released_objects(receipt, ROOT)
    package_value = json.loads(bootstrap["data"]["package.json"])
    assert package_value["launch_authorized"] is False
    assert intent["metadata"]["name"] == release.RELEASE_INTENT
    assert intent["data"]["launch_authorized"] == "true"
    assert intent["data"]["scored_bulk_launch_authorized"] == "false"
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "true"
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert job["spec"]["backoffLimit"] == 0
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"


def _job(*, complete: bool = True, failed: bool = False, active: int = 0) -> dict:
    conditions = []
    if complete:
        conditions.append({"type": "Complete", "status": "True"})
    if failed:
        conditions.append({"type": "Failed", "status": "True"})
    return {
        "metadata": {"name": "g5", "uid": JOB_UID},
        "status": {"conditions": conditions, "active": active},
    }


def _pods(*, phase: str = "Succeeded", restarts: int = 0, exit_code: int = 0) -> dict:
    return {
        "items": [
            {
                "metadata": {"uid": POD_UID},
                "status": {
                    "phase": phase,
                    "containerStatuses": [
                        {
                            "restartCount": restarts,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }
                    ],
                },
            }
        ]
    }


def test_generation5_must_be_exclusively_complete() -> None:
    assert release.validate_terminal_job(_job(), _pods(), job_name="g5") == (
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
            release.validate_terminal_job(job, pods, job_name="g5")


def test_release_writer_is_create_once(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    release.write_json_once(path, {"safe": True})
    with pytest.raises(FileExistsError):
        release.write_json_once(path, {"safe": False})


def test_submitter_has_create_once_and_fixed_evidence_gates() -> None:
    script = (ROOT / release.SUBMIT_PATH).read_text()
    assert "mode=${1:-preview}" in script
    assert '[[ "$mode" == preview || "$mode" == submit ]]' in script
    assert 'kubectl -n "$namespace" create -f "$work/bundle.yaml"' in script
    assert "SFS_OBSERVER_UID" in script
    assert "CANARY-TERMINAL.json" in script
    assert "cell-execution-claims/opencode11827-autocontinue-v1" in script
    assert 'get secret "$secret"' in script
    assert "validate_identity" in script
    assert "task_instance_session_scoring_or_verifier_calls:0" in script
