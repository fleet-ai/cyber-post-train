"""Synthetic read-only event streams only; no Kubernetes or Jobs API calls."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import miles, miles_acceptance
from training import miles_event_evidence as evidence
from training.miles_cluster import cluster_profile


def _owner(kind: str, name: str, uid: str) -> list[dict]:
    return [{"kind": kind, "name": name, "uid": uid, "controller": True}]


def _event(
    kind: str,
    name: str,
    uid: str,
    *,
    owner: list[dict] | None = None,
    spec: dict | None = None,
    status: dict | None = None,
    version: int,
    event_type: str = "MODIFIED",
) -> dict:
    return {
        "type": event_type,
        "object": {
            "kind": kind,
            "metadata": {
                "namespace": evidence.NAMESPACE,
                "name": name,
                "uid": uid,
                "resourceVersion": str(version),
                "creationTimestamp": "2026-09-12T12:00:01Z",
                "ownerReferences": owner or [],
            },
            "spec": spec or {},
            "status": status or {},
        },
    }


@pytest.fixture
def plan(tmp_path: Path) -> dict:
    value = {
        "schema": "cyber_miles_training_v1",
        "run_name": "synthetic-miles-dev3",
        "output_root": str(tmp_path / "run"),
        "model": {"repo": "Qwen/Qwen3.8-27B"},
        "data": {"files": {"train": {"rows": 1}, "dev": {"rows": 1}}},
        "checkpoint": {"image": miles.IMAGE},
        "arguments": {
            "name": "synthetic-miles-dev3",
            "output_root": str(tmp_path / "run"),
            "model": "Qwen/Qwen3.8-27B",
            "nodes": 1,
            "gpus_per_node": 8,
            "steps": 1,
            "groups": 1,
            "samples_per_prompt": 8,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "wandb_entity": "thefleet",
            "wandb_project": "cyber-post-train",
            "wandb_run_id": "synthetic-miles-dev3",
            "lr": 1e-6,
        },
        "runtime_sha256": "8" * 64,
        "native_driver_sha256": "9" * 64,
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "128",
                "memory_request": "1536Gi",
                "memory_limit": "2048Gi",
            },
        },
    }
    return value


@pytest.fixture
def submission(plan: dict) -> dict:
    run_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    value = {
        "schema": miles_acceptance.SUBMISSION_SCHEMA,
        "source_commit": "a" * 40,
        "source_plan_path": "/private/plan.json",
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_plan_file_sha256": "sha256:" + "1" * 64,
        "source_request_path": "/private/request.json",
        "source_request_sha256": "sha256:" + "2" * 64,
        "source_request_file_sha256": "sha256:" + "3" * 64,
        "runtime_bundle_sha256": "sha256:" + "4" * 64,
        "runtime_source_sha256": "sha256:" + plan["runtime_sha256"],
        "runtime_source_commit_match": True,
        "api": {
            "base_url": "https://api.ft.dev.flt.build",
            "run_id": run_id,
            "run_name": plan["run_name"] + "-aaaaaaaa",
        },
        "request": {
            "name": plan["run_name"],
            "run_dir": plan["output_root"],
            "image": miles.IMAGE,
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": plan["execution"]["resources"],
            "priority_class": "c1",
            "automatic_requeue": False,
            "secret_names": ["fleet-api", "wandb-api"],
            "environment_names": sorted({*miles_acceptance._SUBMITTED_ENV, "WANDB_RUN_ID"}),
            "runtime_module": "training.miles_training",
            "runtime_argv": ["--plan", "plan.json", "--sha256", digest(plan)],
        },
        "jobs_api_post_count": 1,
        "submitted_at": "2026-09-12T12:00:00Z",
        "secret_values_included": False,
        "task_content_included": False,
        "private_payload_included": False,
    }
    value["sha256"] = "sha256:" + digest(value)
    return value


def _journal(plan: dict, submission: dict, tmp_path: Path) -> tuple[Path, Path, Path]:
    run_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    api_name = plan["run_name"] + "-aaaaaaaa"
    rayjob_uid = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    workload_uid = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    raycluster_uid = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    pod_uid = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    raycluster_name = "synthetic-raycluster"
    directory = tmp_path / "watch"
    evidence.start_capture(
        plan,
        submission,
        namespace_uid=miles_acceptance.NAMESPACE_UID,
        started_at="2026-09-12T12:00:00Z",
        directory=directory,
    )
    terminal_spec = {
        "shutdownAfterJobFinishes": True,
        "ttlSecondsAfterFinished": 0,
        "rayClusterSpec": {"headGroupSpec": {"template": {"spec": {"priorityClassName": "c1"}}}},
    }
    rayjob = _event(
        "RayJob",
        api_name,
        rayjob_uid,
        spec=terminal_spec,
        status={},
        version=1,
        event_type="ADDED",
    )
    rayjob["object"]["metadata"]["labels"] = {"fleet.ai/run-id": run_id}
    workload = _event(
        "Workload",
        "synthetic-workload",
        workload_uid,
        owner=_owner("RayJob", api_name, rayjob_uid),
        status={
            "conditions": [
                {
                    "type": "Admitted",
                    "status": "True",
                    "lastTransitionTime": "2026-09-12T12:00:02Z",
                }
            ]
        },
        version=2,
    )
    raycluster = _event(
        "RayCluster",
        raycluster_name,
        raycluster_uid,
        owner=_owner("RayJob", api_name, rayjob_uid),
        version=3,
        event_type="ADDED",
    )
    pod_spec = {
        "containers": [
            {
                "name": "ray-head",
                "resources": {"limits": {"nvidia.com/gpu": "8"}},
            }
        ]
    }
    pod_status = {
        "phase": "Succeeded",
        "containerStatuses": [
            {
                "name": "ray-head",
                "imageID": "containerd://registry/image@" + miles.IMAGE.rsplit("@", 1)[1],
                "restartCount": 0,
                "state": {
                    "terminated": {
                        "exitCode": 0,
                        "reason": "Completed",
                        "finishedAt": "2026-09-12T12:00:04Z",
                    }
                },
            }
        ],
    }
    pod = _event(
        "Pod",
        "synthetic-ray-head",
        pod_uid,
        owner=_owner("RayCluster", raycluster_name, raycluster_uid),
        spec=pod_spec,
        status=pod_status,
        version=4,
    )
    terminal_rayjob = copy.deepcopy(rayjob)
    terminal_rayjob["type"] = "DELETED"
    terminal_rayjob["object"]["metadata"]["resourceVersion"] = "5"
    terminal_rayjob["object"]["status"] = {
        "jobStatus": "SUCCEEDED",
        "rayClusterName": raycluster_name,
    }
    for sequence, row in enumerate((rayjob, workload, raycluster, pod, terminal_rayjob)):
        evidence.record_event(
            directory,
            row,
            observed_at=f"2026-09-12T12:00:0{sequence + 1}Z",
            sequence=sequence,
        )
    controller_path = tmp_path / "CONTROLLER.json"
    controller = evidence.compile_controller(
        plan,
        submission,
        directory=directory,
        api_status="SUCCEEDED",
        observed_at="2026-09-12T12:00:06Z",
        output=controller_path,
    )
    assert controller["kubernetes"]["rayjob"]["uid"] == rayjob_uid
    release_path = tmp_path / "RELEASE.json"
    evidence.compile_release(
        plan,
        submission,
        controller_path=controller_path,
        absence={
            "raycluster_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
        },
        observed_at="2026-09-12T12:00:07Z",
        output=release_path,
    )
    return directory, controller_path, release_path


def test_ttl_zero_terminal_and_release_compile_after_all_objects_disappear(
    plan: dict, submission: dict, tmp_path: Path
) -> None:
    directory, controller_path, release_path = _journal(plan, submission, tmp_path)

    # There are intentionally no live Kubernetes objects to read now.  The
    # create-once event journal is sufficient to retain their immutable UIDs.
    assert {path.name for path in directory.iterdir()} == {
        "STARTED.json",
        *(f"EVENT-{index:06d}.json" for index in range(5)),
    }
    controller = __import__("json").loads(controller_path.read_bytes())
    release = __import__("json").loads(release_path.read_bytes())
    miles_acceptance.validate_controller_observation(controller, plan, submission)
    miles_acceptance.validate_release_observation(
        release,
        plan,
        controller,
        miles_acceptance._json_snapshot(controller_path)[1],
        submission,
        not_before=1.0,
    )


def test_capture_start_derives_the_exact_prod_identity_from_the_plan(
    plan: dict, submission: dict, tmp_path: Path
) -> None:
    plan["execution"]["cluster_target"] = "prod"
    profile = cluster_profile("prod")
    submission["source_plan_sha256"] = "sha256:" + digest(plan)
    submission["runtime_source_sha256"] = "sha256:" + plan["runtime_sha256"]
    submission["api"]["base_url"] = profile.api_base_url
    submission["request"]["runtime_argv"][-1] = digest(plan)
    submission.pop("sha256")
    submission["sha256"] = "sha256:" + digest(submission)

    started = evidence.start_capture(
        plan,
        submission,
        namespace_uid=profile.namespace_uid,
        started_at="2026-09-12T12:00:00Z",
        directory=tmp_path / "prod-watch",
        kube_context=profile.kube_context,
    )
    assert started["cluster"] == "prod"
    assert started["api_base_url"] == profile.api_base_url
    assert started["kube_context"] == profile.kube_context
    assert started["namespace_uid"] == profile.namespace_uid

    with pytest.raises(ValueError, match="cluster identity"):
        evidence.start_capture(
            plan,
            submission,
            namespace_uid=cluster_profile("dev").namespace_uid,
            started_at="2026-09-12T12:00:00Z",
            directory=tmp_path / "wrong-watch",
            kube_context=profile.kube_context,
        )


def test_event_journal_fails_closed_if_capture_started_after_admission(
    plan: dict, submission: dict, tmp_path: Path
) -> None:
    directory, _, _ = _journal(plan, submission, tmp_path)
    start = __import__("json").loads((directory / "STARTED.json").read_bytes())
    start["started_at"] = "2026-09-12T12:00:03Z"
    start.pop("sha256")
    start["sha256"] = digest(start)
    (directory / "STARTED.json").write_text(__import__("json").dumps(start))

    with pytest.raises(ValueError, match="began after admission"):
        evidence.compile_controller(
            plan,
            submission,
            directory=directory,
            api_status="SUCCEEDED",
            observed_at="2026-09-12T12:00:08Z",
            output=tmp_path / "late.json",
        )


def test_event_receipts_are_create_once_and_contain_no_raw_private_surfaces(
    plan: dict, submission: dict, tmp_path: Path
) -> None:
    directory, _, _ = _journal(plan, submission, tmp_path)
    first = __import__("json").loads((directory / "EVENT-000000.json").read_bytes())

    assert first["private_logs_included"] is False
    assert first["metric_values_included"] is False
    assert first["task_content_included"] is False
    assert "env" not in first
    with pytest.raises(FileExistsError):
        evidence.record_event(
            directory,
            {
                "type": "ADDED",
                "object": {
                    "kind": "RayJob",
                    "metadata": {
                        "namespace": evidence.NAMESPACE,
                        "name": first["name"],
                        "uid": str(uuid.uuid4()),
                        "resourceVersion": "99",
                        "creationTimestamp": "2026-09-12T12:00:01Z",
                        "labels": {"fleet.ai/run-id": first["api_run_id"]},
                    },
                    "spec": {},
                    "status": {},
                },
            },
            observed_at="2026-09-12T12:00:09Z",
            sequence=0,
        )
