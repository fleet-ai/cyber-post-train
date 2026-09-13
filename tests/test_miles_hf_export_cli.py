"""Operator-ordering tests for pre-watched TTL-zero HF jobs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from training import miles_hf_export_cli as operator


def _submission() -> dict:
    return {
        "api": {
            "run_id": "11111111-1111-4111-8111-111111111111",
            "run_name": "q38-hf-reload-11111111",
        },
        "submitted_at": "2026-09-12T12:00:00Z",
        "sha256": "sha256:" + "a" * 64,
    }


def _watch_events() -> list[dict]:
    run_id = "11111111-1111-4111-8111-111111111111"
    run_name = "q38-hf-reload-11111111"
    rayjob_uid = "22222222-2222-4222-8222-222222222222"
    cluster_name = "exact-cluster"
    cluster_uid = "33333333-3333-4333-8333-333333333333"

    def event(kind: str, name: str, uid: str, *, owner=None, spec=None, status=None) -> dict:
        return {
            "type": "MODIFIED",
            "object": {
                "kind": kind,
                "metadata": {
                    "name": name,
                    "uid": uid,
                    "labels": {"fleet.ai/run-id": run_id} if kind == "RayJob" else {},
                    "ownerReferences": owner or [],
                },
                "spec": spec or {},
                "status": status or {},
            },
        }

    def owner(kind: str, name: str, uid: str) -> list[dict]:
        return [{"kind": kind, "name": name, "uid": uid, "controller": True}]

    return [
        event("RayJob", run_name, rayjob_uid, status={"jobStatus": "SUCCEEDED"}),
        event(
            "Workload",
            "exact-workload",
            "44444444-4444-4444-8444-444444444444",
            owner=owner("RayJob", run_name, rayjob_uid),
        ),
        event(
            "RayCluster",
            cluster_name,
            cluster_uid,
            owner=owner("RayJob", run_name, rayjob_uid),
        ),
        event(
            "Pod",
            "exact-pod",
            "55555555-5555-4555-8555-555555555555",
            owner=owner("RayCluster", cluster_name, cluster_uid),
            spec={"containers": [{"name": "main"}]},
            status={
                "phase": "Succeeded",
                "containerStatuses": [
                    {
                        "name": "main",
                        "state": {"terminated": {"exitCode": 0, "reason": "Completed"}},
                    }
                ],
            },
        ),
    ]


def _direct_watch_events() -> tuple[dict, list[dict]]:
    submission = _submission()
    submission["api"]["run_name"] = "q38-hf-export"
    run = submission["api"]
    owner = [
        {
            "kind": "Job",
            "name": run["run_name"],
            "uid": run["run_id"],
            "controller": True,
        }
    ]

    def event(kind: str, name: str, uid: str, *, owned=False, spec=None, status=None) -> dict:
        return {
            "type": "MODIFIED",
            "object": {
                "kind": kind,
                "metadata": {
                    "name": name,
                    "uid": uid,
                    "ownerReferences": owner if owned else [],
                },
                "spec": spec or {},
                "status": status or {},
            },
        }

    return submission, [
        event(
            "Job",
            run["run_name"],
            run["run_id"],
            status={"conditions": [{"type": "Complete", "status": "True"}]},
        ),
        event(
            "Workload",
            "exact-workload",
            "33333333-3333-4333-8333-333333333333",
            owned=True,
        ),
        event(
            "Pod",
            "exact-pod",
            "44444444-4444-4444-8444-444444444444",
            owned=True,
            spec={"containers": [{"name": "export"}]},
            status={
                "phase": "Succeeded",
                "containerStatuses": [
                    {
                        "name": "export",
                        "state": {"terminated": {"exitCode": 0, "reason": "Completed"}},
                    }
                ],
            },
        ),
    ]


def test_direct_job_watcher_requires_all_three_exact_uid_deletions(
    tmp_path: Path, monkeypatch
) -> None:
    submission, lifecycle = _direct_watch_events()
    watcher = operator.KubernetesJobWatchBuffer()
    recorded: list[dict] = []
    monkeypatch.setattr(
        operator.events,
        "record_event",
        lambda _directory, raw, **_kwargs: recorded.append(raw),
    )
    for raw in lifecycle:
        watcher.events.put((1.0, raw))
    for raw in lifecycle[:-1]:
        deleted = copy.deepcopy(raw)
        deleted["type"] = "DELETED"
        watcher.events.put((2.0, deleted))
    watcher.events.put(RuntimeError("watch ended"))
    with pytest.raises(ValueError, match="watch failed"):
        watcher.record(tmp_path, submission, timeout=1)
    assert {raw["object"]["kind"] for raw in recorded if raw["type"] == "DELETED"} == {
        "Job",
        "Workload",
    }


def test_direct_job_watcher_ignores_a_recreated_job_uid(tmp_path: Path, monkeypatch) -> None:
    submission, lifecycle = _direct_watch_events()
    watcher = operator.KubernetesJobWatchBuffer()
    monkeypatch.setattr(operator.events, "record_event", lambda *_args, **_kwargs: None)
    wrong = copy.deepcopy(lifecycle[0])
    wrong["object"]["metadata"]["uid"] = "99999999-9999-4999-8999-999999999999"
    watcher.events.put((1.0, wrong))
    watcher.events.put(RuntimeError("watch ended"))
    with pytest.raises(ValueError, match="watch failed"):
        watcher.record(tmp_path, submission, timeout=1)


def test_hf_watcher_cannot_accept_terminal_success_without_all_uid_deletions(
    tmp_path: Path, monkeypatch
) -> None:
    watcher = operator.WatchBuffer()
    recorded: list[dict] = []
    monkeypatch.setattr(
        operator.events,
        "record_event",
        lambda _directory, raw, **_kwargs: recorded.append(raw),
    )
    for raw in _watch_events():
        watcher.events.put((1.0, raw))
    watcher.events.put(RuntimeError("watch ended before deletion"))

    with pytest.raises(ValueError, match="watch failed"):
        watcher.record(
            tmp_path,
            _submission(),
            timeout=1,
            require_deletions=True,
        )
    assert len(recorded) == 4


def test_hf_watcher_records_all_four_exact_uid_deletions(tmp_path: Path, monkeypatch) -> None:
    watcher = operator.WatchBuffer()
    recorded: list[dict] = []
    monkeypatch.setattr(
        operator.events,
        "record_event",
        lambda _directory, raw, **_kwargs: recorded.append(raw),
    )
    lifecycle = _watch_events()
    for raw in lifecycle:
        watcher.events.put((1.0, raw))
    for raw in lifecycle:
        deleted = copy.deepcopy(raw)
        deleted["type"] = "DELETED"
        watcher.events.put((2.0, deleted))

    watcher.record(
        tmp_path,
        _submission(),
        timeout=1,
        require_deletions=True,
    )

    assert len(recorded) == 8
    assert {row["object"]["kind"] for row in recorded if row["type"] == "DELETED"} == {
        "RayJob",
        "Workload",
        "RayCluster",
        "Pod",
    }


def test_submit_opens_watches_before_the_only_post_and_waits_for_deletions(
    tmp_path: Path, monkeypatch
) -> None:
    plan = {"stage": "reload", "output_root": str(tmp_path / "runtime")}
    request = {"exact": "request"}
    paths = {
        "root": tmp_path,
        "plan": tmp_path / "plan.json",
        "request": tmp_path / "request.json",
        "preflight": tmp_path / "PREFLIGHT.json",
        "journal": tmp_path / "SUBMISSION.jsonl",
        "submission": tmp_path / "SUBMITTED.json",
        "watch": tmp_path / operator.WATCH_DIRECTORY,
    }
    submission = _submission()
    order: list[object] = []

    class Watcher:
        def start(self) -> None:
            order.append("watch-start")

        def record(self, directory, observed, *, timeout, require_deletions) -> None:
            order.append(("record", directory, observed, timeout, require_deletions))

        def close(self) -> None:
            order.append("watch-close")

    class Jobs:
        def __init__(self, token, *, base_url) -> None:
            assert token == "token" and base_url == operator.API_URLS["dev"]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit_once(self, observed, journal) -> None:
            order.append(("post", observed, journal))

    monkeypatch.setenv("FLEET_API_KEY", "token")
    monkeypatch.setattr(
        operator,
        "_preflight",
        lambda *_args, **_kwargs: (paths, plan, request),
    )
    monkeypatch.setattr(operator, "WatchBuffer", Watcher)
    monkeypatch.setattr(operator, "Jobs", Jobs)
    monkeypatch.setattr(
        operator.job,
        "compile_submission_binding",
        lambda **_kwargs: order.append("bind") or submission,
    )
    monkeypatch.setattr(
        operator.events,
        "start_capture",
        lambda *_args, **_kwargs: order.append("capture-bind"),
    )

    result = operator.submit(tmp_path, "a" * 40, watch_timeout_seconds=1800)

    assert order[0] == "watch-start"
    assert order[1] == ("post", request, paths["journal"])
    assert order[2:4] == ["bind", "capture-bind"]
    assert order[4] == ("record", paths["watch"], submission, 1800, True)
    assert order[5] == "watch-close"
    assert result["status"] == "submitted_and_uid_deletions_recorded"
    assert result["controller_compiled"] is False


def test_export_server_dry_run_precedes_watches_and_single_direct_create(
    tmp_path: Path, monkeypatch
) -> None:
    plan = {"stage": "export", "output_root": str(tmp_path / "runtime")}
    request = {"exact": "zero-gpu-manifest"}
    paths = {
        "root": tmp_path,
        "plan": tmp_path / "plan.json",
        "request": tmp_path / "request.json",
        "preflight": tmp_path / "PREFLIGHT.json",
        "journal": tmp_path / "SUBMISSION.jsonl",
        "submission": tmp_path / "SUBMITTED.json",
        "watch": tmp_path / operator.WATCH_DIRECTORY,
    }
    submission = _submission()
    submission["api"]["run_name"] = "q38-hf-export"
    order: list[object] = []

    class Watcher:
        def start(self) -> None:
            order.append("watch-start")

        def record(self, directory, observed, *, timeout, require_deletions) -> None:
            order.append(("record", directory, observed, timeout, require_deletions))

        def close(self) -> None:
            order.append("watch-close")

    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    monkeypatch.setattr(
        operator,
        "_preflight",
        lambda *_args, **_kwargs: (paths, plan, request),
    )
    monkeypatch.setattr(
        operator,
        "_server_dry_run",
        lambda observed: order.append(("server-dry-run", observed)) or "d" * 64,
    )
    monkeypatch.setattr(operator, "KubernetesJobWatchBuffer", Watcher)
    monkeypatch.setattr(
        operator,
        "_direct_create_once",
        lambda observed_paths, observed_plan, observed_request, **kwargs: order.append(
            ("create", observed_paths, observed_plan, observed_request, kwargs)
        ),
    )
    monkeypatch.setattr(
        operator.job,
        "compile_submission_binding",
        lambda **_kwargs: order.append("bind") or submission,
    )
    monkeypatch.setattr(
        operator.events,
        "start_capture",
        lambda *_args, **_kwargs: order.append("capture-bind"),
    )

    operator.submit(tmp_path, "a" * 40, watch_timeout_seconds=1800)

    assert order[0] == ("server-dry-run", request)
    assert order[1] == "watch-start"
    assert order[2][0] == "create"
    assert order[2][-1] == {
        "server_dry_run_sha256": "d" * 64,
        "recovery_timeout_seconds": 1800,
    }
    assert order[3:5] == ["bind", "capture-bind"]
    assert order[5] == ("record", paths["watch"], submission, 1800, True)
    assert order[6] == "watch-close"


def test_submit_closes_watches_when_the_single_post_is_uncertain(
    tmp_path: Path, monkeypatch
) -> None:
    plan = {"stage": "export", "output_root": str(tmp_path / "runtime")}
    paths = {"journal": tmp_path / "SUBMISSION.jsonl"}
    order: list[str] = []

    class Watcher:
        def start(self) -> None:
            order.append("start")

        def close(self) -> None:
            order.append("close")

    monkeypatch.setattr(
        operator,
        "_preflight",
        lambda *_args, **_kwargs: (paths, plan, {"exact": "request"}),
    )
    monkeypatch.setattr(operator, "_server_dry_run", lambda _request: "a" * 64)
    monkeypatch.setattr(operator, "KubernetesJobWatchBuffer", Watcher)
    monkeypatch.setattr(
        operator,
        "_direct_create_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("uncertain POST; reconcile")),
    )

    with pytest.raises(ValueError, match="uncertain POST"):
        operator.submit(tmp_path, "a" * 40, watch_timeout_seconds=1800)
    assert order == ["start", "close"]


def test_direct_create_lost_response_recovers_by_get_without_second_create(
    tmp_path: Path, monkeypatch
) -> None:
    labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q2",
        "cyber-post-train.fleet.ai/owner": "miles-hf",
        "cyber-post-train.fleet.ai/component": "export",
    }
    annotations = {
        "cyber-post-train.fleet.ai/source-plan-sha256": "1" * 64,
        "cyber-post-train.fleet.ai/runtime-source-sha256": "2" * 64,
        "cyber-post-train.fleet.ai/runtime-bundle-sha256": "3" * 64,
    }
    plan = {"run_name": "q38-hf-export"}
    request = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": plan["run_name"],
            "namespace": operator.job.DEV_NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": 3600,
            "ttlSecondsAfterFinished": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "priorityClassName": "c2",
                    "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
                    "tolerations": [],
                    "volumes": [],
                    "containers": [
                        {
                            "name": "export",
                            "image": operator.job.miles.IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-m", "training.miles_hf_export_job"],
                            "env": [],
                            "resources": {
                                "requests": {"cpu": "64", "memory": "512Gi"},
                                "limits": {"cpu": "64", "memory": "768Gi"},
                            },
                            "volumeMounts": [],
                            "terminationMessagePolicy": "FallbackToLogsOnError",
                        }
                    ],
                },
            },
        },
    }
    response = copy.deepcopy(request)
    response["metadata"].update(
        {
            "uid": "11111111-1111-4111-8111-111111111111",
            "resourceVersion": "12345",
            "creationTimestamp": "2026-09-12T12:00:00Z",
        }
    )
    paths = {"journal": tmp_path / "SUBMISSION.jsonl"}
    calls: list[list[str]] = []

    def kubectl(argv, *, stdin=None):
        calls.append(argv)
        if argv[0] == "create":
            assert stdin is not None
            raise ValueError("response lost after apiserver persisted the Job")
        assert argv[:3] == ["get", "jobs.batch", plan["run_name"]]
        assert stdin is None
        return response

    monkeypatch.setattr(operator, "_kubectl_json", kubectl)

    operator._direct_create_once(
        paths,
        plan,
        request,
        server_dry_run_sha256="4" * 64,
        recovery_timeout_seconds=1800,
    )

    rows = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
    assert [call[0] for call in calls] == ["create", "get"]
    assert [row["state"] for row in rows] == [
        "KUBERNETES_POST_INTENT_DO_NOT_RETRY",
        "KUBERNETES_POST_RESPONSE",
    ]
    assert rows[1]["uid"] == response["metadata"]["uid"]


def test_preflight_rejects_an_existing_runtime_root_before_watching(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    paths = {
        "plan": tmp_path / "plan.json",
        "request": tmp_path / "request.json",
        "preflight": tmp_path / "PREFLIGHT.json",
        "journal": tmp_path / "SUBMISSION.jsonl",
        "submission": tmp_path / "SUBMITTED.json",
        "watch": tmp_path / operator.WATCH_DIRECTORY,
    }
    values = {
        paths["plan"]: {"output_root": str(runtime)},
        paths["request"]: {"exact": "request"},
        paths["preflight"]: {"exact": "proof"},
    }
    monkeypatch.setattr(operator, "_paths", lambda _directory: paths)
    monkeypatch.setattr(operator, "_json", lambda path: values[path])
    monkeypatch.setattr(operator.job, "validate_plan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator.job, "job_request", lambda _plan: values[paths["request"]])
    monkeypatch.setattr(operator.job, "validate_preflight", lambda *_args: None)
    monkeypatch.setattr(operator.job, "_request_projection", lambda *_args, **_kwargs: ({}, "x"))

    with pytest.raises(FileExistsError, match="runtime output root"):
        operator._preflight(tmp_path, "a" * 40, require_unsubmitted=True)


def test_terminal_api_rejects_a_different_api_uuid() -> None:
    submission = _submission()
    plan = {"output_root": "/mnt/sfs/jobs/exact"}

    class Jobs:
        def status(self, _name: str) -> dict:
            return {
                "name": submission["api"]["run_name"],
                "job_id": "22222222-2222-4222-8222-222222222222",
                "run_dir": plan["output_root"],
                "status": "SUCCEEDED",
                "created_at": submission["submitted_at"],
                "finished_at": "2026-09-12T12:01:00Z",
            }

    with pytest.raises(ValueError, match="exact submission"):
        operator._terminal_api(Jobs(), plan, submission)


def test_export_release_query_rejects_any_still_present_exact_uid(monkeypatch) -> None:
    controller = {
        "job_name": "exact-job",
        "job_uid": "11111111-1111-4111-8111-111111111111",
        "workload_name": "exact-workload",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "pods": [
            {
                "name": "exact-pod",
                "uid": "33333333-3333-4333-8333-333333333333",
            }
        ],
    }
    monkeypatch.setattr(
        operator,
        "_kubectl_json",
        lambda _argv: {"metadata": {"uid": operator.job.DEV_NAMESPACE_UID}},
    )
    observed: list[tuple[str, str, str]] = []

    def absent(resource: str, name: str, uid: str) -> bool:
        observed.append((resource, name, uid))
        return resource != "workloads.kueue.x-k8s.io"

    monkeypatch.setattr(operator, "_kubectl_exact_uid_absent", absent)
    with pytest.raises(ValueError, match="still present"):
        operator._export_absence(controller)
    assert observed == [
        ("jobs.batch", "exact-job", controller["job_uid"]),
        ("workloads.kueue.x-k8s.io", "exact-workload", controller["workload_uid"]),
    ]


def test_reload_release_query_rejects_any_still_present_exact_uid(monkeypatch) -> None:
    controller = {
        "rayjob_name": "exact-rayjob",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_name": "exact-workload",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "raycluster_name": "exact-cluster",
        "raycluster_uid": "33333333-3333-4333-8333-333333333333",
        "pods": [
            {
                "name": "exact-pod",
                "uid": "44444444-4444-4444-8444-444444444444",
            }
        ],
    }
    monkeypatch.setattr(
        operator,
        "_kubectl_json",
        lambda _argv: {"metadata": {"uid": operator.job.DEV_NAMESPACE_UID}},
    )
    observed: list[tuple[str, str, str]] = []

    def absent(resource: str, name: str, uid: str) -> bool:
        observed.append((resource, name, uid))
        return resource != "rayclusters.ray.io"

    monkeypatch.setattr(operator, "_kubectl_exact_uid_absent", absent)
    with pytest.raises(ValueError, match="still present"):
        operator._reload_absence(controller)
    assert observed == [
        ("rayjobs.ray.io", "exact-rayjob", controller["rayjob_uid"]),
        ("workloads.kueue.x-k8s.io", "exact-workload", controller["workload_uid"]),
        ("rayclusters.ray.io", "exact-cluster", controller["raycluster_uid"]),
    ]


@pytest.mark.parametrize(
    "schema",
    ("cyber_miles_hf_export_job_plan_v1", "cyber_miles_hf_export_job_plan_v2"),
)
def test_generic_submit_cannot_bypass_the_ttl_zero_watcher(monkeypatch, schema: str) -> None:
    plan = {"schema": schema}
    monkeypatch.setattr(cli, "_prepared", lambda _directory: (plan, {}))
    monkeypatch.setattr(
        cli,
        "_client",
        lambda _cluster: pytest.fail("generic submit must not contact the Jobs API"),
    )

    result = CliRunner().invoke(cli.app, ["submit", "/unused"])

    assert result.exit_code != 0
    assert "pre-POST TTL-zero watcher" in result.output


@pytest.mark.parametrize(
    "schema",
    ("cyber_miles_hf_export_job_plan_v1", "cyber_miles_hf_export_job_plan_v2"),
)
def test_generic_prepared_reader_reopens_direct_job_without_jobs_api_validator(
    tmp_path: Path, monkeypatch, schema: str
) -> None:
    plan = {"schema": schema, "stage": "export"}
    request = {"apiVersion": "batch/v1", "kind": "Job"}
    values = {
        tmp_path / "plan.json": plan,
        tmp_path / "request.json": request,
        tmp_path / "PREPARED.json": {
            "plan_sha256": operator.job.digest(plan),
            "request_sha256": operator.job.digest(request),
        },
    }
    monkeypatch.setattr(cli, "_read", lambda path: values[path])
    monkeypatch.setattr(operator.job, "validate_plan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator.job, "job_request", lambda _plan: request)
    monkeypatch.setattr(
        cli,
        "validate_request",
        lambda _request: pytest.fail("direct batch manifests are not Jobs API requests"),
    )

    assert cli._prepared(tmp_path) == (plan, request)
