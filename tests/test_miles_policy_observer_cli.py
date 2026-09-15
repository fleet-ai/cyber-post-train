"""Operator-rail tests only; no Jobs API, Kubernetes, Ray, or GPU calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import API_URLS
from training import miles_policy_observer_cli as cli


def _object(
    kind: str,
    name: str,
    uid: str,
    *,
    owner: tuple[str, str, str] | None = None,
    run_id: str | None = None,
    status: dict | None = None,
    projectable_pod: bool = False,
) -> dict:
    metadata = {
        "name": name,
        "uid": uid,
        "namespace": "fleet-train-jobs",
        "resourceVersion": "1",
        "creationTimestamp": "2026-09-13T03:07:01Z",
        "ownerReferences": [],
    }
    if owner:
        metadata["ownerReferences"] = [
            {"kind": owner[0], "name": owner[1], "uid": owner[2], "controller": True}
        ]
    if run_id:
        metadata["labels"] = {"fleet.ai/run-id": run_id}
    spec = {}
    value_status = status or {}
    if kind == "Pod":
        spec = {"containers": [{"name": "main", "resources": {"limits": {}}}]}
        if projectable_pod:
            value_status = {
                "phase": "Succeeded",
                "containerStatuses": [
                    {
                        "name": "main",
                        "state": {
                            "terminated": {"exitCode": 0, "reason": "Completed"}
                        },
                    }
                ],
            }
    return {
        "type": "MODIFIED",
        "object": {
            "kind": kind,
            "metadata": metadata,
            "spec": spec,
            "status": value_status,
        },
    }


def test_watch_buffer_replays_owner_chain_and_drops_unready_pod(monkeypatch, tmp_path):
    run_id = "11111111-1111-4111-8111-111111111111"
    run_name = "observer-11111111"
    rayjob_uid = "22222222-2222-4222-8222-222222222222"
    cluster_uid = "33333333-3333-4333-8333-333333333333"
    cluster_name = "cluster"
    owner = ("RayCluster", cluster_name, cluster_uid)
    values = [
        _object("Pod", "pod", "44444444-4444-4444-8444-444444444444", owner=owner),
        _object(
            "Pod",
            "pod",
            "44444444-4444-4444-8444-444444444444",
            owner=owner,
            projectable_pod=True,
        ),
        _object(
            "Workload",
            "workload",
            "55555555-5555-4555-8555-555555555555",
            owner=("RayJob", run_name, rayjob_uid),
        ),
        _object(
            "RayCluster",
            cluster_name,
            cluster_uid,
            owner=("RayJob", run_name, rayjob_uid),
        ),
        _object(
            "RayJob",
            run_name,
            rayjob_uid,
            run_id=run_id,
            status={"jobStatus": "SUCCEEDED"},
        ),
    ]
    watcher = cli.WatchBuffer()
    for index, value in enumerate(values):
        watcher.events.put((1000.0 + index, value))
    recorded = []
    recorded_at = iter((2000.0, 2001.0, 2002.0, 2003.0))
    monkeypatch.setattr(cli.time, "time", lambda: next(recorded_at))
    monkeypatch.setattr(
        cli.events,
        "record_event",
        lambda directory, raw, **kwargs: recorded.append((raw["object"]["kind"], kwargs)),
    )

    watcher.record(
        tmp_path,
        {"api": {"run_id": run_id, "run_name": run_name}},
        timeout=2,
    )

    assert [kind for kind, _ in recorded] == ["RayJob", "Workload", "RayCluster", "Pod"]
    assert [row[1]["sequence"] for row in recorded] == list(range(4))
    assert [row[1]["observed_at"] for row in recorded] == [
        2000.0,
        2001.0,
        2002.0,
        2003.0,
    ]


def test_watch_buffer_ignores_unrelated_identity_and_stops_on_dev_failure(monkeypatch):
    run_id = "11111111-1111-4111-8111-111111111111"
    run_name = "observer-11111111"
    watcher = cli.WatchBuffer()
    watcher.events.put(
        (
            1.0,
            _object(
                "RayJob",
                run_name,
                "22222222-2222-4222-8222-222222222222",
                run_id="99999999-9999-4999-8999-999999999999",
                status={"jobStatus": "SUCCEEDED"},
            ),
        )
    )
    watcher.events.put(
        (
            2.0,
            _object(
                "RayJob",
                run_name,
                "33333333-3333-4333-8333-333333333333",
                run_id=run_id,
                status={"jobStatus": "FAILED"},
            ),
        )
    )
    recorded = []
    monkeypatch.setattr(
        cli.events,
        "record_event",
        lambda *args, **kwargs: recorded.append(args[1]),
    )

    with pytest.raises(RuntimeError, match="terminal failure on dev"):
        watcher.record(
            Path("/unused"),
            {"api": {"run_id": run_id, "run_name": run_name}},
            timeout=2,
        )
    assert len(recorded) == 1
    assert recorded[0]["object"]["metadata"]["labels"]["fleet.ai/run-id"] == run_id


def test_submit_opens_watch_before_exactly_one_post(monkeypatch, tmp_path):
    calls = []
    plan, request = {"plan": True}, {"request": True}
    args = argparse.Namespace(
        plan=tmp_path / "plan.json",
        request=tmp_path / "request.json",
        journal=tmp_path / "journal.jsonl",
        output=tmp_path / "submission.json",
        watch_directory=tmp_path / "watch",
        source_commit="a" * 40,
        watch_timeout_seconds=1800,
    )
    monkeypatch.setattr(cli, "_preflight_submission", lambda value: (plan, request))
    monkeypatch.setattr(
        cli.observer,
        "start_capture_intent",
        lambda *args, **kwargs: calls.append("intent"),
    )
    monkeypatch.setattr(
        cli.observer,
        "compile_submission_binding",
        lambda **kwargs: {"api": {"run_id": "id", "run_name": "name"}},
    )
    monkeypatch.setattr(
        cli.observer,
        "start_capture",
        lambda *args, **kwargs: calls.append("bind"),
    )

    class Watcher:
        def start(self):
            calls.append("watch")

        def record(self, *args, **kwargs):
            calls.append("record")

        def close(self):
            calls.append("close")

    class Jobs:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit_once(self, value, journal):
            calls.append("post")

    monkeypatch.setattr(cli, "WatchBuffer", Watcher)
    monkeypatch.setattr(cli, "Jobs", Jobs)

    cli.submit(args)

    assert calls == ["intent", "watch", "post", "bind", "record", "close"]


def test_api_down_after_lost_post_recovers_exact_run_without_a_second_post(
    tmp_path, monkeypatch
):
    request = {
        "name": "miles-observer",
        "run_dir": "/mnt/sfs/jobs/chris-q38-miles-reload-dev3",
    }
    journal = tmp_path / "submission.jsonl"
    run_id = "11111111-1111-4111-8111-111111111111"
    run_name = request["name"] + "-" + run_id[:8]
    calls: list[object] = []

    class LostResponseJobs:
        list_attempts = 0

        def submit_once(self, observed, path):
            calls.append("post")
            path.write_text(
                json.dumps(
                    {
                        "state": "POST_INTENT_DO_NOT_RETRY",
                            "api_base_url": API_URLS["dev"],
                        "request_sha256": cli.digest(observed),
                    }
                )
                + "\n"
            )
            raise ValueError("response lost after server accepted create")

        def all_runs(self):
            self.list_attempts += 1
            calls.append("list-down" if self.list_attempts == 1 else "list")
            if self.list_attempts == 1:
                raise ValueError("temporary API outage")
            return [{"name": run_name, "run_dir": request["run_dir"]}]

        def status(self, name):
            calls.append(("status", name))
            return {
                "name": run_name,
                "job_id": run_id,
                "run_dir": request["run_dir"],
                "status": "RUNNING",
                "created_at": "2026-09-12T12:00:00Z",
                "finished_at": None,
            }

    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    recovered = cli.submit_once_or_reconcile(LostResponseJobs(), request, journal)

    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert calls == ["post", "list-down", "list", ("status", run_name)]
    assert recovered == {key: value for key, value in rows[1].items() if key != "state"}
    assert [row["state"] for row in rows] == [
        "POST_INTENT_DO_NOT_RETRY",
        "POST_RESPONSE",
    ]


def test_lost_post_response_rejects_name_match_with_different_output(tmp_path):
    request = {"name": "miles-observer", "run_dir": "/mnt/sfs/jobs/exact"}
    journal = tmp_path / "submission.jsonl"
    journal.write_text(
        json.dumps(
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                    "api_base_url": API_URLS["dev"],
                "request_sha256": cli.digest(request),
            }
        )
        + "\n"
    )

    class WrongOutputJobs:
        def all_runs(self):
            return [
                {
                    "name": "miles-observer-11111111",
                    "run_dir": "/mnt/sfs/jobs/different",
                }
            ]

    with pytest.raises(TimeoutError, match="not visible"):
        cli._recover_created_run(
            WrongOutputJobs(),
            request,
            journal,
            timeout_seconds=0,
        )
    assert len(journal.read_text().splitlines()) == 1


def test_final_recovery_timeout_preserves_sanitized_possible_leak(tmp_path):
    request = {
        "name": "miles-observer",
        "run_dir": "/mnt/sfs/jobs/chris-q38-miles-reload-dev3",
    }
    journal = tmp_path / "submission.jsonl"
    journal.write_text(
        json.dumps(
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                    "api_base_url": API_URLS["dev"],
                "request_sha256": cli.digest(request),
            }
        )
        + "\n"
    )
    run_name = "miles-observer-11111111"
    rayjob_uid = "22222222-2222-4222-8222-222222222222"
    watcher = cli.WatchBuffer()
    candidate = _object(
        "RayJob",
        run_name,
        rayjob_uid,
        run_id="11111111-1111-4111-8111-111111111111",
        status={"jobStatus": "RUNNING", "private": "must-not-be-saved"},
    )
    candidate["object"]["spec"] = {
        "runtimeEnvYAML": "API_TOKEN=must-not-be-saved"
    }
    watcher.events.put((1.0, candidate))

    receipt = watcher.preserve_ambiguous(
        tmp_path / "watch",
        request,
        journal,
        TimeoutError("private API error"),
    )

    saved = json.loads((tmp_path / "watch" / cli.AMBIGUOUS_SUBMISSION).read_text())
    assert saved == receipt
    assert saved["status"] == "possible_active_resource_leak"
    assert saved["candidate_events"][0]["name"] == run_name
    assert saved["failure_type"] == "TimeoutError"
    assert "must-not-be-saved" not in json.dumps(saved)


def test_preflight_finishes_source_checks_before_creating_outputs(monkeypatch, tmp_path):
    plan_path, request_path = tmp_path / "plan.json", tmp_path / "request.json"
    plan_path.write_text(json.dumps({"plan": True}))
    request_path.write_text(json.dumps({"request": True}))
    args = SimpleNamespace(
        plan=plan_path,
        request=request_path,
        journal=tmp_path / "journal.jsonl",
        output=tmp_path / "submission.json",
        watch_directory=tmp_path / "watch",
        source_commit="a" * 40,
        watch_timeout_seconds=1800,
    )
    calls = []
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setattr(
        cli.observer,
        "_validate_plan",
        lambda *args, **kwargs: calls.append("plan"),
    )
    monkeypatch.setattr(
        cli.observer,
        "_request_projection",
        lambda *args, **kwargs: calls.append("source"),
    )

    assert cli._preflight_submission(args) == ({"plan": True}, {"request": True})
    assert calls == ["plan", "source"]
    assert not any(path.exists() for path in (args.journal, args.output, args.watch_directory))

    args.journal.write_text("occupied")
    with pytest.raises(FileExistsError, match="create-once"):
        cli._preflight_submission(args)
