import json
import os
import uuid
from datetime import UTC, datetime

import pytest

from training import dev_cleanup_observer as cleanup
from training import miles96_mechanics_launch as launch
from training import miles96_signal_qualification as signal
from training import miles_signal_transition as transition
from training import miles_signal_wave


def _signal_plan() -> dict:
    return {
        "schema": signal.SCHEMA,
        "qualification": {"samples": 8},
        "episode": {
            "max_concurrent_envs": 2,
            "ready_timeout_s": 600,
            "episode_timeout_s": 2400,
            "grade_timeout_s": 900,
            "request_timeout_s": 120,
        },
    }


def _mechanics_plan() -> dict:
    return {
        "schema": launch.mechanics.SCHEMA,
        "identity": {"reload_name": "chris-q38-m96-reload-test-a1"},
        "optimization": {"prompt_groups": 1, "samples_per_prompt": 8},
        "episode": _signal_plan()["episode"],
    }


def _signal_plan_and_request() -> tuple[dict, dict]:
    wave = miles_signal_wave.load()
    row = wave["candidates"][0]
    plan = signal.build_plan(
        name=row["identity"]["name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
        task_binding={
            **miles_signal_wave.task_binding(wave, row),
            "authority_receipt_sha256": row["authority_receipt_sha256"],
        },
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=row["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )
    return plan, signal.job_request(plan)


def _stub_review() -> tuple[dict, dict, dict, dict, dict, float]:
    now = datetime.now(UTC)
    candidate = {"candidate": True}
    review = {"sha256": "sha256:" + "1" * 64}
    live = {"live": True}
    bundle = {"bundle": True}
    approved = {
        "approved_at": transition._stamp(now),
        "sha256": "sha256:" + "2" * 64,
    }
    return approved, candidate, review, live, bundle, now.timestamp()


def test_signal_active_deadline_covers_four_bounded_serial_waves() -> None:
    expected = 1800 + 4 * (600 + 2400 + 900 + 2 * 120 + 60) + 300
    assert launch._maximum_seconds(_signal_plan(), {}) == expected == 18900


def test_operator_capacity_ceiling_is_ten_active_nodes() -> None:
    assert (launch.PROJECT_MAX_NODES, launch.PROJECT_MAX_GPUS) == (10, 80)


def test_mechanics_active_deadline_adds_update_checkpoint_and_export_grace() -> None:
    expected = 1800 + 4 * (600 + 2400 + 900 + 2 * 120 + 60) + 1800 + 1800 + 3600 + 300
    plan = _mechanics_plan()
    assert launch._maximum_seconds(plan, {"name": "train"}) == expected == 26100
    assert launch._maximum_seconds(plan, {"name": plan["identity"]["reload_name"]}) == 7200


def test_signal_post_requires_exact_fresh_reviewed_transition(monkeypatch) -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, bundle, now = _stub_review()

    def approve(*_args, **kwargs):
        if kwargs["expected_parent_review_sha256"] != review["sha256"]:
            raise ValueError("wrong parent review pin")
        return approved

    def validate_live(_receipt, **kwargs):
        assert kwargs == {
            "now": datetime.fromtimestamp(now, UTC),
            "require_fresh": True,
        }
        return live

    monkeypatch.setattr(transition, "approve_review_candidate", approve)
    monkeypatch.setattr(transition, "validate_live_task_receipt", validate_live)
    monkeypatch.setattr(
        transition, "validate_post_receipt_bundle", lambda *_args, **_kwargs: bundle
    )
    assert (
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            post_receipt_bundle=bundle,
            expected_parent_review_sha256=review["sha256"],
            expected_reviewed_transition_sha256=approved["sha256"],
            now=now,
        )
        == approved
    )
    with pytest.raises(launch.JobsError, match="requires"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=None,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            post_receipt_bundle=bundle,
            expected_parent_review_sha256=review["sha256"],
            expected_reviewed_transition_sha256=approved["sha256"],
            now=now,
        )
    with pytest.raises(launch.JobsError, match="invalid or stale"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            post_receipt_bundle=bundle,
            expected_parent_review_sha256="sha256:" + "0" * 64,
            expected_reviewed_transition_sha256=approved["sha256"],
            now=now,
        )


def test_signal_post_rejects_transition_for_a_different_request(monkeypatch) -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, bundle, now = _stub_review()
    original = request

    def approve(_candidate, _review, _live, _bundle, _plan, checked_request, **_kwargs):
        if checked_request != original:
            raise ValueError("different request")
        return approved

    monkeypatch.setattr(transition, "approve_review_candidate", approve)
    monkeypatch.setattr(transition, "validate_live_task_receipt", lambda *_args, **_kwargs: live)
    monkeypatch.setattr(
        transition, "validate_post_receipt_bundle", lambda *_args, **_kwargs: bundle
    )
    request = {**request, "title": request["title"] + " drift"}
    with pytest.raises(launch.JobsError, match="invalid or stale"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            post_receipt_bundle=bundle,
            expected_parent_review_sha256=review["sha256"],
            expected_reviewed_transition_sha256=approved["sha256"],
            now=now,
        )


def test_signal_post_rejects_unpinned_successor(monkeypatch) -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, bundle, now = _stub_review()
    monkeypatch.setattr(transition, "approve_review_candidate", lambda *_args, **_kwargs: approved)
    monkeypatch.setattr(transition, "validate_live_task_receipt", lambda *_args, **_kwargs: live)
    monkeypatch.setattr(
        transition, "validate_post_receipt_bundle", lambda *_args, **_kwargs: bundle
    )
    with pytest.raises(launch.JobsError, match="predecessor"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            post_receipt_bundle=bundle,
            expected_parent_review_sha256=review["sha256"],
            expected_reviewed_transition_sha256="sha256:" + "0" * 64,
            now=now,
        )


def test_signal_submit_fails_before_preview_without_review(tmp_path) -> None:
    plan, request = _signal_plan_and_request()

    class Client:
        def preview(self, _request):
            raise AssertionError("unreviewed signal request reached preview")

    with pytest.raises(launch.JobsError, match="requires"):
        launch.submit_once(plan, request, Client(), tmp_path / "launch")
    assert not (tmp_path / "launch").exists()


def test_signal_submit_reuses_prepared_bundle_and_posts_once(monkeypatch, tmp_path) -> None:
    plan, request = _signal_plan_and_request()
    directory = tmp_path / "prepared"
    directory.mkdir()
    bundle = {
        "server_preview": {"sha256": "sha256:" + "1" * 64},
        "fresh_absence": {"sha256": "sha256:" + "2" * 64},
    }
    bundle_checks = []
    transition_checks = []

    def check_bundle(_plan, _request, _bundle, checked_directory, **kwargs):
        assert checked_directory == directory
        bundle_checks.append(kwargs.get("expect_post_intent", False))
        assert (directory / "SUBMISSION.jsonl").exists() is bundle_checks[-1]
        return bundle

    monkeypatch.setattr(launch, "validate_signal_post_bundle", check_bundle)
    monkeypatch.setattr(
        launch,
        "validate_reviewed_transition",
        lambda *_args, **kwargs: transition_checks.append(kwargs["now"]) or {"launchable": True},
    )
    monkeypatch.setattr(
        launch,
        "_prove_sfs_output_absent",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "3" * 64},
    )
    monkeypatch.setattr(
        launch,
        "capacity_gate",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "4" * 64},
    )
    monkeypatch.setattr(
        launch,
        "final_prepost_checks",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "5" * 64},
    )
    monkeypatch.setattr(
        launch,
        "validate_armed_observer",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "6" * 64},
    )

    class Client:
        calls = []

        def preview(self, _request):
            raise AssertionError("reviewed signal submit regenerated its preview")

        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {
                "name": request["name"] + "-1234abcd",
                "job_id": str(uuid.UUID("33333333-3333-4333-8333-333333333333")),
                "run_dir": request["run_dir"],
                "status": "QUEUED",
            }

    client = Client()
    result = launch.submit_once(
        plan,
        request,
        client,
        directory,
        now=lambda: 100.0,
        start_observer=lambda _directory: (_ for _ in ()).throw(
            AssertionError("reviewed signal submit started a second observer")
        ),
        reviewed_transition={"approved": True},
        transition_candidate={"candidate": True},
        parent_review={"review": True},
        live_task_receipt={"live": True},
        post_receipt_bundle=bundle,
        expected_parent_review_sha256="sha256:" + "7" * 64,
        expected_reviewed_transition_sha256="sha256:" + "8" * 64,
    )
    assert result["job_id"] == "33333333-3333-4333-8333-333333333333"
    assert bundle_checks == [False, True]
    assert transition_checks == [100.0, 100.0]
    assert client.calls == [("POST", "/v1/runs", {"json": request})]
    rows = [json.loads(line) for line in (directory / "SUBMISSION.jsonl").read_text().splitlines()]
    assert [row["state"] for row in rows] == ["POST_INTENT_DO_NOT_RETRY", "POST_RESPONSE"]


def test_final_prepost_gate_rechecks_observer_jobs_and_kubernetes(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        launch,
        "validate_armed_observer",
        lambda *_args, **_kwargs: calls.append("observer") or {"sha256": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        launch,
        "_jobs_absent",
        lambda *_args, **_kwargs: calls.append("jobs") or 1038,
    )
    monkeypatch.setattr(
        launch,
        "_kubernetes_absent",
        lambda *_args, **_kwargs: calls.append("kubernetes") or 27,
    )
    times = iter([100.0, 101.0])
    proof = launch.final_prepost_checks(
        {"identity": {"name": "test"}},
        {"name": "test"},
        object(),
        tmp_path,
        now=lambda: next(times),
    )
    assert calls == ["observer", "jobs", "kubernetes"]
    assert proof["status"] == "passed"
    assert proof["jobs_history_rows_checked"] == 1038
    assert proof["kubernetes_objects_checked"] == 27


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generated_name_pattern", "^different-[a-f0-9]{8}$"),
        ("manifest_sha256", "sha256:" + "b" * 64),
        ("maximum_seconds", 7199),
    ],
)
def test_armed_observer_receipt_binds_preview_pattern_and_deadline(tmp_path, field, value) -> None:
    plan = {
        "schema": "mechanics",
        "execution": {
            "kubernetes_context": cleanup.PROD_CONTEXT,
            "namespace": cleanup.NAMESPACE,
        },
    }
    request = {
        "name": "chris-q38-m96-test-a1",
        "run_dir": "/mnt/sfs/jobs/chris-q38-m96-test-a1",
        "image": "registry/image@sha256:" + "a" * 64,
        "workers": 1,
        "gpus_per_worker": 8,
    }
    preview = launch._seal(
        {
            "schema": "cyber_miles96_live_server_preview_v1",
            "plan_sha256": "sha256:" + launch.mechanics.digest(plan),
            "request_sha256": "sha256:" + launch.digest(request),
            "manifest_sha256": "sha256:" + "a" * 64,
            "root_failure_alerts": "off",
            "backoff_limit": 0,
            "shutdown_after_job_finishes": True,
            "nodes": 1,
            "gpus": 8,
            "observed_at_unix": 100.0,
            "preview_count": 2,
            "priority_class": "c1",
            "queue_priority": "q1",
            "requeue_if_preempted": False,
        }
    )
    (tmp_path / "SERVER_PREVIEW.json").write_text(json.dumps(preview))
    body = {
        "schema": cleanup.JOBS_API_PREFIX_GUARD_SCHEMA,
        "status": "armed_non_destructive_prefix_guard",
        "context": cleanup.PROD_CONTEXT,
        "namespace": cleanup.NAMESPACE,
        "run_name_prefix": request["name"],
        "generated_name_pattern": "^" + launch.re.escape(request["name"]) + "-[a-f0-9]{8}$",
        "run_dir": request["run_dir"],
        "image": request["image"],
        "plan_sha256": "sha256:" + launch.mechanics.digest(plan),
        "manifest_sha256": preview["manifest_sha256"],
        "maximum_seconds": 7200,
        "expected_gpus": 8,
        "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observer_pid": os.getpid(),
        "prefix_collision_count_before_post": 0,
    }
    body[field] = value
    (tmp_path / "OBSERVER_ARMED.json").write_text(json.dumps(launch._seal(body)))
    with pytest.raises(launch.JobsError, match="differs"):
        launch.validate_armed_observer(
            plan,
            request,
            tmp_path,
            now=datetime.now(UTC).timestamp(),
        )


class _History:
    rows = []

    def __init__(self, token, *, base_url):
        assert token == "secret" and base_url == "https://jobs.example"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def all_runs(self):
        return self.rows


def _request() -> dict:
    return {
        "name": "chris-q38-m96-sig-test-a1",
        "title": "Miles96 signal test",
        "run_dir": "/mnt/sfs/jobs/chris-q38-m96-sig-test-a1",
        "image": "registry/image@sha256:" + "a" * 64,
        "priority_class": "c1",
        "requeueIfPreempted": False,
    }


def test_uncertain_post_reconciles_one_exact_authenticated_history_row(
    tmp_path, monkeypatch
) -> None:
    request = _request()
    job_id = str(uuid.UUID("33333333-3333-4333-8333-333333333333"))
    _History.rows = [
        {
            **request,
            "name": request["name"] + "-1234abcd",
            "job_id": job_id,
            "status": "QUEUED",
        }
    ]
    monkeypatch.setenv("FLEET_API_KEY", "secret")
    monkeypatch.setattr(launch, "Jobs", _History)
    journal = tmp_path / "SUBMISSION.jsonl"
    journal.write_text(
        json.dumps(
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "request_sha256": launch.digest(request),
            }
        )
        + "\n"
    )
    result = launch._reconcile_post_response(
        {"execution": {"jobs_api_base_url": "https://jobs.example"}},
        request,
        journal,
    )
    assert result["state"] == "POST_RESPONSE_RECONCILED"
    assert result["job_id"] == job_id
    assert len(journal.read_text().splitlines()) == 2


def test_uncertain_post_reconciliation_polls_until_history_is_consistent(
    tmp_path, monkeypatch
) -> None:
    request = _request()
    job_id = str(uuid.UUID("33333333-3333-4333-8333-333333333333"))

    class DelayedHistory(_History):
        calls = 0

        def all_runs(self):
            self.__class__.calls += 1
            if self.calls == 1:
                return []
            return [
                {
                    **request,
                    "name": request["name"] + "-1234abcd",
                    "job_id": job_id,
                    "status": "QUEUED",
                }
            ]

    monkeypatch.setenv("FLEET_API_KEY", "secret")
    monkeypatch.setattr(launch, "Jobs", DelayedHistory)
    journal = tmp_path / "SUBMISSION.jsonl"
    journal.write_text(
        json.dumps(
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "request_sha256": launch.digest(request),
            }
        )
        + "\n"
    )
    result = launch._reconcile_post_response(
        {"execution": {"jobs_api_base_url": "https://jobs.example"}},
        request,
        journal,
        deadline=launch.time.monotonic() + 1,
        sleep=lambda _seconds: None,
    )
    assert result["job_id"] == job_id
    assert DelayedHistory.calls == 2


@pytest.mark.parametrize("count", [0, 2])
def test_uncertain_post_reconciliation_fails_closed_unless_unique(
    tmp_path, monkeypatch, count
) -> None:
    request = _request()
    _History.rows = [
        {
            **request,
            "name": request["name"] + f"-{index + 1:08x}",
            "job_id": str(uuid.UUID(int=index + 1)),
            "status": "QUEUED",
        }
        for index in range(count)
    ]
    monkeypatch.setenv("FLEET_API_KEY", "secret")
    monkeypatch.setattr(launch, "Jobs", _History)
    journal = tmp_path / "SUBMISSION.jsonl"
    journal.write_text(
        json.dumps(
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "request_sha256": launch.digest(request),
            }
        )
        + "\n"
    )
    with pytest.raises(cleanup.ObserverError, match="exactly one|multiple"):
        launch._reconcile_post_response(
            {"execution": {"jobs_api_base_url": "https://jobs.example"}},
            request,
            journal,
        )
    assert len(journal.read_text().splitlines()) == 1
