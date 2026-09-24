import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

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


def _reviewed_artifacts(plan: dict, request: dict) -> tuple[dict, dict, dict, dict, float]:
    wave = miles_signal_wave.load()
    live = json.loads(
        Path(
            "configs/qualification/qwen38-miles-signal-live-task-receipt-20260924.json"
        ).read_text()
    )
    observed = transition._time(live["observed_at"])

    def sha(value: str) -> str:
        return "sha256:" + launch.digest(value)

    lanes = []
    for index, row in enumerate(wave["candidates"]):
        row_plan = "sha256:" + launch.mechanics.digest(plan) if index == 0 else sha(f"p{index}")
        row_request = "sha256:" + launch.digest(request) if index == 0 else sha(f"r{index}")
        lanes.append(
            {
                "name": row["identity"]["name"],
                "plan_sha256": row_plan,
                "request_sha256": row_request,
                "optimizer_steps": 0,
                "checkpoint": False,
                "authority_config_sha256": wave["sha256"],
                "live_binding_receipt_sha256": row["live_binding_receipt_sha256"],
                "preview": {
                    "request_sha256": row_request,
                    "receipt_sha256": sha(f"preview{index}"),
                    "observed_at": transition._stamp(observed),
                    "root_annotations": {"fleet.ai/failure-alerts": "off"},
                    "priority_class": "c1",
                    "queue_priority": "q1",
                    "backoff_limit": 0,
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "requeue_if_preempted": False,
                    "optimizer_steps": 0,
                    "checkpoint": False,
                },
                "absence": {
                    "receipt_sha256": sha(f"absence{index}"),
                    "observed_at": transition._stamp(observed),
                    "jobs_api_duplicates": 0,
                    "kubernetes_duplicates": 0,
                    "sfs_output_absent": True,
                },
                "observer": {
                    "receipt_sha256": sha(f"observer{index}"),
                    "armed": True,
                    "healthy": True,
                    "uid_bound_release": True,
                    "delete_drain_supervised": True,
                    "active_deadline_s": 18_900,
                },
            }
        )
    evidence = {
        "adapter": {
            "commit": "1" * 40,
            "clean": True,
            "source_closure_sha256": "sha256:" + launch.mechanics.digest(plan["runtime_sources"]),
            "runtime_image": request["image"],
            "tests_receipt_sha256": sha("tests"),
            "exact_image_preflight_sha256": sha("image"),
            "sample_indexes": list(range(8)),
            "max_concurrent_episodes": 2,
            "optimizer_steps": 0,
            "checkpoint": False,
            "live_session_open_tool_schema_gate": True,
            "tool_catalog_sha256": wave["authorities"]["tool_catalog"]["self_sha256"],
            "current_binding_source": "live_binding_receipt_sha256",
            "no_outer_retry_or_replacement": True,
            "all_slots_terminally_accounted": True,
            "unique_verifier_execution_ids": True,
            "exact_cleanup_required": True,
        },
        "operator": {
            "commit": "2" * 40,
            "clean": True,
            "tests_receipt_sha256": sha("operator"),
            "source_manifest": launch.operator_source_manifest(),
            "source_closure_sha256": "sha256:" + launch.digest(launch.operator_source_manifest()),
        },
        "lanes": lanes,
        "cleanup": {
            "receipt_sha256": sha("cleanup"),
            "prompt_removed_after_terminal": True,
            "private_episode_material_restricted": True,
            "output_create_once": True,
        },
    }
    candidate = transition.build_review_candidate(live, evidence, observed_at=observed)
    review = transition._sealed(
        {
            "schema": transition.REVIEW_SCHEMA,
            "candidate_sha256": candidate["sha256"],
            "approved": True,
            "reviewer": "root",
            "reviewed_at": transition._stamp(observed),
        }
    )
    approved = transition.approve_review_candidate(candidate, review, live, reviewed_at=observed)
    return approved, candidate, review, live, observed.timestamp()


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


def test_signal_post_requires_exact_fresh_reviewed_transition() -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, now = _reviewed_artifacts(plan, request)
    assert (
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
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
            now=now,
        )
    with pytest.raises(launch.JobsError, match="stale"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            now=now + 301,
        )


def test_signal_post_rejects_transition_for_a_different_request() -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, now = _reviewed_artifacts(plan, request)
    request = {**request, "title": request["title"] + " drift"}
    with pytest.raises(launch.JobsError, match="exact request"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
            now=now,
        )


def test_signal_post_rejects_operator_source_drift(monkeypatch) -> None:
    plan, request = _signal_plan_and_request()
    approved, candidate, review, live, now = _reviewed_artifacts(plan, request)
    monkeypatch.setattr(
        launch,
        "operator_source_manifest",
        lambda: {"training/miles96_mechanics_launch.py": "sha256:" + "0" * 64},
    )
    with pytest.raises(launch.JobsError, match="exact request"):
        launch.validate_reviewed_transition(
            plan,
            request,
            approved=approved,
            candidate=candidate,
            parent_review=review,
            live_task_receipt=live,
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
