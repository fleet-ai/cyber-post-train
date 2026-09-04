from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import opencode_train_sweep_runner as legacy


def _drain_request(plan: dict, *, job_uid: str = "job-uid", pod_uid: str = "pod-uid") -> dict:
    request = {
        "schema_version": legacy.DRAIN_REQUEST_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "target_job_uid": job_uid,
        "target_pod_uid": pod_uid,
        "reason": "protocol_change",
    }
    request["request_sha256"] = self_hosted.digest_without(request, "request_sha256")
    return request


def test_hosted_uid_bound_drain_closes_task_and_attempt_claim_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "sha256:plan", "campaign_id": "campaign-v1"}
    request = _drain_request(plan)
    (tmp_path / "DRAIN-REQUEST.json").write_text(json.dumps(request))
    (tmp_path / "task-claims").mkdir()
    (tmp_path / "claims").mkdir()
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "pod-uid")

    paths = [
        tmp_path / "task-claims" / "rank-001.json",
        tmp_path / "claims" / "run-1.json",
    ]
    for path in paths:
        with pytest.raises(legacy.DrainRequested) as raised:
            hosted._claim_or_raise_drained(plan, tmp_path, path, {"claim": True})
        assert raised.value.request["request_sha256"] == request["request_sha256"]
        assert not path.exists()

    with pytest.raises(legacy.DrainRequested):
        hosted._claim_task_and_first_attempt_or_raise_drained(
            plan,
            tmp_path,
            paths[0],
            {"task": True},
            paths[1],
            {"attempt": True},
        )
    assert not any(path.exists() for path in paths)


def _accepted_drain_root(tmp_path: Path) -> tuple[dict, Path, dict]:
    plan = {
        "campaign_id": "campaign-v1",
        "tasks": [
            {
                "rank": 1,
                "source_rank": 11,
                "task": {"key": "task-1", "version_id": "version-1"},
            }
        ],
        "attempts": [
            {
                "run_id": "run-1",
                "rank": 1,
                "source_rank": 11,
                "attempt": 1,
                "network": "network-1",
            }
        ],
    }
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    root = tmp_path / "campaign"
    (root / "claims").mkdir(parents=True)
    attempt = root / "attempts" / "run-1"
    attempt.mkdir(parents=True)
    (root / "PLAN.json").write_text(json.dumps(plan))
    claim = {
        "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "run_id": "run-1",
        "rank": 1,
        "source_rank": 11,
        "attempt": 1,
        "network": "network-1",
        "task_key": "task-1",
        "task_version_id": "version-1",
        "config_sha256": "sha256:config",
    }
    claim["claim_sha256"] = self_hosted.digest_without(claim, "claim_sha256")
    (root / "claims" / "run-1.json").write_text(json.dumps(claim))
    accepted = {
        "schema_version": "fleet-hosted-opencode-attempt-accepted-v1",
        "accepted": True,
        "credited": True,
        "run_id": "run-1",
        "rank": 1,
        "source_rank": 11,
        "attempt": 1,
        "task_key": "task-1",
        "task_version_id": "version-1",
        "config_sha256": "sha256:config",
        "claim_sha256": claim["claim_sha256"],
    }
    accepted["receipt_sha256"] = self_hosted.digest_without(accepted, "receipt_sha256")
    (attempt / "ACCEPTED.json").write_text(json.dumps(accepted))
    return plan, root, accepted


def test_hosted_drained_receipt_requires_exact_claim_acceptance_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, root, accepted = _accepted_drain_root(tmp_path)
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "pod-uid")
    request = _drain_request(plan)

    receipt = hosted._write_hosted_drained_unlocked(plan, root, request)
    assert receipt["claimed_attempts"] == receipt["accepted_attempts"] == 1
    assert receipt["claim_acceptance_identity_exact"] is True
    assert receipt["completed_claim_receipts"] == [
        {
            "run_id": "run-1",
            "rank": 1,
            "source_rank": 11,
            "attempt": 1,
            "claim_sha256": accepted["claim_sha256"],
            "acceptance_receipt_sha256": accepted["receipt_sha256"],
        }
    ]

    (root / "DRAINED.json").unlink()
    accepted["source_rank"] = 12
    accepted["receipt_sha256"] = self_hosted.digest_without(accepted, "receipt_sha256")
    (root / "attempts" / "run-1" / "ACCEPTED.json").write_text(json.dumps(accepted))
    with pytest.raises(RuntimeError, match="not bound to its exact claim"):
        hosted._write_hosted_drained_unlocked(plan, root, request)


def test_hosted_scheduler_submits_no_task_after_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "campaign_id": "campaign-v1",
        "plan_sha256": "sha256:plan",
        "task_count": 2,
        "tasks": [
            {
                "rank": rank,
                "source_rank": 10 + rank,
                "task": {"key": f"task-{rank}", "version_id": f"version-{rank}"},
            }
            for rank in (1, 2)
        ],
        "attempts": [],
        "credited_sessions": [],
    }
    for name in (
        "validate_plan",
        "validate_dedicated_scoring_release",
        "validate_hosted_replacement_scoring_release",
        "validate_qwen_http500_scoring_release",
        "validate_qwen_post_partial_tail_release",
        "validate_glm_http500_scoring_release",
        "validate_glm_dedicated_b_v5_scoring_release",
        "validate_glm_dedicated_a_v5_scoring_release",
        "validate_completed_exit1_gap_scoring_release",
        "validate_dedicated_a_completed_exit1_gap_release",
    ):
        monkeypatch.setattr(hosted, name, lambda *_args: None)
    monkeypatch.setattr(hosted, "_validate_plan_identity_absence", lambda *_args: 0)
    monkeypatch.setattr(hosted, "_validate_inventory_for_task", lambda *_args: 0)
    monkeypatch.setattr(
        self_hosted,
        "_request",
        lambda *_args, **_kwargs: {
            "team_name": "fleet",
            "team_id": self_hosted.FLEET_TEAM_ID,
        },
    )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setenv("FLEET_API_KEY", "test")
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "pod-uid")
    seen: list[int] = []

    def run_task(_plan, task, _items, root, *_args):
        seen.append(int(task["rank"]))
        request = _drain_request(plan)
        (root / "DRAIN-REQUEST.json").write_text(json.dumps(request))
        raise legacy.DrainRequested(request)

    monkeypatch.setattr(hosted, "_run_task", run_task)
    result = hosted.run_plan(plan, tmp_path / "ledger", tmp_path / "proxy.py", {})
    assert seen == [1]
    assert result["schema_version"] == "fleet-hosted-opencode-drained-v1"
    assert result["claimed_attempts"] == result["accepted_attempts"] == 0
    assert not (tmp_path / "ledger" / "task-claims" / "rank-002.json").exists()


def test_endpoint_stream_leases_are_atomic_and_reusable(tmp_path: Path) -> None:
    lease1 = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key="hosted-qwen", maximum_streams=2
    )
    lease2 = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key="hosted-qwen", maximum_streams=2
    )
    assert {lease1.slot, lease2.slot} == {1, 2}
    with pytest.raises(RuntimeError, match="cap is already full"):
        endpoint_lease.acquire_endpoint_lease(
            lease_root=tmp_path, endpoint_key="hosted-qwen", maximum_streams=2
        )
    lease1.close()
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key="hosted-qwen", maximum_streams=2
    ) as replacement:
        assert replacement.slot == 1
    lease2.close()
