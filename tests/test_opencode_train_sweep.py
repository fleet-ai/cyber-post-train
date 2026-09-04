from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from evals.fleet import (
    opencode_train_sweep,
    opencode_train_sweep_drain,
    opencode_train_sweep_runner,
    self_hosted,
)


def _source_state(plan: dict, accepted: int) -> dict:
    value = {
        "schema_version": "fleet-selfhosted-opencode-v3-fenced-state-v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "job_uid": "job-uid",
        "pod_uid": "pod-uid",
        "quiesced": True,
        "accepted_attempts": accepted,
        "unresolved_attempts": 0,
        "attempt_directories": accepted,
        "model_rollouts_started": accepted,
        "scored_sessions_created": accepted,
        "sfs_root_created": bool(accepted),
        "original_artifacts_preserved": True,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_duplicate_preflight_normalizes_persisted_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "sessions": [{"model": "qwen3.8-27b"}],
            "has_more": False,
        }

    monkeypatch.setattr(self_hosted, "_request", fake_request)
    with pytest.raises(RuntimeError, match="already exist"):
        opencode_train_sweep.duplicate_preflight(
            httpx.Client(),
            [{"task_key": "task-a", "task_version_id": "version-a"}],
            "fleet-cluster-opencode-1.18.27/qwen3.8-27b",
        )
    assert calls == 1


def test_full_plan_validator_requires_exact_pass4_arithmetic() -> None:
    plan = {
        "schema_version": opencode_train_sweep_runner.PLAN_SCHEMA,
        "task_count": 50,
        "pass_k": 4,
        "tasks": [{}] * 50,
        "attempts": [{}] * 199,
        "execution": {
            "max_concurrent": 1,
            "required_task_tools": ["bash", "submit_report"],
        },
    }
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    opencode_train_sweep_runner.validate_plan(plan)
    broken = json.loads(json.dumps(plan))
    broken["attempts"].pop()
    broken["plan_sha256"] = self_hosted.digest_without(broken, "plan_sha256")
    with pytest.raises(ValueError, match="attempt count"):
        opencode_train_sweep_runner.validate_plan(broken)


def test_infrastructure_successor_rekeys_every_attempt() -> None:
    plan = {
        "schema_version": opencode_train_sweep_runner.PLAN_SCHEMA,
        "created_at": "2026-09-03T00:00:00Z",
        "campaign_id": "chris-example-v1",
        "task_count": 50,
        "pass_k": 4,
        "tasks": [{}] * 50,
        "attempts": [
            {
                "ordinal": index,
                "run_id": f"chris-example-v1-r{index:03d}",
                "network": f"example-v1-r{index:03d}",
            }
            for index in range(1, 200)
        ],
        "execution": {
            "max_concurrent": 1,
            "required_task_tools": ["bash", "submit_report"],
        },
    }
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    successor = opencode_train_sweep.build_infrastructure_successor(
        plan,
        generation="v2",
        reason="dind_missing_shared_bind_mounts_pre_agent",
    )
    assert successor["campaign_id"] == "chris-example-v2"
    assert successor["supersedes"]["plan_sha256"] == plan["plan_sha256"]
    assert successor["supersedes"]["scored_sessions_created"] == 0
    assert all("-v2-" in row["run_id"] for row in successor["attempts"])
    assert all("-v2-" in row["network"] for row in successor["attempts"])
    opencode_train_sweep_runner.validate_plan(successor)


def test_full_cluster_jobs_share_bind_mount_sources_with_dind() -> None:
    manifest = (
        __import__("pathlib").Path(__file__).parents[1]
        / "evals/fleet/cluster/opencode-train-sweep-full-jobs.yaml"
    ).read_text()
    documents = manifest.split("\n---\n")
    assert len(documents) == 2
    for document in documents:
        dind = document.split("containers:", 1)[0]
        assert "{name: workspace, mountPath: /workspace}" in dind
        assert "{name: sfs, mountPath: /mnt/sfs}" in dind


def test_full_cluster_plan_keys_match_bootstrap_and_runtime() -> None:
    manifest = (
        __import__("pathlib").Path(__file__).parents[1]
        / "evals/fleet/cluster/opencode-train-sweep-full-jobs.yaml"
    ).read_text()
    submitter = (
        __import__("pathlib").Path(__file__).parents[1]
        / "evals/fleet/scripts/submit_opencode_train_sweep_full.sh"
    ).read_text()
    for model in ("qwen", "glm"):
        key = f"{model}-plan-v4.json"
        assert f"plan_key={key}" in submitter
        assert f'/bootstrap/{key} "$root/evals/fleet/configs/{key}"' in manifest
        assert f"value: {key}" in manifest
    assert '--from-file="$plan_key=$plan_path"' in submitter
    assert "requests: {cpu: 750m, memory: 2Gi" in manifest
    assert "requests: {cpu: 250m, memory: 1Gi" in manifest
    assert (
        "--parallel"
        in (
            Path(__file__).parents[1] / "evals/fleet/scripts/run_opencode_train_sweep_full.sh"
        ).read_text()
    )


def test_parallel_successor_preserves_exact_cartesian_cells() -> None:
    path = Path(__file__).parents[1] / "evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"
    original = json.loads(path.read_text())
    source_attempt = original["attempts"][0]
    receipt = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-accepted-v1",
        "accepted": True,
        "run_id": source_attempt["run_id"],
        "session_id": "session-v3-r1-a2",
        "verifier_execution_id": "verifier-v3-r1-a2",
        "cleanup_completed": True,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    successor = opencode_train_sweep.build_parallel_successor(
        original,
        accepted_receipts=[receipt],
        source_state=_source_state(original, 1),
    )
    assert successor["campaign_id"].endswith("-v4")
    assert successor["new_session_count"] == 198
    assert successor["prior_accepted"][0]["attempt"] == 2
    assert all("-v4-" in row["network"] for row in successor["attempts"])
    opencode_train_sweep_runner.validate_parallel_plan(successor)

    broken = json.loads(json.dumps(successor))
    broken["attempts"][0]["attempt"] = 4
    broken["plan_sha256"] = self_hosted.digest_without(broken, "plan_sha256")
    with pytest.raises(ValueError, match="Cartesian"):
        opencode_train_sweep_runner.validate_parallel_plan(broken)

    broken_total = json.loads(json.dumps(successor))
    broken_total["total_session_count"] = 401
    broken_total["plan_sha256"] = self_hosted.digest_without(broken_total, "plan_sha256")
    with pytest.raises(ValueError, match="total-session"):
        opencode_train_sweep_runner.validate_parallel_plan(broken_total)


def test_parallel_task_groups_never_overlap_one_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    active_ranks: set[int] = set()
    lock = threading.Lock()

    def fake_attempt(_plan, _task, item, _root, _proxy, _key):
        nonlocal active, maximum
        rank = int(item["rank"])
        with lock:
            assert rank not in active_ranks
            active_ranks.add(rank)
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
            active_ranks.remove(rank)
        return {}

    monkeypatch.setattr(opencode_train_sweep_runner, "_run_parallel_attempt", fake_attempt)
    groups = [
        [{"rank": rank, "attempt": attempt} for attempt in (1, 2, 3, 4)] for rank in range(1, 9)
    ]
    opencode_train_sweep_runner._run_task_groups(
        {},
        {rank: {} for rank in range(1, 9)},
        groups,
        Path("/unused"),
        Path("/unused"),
        "unused",
        4,
    )
    assert maximum == 4


def test_parallel_canary_stops_new_claims_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[int] = []
    lock = threading.Lock()

    def fake_attempt(_plan, _task, item, _root, _proxy, _key):
        with lock:
            started.append(int(item["rank"]))
        if int(item["rank"]) == 1:
            raise RuntimeError("infrastructure failure")
        time.sleep(0.03)
        return {}

    monkeypatch.setattr(opencode_train_sweep_runner, "_run_parallel_attempt", fake_attempt)
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        opencode_train_sweep_runner._run_attempt_wave(
            {},
            {rank: {} for rank in range(1, 5)},
            [{"rank": rank} for rank in range(1, 5)],
            Path("/unused"),
            Path("/unused"),
            "unused",
            2,
        )
    assert set(started) == {1, 2}


def test_uid_bound_drain_closes_claim_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {"plan_sha256": "sha256:plan", "campaign_id": "campaign-v1"}
    request = {
        "schema_version": opencode_train_sweep_runner.DRAIN_REQUEST_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "target_job_uid": "job-uid",
        "target_pod_uid": "pod-uid",
        "reason": "protocol_change",
    }
    request["request_sha256"] = self_hosted.digest_without(request, "request_sha256")
    (tmp_path / "DRAIN-REQUEST.json").write_text(json.dumps(request))
    (tmp_path / "claims").mkdir()
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "pod-uid")

    claim_path = tmp_path / "claims" / "attempt.json"
    with pytest.raises(opencode_train_sweep_runner.DrainRequested) as raised:
        opencode_train_sweep_runner._claim_or_raise_drained(
            plan, tmp_path, claim_path, {"claim": True}
        )

    assert raised.value.request["request_sha256"] == request["request_sha256"]
    assert not claim_path.exists()


def test_drain_rejects_stale_pod_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {"plan_sha256": "sha256:plan", "campaign_id": "campaign-v1"}
    request = {
        "schema_version": opencode_train_sweep_runner.DRAIN_REQUEST_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "target_job_uid": "job-uid",
        "target_pod_uid": "old-pod-uid",
        "reason": "protocol_change",
    }
    request["request_sha256"] = self_hosted.digest_without(request, "request_sha256")
    (tmp_path / "DRAIN-REQUEST.json").write_text(json.dumps(request))
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "new-pod-uid")

    with pytest.raises(RuntimeError, match="different Job or Pod"):
        opencode_train_sweep_runner._load_drain_request(plan, tmp_path)


def test_drain_writer_serializes_request_with_claim_gate(tmp_path: Path) -> None:
    plan = {"campaign_id": "campaign-v1", "tasks": [], "attempts": []}
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "PLAN.json").write_text(json.dumps(plan))

    request = opencode_train_sweep_drain.request_drain(
        plan=plan,
        root=root,
        target_job_uid="job-uid",
        target_pod_uid="pod-uid",
        reason="protocol_change",
    )

    assert request["request_sha256"] == self_hosted.digest_without(request, "request_sha256")
    assert json.loads((root / "DRAIN-REQUEST.json").read_text()) == request
    with pytest.raises(FileExistsError):
        opencode_train_sweep_drain.request_drain(
            plan=plan,
            root=root,
            target_job_uid="job-uid",
            target_pod_uid="pod-uid",
            reason="protocol_change",
        )


def test_terminal_acceptance_yields_to_existing_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"campaign_id": "campaign-v1"}
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    (tmp_path / "claims").mkdir()
    (tmp_path / "attempts").mkdir()
    monkeypatch.setenv("JOB_UID", "job-uid")
    monkeypatch.setenv("POD_UID", "pod-uid")
    request = {
        "schema_version": opencode_train_sweep_runner.DRAIN_REQUEST_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "target_job_uid": "job-uid",
        "target_pod_uid": "pod-uid",
        "reason": "protocol_change",
    }
    request["request_sha256"] = self_hosted.digest_without(request, "request_sha256")
    (tmp_path / "DRAIN-REQUEST.json").write_text(json.dumps(request))
    final = {"accepted": True, "receipt_sha256": "sha256:final"}

    result = opencode_train_sweep_runner._write_accepted_or_drained(plan, tmp_path, final)

    assert result["schema_version"] == opencode_train_sweep_runner.DRAINED_SCHEMA
    assert result["completed_claim_receipts"] == []
    assert (tmp_path / "DRAINED.json").is_file()
    assert not (tmp_path / "ACCEPTED.json").exists()


def test_attempt_wave_drain_waits_for_already_started_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed: list[int] = []
    request = {"request_sha256": "sha256:drain"}

    def fake_attempt(_plan, _task, item, _root, _proxy, _key):
        rank = int(item["rank"])
        if rank == 2:
            raise opencode_train_sweep_runner.DrainRequested(request)
        time.sleep(0.03)
        completed.append(rank)
        return {}

    monkeypatch.setattr(opencode_train_sweep_runner, "_run_parallel_attempt", fake_attempt)
    observed = opencode_train_sweep_runner._run_attempt_wave(
        {},
        {1: {}, 2: {}, 3: {}},
        [{"rank": 1}, {"rank": 2}, {"rank": 3}],
        Path("/unused"),
        Path("/unused"),
        "unused",
        2,
    )

    assert observed == request
    assert completed == [1]
