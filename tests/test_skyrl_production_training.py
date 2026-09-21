"""Offline contracts for the five full Qwen3.8 SkyRL arms."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError, digest
from scripts import audit_qwen38_skyrl_launch_readiness as readiness
from scripts import prepare_qwen38_skyrl_production_queue as queue
from training import (
    rl_runtime,
    sft_runtime,
    skyrl_launch_guard,
    skyrl_production,
    skyrl_production_training,
)


def _artifacts() -> dict[Path, dict]:
    return queue.expected()


def test_exact_plans_preserve_science_shape_and_final_test_exclusion() -> None:
    artifacts = _artifacts()
    qualification = artifacts[queue.QUALIFICATION]
    assert qualification["fixed_controls"] == skyrl_production.FIXED_CONTROLS
    assert qualification["fixed_controls"]["reward"] == ("authoritative Fleet partial cyber score")
    split = artifacts[queue.SPLIT]
    final_test = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] == "test"
    }
    assert len(final_test) == 10
    for arm in queue.ARMS:
        plan = artifacts[queue.path_for("plan", arm)]
        compiled, request = queue.compile_arm(artifacts, arm)
        preview = artifacts[queue.path_for("preview", arm)]
        observer = artifacts[queue.path_for("observer", arm)]
        assert plan == compiled
        assert plan["schema"] == skyrl_production_training.SCHEMA
        assert plan["model"]["revision"] == skyrl_production.MODEL["revision"]
        assert "sha256:" + digest(plan["model"]) == (skyrl_production.MODEL_INVENTORY_SHA256)
        assert plan["data"] == artifacts[queue.path_for("manifest", arm)]
        assert set(plan["data"]["files"]) == {"train", "dev"}
        assert request["workers"] == 1
        assert request["gpus_per_worker"] == 8
        assert request["priority_class"] == "c1"
        assert request["requeueIfPreempted"] is False
        assert request["image"] == skyrl_production.IMAGE
        assert preview == skyrl_production.offline_preview(plan, request)
        assert preview["server_preview_requested"] is False
        assert preview["external_reads"] == preview["external_mutations"] == 0
        assert observer == skyrl_production.release_observer_contract(plan, request)
        assert observer["status"] == "contract_only_not_armed"
        assert observer["expected_gpus"] == 8
        assert observer["bind_after_jobs_response"] == [
            "jobs_run_name",
            "jobs_run_id",
            "rayjob_uid",
            "raycluster_uid",
            "workload_uid",
            "pod_uids",
            "runtime_image_ids",
        ]
    selected = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] in {"train", "dev"}
    }
    assert not selected & final_test


def test_exact_plan_binding_rejects_model_and_data_drift() -> None:
    artifacts = _artifacts()
    plan = copy.deepcopy(artifacts[queue.path_for("plan", queue.ARMS[0])])
    plan["model"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="plan binding changed"):
        skyrl_production_training.job_request(plan)

    plan = copy.deepcopy(artifacts[queue.path_for("plan", queue.ARMS[0])])
    plan["data"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="closure digest mismatch|plan binding changed"):
        skyrl_production_training.job_request(plan)


def test_qualification_rejects_resealed_reward_drift() -> None:
    qualification = copy.deepcopy(_artifacts()[queue.QUALIFICATION])
    qualification["fixed_controls"]["reward"] = "different reward"
    qualification["sha256"] = "sha256:" + digest(
        {key: value for key, value in qualification.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="controls changed"):
        skyrl_production.validate_qualification(qualification)


def test_plan_watchdogs_cover_legal_episode_ceiling_without_shortening_episodes() -> None:
    artifacts = _artifacts()
    for arm in queue.ARMS:
        plan = artifacts[queue.path_for("plan", arm)]
        watchdog = plan["watchdog"]
        expected_ceiling = 38400 if arm["steps"] == 10 else 163200
        assert plan["data"]["limits"]["episode_seconds"] == 2400
        assert watchdog["episode_ceiling_seconds"] == expected_ceiling
        assert watchdog["hard_seconds"] >= (
            expected_ceiling + watchdog["startup_seconds"] + watchdog["drain_seconds"]
        )


def test_production_runtime_applies_and_restores_only_its_plan_watchdog(monkeypatch) -> None:
    artifacts = _artifacts()
    plan = artifacts[queue.path_for("plan", queue.ARMS[-1])]
    previous = {
        name: getattr(sft_runtime, name)
        for name in (
            "WATCHDOG_POLL_SECONDS",
            "WATCHDOG_STARTUP_SECONDS",
            "WATCHDOG_IDLE_SECONDS",
            "WATCHDOG_HARD_SECONDS",
            "WATCHDOG_DRAIN_SECONDS",
        )
    }
    captured = {}

    def fake_run(received, path, *, backend):
        assert received is plan
        assert path == Path("plan.json")
        assert backend is skyrl_production_training
        captured.update({name: getattr(sft_runtime, name) for name in previous})
        return {"status": "not_started"}

    monkeypatch.setattr(rl_runtime, "run", fake_run)
    assert skyrl_production_training.run(plan, Path("plan.json")) == {"status": "not_started"}
    assert captured["WATCHDOG_HARD_SECONDS"] == 172800
    assert captured["WATCHDOG_STARTUP_SECONDS"] == 1800
    assert {name: getattr(sft_runtime, name) for name in previous} == previous


class _FakeJobs:
    def __init__(self, rows=None):
        self.rows = rows or []

    def all_runs(self):
        return copy.deepcopy(self.rows)


def _armed(plan: dict, request: dict, now: float) -> tuple[dict, dict]:
    contract = skyrl_production.release_observer_contract(plan, request)
    value = {
        "schema": skyrl_launch_guard.ARMED_SCHEMA,
        "status": "armed",
        "contract_sha256": contract["sha256"],
        "plan_sha256": contract["plan_sha256"],
        "request_sha256": contract["request_sha256"],
        "kubernetes_context": contract["kubernetes_context"],
        "namespace": contract["namespace"],
        "expected_gpus": 8,
        "observer_pid": os.getpid(),
        "armed_at": datetime.fromtimestamp(now, UTC).isoformat().replace("+00:00", "Z"),
    }
    value["sha256"] = "sha256:" + digest(value)
    return contract, value


def test_fresh_guard_checks_jobs_kubernetes_sfs_wandb_and_observer(monkeypatch) -> None:
    artifacts = _artifacts()
    plan, request = queue.compile_arm(artifacts, queue.ARMS[0])
    current = 1_800_000_000.0
    contract, armed = _armed(plan, request, current)
    monkeypatch.setattr(
        skyrl_production,
        "validate_staged_data",
        lambda received: {
            "sha256": "sha256:" + "a" * 64,
            "output_absent": received is plan,
        },
    )

    def runner(*args, **kwargs):
        assert args[0][:5] == [
            "kubectl",
            "--context",
            skyrl_production.PROD_CONTEXT,
            "--namespace",
            skyrl_production.NAMESPACE,
        ]
        return subprocess.CompletedProcess(args[0], 0, json.dumps({"items": []}), "")

    proof = skyrl_launch_guard.fresh_absence_checks(
        plan,
        request,
        _FakeJobs(),
        contract,
        armed,
        runner=runner,
        wandb_exists=lambda entity, project, run_id: False,
        now=lambda: current,
    )
    assert proof["status"] == "passed"
    assert proof["jobs_history_rows_checked"] == 0
    assert proof["kubernetes_objects_checked"] == 0
    assert proof["sfs_output_absent"] is True
    assert proof["wandb_run_id_absent"] is True


def test_prod8_uses_same_fresh_caller_surfaces_without_changing_its_plan(monkeypatch) -> None:
    next_gates = readiness.load(readiness.NEXT_GATES)
    run = readiness.load(readiness.CANARY_RUN)
    plan, request = readiness.compile_prod8(run, readiness.prod8_metadata(run, next_gates))
    current = 1_800_000_000.0
    monkeypatch.setattr(
        skyrl_launch_guard,
        "_canary_sfs_absent",
        lambda received: {
            "manifest_sha256": received["data"]["sha256"],
            "payloads_checked": 2,
            "output_absent": True,
        },
    )

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps({"items": []}), "")

    proof = skyrl_launch_guard.fresh_canary_absence_checks(
        plan,
        request,
        _FakeJobs(),
        runner=runner,
        wandb_exists=lambda *args: False,
        now=lambda: current,
    )
    assert proof["schema"] == "cyber_skyrl_reward_canary_fresh_absence_v1"
    assert proof["sfs"]["output_absent"] is True
    assert proof["wandb_run_id_absent"] is True


@pytest.mark.parametrize("duplicate", ["jobs", "kubernetes", "wandb"])
def test_fresh_guard_rejects_each_duplicate_surface(monkeypatch, duplicate: str) -> None:
    artifacts = _artifacts()
    plan, request = queue.compile_arm(artifacts, queue.ARMS[0])
    current = 1_800_000_000.0
    contract, armed = _armed(plan, request, current)
    monkeypatch.setattr(
        skyrl_production,
        "validate_staged_data",
        lambda plan: {"sha256": "sha256:" + "a" * 64, "output_absent": True},
    )
    rows = [{"name": request["name"] + "-1234abcd", "run_dir": request["run_dir"]}]
    items = []
    if duplicate == "kubernetes":
        items = [{"metadata": {"name": request["name"] + "-1234abcd"}}]

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps({"items": items}), "")

    with pytest.raises(JobsError):
        skyrl_launch_guard.fresh_absence_checks(
            plan,
            request,
            _FakeJobs(rows if duplicate == "jobs" else []),
            contract,
            armed,
            runner=runner,
            wandb_exists=lambda *args: duplicate == "wandb",
            now=lambda: current,
        )
