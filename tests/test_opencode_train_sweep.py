from __future__ import annotations

import json

import httpx
import pytest

from evals.fleet import opencode_train_sweep, opencode_train_sweep_runner, self_hosted


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
        key = f"{model}-plan-v3.json"
        assert f"--from-file={key}=" in submitter
        assert f'/bootstrap/{key} "$root/evals/fleet/configs/{key}"' in manifest
        assert f"value: {key}" in manifest
